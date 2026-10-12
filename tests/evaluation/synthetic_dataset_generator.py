"""
synthetic_dataset_generator.py

Generates a synthetic corpus of financial/regulatory documents (contracts,
SEC filings, compliance manuals, loan agreements, insurance policies,
procurement contracts) PLUS a golden question/answer set with known-correct
chunk references, which is what every downstream eval script depends on.

IMPORTANT HONESTY NOTE:
This generates *synthetic* documents from templates with randomized clauses,
not real filings. That's deliberate: it gives us ground-truth answers (we
know exactly which chunk contains the answer to each generated question,
because we generated both from the same template slot), which is what makes
Recall@K / Precision@K / MRR mathematically meaningful without needing
expensive human annotation. This is a standard technique for building
IR/RAG eval harnesses when you don't have a labeled production dataset yet.

Usage:
    python synthetic_dataset_generator.py --n-docs 60 --n-questions 180
"""
from __future__ import annotations
import argparse
import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

random.seed(42)  # reproducibility is the whole point of this framework

DOC_TYPES = ["contract", "sec_filing", "compliance_manual", "loan_agreement",
             "insurance_policy", "procurement_contract"]

COMPANIES = ["Meridian Capital", "Northbridge Holdings", "Cascade Financial",
             "Sterling & Vance LLP", "Harbor Point Insurance", "Vantage Logistics",
             "Ashford Partners", "Beacon Regulatory Group", "Falcon Trust Co",
             "Ironwood Ventures"]

JURISDICTIONS = ["Delaware", "New York", "California", "Texas", "Illinois"]

# Each "clause template" has a slot type, a fill-in generator, and a question
# template whose answer is exactly the filled value. This is what gives us
# ground truth: we know precisely which chunk/document holds the answer.
CLAUSE_TEMPLATES = [
    {
        "clause_type": "indemnification_cap",
        "text": "The Indemnifying Party's aggregate liability under this Section shall not exceed {value}, "
                "except in cases of gross negligence or willful misconduct.",
        "gen_value": lambda: f"${random.choice([500_000, 1_000_000, 2_500_000, 5_000_000]):,}",
        "question": "What is the indemnification liability cap in the {doc_ref}?",
    },
    {
        "clause_type": "auto_renewal",
        "text": "This Agreement shall automatically renew for successive {value} periods unless either "
                "party provides written notice of non-renewal at least 60 days prior to the renewal date.",
        "gen_value": lambda: random.choice(["12-month", "24-month", "36-month"]),
        "question": "What is the auto-renewal term specified in the {doc_ref}?",
    },
    {
        "clause_type": "governing_law",
        "text": "This Agreement shall be governed by and construed in accordance with the laws of the "
                "State of {value}, without regard to its conflict of laws principles.",
        "gen_value": lambda: random.choice(JURISDICTIONS),
        "question": "Which state's law governs the {doc_ref}?",
    },
    {
        "clause_type": "data_residency",
        "text": "All Customer Data processed under this Agreement shall be stored and processed exclusively "
                "within {value}, and shall not be transferred outside this region without prior written consent.",
        "gen_value": lambda: random.choice(["the United States", "the European Economic Area",
                                             "the United Kingdom", "Canada"]),
        "question": "Where must Customer Data be stored under the {doc_ref}?",
    },
    {
        "clause_type": "termination_notice",
        "text": "Either party may terminate this Agreement for convenience upon {value} written notice to the "
                "other party.",
        "gen_value": lambda: random.choice(["30 days'", "60 days'", "90 days'"]),
        "question": "How much notice is required to terminate the {doc_ref} for convenience?",
    },
    {
        "clause_type": "interest_rate",
        "text": "The outstanding principal balance shall accrue interest at a rate of {value} per annum, "
                "calculated on a 360-day basis.",
        "gen_value": lambda: f"{random.choice([4.25, 5.5, 6.75, 7.9, 8.25])}%",
        "question": "What annual interest rate applies to the outstanding balance in the {doc_ref}?",
    },
    {
        "clause_type": "confidentiality_term",
        "text": "The confidentiality obligations set forth herein shall survive termination of this Agreement "
                "for a period of {value}.",
        "gen_value": lambda: random.choice(["3 years", "5 years", "7 years", "10 years"]),
        "question": "How long do confidentiality obligations survive termination in the {doc_ref}?",
    },
    {
        "clause_type": "insurance_deductible",
        "text": "The Policyholder shall be responsible for a deductible of {value} per claim before coverage "
                "under this Policy applies.",
        "gen_value": lambda: f"${random.choice([1_000, 2_500, 5_000, 10_000]):,}",
        "question": "What is the per-claim deductible in the {doc_ref}?",
    },
    {
        "clause_type": "audit_rights",
        "text": "The Company reserves the right to audit Vendor's compliance with this Agreement upon "
                "{value} prior notice, no more than twice per calendar year.",
        "gen_value": lambda: random.choice(["10 business days'", "15 business days'", "30 days'"]),
        "question": "What notice period applies to audit rights under the {doc_ref}?",
    },
    {
        "clause_type": "limitation_of_liability",
        "text": "In no event shall either party's total liability arising out of this Agreement exceed the "
                "total fees paid in the {value} preceding the claim.",
        "gen_value": lambda: random.choice(["12 months", "6 months", "24 months"]),
        "question": "Over what preceding period is the liability cap calculated in the {doc_ref}?",
    },
]

FILLER_PARAGRAPHS = [
    "The parties acknowledge that they have read and understood the terms of this Agreement and agree "
    "to be bound by its provisions. Each party represents that it has full authority to enter into this Agreement.",
    "Nothing in this Agreement shall be construed to create a partnership, joint venture, or agency "
    "relationship between the parties. Each party remains an independent contractor at all times.",
    "This Agreement, together with all exhibits and schedules attached hereto, constitutes the entire "
    "agreement between the parties and supersedes all prior negotiations, representations, or agreements.",
    "Any notice required under this Agreement shall be in writing and delivered by certified mail, "
    "overnight courier, or email with confirmation of receipt to the addresses set forth herein.",
    "If any provision of this Agreement is held to be invalid or unenforceable, the remaining provisions "
    "shall continue in full force and effect, and the invalid provision shall be modified to the minimum extent necessary.",
]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    doc_type: str
    page_number: int
    chunk_index: int
    text: str
    clause_type: str | None  # None for filler chunks


@dataclass
class GoldenQuestion:
    question_id: str
    question: str
    document_id: str
    relevant_chunk_ids: List[str]
    answer: str
    clause_type: str


def _make_document(doc_idx: int) -> Dict:
    doc_type = DOC_TYPES[doc_idx % len(DOC_TYPES)]
    company = random.choice(COMPANIES)
    counterparty = random.choice([c for c in COMPANIES if c != company])
    document_id = f"doc_{doc_idx:04d}"

    n_clauses = random.randint(4, 7)
    chosen_clauses = random.sample(CLAUSE_TEMPLATES, k=n_clauses)

    chunks: List[Chunk] = []
    questions: List[GoldenQuestion] = []

    page = 1
    chunk_idx = 0
    header = (f"{doc_type.replace('_', ' ').upper()}\n\nThis {doc_type.replace('_', ' ')} "
              f"(\"Agreement\") is entered into between {company} (\"Company\") and {counterparty} "
              f"(\"Counterparty\").\n\n")
    chunks.append(Chunk(f"{document_id}_c{chunk_idx:03d}", document_id, doc_type, page, chunk_idx, header, None))
    chunk_idx += 1

    doc_ref = f"{company} {doc_type.replace('_', ' ')}"

    for i, clause in enumerate(chosen_clauses):
        # occasional filler chunk between clauses, simulating realistic document noise
        if i > 0 and random.random() < 0.5:
            filler = random.choice(FILLER_PARAGRAPHS)
            chunks.append(Chunk(f"{document_id}_c{chunk_idx:03d}", document_id, doc_type, page,
                                 chunk_idx, filler, None))
            chunk_idx += 1
            if chunk_idx % 4 == 0:
                page += 1

        value = clause["gen_value"]()
        clause_text = clause["text"].format(value=value)
        cid = f"{document_id}_c{chunk_idx:03d}"
        chunks.append(Chunk(cid, document_id, doc_type, page, chunk_idx, clause_text, clause["clause_type"]))

        q_id = f"{document_id}_q{i:03d}"
        questions.append(GoldenQuestion(
            question_id=q_id,
            question=clause["question"].format(doc_ref=doc_ref),
            document_id=document_id,
            relevant_chunk_ids=[cid],
            answer=value,
            clause_type=clause["clause_type"],
        ))

        chunk_idx += 1
        if chunk_idx % 4 == 0:
            page += 1

    return {
        "document_id": document_id,
        "doc_type": doc_type,
        "company": company,
        "counterparty": counterparty,
        "page_count": page,
        "chunks": [asdict(c) for c in chunks],
        "questions": [asdict(q) for q in questions],
    }


def generate(n_docs: int, n_questions: int, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    all_questions = []
    documents_meta = []

    for i in range(n_docs):
        doc = _make_document(i)
        documents_meta.append({k: v for k, v in doc.items() if k not in ("chunks", "questions")})
        all_chunks.extend(doc["chunks"])
        all_questions.extend(doc["questions"])

    random.shuffle(all_questions)
    all_questions = all_questions[:n_questions] if n_questions < len(all_questions) else all_questions

    (out_dir / "documents.json").write_text(json.dumps(documents_meta, indent=2))
    (out_dir / "chunks.json").write_text(json.dumps(all_chunks, indent=2))
    (out_dir / "golden_questions.json").write_text(json.dumps(all_questions, indent=2))

    summary = {
        "n_documents": n_docs,
        "n_chunks": len(all_chunks),
        "n_questions": len(all_questions),
        "doc_type_distribution": {dt: sum(1 for d in documents_meta if d["doc_type"] == dt) for dt in DOC_TYPES},
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--n-questions", type=int, default=180)
    ap.add_argument("--out-dir", type=str, default="data")
    args = ap.parse_args()

    summary = generate(args.n_docs, args.n_questions, Path(args.out_dir))
    print(json.dumps(summary, indent=2))
