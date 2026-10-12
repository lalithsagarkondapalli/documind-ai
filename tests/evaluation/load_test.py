"""
load_test.py

Locust load test against a LIVE DocuMind API deployment. This is real,
runnable Locust code -- but it requires an actual running API endpoint
(local docker-compose or deployed), which this sandbox doesn't have. That's
an honest limitation: retrieval-engine benchmarks above can run fully
offline because they test the algorithm; load testing tests the deployed
SYSTEM (network, ASGI server, DB connection pool, etc.) and can't be
faked without one.

RUN AGAINST A LOCAL STACK:
    docker-compose up -d
    locust -f load_test.py --host http://localhost:8000 --headless \
        --users 100 --spawn-rate 10 --run-time 2m \
        --csv reports/load_100u

Then repeat for each concurrency tier requested (10/50/100/500/1000 users),
and feed the resulting CSVs into generate_load_report() below to produce the
comparison graphs. Locust already writes percentile latency, RPS, and
failure-rate CSVs natively -- no need to hand-roll that math.

Usage (staged run across all tiers, requires `--host` reachable):
    python load_test.py --host http://localhost:8000 --stages 10,50,100,500,1000 --duration 60
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from locust import HttpUser, task, between


class DocuMindUser(HttpUser):
    """Simulated reviewer: mixes document-scoped queries with review-queue polling."""
    wait_time = between(1, 3)

    @task(3)
    def query_document(self):
        self.client.post("/v1/query/", json={
            "question": "What is the indemnification liability cap?",
            "document_id": "doc_0001",
        }, name="/v1/query [doc-scoped]")

    @task(1)
    def poll_review_queue(self):
        self.client.get("/v1/reviews/queue", name="/v1/reviews/queue")

    @task(1)
    def get_document_status(self):
        self.client.get("/v1/documents/doc_0001/status", name="/v1/documents/{id}/status")


def run_staged(host: str, stages: list[int], duration_s: int, out_dir: Path) -> dict:
    """Runs Locust headless once per concurrency tier and collects the summary stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for users in stages:
        prefix = out_dir / f"load_{users}u"
        cmd = [
            "locust", "-f", __file__, "--host", host, "--headless",
            "--users", str(users), "--spawn-rate", str(max(1, users // 10)),
            "--run-time", f"{duration_s}s", "--csv", str(prefix), "--only-summary",
        ]
        print(f"Running load stage: {users} users -> {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=False)

        stats_file = Path(f"{prefix}_stats.csv")
        if stats_file.exists():
            with open(stats_file) as f:
                rows = list(csv.DictReader(f))
            agg = next((r for r in rows if r["Name"] == "Aggregated"), rows[-1] if rows else {})
            results[users] = {
                "requests": agg.get("Request Count"),
                "failures": agg.get("Failure Count"),
                "p50_ms": agg.get("50%"),
                "p95_ms": agg.get("95%"),
                "p99_ms": agg.get("99%"),
                "rps": agg.get("Requests/s"),
            }
        else:
            results[users] = {"status": "stats file not produced -- host likely unreachable"}
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, required=True)
    ap.add_argument("--stages", type=str, default="10,50,100,500,1000")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--out-dir", type=str, default="../reports/load")
    args = ap.parse_args()

    stages = [int(s) for s in args.stages.split(",")]
    results = run_staged(args.host, stages, args.duration, Path(args.out_dir))
    print(json.dumps(results, indent=2))
