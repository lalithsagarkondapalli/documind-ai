"""Shim: the evaluation harness scores the SAME retrieval code the API runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.retrieval import *  # noqa: F401,F403,E402
from app.services.retrieval import (  # noqa: F401,E402
    EMBEDDING_PROVIDER, Chunk, RetrievalEngine, bm25_score, embed, load_engine)
from app.services.retrieval import tokenize as _tokenize  # noqa: F401,E402
