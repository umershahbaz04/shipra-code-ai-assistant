"""Run before deployment. Use --fast when model downloads are unavailable."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent / "data"
chunks = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
assert len(chunks) == len(metadata) > 0, "chunks/metadata mismatch"
assert all(m.get("file_path") for m in metadata), "metadata without paths"
assert not any("[REDACTED]" not in c and "GROQ_API_KEY=" in c for c in chunks), "possible indexed secret"
import faiss
index = faiss.read_index(str(root / "faiss.index"))
assert index.ntotal == len(chunks), "FAISS count mismatch"
if "--fast" in sys.argv:
    # Dependency-free regression: validates that the expected code exists in
    # this exact snapshot without downloading embedding/reranker models.
    corpus = "\n".join(chunks).lower()
    for required in ("syncpolicies", "createsalechannelconfig", "deleteinventorysyncpolicy"):
        assert required in corpus, f"missing expected source identifier: {required}"
    print("PASS: fast source/index regression checks")
    raise SystemExit(0)

from rag_engine import ShipraRag
engine = ShipraRag(root)
tests = {
    "sync policies kon c hain": "syncpolic",
    "sale channel connect flow": "salechannel",
    "delete inventory sync policy": "inventorysyncpolicy",
}
only = next((value.split("=", 1)[1] for value in sys.argv if value.startswith("--only=")), None)
for question, expected in tests.items():
    if only and only != expected:
        continue
    paths = " ".join(e.meta.get("file_path", "").lower() for e in engine.retrieve(question))
    assert expected in paths, f"retrieval regression: {question} -> {paths[:400]}"
    print("PASS", question)
print("PASS: index and regression checks")
