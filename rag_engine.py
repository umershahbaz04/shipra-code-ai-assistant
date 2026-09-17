"""Evidence-first retrieval for the Shipra code assistant."""
from __future__ import annotations

import json, math, re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

STOP = {"a","an","and","are","can","do","for","from","how","in","is","it","of","on","or","the","this","to","what","when","where","which","with","you","your","kia","kya","ka","ki","ke","ko","mai","main","mujhe","mjhy","batao","btao","hain","hai","hy","kesy","kaise"}

def tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text) if len(x) > 1 and x.lower() not in STOP}

@dataclass
class Evidence:
    score: float
    chunk_id: int
    meta: dict
    text: str

class ShipraRag:
    def __init__(self, data_dir: str | Path):
        root = Path(data_dir)
        self.chunks = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(root / "faiss.index"))
        if not (len(self.chunks) == len(self.metadata) == self.index.ntotal):
            raise ValueError("Index mismatch: rebuild chunks, metadata and FAISS together.")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.doc_tokens = [tokens(f"{m.get('file_path','')} {m.get('symbol','')} {c}") for c, m in zip(self.chunks, self.metadata)]
        self.df = Counter(t for doc in self.doc_tokens for t in doc)
        self.by_file = defaultdict(list)
        for i, m in enumerate(self.metadata): self.by_file[m.get("file_path", "")].append(i)

    def retrieve(self, question: str, limit: int = 8) -> list[Evidence]:
        q = tokens(question)
        vector = self.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, dense = self.index.search(vector, min(80, self.index.ntotal))
        ranks = {int(i): rank for rank, i in enumerate(dense[0]) if i >= 0}
        candidates = set(ranks)
        lexical = []
        for i, doc in enumerate(self.doc_tokens):
            overlap = q & doc
            if overlap:
                score = sum(math.log((len(self.chunks)+1)/(self.df[t]+1)) for t in overlap)
                lexical.append((score, i))
        lexical.sort(reverse=True)
        candidates.update(i for _, i in lexical[:80])
        # Reciprocal-rank fusion: semantic wording + exact identifiers both matter.
        lexical_rank = {i: r for r, (_, i) in enumerate(lexical[:80])}
        fused = []
        for i in candidates:
            score = 1/(60+ranks.get(i, 1000)) + 1/(60+lexical_rank.get(i, 1000))
            path = self.metadata[i].get("file_path", "").lower()
            if any(t in path for t in q): score += .03
            fused.append((score, i))
        fused.sort(reverse=True)
        pool = fused[:35]
        rerank_scores = self.reranker.predict([(question, self.chunks[i]) for _, i in pool])
        ordered = sorted(zip(rerank_scores, pool), key=lambda x: float(x[0]), reverse=True)
        output, seen_files = [], set()
        for score, (_, i) in ordered:
            path = self.metadata[i].get("file_path", "")
            # Keep one strongest excerpt per file first; avoid a single huge file dominating.
            if path in seen_files and len(output) < 4: continue
            seen_files.add(path)
            output.append(Evidence(float(score), i, self.metadata[i], self.chunks[i]))
            if len(output) >= limit: break
        return output

    @staticmethod
    def context(evidence: list[Evidence]) -> str:
        blocks = []
        for n, item in enumerate(evidence, 1):
            m = item.meta
            blocks.append(f"[S{n}] FILE: {m.get('file_path')}\nLINES: {m.get('start_line')}-{m.get('end_line')}\nSYMBOL: {m.get('symbol')}\nCODE:\n{item.text[:4200]}")
        return "\n\n".join(blocks)
