"""Retrieval, always scoped to (company, active version).

vector : embed the question, cosine search in the company's collection.
hybrid : vector + BM25 keyword ranking fused with Reciprocal Rank Fusion (the bonus improvement).
"""
import math
import re
from dataclasses import dataclass

STOP = set("a an the is are am be to of for on in at by do does can i my me we you it this that and or if with from what when who how will get".split())


def tokenize(text: str) -> list:
    toks = re.findall(r"[a-z0-9]+(?::[0-9]+)?", text.lower())
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if t not in STOP]


class BM25:
    """Okapi BM25 over a handful of chunks. ~20 lines, no dependency, easy to explain."""

    def __init__(self, docs: list, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(len(self.docs), 1) or 1.0
        n = len(self.docs)
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list:
        q = tokenize(query)
        out = []
        for d in self.docs:
            s = 0.0
            for t in q:
                f = d.count(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return out


@dataclass
class Retrieved:
    version: int | None
    hits: list  # ordered best first, each a dict with id/text/section/title/sim...
    best_sim: float  # best vector similarity (the "is there any evidence at all" signal)


class IsolationError(Exception):
    """A chunk from the wrong company/version came back. Fail closed."""


class Retriever:
    def __init__(self, store, embedder, mode: str, top_k: int):
        self._store, self._embedder, self._mode, self._k = store, embedder, mode, top_k
        self._bm25_cache = {}  # (company, version) -> (chunks, BM25); key includes version so updates never reuse stale data

    def _bm25(self, company, version):
        key = (company, version)
        if key not in self._bm25_cache:
            for k in [k for k in self._bm25_cache if k[0] == company]:
                del self._bm25_cache[k]
            chunks = self._store.get_chunks(company, version)
            self._bm25_cache[key] = (chunks, BM25([f"{c['heading']} {c['title']} {c['text']}" for c in chunks]))
        return self._bm25_cache[key]

    def search(self, company: str, question: str) -> Retrieved:
        active = self._store.active(company)
        if active is None:
            return Retrieved(None, [], 0.0)
        version = active[0]
        qvec = self._embedder.embed([question])[0]
        vec_hits = self._store.query(company, version, qvec, 10)
        best = max((h["sim"] for h in vec_hits), default=0.0)

        if self._mode == "hybrid" and vec_hits:
            chunks, bm = self._bm25(company, version)
            sc = bm.scores(question)
            lex = [c for s, c in sorted(zip(sc, chunks), key=lambda x: -x[0]) if s > 0][:10]
            fused, by_id = {}, {}
            for ranking in (vec_hits, lex):
                for rank, h in enumerate(ranking):
                    fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (60 + rank)
                    by_id[h["id"]] = {**by_id.get(h["id"], {}), **h}
            ordered = [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)]
            hits = ordered[: self._k]
            sim_by_id = {h["id"]: h["sim"] for h in vec_hits}
            for h in hits:
                h["sim"] = sim_by_id.get(h["id"], 0.0)
        else:
            hits = vec_hits[: self._k]

        for h in hits:  # defence in depth: never trust the filter alone
            if h.get("company") != company or h.get("version") != version:
                raise IsolationError(f"chunk {h.get('id')} does not belong to {company} v{version}")
        return Retrieved(version, hits, best)
