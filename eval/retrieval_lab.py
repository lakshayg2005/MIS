"""Compare retrieval techniques on the hard benchmark (eval/retrieval_hard.jsonl), no answer generation involved.

    python -m eval.retrieval_lab                       # techniques that need no LLM
    python -m eval.retrieval_lab --llm                 # also query rewriting, HyDE, doc2query (needs the Groq key)
    python -m eval.retrieval_lab --only baseline,hybrid

Method: each technique gets a FRESH store built from the same three PDF policies. Every query is scored on whether the
labelled section(s) are in the top 3. Settings are chosen on the DEV half and reported on the TEST half so we do not fool ourselves.
Latency = retrieval stage only, cold (question cache cleared), in milliseconds.
"""
import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from eval import metrics as M

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "eval" / "retrieval_hard.jsonl"

# name -> options.  embedder: minilm (chroma runtime) | minilm-fast | bge-small ; mode: vector | hybrid ; rerank: bool ;
# chunking: section | sentence | heading ; transform: none | rewrite | hyde ; expand: doc2query
TECHNIQUES = {
    "baseline":         dict(),
    "hybrid":           dict(mode="hybrid"),
    "rerank":           dict(rerank=True),
    "hybrid+rerank":    dict(mode="hybrid", rerank=True),
    "chunk-sentence":   dict(chunking="sentence"),
    "chunk-heading":    dict(chunking="heading"),
    "embed-minilm-fast": dict(embedder="minilm-fast"),
    "embed-bge-small":  dict(embedder="bge-small"),
    "fast+rerank":      dict(embedder="minilm-fast", rerank=True),
    "fast+rerank(pool5)": dict(embedder="minilm-fast", rerank=True, pool=5),
    "fast+hybrid+rerank": dict(embedder="minilm-fast", mode="hybrid", rerank=True),
    "bge+hybrid":       dict(embedder="bge-small", mode="hybrid"),
    "bge+rerank":       dict(embedder="bge-small", rerank=True),
    "bge+hybrid+rerank": dict(embedder="bge-small", mode="hybrid", rerank=True),
    "rewrite-llm":      dict(transform="rewrite", llm=True),
    "hyde-llm":         dict(transform="hyde", llm=True),
    "doc2query-llm":    dict(expand=True, llm=True),
    "fast+doc2query":   dict(embedder="minilm-fast", expand=True, llm=True),
    "fast+rewrite":     dict(embedder="minilm-fast", transform="rewrite", llm=True),
    "fast+rerank+doc2query": dict(embedder="minilm-fast", rerank=True, expand=True, llm=True),
}


class CheckedLLM:
    """Wraps the LLM for the lab: retries transient network errors and COUNTS failures, so a technique whose LLM calls
    silently failed (and fell back to the plain query) is reported as INVALID instead of looking like a real result."""

    def __init__(self, llm):
        self._llm, self.calls, self.errors = llm, 0, 0

    def complete(self, system, user, timeout_s, **kw):
        from app.llm import LLMError
        self.calls += 1
        for attempt in range(3):
            try:
                return self._llm.complete(system, user, max(timeout_s, 6.0), **kw)
            except LLMError:
                time.sleep(2 * (attempt + 1))
        self.errors += 1
        raise LLMError("lab: gave up after 3 attempts")


def relevant(retrieved: str, rel: list) -> bool:
    """A retrieved section counts if it is a labelled one, or (heading-level chunks) the heading that contains it."""
    return retrieved in rel or any(r.startswith(retrieved + ".") for r in rel)


def build(opts: dict, llm, sleep_s: float):
    from app.config import Settings
    from app.embeddings import make_embedder
    from app.expansion import Doc2Query
    from app.ingest import Ingestor
    from app.query_transforms import make_hyde, make_rewriter
    from app.retrieval import Retriever
    from app.store import PolicyStore

    s = Settings(data_dir=tempfile.mkdtemp(), embedder=opts.get("embedder", "minilm"))
    emb, store = make_embedder(s.embedder), PolicyStore(s.data_dir)
    exp = Doc2Query(llm, str(ROOT / "eval" / "cache" / "doc2query.json"), sleep_s) if opts.get("expand") else None
    ing = Ingestor(store, emb, s.max_policy_bytes, s.embedder, opts.get("chunking", "section"), exp)
    t0 = time.perf_counter()
    for co in ("acme", "borealis", "cygnus"):
        ing.ingest(co, (ROOT / "data" / "policies" / f"{co}_v1.pdf").read_bytes(), "pdf")
    ingest_s = time.perf_counter() - t0
    rer = None
    if opts.get("rerank"):
        from app.rerank import CrossEncoderReranker
        rer = CrossEncoderReranker()
    tr = {"rewrite": make_rewriter, "hyde": make_hyde}.get(opts.get("transform", ""), None)
    r = Retriever(store, emb, opts.get("mode", "vector"), 3, reranker=rer, pool=opts.get("pool", 10), transform=tr(llm) if tr else None)
    emb.embed(["warm up"])
    if rer:
        rer.rerank("warm", [{"title": "a", "text": "b"}, {"title": "c", "text": "d"}])
    return r, ingest_s


def evaluate(name: str, opts: dict, rows: list, llm, sleep_s: float) -> dict:
    r, ingest_s = build(opts, llm, sleep_s)
    per, lat, calls = [], [], 0
    for row in rows:
        r._qcache.clear()
        t = time.perf_counter()
        res = r.search(row["company"], row["query"])
        lat.append((time.perf_counter() - t) * 1000)
        calls += res.llm_calls
        got = [h["section"] for h in res.hits]
        # score on relevance-mapped section list (so heading-level chunks are compared fairly)
        rel_hits = [g for g in got]
        rel = row["relevant"]
        hit_flags = [relevant(g, rel) for g in rel_hits]
        found = {x for x in rel if any(g == x or x.startswith(g + ".") for g in rel_hits[:3])}
        per.append({"split": row["split"], "kind": row["kind"],
                    "recall@3": len(found) / len(rel),
                    "recall@1": len({x for x in rel if any(g == x or x.startswith(g + ".") for g in rel_hits[:1])}) / len(rel),
                    "mrr": next((1 / (i + 1) for i, f in enumerate(hit_flags) if f), 0.0),
                    "ndcg@3": M.ndcg_at_k([g for g in rel_hits if relevant(g, rel)] and [g for g in rel_hits if relevant(g, rel)], 3) if False else
                              (sum(1 / __import__("math").log2(i + 2) for i, f in enumerate(hit_flags[:3]) if f) /
                               sum(1 / __import__("math").log2(i + 2) for i in range(min(len(rel), 3)))),
                    "ms": lat[-1]})
        if opts.get("llm") and not opts.get("expand"):
            time.sleep(sleep_s)
    def agg(sub):
        return {k: round(statistics.mean(p[k] for p in sub), 3) for k in ("recall@1", "recall@3", "mrr", "ndcg@3")} if sub else {}
    dev, test = [p for p in per if p["split"] == "dev"], [p for p in per if p["split"] == "test"]
    ms = sorted(p["ms"] for p in per)
    llm_calls = getattr(llm, "calls", 0)
    llm_errors = getattr(llm, "errors", 0)
    return {"technique": name, "valid": llm_errors <= 0.05 * max(llm_calls, 1), "llm_calls_made": llm_calls, "llm_errors": llm_errors, "dev": agg(dev), "test": agg(test), "all": agg(per),
            "latency_ms_p50": round(statistics.median(ms), 1), "latency_ms_p95": round(ms[int(0.95 * len(ms))], 1),
            "extra_llm_calls_per_query": round(calls / len(rows), 2), "ingest_s": round(ingest_s, 1),
            "by_kind_recall@3": {k: round(statistics.mean(p["recall@3"] for p in per if p["kind"] == k), 2) for k in sorted({p["kind"] for p in per})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="include techniques that call the Groq LLM")
    ap.add_argument("--only", default="")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    rows = [json.loads(line) for line in BENCH.read_text().splitlines() if line.strip()]
    llm = None
    if a.llm:
        from app.llm import make_llm
        llm = make_llm("groq", os.environ["LLM_API_KEY"], "")
    names = [n for n in (a.only.split(",") if a.only else TECHNIQUES) if a.llm or not TECHNIQUES[n].get("llm")]
    out = []
    (ROOT / "eval" / "cache").mkdir(exist_ok=True)
    for n in names:
        checked = CheckedLLM(llm) if llm else None
        res = evaluate(n, TECHNIQUES[n], rows, checked, a.sleep)
        out.append(res)
        print(f"{n:<20} dev R@3={res['dev']['recall@3']:.2f} MRR={res['dev']['mrr']:.2f} | TEST R@1={res['test']['recall@1']:.2f} "
              f"R@3={res['test']['recall@3']:.2f} MRR={res['test']['mrr']:.2f} nDCG={res['test']['ndcg@3']:.2f} | "
              f"{res['latency_ms_p50']:>6.1f}ms p50 {res['extra_llm_calls_per_query']} extra calls"
              + ("" if res["valid"] else f"  *** INVALID: {res['llm_errors']}/{res['llm_calls_made']} LLM calls failed ***"), flush=True)
    path = ROOT / "eval" / "results" / (a.out or ("retrieval_lab_llm.json" if a.llm else "retrieval_lab.json"))
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
