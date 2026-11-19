"""Run the fixed evaluation set and score it the same way every time.

    python -m eval.run                       # real LLM from .env (Groq), vector vs hybrid, updates evalset.jsonl
    python -m eval.run --llm fake            # offline: no network, answer text is a crude stand-in
    python -m eval.run --modes hybrid        # one retrieval mode only

Scoring (all defined here, nothing hidden):
  retrieval  hit@1/3/5, MRR: an answerable case counts if its expected section is among the top 3 / 5
                             retrieved chunks. Measured directly on the retriever (no LLM involved).
  case pass                : status is one of the expected ones AND (if answered) it cites an expected
                             section AND every must_include fact appears AND no must_not_include text
                             appears AND every citation belongs to the asker's own company.
  not-covered precision / recall, cross-company leak count (must be 0), latency p50 / p95, max LLM calls.
"""
import argparse
import json
import os
import re
import statistics
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALSET = ROOT / "eval" / "evalset.jsonl"
RESULTS = ROOT / "eval" / "results"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("‑", "-").replace(" ", " ")).lower()


def has_any(text: str, alternatives: str) -> bool:
    return any(norm(a) in norm(text) for a in alternatives.split("|"))


def judge(case: dict, ans, top_ids: list) -> tuple:
    exp, company = case["expected"], case["input"]["company"]
    allowed = exp["status"] if isinstance(exp["status"], list) else [exp["status"]]
    notes = []
    if ans.status not in allowed:
        notes.append(f"status {ans.status}({ans.reason}) not in {allowed}")
    cited = [c["section"] for c in ans.citations]
    leaked = [c for c in ans.citations if not c["document"].lower().startswith(company)]
    if leaked:
        notes.append("LEAK: citation from another company")
    if ans.status == "answered":
        if exp["sections"] and not set(cited) & set(exp["sections"]):
            notes.append(f"cited {cited}, expected one of {exp['sections']}")
        for inc in exp["must_include"]:
            if not has_any(ans.message, inc):
                notes.append(f"missing fact '{inc}'")
    for bad in exp["must_not_include"]:
        if norm(bad) in norm(ans.message):
            notes.append(f"LEAK/FORBIDDEN text '{bad}' in answer")
    return (not notes), notes, bool(leaked) or any(n.startswith("LEAK") for n in notes)


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def run_mode(mode: str, cases: list, llm_name: str, sleep_s: float) -> tuple:
    from app.config import Settings
    from app.embeddings import make_embedder
    from app.ingest import Ingestor
    from app.llm import make_llm
    from app.retrieval import Retriever
    from app.service import QuestionService
    from app.store import PolicyStore

    s = Settings(data_dir=tempfile.mkdtemp(), retrieval_mode=mode, llm_provider=llm_name)
    emb, store = make_embedder(s.embedder), PolicyStore(s.data_dir)
    ing = Ingestor(store, emb, s.max_policy_bytes)
    from app.main import seed_if_empty
    seed_if_empty(s, store, ing)
    retriever = Retriever(store, emb, mode, s.top_k)
    svc = QuestionService(s, store, retriever, make_llm(llm_name, s.llm_api_key, s.llm_model))
    emb.embed(["warm up"])

    rows, hit1, hit3, hit5, rr, answerable = [], 0, 0, 0, 0.0, 0
    for case in cases:
        co, q = case["input"]["company"], case["input"]["question"]
        if case["input"].get("setup_ingest"):
            f = ROOT / "data" / "policies" / case["input"]["setup_ingest"]
            try:
                ing.ingest(co, f.read_bytes(), f.suffix.lstrip("."))
            except Exception:
                pass  # already ingested by an earlier case in this run
        exp_secs = case["expected"]["sections"]
        res = retriever.search(co, q)
        got = [h["section"] for h in res.hits]
        if exp_secs:
            answerable += 1
            hit1 += bool(set(got[:1]) & set(exp_secs))
            rr += next((1 / (i + 1) for i, g in enumerate(got) if g in exp_secs), 0.0)
            hit3 += bool(set(got[:3]) & set(exp_secs))
            hit5 += bool(set(got[:5]) & set(exp_secs))
        ans = svc.ask(co, q)
        if ans.status == "error" and llm_name != "fake":  # harness-only: transient quota/network, retry once
            time.sleep(6)
            ans = svc.ask(co, q)
        ok, notes, leak = judge(case, ans, got)
        rows.append({"case": case, "ok": ok, "notes": notes, "leak": leak, "ans": ans, "top3": got[:3]})
        print(f"[{mode:6}] {'PASS' if ok else 'FAIL'} {case['id']:<18} {ans.status:<12} {ans.latency_ms:>5}ms  {'; '.join(notes)}")
        if llm_name != "fake" and ans.llm_calls:
            time.sleep(sleep_s)  # stay under the free-tier requests/minute limit

    lat = sorted(r["ans"].latency_ms for r in rows if r["ans"].status != "error")
    exp_nc = [r for r in rows if r["case"]["expected"]["status"] == "not_covered"]
    got_nc = [r for r in rows if r["ans"].status == "not_covered"]
    tp = [r for r in got_nc if r["case"]["expected"]["status"] == "not_covered"]
    summary = {
        "mode": mode, "llm": llm_name, "cases": len(rows),
        "case_pass_pct": pct(sum(r["ok"] for r in rows), len(rows)),
        "retrieval_hit@1_pct": pct(hit1, answerable), "retrieval_mrr": round(rr / answerable, 3),
        "retrieval_hit@3_pct": pct(hit3, answerable), "retrieval_hit@5_pct": pct(hit5, answerable),
        "not_covered_recall_pct": pct(len(tp), len(exp_nc)), "not_covered_precision_pct": pct(len(tp), len(got_nc)),
        "cross_company_leaks": sum(r["leak"] for r in rows),
        "errors_503": sum(r["ans"].status == "error" for r in rows),
        "latency_ms_p50": int(statistics.median(lat)) if lat else None,
        "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
        "max_llm_calls": max(r["ans"].llm_calls for r in rows), "max_citations": max(len(r["ans"].citations) for r in rows),
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="vector,hybrid")
    ap.add_argument("--llm", default=None, help="groq|gemini|fake (default: LLM_PROVIDER from .env, else fake)")
    ap.add_argument("--sleep", type=float, default=2.5)
    ap.add_argument("--no-write", action="store_true", help="do not update evalset.jsonl")
    args = ap.parse_args()

    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    llm = args.llm or os.getenv("LLM_PROVIDER", "fake")

    cases = [json.loads(line) for line in EVALSET.read_text().splitlines() if line.strip()]
    for c in cases:
        c["actual"], c["pass"], c["notes"] = None, None, c["notes"].split(" || ")[0]
    RESULTS.mkdir(exist_ok=True)
    summaries, last_rows = [], None
    for mode in args.modes.split(","):
        rows, summary = run_mode(mode, cases, llm, args.sleep)
        summaries.append(summary)
        last_rows = rows
        print(json.dumps(summary, indent=1))
    (RESULTS / f"summary_{llm}.json").write_text(json.dumps(summaries, indent=1) + "\n")

    if not args.no_write and last_rows:  # the evalset file records the last mode run (hybrid, the shipped default)
        with open(EVALSET, "w") as f:
            for r in last_rows:
                c, a = r["case"], r["ans"]
                c["actual"] = {"status": a.status, "reason": a.reason, "answer": a.message,
                               "cited_sections": [x["section"] for x in a.citations],
                               "retrieved_top3": r["top3"], "latency_ms": a.latency_ms, "llm_calls": a.llm_calls}
                c["pass"] = r["ok"]
                c["notes"] = c["notes"] + (" || " + "; ".join(r["notes"]) if r["notes"] else "")
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
