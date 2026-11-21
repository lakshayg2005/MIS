"""Groundedness report from the stored evaluation results (no LLM, no network):

    python -m eval.groundedness

For every ANSWERED case in eval/evalset.jsonl it takes the model's answer and the policy text of the sections it cited
(looked up from the same policy files the app ingested) and checks each sentence. Reports:
  sentence groundedness  = grounded sentences / all sentences
  fully grounded answers = answers in which EVERY sentence is grounded
  citation validity      = answers whose cited sections really exist in the company's policy
  numeric grounding      = answers whose numbers all appear in the cited text or the question
and lists every flagged sentence so a human can read it (a metric you cannot inspect is not worth trusting).
"""
import json
from pathlib import Path

from app.chunking import parse_policy
from app.documents import extract_lines
from app.guardrails import sentence_groundedness, ungrounded_numbers

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ROOT / "data" / "policies"


def sections(company: str, version_file: str = "") -> dict:
    f = POLICIES / (version_file or f"{company}_v1.md")
    return {c.section: c.text for c in parse_policy(extract_lines(f.read_bytes(), "md")).chunks}


def main():
    cases = [json.loads(line) for line in (ROOT / "eval" / "evalset.jsonl").read_text().splitlines() if line.strip()]
    n_ans = n_sent = n_ground = n_full = n_valid = n_num = 0
    flagged = []
    for c in cases:
        a = c["actual"]
        if not a or a["status"] != "answered":
            continue
        co, q = c["input"]["company"], c["input"]["question"]
        secs = sections(co, (c["input"].get("setup_ingest") or "").replace(".pdf", ".md"))
        cited = a["cited_sections"]
        evidence = [secs[s] for s in cited if s in secs]
        n_ans += 1
        n_valid += bool(cited) and all(s in secs for s in cited)
        g = sentence_groundedness(a["answer"], evidence, q)
        n_sent += g["sentences"]
        n_ground += g["grounded"]
        n_full += g["grounded"] == g["sentences"]
        n_num += not ungrounded_numbers(a["answer"], evidence, q)
        for s in g["ungrounded"]:
            flagged.append({"id": c["id"], "sentence": s, "cited": cited})
    out = {"answered_cases": n_ans, "sentences": n_sent,
           "sentence_groundedness_pct": round(100 * n_ground / n_sent, 1),
           "fully_grounded_answers_pct": round(100 * n_full / n_ans, 1),
           "citation_validity_pct": round(100 * n_valid / n_ans, 1),
           "numeric_grounding_pct": round(100 * n_num / n_ans, 1),
           "flagged_sentences": flagged}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    (ROOT / "eval" / "results" / "groundedness.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
