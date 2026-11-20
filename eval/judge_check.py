"""Does our LLM judge actually catch bad answers?  (A metric that cannot fail is worthless.)

Feeds the judge answers that are correct and answers that are deliberately WRONG, then reports how many it got right.
    python -m eval.judge_check [--model openai/gpt-oss-120b]
"""
import argparse
import os
import time
from pathlib import Path

from eval.run import llm_judge

EX = "A security escort is mandatory for every woman employee on drops after 8:00 PM. The transport desk assigns the escort automatically when the drop is booked."
EX2 = "Airport pickups and drops need approval from the department head. The reporting manager cannot approve an airport trip."
EX3 = "The first no-show in a calendar month results in a warning. From the second no-show in the same month, a fee of Rs 200 is charged for each no-show."

# (excerpt, question, answer, should_be_faithful)
CASES = [
    (EX, "When is an escort needed?", "An escort is required for women on drops after 8:00 PM, and the desk assigns it automatically.", True),
    (EX2, "Who approves airport trips?", "The department head approves them; your reporting manager cannot.", True),
    (EX3, "What if I no-show twice?", "The second no-show in the month costs Rs 200.", True),
    (EX, "When is an escort needed?", "An escort is required for women on drops after 10:00 PM.", False),  # wrong number
    (EX2, "Who approves airport trips?", "Your reporting manager approves them.", False),  # contradicts the text
    (EX3, "What if I no-show twice?", "The second no-show costs Rs 200 and you are also banned from transport for a week.", False),  # invented extra
    (EX, "When is an escort needed?", "An escort is required for all employees, men and women, on every drop.", False),  # overgeneralised
    (EX3, "What if I no-show twice?", "The fee is Rs 200 per no-show, and the policy also gives a free cab voucher after three.", False),  # invented benefit
]


class A:  # minimal stand-in for an Answer
    def __init__(self, ex, msg):
        self.citations, self.message = [{"excerpt": ex}], msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    a = ap.parse_args()
    env = Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    from app.llm import make_llm
    judge = make_llm("groq", os.environ["LLM_API_KEY"], a.model)
    right = 0
    for ex, q, ans, truth in CASES:
        faithful, _ = llm_judge(judge, A(ex, ans), q)
        ok = faithful == truth
        right += ok
        print(f"{'OK ' if ok else 'MISS'} expected faithful={truth!s:5} judge said {faithful!s:5} | {ans[:70]}")
        time.sleep(2)
    print(f"judge correct on {right}/{len(CASES)} (3 truthful, 5 deliberately wrong)")


if __name__ == "__main__":
    main()
