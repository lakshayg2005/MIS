"""Query-time rewriting with an LLM. Each costs ONE LLM call, which counts against the 2-calls-per-question cap
(rewrite + answer = 2, so the answer step may not retry). On any failure we fall back to the original question."""
import re

from .llm import LLMError

REWRITE_SYSTEM = ("You rewrite an employee's question so it matches the wording of a corporate transport policy "
                  "(terms like: eligibility, grade, distance, shift, booking cut-off, cancellation, security escort, night drop, "
                  "weekend, public holiday, ad-hoc trip approval, airport trip, cost cap, guest, no-show penalty, suspension). "
                  "Fix typos, replace slang with policy terms, keep every number and time. Reply with ONLY the rewritten question.")
HYDE_SYSTEM = ("Write ONE sentence, in the style of a corporate transport policy, that would directly answer the employee's question. "
               "Do not invent specific numbers; use the employee's own numbers if any. Reply with ONLY that sentence.")


def _llm_transform(llm, system: str, timeout_s: float):
    def run(question: str):
        try:
            out = llm.complete(system, question, timeout_s).strip()
            out = re.sub(r"^```.*?\n|```$", "", out, flags=re.S).strip().strip('"')
            return (out if 3 <= len(out) <= 400 else question), 1
        except LLMError:
            return question, 1  # the call was spent even though it failed
    return run


def make_rewriter(llm, timeout_s: float = 1.2):
    return _llm_transform(llm, REWRITE_SYSTEM, timeout_s)


def make_hyde(llm, timeout_s: float = 1.2):
    return _llm_transform(llm, HYDE_SYSTEM, timeout_s)
