"""Deterministic checks on a model answer. Used at RUNTIME (reject) and in the EVAL (measure).
No LLM involved: free, instant, explainable.

Idea: policy answers are mostly numbers (times, amounts, hours, distances). If the answer contains a number
that is in neither the cited policy text nor the user's own question, the model made it up (or took it from
memory). We refuse to show such an answer.
"""
import re

MIN_SUPPORT = 0.35  # answers whose wording is <35 % from the cited text are rejected (lowest real answer in the eval: 0.545)

_NUM = re.compile(r"\d[\d,]*(?:[:.]\d+)?")
_WORD = re.compile(r"[a-z]{4,}")
_STOP = set("this that with from your have will which their there been into than then them they when what where "
            "also only such must shall being does need needs".split())


def numeric_tokens(text: str) -> set:
    """Numbers as comparable strings: '1,500' -> '1500', '8:00' -> '8', '9:30' stays, '5.0' -> '5'."""
    out = set()
    for m in _NUM.findall(text):
        t = m.replace(",", "").rstrip(":.")
        t = re.sub(r"[:.]0+$", "", t)
        if t:
            out.add(t)
    return out


def ungrounded_numbers(answer: str, evidence_texts: list, question: str = "") -> set:
    """Numbers in the answer that appear neither in the cited excerpts nor in the question."""
    allowed = numeric_tokens(question)
    for t in evidence_texts:
        allowed |= numeric_tokens(t)
    return numeric_tokens(answer) - allowed


def support_ratio(answer: str, evidence_texts: list, question: str = "") -> float:
    """Share of the answer's content words (4+ letters, not stop words) found in the evidence or question.
    A rough proxy for 'is the answer made of what the policy said'. 1.0 when there are no content words."""
    ev = " ".join(evidence_texts + [question]).lower()
    ev_words = set(_WORD.findall(ev))
    words = [w for w in _WORD.findall(answer.lower()) if w not in _STOP]
    if not words:
        return 1.0
    stem = lambda w: w.rstrip("s")  # noqa: E731  (cheap plural handling)
    ev_stems = {stem(w) for w in ev_words}
    return sum(stem(w) in ev_stems for w in words) / len(words)


SENTENCE_MIN_SUPPORT = 0.5  # a sentence counts as grounded if at least half of its content words come from the cited text


def sentence_groundedness(answer: str, evidence_texts: list, question: str = "") -> dict:
    """Sentence-level groundedness, deterministic (no LLM). A sentence is GROUNDED when
       (a) at least half of its content words appear in the cited policy text (or the question), and
       (b) every number in it appears in the cited policy text or the question.
    Returns {"sentences": n, "grounded": g, "ungrounded": [sentence, ...]}."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer.strip()) if x.strip()]
    bad = [x for x in sents if support_ratio(x, evidence_texts, question) < SENTENCE_MIN_SUPPORT
           or ungrounded_numbers(x, evidence_texts, question)]
    return {"sentences": len(sents), "grounded": len(sents) - len(bad), "ungrounded": bad}
