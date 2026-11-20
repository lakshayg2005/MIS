import pytest

from app.guardrails import numeric_tokens, support_ratio, ungrounded_numbers
from tests.conftest import ScriptedLLM

EVIDENCE = ["A security escort is mandatory for every woman employee on drops after 8:00 PM. A fee of Rs 1,500 applies."]


def test_number_normalisation():
    assert numeric_tokens("Rs 1,500 at 8:00 PM and 9:30 PM, 5.0 km, 24 hours.") == {"1500", "8", "9:30", "5", "24"}


@pytest.mark.parametrize("answer,bad", [
    ("An escort is needed after 8 PM.", set()),  # 8 PM vs 8:00 PM is the same number
    ("The cap is Rs 1500.", set()),  # 1,500 vs 1500
    ("For your 9:30 PM drop an escort is needed.", set()),  # number taken from the question
    ("An escort is needed after 10 PM.", {"10"}),  # invented
    ("The fee is Rs 2,000.", {"2000"}),
])
def test_ungrounded_numbers(answer, bad):
    assert ungrounded_numbers(answer, EVIDENCE, "Is a 9:30 PM drop covered?") == bad


def test_support_ratio():
    assert support_ratio("escort mandatory after drops", EVIDENCE) == 1.0
    assert support_ratio("helicopters landing nearby tomorrow", EVIDENCE) < 0.3


def ask(c, tok, llm_answer, company="acme"):
    llm = ScriptedLLM('{"status":"answered","answer":"%s","citations":[1]}' % llm_answer)
    c.app.state.service.llm = llm
    return c.post("/ask", json={"question": "At what time is a security escort mandatory for women on drops?"}, headers=tok(company)), llm


def test_runtime_rejects_invented_number(make_client, tok):
    r, llm = ask(make_client(), tok, "An escort is mandatory after 11 PM.")
    assert r.status_code == 503 and r.json()["citations"] == [] and llm.calls == 2  # repaired once, still invalid
    assert "11" not in r.json()["answer"]


def test_runtime_accepts_grounded_number(make_client, tok):
    r, llm = ask(make_client(), tok, "An escort is mandatory after 8 PM.")
    assert r.status_code == 200 and r.json()["status"] == "answered" and llm.calls == 1
