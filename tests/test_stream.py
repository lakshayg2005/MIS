import json

from tests.conftest import ANSWER_1, ScriptedLLM
from app.llm import LLMError

Q = {"question": "At what time is a security escort mandatory for women on drops?"}


def events(resp):
    out, name = [], None
    for line in resp.text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[6:])))
    return out


def test_stream_sends_stages_then_one_validated_final(make_client, tok):
    c = make_client(ScriptedLLM(ANSWER_1))
    r = c.post("/ask/stream", json=Q, headers=tok("acme"))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    assert [n for n, _ in ev] == ["stage", "stage", "final"]
    assert ev[0][1]["step"] == "retrieving" and ev[1][1]["step"] == "drafting" and len(ev[1][1]["sections"]) <= 3
    final = ev[-1][1]
    assert final["status"] == "answered" and final["citations"] and final["llm_calls"] == 1


def test_stream_never_leaks_unvalidated_model_text(make_client, tok):
    c = make_client(ScriptedLLM('{"status":"answered","answer":"Bananas are a wonderful fruit rich in potassium","citations":[1]}'))
    r = c.post("/ask/stream", json=Q, headers=tok("acme"))
    assert "Bananas" not in r.text and events(r)[-1][1]["status"] == "error"


def test_stream_llm_failure_is_a_clean_final_event(make_client, tok):
    r = make_client(ScriptedLLM(LLMError("quota"))).post("/ask/stream", json=Q, headers=tok("acme"))
    f = events(r)[-1][1]
    assert f["status"] == "error" and "try again" in f["answer"].lower()


def test_stream_requires_token_and_applies_limits(make_client, tok, settings):
    settings.rate_limit_per_min = 1
    c = make_client()
    assert c.post("/ask/stream", json=Q).status_code == 401
    assert c.post("/ask/stream", json={"question": "x" * 600}, headers=tok("acme")).status_code == 400
    assert c.post("/ask/stream", json=Q, headers=tok("acme")).status_code == 200
    assert c.post("/ask/stream", json=Q, headers=tok("acme")).status_code == 429
