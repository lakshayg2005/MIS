import time

import jwt
import pytest

from app.retrieval import IsolationError

QUESTIONS = ["escort after 10 PM", "weekend cab booking", "airport trip approval", "no-show penalty",
             "cost cap per trip", "guest travel", "booking cut-off", "who is eligible"]


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
@pytest.mark.parametrize("company", ["acme", "borealis", "cygnus"])
def test_retrieval_never_returns_another_companys_chunk(make_client, tok, company, mode):
    c = make_client(retrieval_mode=mode)
    svc = c.app.state.service
    for q in QUESTIONS:
        res = svc.retriever.search(company, q)
        assert res.hits and all(h["id"].startswith(f"{company}:") and h["company"] == company for h in res.hits)
        for cit in svc.ask(company, q).citations:
            assert cit["document"].lower().startswith(company)


def test_client_cannot_choose_company(make_client, tok):
    c = make_client()
    body = {"question": "when is escort mandatory?", "company": "borealis", "company_id": "borealis"}
    r = c.post("/ask", json=body, headers=tok("acme"))
    assert r.status_code == 200 and all("Acme" in x["document"] for x in r.json()["citations"])
    r = c.post("/ask?company=borealis", json=body, headers={**tok("acme"), "X-Company": "borealis"})
    assert all("Acme" in x["document"] for x in r.json()["citations"])


def test_isolation_violation_fails_closed(make_client, monkeypatch):
    c = make_client(retrieval_mode="vector")
    svc = c.app.state.service
    real = svc.store.query
    monkeypatch.setattr(svc.store, "query", lambda co, v, e, k: [{**h, "company": "borealis"} for h in real(co, v, e, k)])
    a = svc.ask("acme", "escort drop")
    assert a.status == "error" and a.reason == "isolation_violation" and a.citations == []


def test_unknown_company_gets_not_covered(make_client, settings):
    c = make_client()
    a = c.app.state.service.ask("newco", "escort?")
    assert a.status == "not_covered" and a.reason == "no_policy"


def hdr(settings, **over):
    now = int(time.time())
    claims = {"sub": "u", "company": "acme", "role": "employee", "iss": settings.jwt_issuer,
              "aud": settings.jwt_audience, "exp": now + 60, **over}
    claims = {k: v for k, v in claims.items() if v is not None}
    return {"Authorization": "Bearer " + jwt.encode(claims, over.get("_key", settings.jwt_secret), algorithm="HS256")}


def test_auth_rejections(make_client, settings):
    c = make_client()
    q = {"question": "escort?"}
    assert c.post("/ask", json=q).status_code == 401
    assert c.post("/ask", json=q, headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert c.post("/ask", json=q, headers=hdr(settings, exp=int(time.time()) - 10)).status_code == 401  # expired
    assert c.post("/ask", json=q, headers=hdr(settings, aud="other")).status_code == 401
    assert c.post("/ask", json=q, headers=hdr(settings, iss="evil")).status_code == 401
    assert c.post("/ask", json=q, headers=hdr(settings, company=None)).status_code == 401
    assert c.post("/ask", json=q, headers=hdr(settings, company="../etc")).status_code == 401
    forged = jwt.encode({"sub": "u", "company": "borealis", "iss": settings.jwt_issuer, "aud": settings.jwt_audience,
                         "exp": int(time.time()) + 60}, "wrong-secret-wrong-secret-wrong-secret", algorithm="HS256")
    assert c.post("/ask", json=q, headers={"Authorization": "Bearer " + forged}).status_code == 401
    none_tok = jwt.encode({"sub": "u", "company": "acme"}, None, algorithm="none")  # alg=none attack
    assert c.post("/ask", json=q, headers={"Authorization": "Bearer " + none_tok}).status_code == 401


def test_admin_rules(make_client, tok, policy_text):
    c = make_client()
    md = policy_text("acme_v2.md")
    H = {"Content-Type": "text/markdown"}
    assert c.put("/admin/policy", content=md, headers={**tok("acme"), **H}).status_code == 403  # employee
    # an admin of borealis cannot overwrite acme: ingestion always targets the token's own company
    r = c.put("/admin/policy", content=md, headers={**tok("borealis", "admin"), **H})
    assert r.status_code == 200 and r.json()["company"] == "borealis"
    assert c.app.state.service.store.active("acme")[0] == 1


def test_bad_questions_rejected(make_client, tok):
    c = make_client()
    assert c.post("/ask", json={"question": "   "}, headers=tok("acme")).status_code == 400
    assert c.post("/ask", json={"question": "x" * 501}, headers=tok("acme")).status_code == 400
    assert c.post("/ask", json={}, headers=tok("acme")).status_code == 422
