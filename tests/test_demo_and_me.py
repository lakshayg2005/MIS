def test_demo_endpoints_are_off_by_default(make_client):
    c = make_client()
    assert c.post("/demo/token", json={"company": "acme"}).status_code == 404
    assert c.get("/").status_code == 404


def test_demo_token_is_employee_only_and_known_companies_only(make_client, settings):
    settings.demo_mode = True
    c = make_client()
    r = c.post("/demo/token", json={"company": "acme"})
    assert r.status_code == 200
    h = {"Authorization": "Bearer " + r.json()["token"]}
    me = c.get("/me", headers=h).json()
    assert me["company"] == "acme" and me["role"] == "employee" and me["policy"]["version"] == 1
    assert c.put("/admin/policy", content=b"x", headers={**h, "Content-Type": "text/markdown"}).status_code == 403
    assert c.post("/demo/token", json={"company": "evilcorp"}).status_code == 404
    assert c.get("/").status_code == 200 and "Commute Policy Assistant" in c.get("/").text


def test_me_requires_token(make_client):
    assert make_client().get("/me").status_code == 401
