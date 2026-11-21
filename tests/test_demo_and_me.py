import pytest

from app.users import UserStore, hash_password


def test_demo_endpoints_are_off_by_default(make_client):
    c = make_client()
    assert c.get("/demo/accounts").status_code == 404
    assert c.get("/").status_code == 404
    # and no demo accounts exist, so nobody can log in with the well-known demo password
    assert c.post("/auth/login", json={"email": "employee@acme.example", "password": "demo-pass-123"}).status_code == 401


def login(c, email, password="demo-pass-123"):
    return c.post("/auth/login", json={"email": email, "password": password})


def test_login_returns_a_company_token_that_works_everywhere(make_client, settings):
    settings.demo_mode = True
    c = make_client()
    r = login(c, "employee@acme.example")
    assert r.status_code == 200 and r.json()["company"] == "acme" and r.json()["role"] == "employee"
    h = {"Authorization": "Bearer " + r.json()["token"]}
    me = c.get("/me", headers=h).json()
    assert me["company"] == "acme" and me["role"] == "employee" and me["policy"]["version"] == 1
    assert c.post("/ask", json={"question": "escort after what time"}, headers=h).status_code == 200
    # an employee token can never ingest
    assert c.put("/admin/policy", content=b"x", headers={**h, "Content-Type": "text/markdown"}).status_code == 403
    assert c.get("/").status_code == 200 and "Commute Policy Assistant" in c.get("/").text


def test_admin_login_can_ingest_only_its_own_company(make_client, settings, policy_text):
    settings.demo_mode = True
    c = make_client()
    h = {"Authorization": "Bearer " + login(c, "admin@borealis.example").json()["token"], "Content-Type": "text/markdown"}
    r = c.put("/admin/policy", content=policy_text("acme_v2.md"), headers=h)
    assert r.status_code == 200 and r.json()["company"] == "borealis"  # the token decides; the file's content does not


@pytest.mark.parametrize("email,password", [("employee@acme.example", "wrong"), ("nobody@acme.example", "demo-pass-123"),
                                            ("", "x"), ("employee@acme.example", "")])
def test_bad_logins_get_the_same_generic_401(make_client, settings, email, password):
    settings.demo_mode = True
    r = make_client().post("/auth/login", json={"email": email or "x@y", "password": password or "x"})
    assert r.status_code == 401 and r.json()["detail"] == "Invalid email or password"  # never says which part was wrong


def test_login_is_rate_limited_per_client_and_email(make_client, settings):
    settings.demo_mode = True
    c = make_client()
    codes = [login(c, "employee@acme.example", "wrong").status_code for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]
    assert login(c, "employee@borealis.example").status_code == 200  # another account is not locked out


def test_passwords_are_stored_hashed_and_salted(tmp_path):
    us = UserStore(str(tmp_path / "u.db"))
    us.create("a@x.example", "acme", "employee", "correct horse battery")
    import sqlite3
    stored = sqlite3.connect(str(tmp_path / "u.db")).execute("SELECT pw_hash FROM users").fetchone()[0]
    assert "correct horse" not in stored and stored.startswith("scrypt$")
    assert hash_password("same") != hash_password("same")  # random salt
    assert us.authenticate("A@X.example", "correct horse battery") == ("acme", "employee")  # email is case-insensitive
    assert us.authenticate("a@x.example", "nope") is None


def test_weak_passwords_and_bad_companies_are_refused(tmp_path):
    us = UserStore(str(tmp_path / "u.db"))
    for args in (("a@x", "acme", "employee", "short"), ("a@x", "../etc", "employee", "longenough1"), ("a@x", "acme", "root", "longenough1")):
        with pytest.raises(ValueError):
            us.create(*args)


def test_me_requires_token(make_client):
    assert make_client().get("/me").status_code == 401
