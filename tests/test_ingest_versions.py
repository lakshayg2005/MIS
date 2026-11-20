import pytest

from app.embeddings import HashEmbedder
from app.ingest import Ingestor, IngestError
from app.store import PolicyStore

Q = {"question": "At what time is a security escort mandatory for women on drops?"}


def test_new_version_replaces_old_everywhere(make_client, tok, policy_text):
    c = make_client()
    before = c.post("/ask", json=Q, headers=tok("acme")).json()
    assert before["citations"][0]["version"] == 1 and "8:00 PM" in before["citations"][0]["excerpt"]

    r = c.put("/admin/policy", content=policy_text("acme_v2.md"), headers={**tok("acme", "admin"), "Content-Type": "text/markdown"})
    assert r.status_code == 200 and r.json()["version"] == 2 and r.json()["previous_version"] == 1

    after = c.post("/ask", json=Q, headers=tok("acme")).json()
    assert all(x["version"] == 2 for x in after["citations"])
    assert "9:00 PM" in after["citations"][0]["excerpt"]
    # the old version's chunks are physically gone, not just hidden
    store = c.app.state.service.store
    assert {ch["version"] for ch in store.get_chunks("acme", 1)} == set()


def test_same_file_is_noop_and_older_version_rejected(make_client, tok, policy_text):
    c = make_client()
    h = {**tok("acme", "admin"), "Content-Type": "text/markdown"}
    r = c.put("/admin/policy", content=policy_text("acme_v1.md"), headers=h)
    assert r.status_code == 200 and r.json()["unchanged"] is True  # identical file: nothing happens
    assert c.put("/admin/policy", content=policy_text("acme_v2.md"), headers=h).status_code == 200
    assert c.put("/admin/policy", content=policy_text("acme_v1.md"), headers=h).status_code == 409  # now v1 is older


def test_malformed_document_leaves_policy_untouched(make_client, tok):
    c = make_client()
    r = c.put("/admin/policy", content=b"# T\nVersion: 5\nno sections", headers={**tok("acme", "admin"), "Content-Type": "text/markdown"})
    assert r.status_code == 422
    assert c.app.state.service.store.active("acme")[0] == 1


class Boom(HashEmbedder):
    def embed(self, texts):
        raise RuntimeError("embedding service down")


def test_embedding_failure_is_all_or_nothing(tmp_path, policy_text):
    store = PolicyStore(str(tmp_path))
    Ingestor(store, HashEmbedder(), 10**6).ingest("acme", policy_text("acme_v1.md"))
    with pytest.raises(IngestError) as e:
        Ingestor(store, Boom(), 10**6).ingest("acme", policy_text("acme_v2.md"))
    assert e.value.http_status == 503
    assert store.active("acme")[0] == 1 and len(store.get_chunks("acme", 1)) > 0
    assert store.get_chunks("acme", 2) == []


def test_crash_between_stage_and_activate_rolls_back(tmp_path, policy_text, monkeypatch):
    store = PolicyStore(str(tmp_path))
    ing = Ingestor(store, HashEmbedder(), 10**6)
    ing.ingest("acme", policy_text("acme_v1.md"))
    monkeypatch.setattr(store, "activate", lambda *a: (_ for _ in ()).throw(RuntimeError("db locked")))
    with pytest.raises(IngestError):
        ing.ingest("acme", policy_text("acme_v2.md"))
    assert store.active("acme")[0] == 1
    assert store.get_chunks("acme", 2) == []  # staged chunks removed


def test_oversized_document_rejected(tmp_path):
    ing = Ingestor(PolicyStore(str(tmp_path)), HashEmbedder(), 100)
    with pytest.raises(IngestError) as e:
        ing.ingest("acme", b"x" * 101)
    assert e.value.http_status == 413
