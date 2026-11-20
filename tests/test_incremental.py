"""Only new or changed text may be embedded; unchanged chunks reuse stored vectors."""
from app.embeddings import HashEmbedder
from app.ingest import Ingestor
from app.store import PolicyStore


class Counting(HashEmbedder):
    def __init__(self):
        self.texts = []

    def embed(self, texts):
        self.texts += list(texts)
        return super().embed(texts)


def setup(tmp_path, name="hash"):
    emb = Counting()
    store = PolicyStore(str(tmp_path))
    return store, emb, Ingestor(store, emb, 10**6, name)


def test_first_ingest_embeds_everything(tmp_path, policy_text):
    store, emb, ing = setup(tmp_path)
    r = ing.ingest("acme", policy_text("acme_v1.md"))
    assert r.embedded == r.chunks == 15 and r.reused == 0 and len(emb.texts) == 15


def test_new_version_embeds_only_changed_chunks(tmp_path, policy_text):
    store, emb, ing = setup(tmp_path)
    ing.ingest("acme", policy_text("acme_v1.md"))
    emb.texts.clear()
    r = ing.ingest("acme", policy_text("acme_v2.md"))
    # v1 -> v2 changes: 1.1 (8->6 km, interns), 2.1 (4->3 h), 3.1 and 3.2 (8->9 PM), 5.3 (cap), 7.1 (fee) = 6 chunks
    assert r.embedded == 6 and r.reused == 9 and len(emb.texts) == 6
    assert any("9:00 PM" in t for t in emb.texts) and not any("Guests are not allowed" in t for t in emb.texts)


def test_reused_vectors_are_identical_to_fresh_ones(tmp_path, policy_text):
    store, emb, ing = setup(tmp_path)
    ing.ingest("acme", policy_text("acme_v1.md"))
    ing.ingest("acme", policy_text("acme_v2.md"))
    store2, emb2, ing2 = setup(tmp_path / "fresh")
    ing2.ingest("acme", policy_text("acme_v2.md"))  # everything embedded from scratch
    a = {c["id"]: c for c in store.get_chunks("acme", 2)}
    assert set(a) == {c["id"] for c in store2.get_chunks("acme", 2)}
    va, vb = store.vectors_by_hash("acme"), store2.vectors_by_hash("acme")
    assert va.keys() == vb.keys()
    assert all(max(abs(x - y) for x, y in zip(va[k], vb[k])) < 1e-6 for k in va)  # float32 storage: equal to rounding


def test_same_file_twice_is_a_noop(tmp_path, policy_text):
    store, emb, ing = setup(tmp_path)
    ing.ingest("acme", policy_text("acme_v1.md"))
    emb.texts.clear()
    r = ing.ingest("acme", policy_text("acme_v1.md"))
    assert r.unchanged and r.embedded == 0 and emb.texts == [] and store.active("acme")[0] == 1


def test_changed_text_with_same_version_is_still_rejected(tmp_path, policy_text):
    import pytest

    from app.ingest import IngestError
    store, emb, ing = setup(tmp_path)
    ing.ingest("acme", policy_text("acme_v1.md"))
    edited = policy_text("acme_v1.md").replace(b"Rs 200", b"Rs 999")
    with pytest.raises(IngestError) as e:
        ing.ingest("acme", edited)
    assert e.value.http_status == 409


def test_other_company_vectors_are_never_reused(tmp_path, policy_text):
    store, emb, ing = setup(tmp_path)
    ing.ingest("acme", policy_text("acme_v1.md"))
    emb.texts.clear()
    r = ing.ingest("borealis", policy_text("borealis_v1.md"))
    assert r.reused == 0 and r.embedded == r.chunks  # reuse is per company only


def test_different_embedding_model_never_reuses_old_vectors(tmp_path, policy_text):
    store = PolicyStore(str(tmp_path))
    Ingestor(store, Counting(), 10**6, "model-A").ingest("acme", policy_text("acme_v1.md"))
    emb_b = Counting()
    r = Ingestor(store, emb_b, 10**6, "model-B").ingest("acme", policy_text("acme_v2.md"))
    assert r.reused == 0 and len(emb_b.texts) == r.chunks  # hash includes the model name


def test_api_reports_embedded_and_reused(make_client, tok, policy_text):
    c = make_client()
    r = c.put("/admin/policy", content=policy_text("acme_v2.md"), headers={**tok("acme", "admin"), "Content-Type": "text/markdown"})
    assert r.status_code == 200 and r.json()["embedded_chunks"] == 6 and r.json()["reused_chunks"] == 9
