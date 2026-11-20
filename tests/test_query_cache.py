from app.embeddings import HashEmbedder
from app.retrieval import Retriever
from tests.test_incremental import Counting


def make(settings_store, size=512):
    emb = Counting()
    return emb, Retriever(settings_store, emb, "vector", 3, cache_size=size)


def test_repeated_question_embeds_once_and_normalises_case_and_spaces(make_client):
    svc = make_client().app.state.service
    emb, r = make(svc.store)
    r.search("acme", "When is an escort mandatory?")
    r.search("acme", "  when is an ESCORT   mandatory? ")
    r.search("borealis", "When is an escort mandatory?")  # other company: same vector, different results
    assert len(emb.texts) == 1 and r.cache_hits == 2 and r.cache_misses == 1


def test_cache_never_mixes_company_results(make_client):
    svc = make_client().app.state.service
    emb, r = make(svc.store)
    a = r.search("acme", "escort after what time")
    b = r.search("borealis", "escort after what time")
    assert all(h["company"] == "acme" for h in a.hits) and all(h["company"] == "borealis" for h in b.hits)


def test_cache_is_bounded(make_client):
    svc = make_client().app.state.service
    emb, r = make(svc.store, size=2)
    for q in ["a one", "b two", "c three"]:
        r.search("acme", q)
    assert len(r._qcache) == 2
    r.search("acme", "a one")  # was evicted -> embedded again
    assert len(emb.texts) == 4
