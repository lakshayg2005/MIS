import json

from app.chunking import parse_policy
from app.embeddings import HashEmbedder
from app.expansion import Doc2Query
from app.ingest import Ingestor
from app.llm import LLMError
from app.query_transforms import make_hyde, make_rewriter
from app.retrieval import Retriever
from app.store import PolicyStore

DOC = [(1, "T"), (1, "Version: 1"), (1, "3. Night shift"), (1, "3.1 Escort"), (1, "An escort is mandatory after 8 PM. It is assigned automatically."),
       (1, "3.2 Timings"), (1, "The last drop is at 1 AM."), (1, "4. Weekends"), (1, "4.1 Saturday"), (1, "Manager approval needed.")]


def test_sentence_chunks_keep_their_parent_section():
    ch = parse_policy(DOC, "sentence").chunks
    esc = [c for c in ch if c.section == "3.1"]
    assert [c.text for c in esc] == ["An escort is mandatory after 8 PM.", "It is assigned automatically."]
    assert all(c.parent.startswith("An escort is mandatory after 8 PM. It is assigned") for c in esc)
    assert len({c.key for c in ch}) == len(ch)  # ids stay unique


def test_heading_chunks_group_subsections():
    ch = parse_policy(DOC, "heading").chunks
    assert [(c.section, c.title) for c in ch] == [("3", "Night shift"), ("4", "Weekends")]
    assert "3.1 Escort:" in ch[0].text and "3.2 Timings:" in ch[0].text


def test_default_granularity_is_unchanged():
    assert [c.section for c in parse_policy(DOC).chunks] == ["3.1", "3.2", "4.1"]


def setup(tmp_path, policy_text, **kw):
    store = PolicyStore(str(tmp_path))
    Ingestor(store, HashEmbedder(), 10**6, "hash").ingest("acme", policy_text("acme_v1.md"))
    return store, Retriever(store, HashEmbedder(), "vector", 3, **kw)


class Reverse:  # a fake reranker that visibly reorders the candidates
    def rerank(self, question, hits):
        return list(reversed(hits))


def test_reranker_reorders_candidates_and_still_respects_top_k(tmp_path, policy_text):
    store, plain = setup(tmp_path, policy_text)
    _, rer = setup(tmp_path / "b", policy_text, reranker=Reverse(), pool=6)
    q = "when is an escort mandatory for women"
    a, b = plain.search("acme", q).hits, rer.search("acme", q).hits
    assert len(a) == len(b) == 3 and [h["id"] for h in a] != [h["id"] for h in b]
    assert all(h["company"] == "acme" for h in b)  # reranking never widens the candidate set beyond the company


def test_sentence_mode_returns_one_hit_per_section_with_full_text(tmp_path, policy_text):
    store = PolicyStore(str(tmp_path))
    Ingestor(store, HashEmbedder(), 10**6, "hash", "sentence").ingest("acme", policy_text("acme_v1.md"))
    hits = Retriever(store, HashEmbedder(), "vector", 3).search("acme", "when is an escort mandatory").hits
    assert len({h["section"] for h in hits}) == len(hits)  # no duplicate sections
    assert any(len(h["text"]) > 80 for h in hits)  # parent (whole section) text, not one short sentence


class Said:
    def __init__(self, *outs):
        self.outs, self.calls = list(outs), 0

    def complete(self, system, user, timeout_s, **kw):
        o = self.outs[min(self.calls, len(self.outs) - 1)]
        self.calls += 1
        if isinstance(o, Exception):
            raise o
        return o


def test_rewrite_uses_one_call_and_falls_back_on_failure():
    llm = Said("What is the booking cut-off?")
    assert make_rewriter(llm)("how early do i book") == ("What is the booking cut-off?", 1)
    assert make_rewriter(Said(LLMError("quota")))("how early do i book") == ("how early do i book", 1)  # fallback, call still counted
    assert make_rewriter(Said(""))("x y z") == ("x y z", 1)  # empty rewrite ignored
    assert make_hyde(Said("Bookings close 4 hours before."))("when to book")[1] == 1


def test_rewritten_query_is_what_gets_searched(tmp_path, policy_text):
    store, _ = setup(tmp_path, policy_text)
    seen = []

    class Spy(HashEmbedder):
        def embed_query(self, text):
            seen.append(text)
            return super().embed([text])[0]

    r = Retriever(store, Spy(), "vector", 3, transform=lambda q: ("security escort", 1))
    res = r.search("acme", "guard thing at night??")
    assert seen == ["security escort"] and res.llm_calls == 1


def test_doc2query_caches_by_text_and_survives_bad_output(tmp_path):
    llm = Said('["a?", "b?"]', "not json at all")
    d = Doc2Query(llm, str(tmp_path / "c.json"))
    assert d.questions("T", "text one") == ["a?", "b?"] and llm.calls == 1
    assert d.questions("T", "text one") == ["a?", "b?"] and llm.calls == 1  # cache hit: no new call
    assert d.questions("T", "text two") == [] and llm.calls == 2  # garbage -> no expansion, and not cached
    assert Doc2Query(Said("x"), str(tmp_path / "c.json")).questions("T", "text one") == ["a?", "b?"]  # survives restart


def test_expansion_is_embedded_and_keeps_incremental_hashes_stable(tmp_path, policy_text):
    class Counting(HashEmbedder):
        texts = []

        def embed(self, texts):
            Counting.texts += list(texts)
            return super().embed(texts)

    llm = Said('["will someone ride along at night?"]')
    exp = Doc2Query(llm, str(tmp_path / "c.json"))
    store = PolicyStore(str(tmp_path / "s"))
    ing = Ingestor(store, Counting(), 10**6, "hash", "section", exp)
    ing.ingest("acme", policy_text("acme_v1.md"))
    assert any("will someone ride along at night?" in t for t in Counting.texts)
    Counting.texts.clear()
    r = ing.ingest("acme", policy_text("acme_v2.md"))  # 6 chunks changed: only those are re-embedded, others keep cached questions
    assert r.embedded == 6 and r.reused == 9
