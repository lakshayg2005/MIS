# Viva cheat-sheet (own use)

## 60-second pitch
"Employees ask the desk about their company's transport policy. My assistant answers only from **their own
company's current policy**, shows the section it came from, says so when the policy is silent, and says
'try again' instead of hanging. The key idea: **the LLM only phrases the answer. Everything that must be
correct or safe is plain code**: who you are (JWT), which company's data you can reach (one collection per
company, re-checked), which version is live (an atomic pointer), whether there is evidence (similarity gate),
and whether the answer is valid (citations built from my metadata). I measure it with a fixed 42-case set."

## Walk the code in this order (each is short)
1. `auth.py` – company comes from the token; algorithm pinned; required claims.
2. `store.py` – collection per company; SQLite `one_active` index; `activate()` is one transaction.
3. `ingest.py` – steps 1-4 in the docstring; what happens on each failure.
4. `retrieval.py` – vector, BM25, RRF; the defensive company check.
5. `service.py` – the pipeline; `parse_model_output`; the retry rule.
6. `eval/run.py` – how a case passes.

## Questions you will get
**Why not just put the company in a metadata filter?** A filter is one forgotten `where` from a leak; per-company
collections make a leak structurally impossible, and I still verify every chunk. Cost: more collections, irrelevant at this scale.

**What if ingestion fails halfway?** Everything is built in memory first. Chunks of the new version are staged
(queries filter on the active version, so they are invisible). One SQLite transaction flips the pointer. A
failure before the flip deletes the staged chunks and the old version stays live. Tests simulate an embedder crash and a DB crash.

**How do you stay under 2 LLM calls?** Normal path is 1. The 2nd call only repairs a *malformed* reply and only if
≥ 1 s of the 2.8 s budget is left. Quota/timeout errors are not retried (time is gone). A `while calls < 2` loop plus a test.

**Why a similarity gate before the LLM?** "Not covered" becomes instant, free, and cannot hallucinate. Risk: threshold too high hides real
answers; I tune `MIN_SIMILARITY` on the eval set. (Live change: it is one number in `config.py`.)

**How do you handle prompt injection?** Two layers. Architecture: the model has no tools and only ever sees the
caller's own company's chunks, so even a fully obedient model cannot leak or act. Checks: excerpts fenced as data;
output validated; an answer naming another company is rejected. Regex flagging at ingestion is only a *report*.
Two sample policies contain real injection text and are in the eval set.

**Why is hybrid search not the default / did it help?** Answer from EVALUATION.md: be honest, quote the numbers.
With 15 chunks per company, plain vector search already ranks the right section first almost always; hybrid
cost a little latency and complexity and did not clearly help. I report the negative result rather than tune until it wins.

**Why top 3 to the LLM?** Equals the citation cap, ~40 % fewer tokens (Groq free tier is 8k tokens/min), and the eval shows the right section is always in the top 3.

**Why not LLM-as-judge / query rewriting?** Latency budget and the 2-call cap; deterministic checks are free and explainable.

**Why is the JWT HS256?** Mock issuer = simplest verifiable token. Production: RS256, verify with the IdP's public key; only `auth.py` changes.

**What would you do with more time?** Fallback LLM provider on quota errors; per-user rate limits; audit log; OCR behind a review step;
bigger eval set with real employee questions; a cross-encoder reranker if documents get large.

**What are the failures?** Say them first: model may mis-phrase within valid citations; free-tier token limit; structured-PDF requirement; known failing eval cases (see EVALUATION.md).

**Why no token streaming?** Guardrails need the whole answer (valid JSON, numbers, citations). Streaming raw tokens would show text we might then retract. I stream *progress* (real stages) and reveal the validated answer word by word. Cost: less "instant" feel; total is ~0.7 s anyway.

**Why not store conversations?** Privacy and isolation (nothing to leak, no retention duty) and it keeps each question independent so the 2-call / 3-chunk limits hold. Cost: no follow-up questions. Chat is kept only in the browser tab. Future: opt-in server history with per-user encryption and a retention limit.

**Are the guardrails strong enough?** Say: isolation is architectural (strong); hallucination/injection are reduced and measured, not eliminated; the real gap is a malicious admin. See GUARDRAILS.md table.

**Which bonus techniques did you try and why did you ship reranking?** Hybrid, reranking, query rewriting, sentence/heading chunking, bge embeddings, doc2query, combos; all on a 63-query hard benchmark with a dev/test split, because our first set had a ceiling effect. Reranking gave recall@3 0.87 to 0.94 for ~70-100 ms and no LLM call. Rewriting gave the same quality but +1 LLM call and +0.5 s, and halves free-tier throughput. sentence chunks hurt. Mention the same-model-faster-runtime finding (145 ms to 5 ms) and the honest wrinkle (rank-1 sometimes 2nd on clean queries).

**What went wrong in that experiment?** My first LLM runs matched the baseline exactly: Groq's JSON mode needs the word "json", so every rewrite call failed with 400 and silently fell back. I fixed it, added failure counting that marks a technique INVALID, and discarded those numbers.

**What does "a JWT that carries the company, no client-controlled override" mean?** The company is a signed claim inside the token; the server reads it from the *verified* token and searches only that company. There is no company field the client can send, and tampering with the token breaks its signature. See README "How the JWT works here" and `auth.py`.

**How do you measure groundedness?** Sentence-level check against the cited text (content-word support plus numbers), citation validity, numeric grounding, and an independent LLM judge; the metric is tested with deliberately ungrounded answers so it can fail. Honest limit: it is lexical, so a wrong claim without new numbers can slip through.

## Live-change cookbook (practise each once, then run `pytest -q`)
| They ask | Change | Check |
|---|---|---|
| Be stricter about "not covered" | `min_similarity` in `config.py` (or `MIN_SIMILARITY=0.4`) | eval: not-covered precision/recall |
| Cite at most 2 chunks | `top_k = 2` in `config.py`; update the `<= 3` test to `<= 2` | `pytest tests/test_service_robustness.py` |
| Allow only 1 LLM call | `max_llm_calls = 1` in `config.py` | tests expecting 2 calls will fail: say why and update them |
| Different "not covered" wording | `NOT_COVERED_MSG` in `service.py` | `pytest -k not_covered` |
| Question length limit 300 | `max_question_chars` in `config.py` | `pytest -k bad_questions` (update 501 -> 301) |
| Escort after 9 PM at Borealis | edit `data/policies/borealis_v1.md` section 3.1; regenerate with `python -m scripts.make_pdfs`; then ingest as v2 (bump `Version:`) | ask the escort question |
| Switch to Gemini | `LLM_PROVIDER=gemini LLM_API_KEY=...` (already supported in `llm.py`) | one real question |
| Vector only vs hybrid | `RETRIEVAL_MODE=vector` | `python -m eval.run --modes vector --llm fake` |
| Add an eval case | append one JSON line to `eval/evalset.jsonl` | `python -m eval.run --llm fake --no-write` |
| Make tokens expire in 5 min | `ttl_s` default in `auth.issue_token` | `pytest tests/test_isolation_auth.py` |
| Reject questions with URLs | add a check next to the length check in `main.py` `/ask` | curl it |
| Add a `GET /policy/versions` | query `policies` table in `store.py`, route in `main.py` | curl with a token |
