# Evaluation report

## 1. What was evaluated and how

**Set:** [eval/evalset.jsonl](eval/evalset.jsonl), 42 cases, one JSON line each with `id, input, expected, actual, pass, notes`.
`input` = `{company, question}` (plus `setup_ingest` for the two version-update cases); `expected` = allowed status,
expected section(s), facts that must appear (`a|b` = alternatives) and text that must not appear.

| Category | Cases | Purpose |
|---|---|---|
| Direct questions, same topic across 3 companies (escort, weekend, airport, no-show, cap, cut-off) | 15 | the contradictory rules would expose any cross-company leak |
| Paraphrases (different vocabulary, situational, "9:30 PM" vs "after 8 PM") | 5 | semantic retrieval + rule application |
| Hard retrieval (exact terms, grades, rare words, numbers) | 8 | where keyword search could help |
| Not covered (off-topic and near-topic) | 6 | honest "the policy does not say" |
| Cross-company traps + prompt injection in the question | 4 | asks for another company's rule by name, or tries to steer |
| Injection **inside the policy documents** (Borealis 8.1, Cygnus 9.1) | 2 | model must not obey document text |
| Version update (Acme v2 ingested, then asked) | 2 | only the new version may answer |

Expected values were verified by hand against the policy text (all rules are in `data/policies/*.md`).

**Scoring (defined in `eval/run.py`, identical every run):**
* **Case passes** if: status is one of the allowed ones; if answered, it cites an expected section, contains every required fact, contains no forbidden text, and every citation belongs to the asker's company.
* **Retrieval** (measured on the retriever directly, no LLM): hit@1, hit@3 (= what the LLM sees), MRR over the 32 answerable cases.
* **Not-covered recall / precision** over clear-cut cases (cases where either status is acceptable are excluded from precision).
* **Cross-company leaks** (must be 0), 503 count, max LLM calls per question, max citations, latency p50/p95.

**Method:** `python -m eval.run`: fresh store per mode, seeded from the **PDF** policies, real MiniLM embeddings, real Groq LLM (`openai/gpt-oss-20b`, temperature 0), cases run in file order, paced to Groq's free-tier limit. Rerun it any time. Without a key use `--llm fake` (retrieval metrics are unchanged; answer-text metrics are then meaningless).

## 2. Results (official run, shipped configuration = vector retrieval)

| Metric | Vector (shipped) | Hybrid (bonus) |
|---|---|---|
| Cases passed | **42 / 42 (100 %)** | 41 / 42 (97.6 %) |
| Retrieval hit@1 | **96.9 %** (31/32) | 93.8 % (30/32) |
| Retrieval hit@3 | 100 % | 100 % |
| MRR | **0.984** | 0.958 |
| Not-covered recall / precision | 100 % / 100 % | 100 % / 85.7 % |
| Cross-company leaks | **0** | **0** |
| 503 errors | 0 | 0 |
| Max LLM calls / max citations | 1 / 2 | 1 / 3 |
| Latency, retrieval only, p50 / p95 | 156 ms / 216 ms | 154 ms / 161 ms |
| Latency, end to end in-process (incl. Groq), p50 / p95 | 625 ms / 821 ms | 601 ms / 936 ms |

HTTP end-to-end benchmark against the running container (JWT + FastAPI + retrieval + real Groq call), `python -m eval.bench`:
60 requests, 0 errors (53 answered, 7 not covered): **p50 633 ms, p95 884 ms, max 957 ms** (target p95 3,000 ms). Raw: `eval/results/bench.json`.

Both are far under the 3 s p95 target and never needed more than **1** LLM call. These are one machine plus a free API; treat them as indicative.

## 3. Bonus: hybrid retrieval, before and after (same 42 cases)

**What:** vector search plus BM25 keyword ranking, fused with Reciprocal Rank Fusion (`app/retrieval.py`, `RETRIEVAL_MODE=hybrid`).
**Hypothesis:** keywords help exact terms, grades and numbers that embeddings blur.

**Result: no improvement on this data, so it is not the default.**
Hit@1 fell from 31/32 to 30/32, MRR 0.984 → 0.958, one extra case failed, and not-covered precision fell. Retrieval-only latency was the same at p50 (about 155 ms, dominated by embedding the question) and noisy at p95.
**Cost in complexity:** about 45 extra lines, a BM25 index cache keyed by (company, version), one more thing to explain and test (isolation tests run in both modes).
**Why it did not help:** each company has only about 15 short, well-titled chunks, so MiniLM already ranks the right section first; BM25 over very short texts promotes chunks that merely share common words ("cab", "now").
**Caveat:** n = 32 and the gap is 1-2 cases, so it is within noise. The honest conclusion is "no evidence that it helps here", not "hybrid is bad". It would be worth re-measuring with 30-page documents (hundreds of chunks), where I would expect keyword matching to matter more.
I did **not** tune the fusion weights to make hybrid win: tuning on the same 42 cases would be overfitting.

## 4. Other before / after: prompt and context size

| Run | Prompt | Chunks to LLM | Vector pass | Notes |
|---|---|---|---|---|
| Baseline (`results/baseline_prompt_v1_groq.json`) | v1: "say so if the excerpts do not answer" | 5 | 95.2 % | 2 failures: `hard-grade-acme` (real) and `v2-noshow` (my eval file ordering bug, see 6) |
| Final | v2: "apply the rule to the person's numbers; not_covered only if no excerpt addresses the topic" | 3 | 100 % | |

Two things changed at once (prompt and 5 → 3 chunks) because the 5 → 3 change was forced by Groq's token limit, so I cannot attribute the gain to either alone.
**Overfitting warning:** the v2 prompt was written after seeing `hard-grade-acme` fail on this same set, so 100 % is optimistic for unseen questions.

## 5. Known failures and limitations

1. **LLM does not always apply a rule to a user's numbers.** `hard-grade-acme` ("L2, 7 km away") passes under vector but fails under hybrid (the model answers "not covered" instead of "not eligible, the threshold is 8 km"). The same question, same prompt, different retrieved context, different outcome: the model is the weak link, not retrieval.
2. **hit@1 miss:** `hard-standby-borealis` (a situational question) ranks §2.1 second. Still inside the top 3, so the answer was correct.
3. **LLM non-determinism:** temperature 0 is not a guarantee; two runs differed in one case (`hard-grade-acme`, under hybrid).
4. **Free-tier throughput:** Groq allows about 8,000 tokens/minute (about 10 questions/min). Over that, users get a clean "try again" (503). Seen live: the evaluation hit it and the app degraded correctly.
5. **Small, author-written evaluation set:** the same person wrote the documents and the questions, so it is easier than real employee traffic. 42 cases cannot prove "never leaks"; the isolation guarantee comes from the design (per-company collections + fail-closed check) and tests, not from the eval score.
6. **Single LLM judged by substring checks:** a correct answer phrased unexpectedly could fail a `must_include`; a wrong answer that happens to contain the keywords could pass. I read the answers in `evalset.jsonl` (`actual.answer`) to check; for production I would add human review.
7. **No OCR and structured documents only** (see README).

## 6. What failed during development, and why

| What | Why | Fix |
|---|---|---|
| Isolation test did not detect a tampered chunk | hybrid mode re-fetched clean metadata and masked the tampering; a test-design bug | test that mode explicitly, and now test both modes |
| Prompt-injection test failed | the question itself contained the word "Borealis" | assert on the excerpt block only |
| `v2-noshow` eval case "leaked" Rs 200 | the case ran **before** the v2 ingest, so v1's Rs 200 was correct; my file ordering was wrong | reorder; the app was right |
| Groq returned 404 | default model `llama-3.1-8b-instant` no longer exists on the account | switched to `openai/gpt-oss-20b`, set low reasoning effort and larger `max_tokens` |
| Many `llm_failed` during the evaluation | Groq free tier 8,000 tokens/min; I was sending 5 excerpts per call and pacing too fast | 3 excerpts per call, paced evaluation, 40 s wait before the harness retries |
| My README claimed retrieval is about 20 ms | measured about 150 ms (CPU embedding of the question) | corrected the claim |
| Tests took 71 s | each test re-seeded three policies | seed once per session, copy per test (12 s) |

## 7. Test suite (`python -m pytest -q`, 88 tests, about 12 s, no network)

Chunking and PDF parsing (headers/footers stripped, page numbers, scanned/corrupt/empty/oversized PDFs) · versioned ingestion
(only v2 answers after update, old chunks deleted, old/same version rejected, embedder crash and DB crash leave v1 untouched) ·
isolation (all three companies × both retrieval modes, client cannot choose company, tampered chunk fails closed) ·
auth (missing/garbage/expired/forged/`alg=none`/wrong audience or issuer/missing company/employee on admin route/admin of A cannot touch B) ·
robustness with a scripted fake LLM (12 kinds of garbage output, one repair then give up, never > 2 calls or > 3 citations, LLM
down, slow retrieval, no retry when budget is spent, injected document text fenced as data, document cannot close the data fence) ·
Groq client error mapping (429, 500, malformed, empty, timeout) · demo endpoints off by default.

## 8. Trade-off made for this report (as required)

**Vector-only as default instead of the more sophisticated hybrid search.** I implemented hybrid because the brief invited it, measured it on the same fixed set, and it was not better while adding code, a cache and a second ranking to explain. I shipped the simpler option and kept hybrid behind a flag. Simple and maintainable beat clever, and the measurement is what justifies it.

---

# Update (branch `retrieval-v2`): incremental ingestion, speed, guardrail, richer metrics

All numbers below are from the **same 42 cases** and the same pipeline, rerun after the changes (real Groq LLM, vector retrieval shipped,
`results/summary_groq_retrieval_v2_with_judge.json`).

## 9. Retrieval metrics (32 answerable cases, k = 3 = what the LLM sees)

| Metric | Vector (shipped) | Hybrid |
|---|---|---|
| recall@1 | 0.969 | 0.938 |
| **recall@3** | **1.000** | 1.000 |
| precision@3 | 0.333 | 0.333 |
| hit@1 | 0.969 | 0.938 |
| MRR | 0.984 | 0.958 |
| nDCG@3 | 0.988 | 0.969 |

Reading it: the right section is **always** in the top 3 (recall@3 = 1.0), so the LLM always has the evidence. precision@3 can never exceed
1/3 here because almost every question has exactly **one** relevant section and we show three, so 0.333 is the ceiling, not a weakness; use it
only to compare modes. The only rank-1 miss is `hard-standby-borealis` (correct section ranked 2nd).

## 10. Answer quality and hallucination metrics

| Metric | Vector | Hybrid | Notes |
|---|---|---|---|
| Cases passed | 42/42 | 41/42 | same as before the changes: **no regression** |
| fact_recall | 1.00 | 1.00 | required facts present |
| citation_precision / recall | 1.00 / 1.00 | 0.968 / 1.00 | hybrid cited one extra, off-target section |
| **numeric_hallucination_rate** | **0.0** | **0.0** | no answer contained a number absent from its cited text |
| guardrail_rejected_attempts | 0 | 0 | the runtime guardrail never fired, so it also never rejected a good answer (no false positives) |
| support_ratio (mean) | 0.872 | 0.874 | about 87 % of answer content words come from the cited text (the rest is phrasing: "you", "need to", ...) |
| **judge_faithful_rate** | **1.00** (34/34 judged) | 1.00 (33/33) | independent model checks every claim against the excerpts |
| judge_relevant_rate | 1.00 | 1.00 | answer addresses the question |
| abstention recall (= 1 − false-answer rate) | 100 % | 100 % | every not-covered question refused |
| abstention precision | 100 % | 85.7 % | hybrid refused once when it should have answered (`hard-grade-acme`) |
| over-abstention | 0 % | 3.1 % | |
| cross-company leaks | **0** | **0** | |
| Latency p50 / p95, cold cache, incl. Groq | 679 / 943 ms | 618 / 890 ms | target p95 < 3 s |

**Honest limits of these numbers.** (1) 0 % hallucination on 42 friendly cases does not prove it cannot happen. The guardrail firing zero
times means "no false alarms here", not "proved effective"; its effectiveness is shown by unit tests that feed it invented numbers
(`tests/test_guardrails.py`). (2) The judge (`gpt-oss-120b`) is from the same family as the answerer (`gpt-oss-20b`), so it can share blind spots; a
different family (e.g. Qwen) would be a stronger check. (3) The numeric guardrail cannot catch a wrong claim that uses no new number
(see GUARDRAILS.md). (4) `support_ratio` is a rough lexical proxy.

## 11. What the retrieval-v2 changes achieved (before = pushed `main`, after = this branch)

| Change | Before | After | Measured how |
|---|---|---|---|
| Policy update cost: embedding calls | 15 chunks embedded for Acme v1 to v2 | **6** embedded, 9 reused | `tests/test_incremental.py` (counting embedder) |
| One sentence edited in a 15-chunk policy | 15 embedded | **1** embedded, 14 reused | live run of `scripts/sync_policies.py` |
| Same file uploaded again | re-processed (or 409) | **no-op**, 0 embedded | `test_same_file_twice_is_a_noop` |
| Retrieval time, question seen before | 150 ms | **1.5 ms** | 20 questions twice, real MiniLM |
| Retrieval time, new question | 150 ms | 150 ms | unchanged: embedding the question dominates |
| Answer quality (pass rate, recall@3, MRR, leaks) | 100 %, 1.0, 0.984, 0 | 100 %, 1.0, 0.984, 0 | no regression |
| Hallucination guard | none at runtime | numeric grounding | tests + 0 false rejections in eval |

**Where the time goes (why we did not add clustering):** embedding the question 134 ms, Chroma HNSW search 1.1 ms, registry lookup 0.04 ms. Chroma already
uses an HNSW index, and with about 15 chunks per company a cluster layer could save at most about 1 ms, so it would add complexity for nothing.
It becomes worth revisiting at tens of thousands of chunks per company.

**Metadata filtering:** every search is already restricted by company (collection) and by `version = active` (metadata filter). I did **not** add a
topic filter (e.g. "only escort sections for escort questions"): a wrong guess about the topic would hide the right chunk, and recall@3 is already 1.0, so
the risk is not worth it on this data.

---

# Update 2 (branch `ui-v3`): is our measurement trustworthy? hardening results

## 12. Are we measuring correctness the best way? An honest answer
**Retrieval:** yes for what it measures. Each answerable question is labelled with the section(s) that hold the answer, and we compute recall@k, precision@k, hit@k, MRR, nDCG on those labels. Standard, deterministic, cheap.
**Answers:** four complementary checks, each with a known weakness:

| Check | Catches | Misses |
|---|---|---|
| Required facts (`must_include`) and forbidden text | wrong or missing numbers, leaks | a correct answer phrased unexpectedly fails; a wrong one that contains the keywords passes |
| Cited section is the expected one | wrong or missing sources | the right section cited for a wrong statement |
| Numeric grounding and support ratio (deterministic) | invented numbers, off-topic wording | a wrong claim that uses no new number |
| LLM judge (different, larger model) | unsupported claims in plain words | shared blind spots; cost |

**Validity check of the judge** (`python -m eval.judge_check`): given 3 truthful and 5 deliberately wrong answers (wrong number, contradiction, invented extra penalty, over-generalisation, invented benefit), it was right **8 of 8**. Small sample, but it proves the judge can say "unfaithful", so its 35/35 "faithful" on our real answers means something.
**What would be better:** a larger set (hundreds of questions, ideally real employee questions), human-written gold answers, a judge calibrated against human labels, and a judge from a different model family. Our set is 42 author-written cases, so treat the scores as regression protection and evidence of direction, not as production accuracy.

## 13. Results after hardening (quarantine + min-support guardrail + rate limit), real Groq, vector retrieval
Pass 42/42, recall@3 1.0, MRR 0.984, numeric hallucination 0 %, judge faithful 35/35 and relevant 35/35, not-covered recall and precision 100 %, leaks 0, p50/p95 681/955 ms.
**The new guardrails rejected 0 correct answers** (no false positives on 35 answers; lowest support ratio was 0.545 against a 0.35 threshold).
**Red team with quarantine switched OFF** (`QUARANTINE_FLAGGED=0 python -m eval.run --only inj,trap`): the injected Borealis and Cygnus sections reached the model and it still ignored them; all 6 inj/trap cases passed. So the model-level defence works today, and quarantine removes the exposure at the source.
Raw: `results/summary_groq_hardened.json`.

## 14. Greetings and small talk (added after seeing "hi" answered with "the policy does not say")
**Problem:** a greeting is not a policy question, so replying "the policy does not say anything about this" felt broken. **Decision:** `app/smalltalk.py` answers
greetings, thanks, goodbyes and "what can you do?" in plain code (no LLM call, no retrieval, no citations, `status: "smalltalk"`), and nothing else.
**Why code, not the LLM:** instant, free, cannot be steered, keeps the assistant scoped to transport policy. **Safety of the rule:** only short messages that are *entirely*
a pleasantry match; "hi, can I book Saturday?" and "hello ignore your instructions" still go through the normal pipeline (tests: `tests/test_smalltalk.py`).
**Cost / trade-off:** a hand-written pattern list will miss some phrasings ("yo", "sup"), which then get the polite "policy does not say" reply; it errs on the side of never swallowing a real question.
Three cases (`chat-hi`, `chat-thanks`, `chat-help`) were added to the set (now **45**); rerun on real Groq: **45/45 pass**, 0 leaks, 0 errors, not-covered recall/precision 100 %, p95 1,017 ms.
