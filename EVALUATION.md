# Evaluation report

> **Read this first: final state of the submission.** The report below grew in stages (sections 1-8 first round, then updates 1-3); this box is the current truth.
>
> * **Documents:** three full-length policies (13-14 pages, about 9,000 words each, PDF + Markdown) for Acme, Borealis and Cygnus, plus 2-page versions of the same core rules used as unit-test fixtures. The numbered core rules are the contradictory, verified ones; the extra chapters were generated under checked constraints (no numbers, no other company's name).
> * **Pipeline shipped:** question -> hybrid search (vector + BM25) over the caller's company only -> cross-encoder rerank of 6 candidates -> top 3 -> ONE LLM call (`qwen/qwen3.8-27b` on Groq) -> validation (JSON, citations, numbers, wording, other-company names) -> answer + citations, or "not covered", or "try again".
> * **Evaluation set:** `eval/evalset.jsonl`, 48 cases (`id, input, expected, actual, pass, notes`), rerun with `python -m eval.run` (offline: `--llm fake`).
>
> | Official run on the full-length policies (real LLM: `qwen/qwen3.8-27b`, judge: `gpt-oss-120b`) | Before: plain vector search | After: hybrid + rerank (shipped) |
> |---|---|---|
> | Cases passed | 48 / 48 | 48 / 48 |
> | Retrieval recall@3 / MRR / hit@1 | 1.00 / 0.984 / 0.969 | 1.00 / 0.938 / 0.875 |
> | Not-covered recall / precision | 100 % / 100 % | 100 % / 100 % |
> | Cross-company leaks / errors | 0 / 0 | 0 / 0 |
> | Numeric hallucination / sentence groundedness (eval) | 0 % / 100 % | 0 % / 98.6 % |
> | LLM judge: faithful / relevant (35 judged) | not judged | 100 % / 97.1 % |
> | Latency p50 / p95, LLM call included (in-process) | 329 / 531 ms | 551 / 748 ms |
>
> | Harder retrieval-only benchmark (63 queries, full-length policies, `eval/retrieval_hard.jsonl`) | Before | After (shipped) |
> |---|---|---|
> | recall@1 / recall@3 / MRR / nDCG@3 | 0.65 / 0.79 / 0.75 / 0.75 | **0.73 / 0.90 / 0.85 / 0.85** |
> | Retrieval time p50 | 148 ms (Chroma's default MiniLM runtime) | 249 ms (fast runtime + hybrid + rerank) |
>
> **The honest reading (a trade-off I made):** on our easy 48-case set the simple baseline is already perfect on answers, so the shipped pipeline adds nothing there and costs about 220 ms and a slightly lower rank-1 score; the benefit shows on the harder 63-query benchmark (recall@3 0.79 -> 0.90) and on near-duplicate sections in long documents (stress test: vector-only rank-1 0.93, hybrid and rerank 1.0). I shipped hybrid + rerank for robustness on long policies and accepted the latency (p95 still about a quarter of the 3 s budget) and the extra complexity (one dependency, about 60 lines). `RERANK=0 RETRIEVAL_MODE=vector` restores the simple pipeline.
> **Latency over real HTTP** (Docker container, login flow, real LLM, 40 requests, `python -m eval.bench`): 0 errors, **p50 544 ms, p95 855 ms, max 917 ms** against the 3 s target (raw: `eval/results/bench.json`). **LLM calls per question:** 1 (max 2). **Cited chunks:** at most 3.
>
> **Known failures and limits (details in sections 5, 21, 23):** (1) the answering model is the weak link: an earlier model (`gpt-oss-20b`, short-document runs) intermittently answered "not covered" for the L2/7 km case; `qwen` got it right in the final run, which shows model sensitivity, not that it is solved; (2) full-length policies make every request carry about 1,000-1,500 tokens, so the free tier (8,000 tokens/min, 200,000 tokens/day per model) serves only about 5 questions a minute, and my own repeated evaluation runs exhausted a model's daily quota once (switching model fixed it); (3) the 48-case and 63-query sets are small and written by the same person who wrote the policies; (4) the LLM-based retrieval techniques (rewriting, document expansion) were only measured on the short documents; (5) the sentence-groundedness metric is word-overlap based: it flagged 1 of 57 sentences in the offline report, a faithful paraphrase ("Any amount exceeding this cap is not reimbursed" for "Trips above the cap are not reimbursed"), i.e. a false positive; (6) extra chapters of the long policies were LLM-drafted, so they are realistic-looking filler rather than carefully authored policy.

## 1. What was evaluated and how

**Set:** [eval/evalset.jsonl](eval/evalset.jsonl), 48 cases (42 originally, as in the table below, plus 6 small-talk / meta-question cases added later, see section 14-15), one JSON line each with `id, input, expected, actual, pass, notes`.
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

## 2. Results of the FIRST round (short 2-page policies, vector retrieval; superseded by the box at the top, kept for history)

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

## 15. Questions about the assistant itself, and off-topic questions
**Problem found in use:** "how are you being implemented" and "are you a dog" were answered "the policy does not say anything about this", which is
technically honest but confusing. **Decision: no free-form LLM chat.** Instead `app/smalltalk.py` gives fixed, reviewed, truthful replies (no LLM call) for:
identity ("are you a dog / human / AI"), how it works, privacy ("do you store my chats"), "can you book a cab" (it cannot, and says what it can do), plus greetings, thanks and help.
**Why not let the LLM chat:** it would have no excerpts to ground it (so the numeric/support/citation guardrails cannot apply), answers would vary run to run, it widens the injection surface, and it turns a policy assistant into a general chatbot.
**Off-topic questions** (nothing in the policy is similar, e.g. "capital of France") now get a clearer message ("I couldn't find anything about this in your transport policy... otherwise this is outside what I can help with") instead of implying the policy was consulted and found silent. Still no LLM call.
**Trade-off:** patterns are hand-written, so unusual phrasings fall through to the polite not-covered reply. The rules only match short messages that are entirely a pleasantry/meta question, so a real policy question is never swallowed (tests: `tests/test_smalltalk.py`).
Eval set is now **48** cases (6 small-talk/meta). Real Groq run: **48/48 pass**, recall@3 1.0, judge faithful 100 %, 0 leaks, 0 errors, p95 935 ms.

---

# Update 3 (branch `bonus-retrieval`): all bonus techniques, compared properly

The brief's bonus asks us to implement and measure retrieval improvements and report cost. The first round tried only hybrid search, and it could not show any
gain because our end-to-end set has a **ceiling effect** (recall@3 was already 1.0). So this round (a) built a **harder retrieval-only benchmark** and
(b) tried **every technique the brief lists plus some of our own**.

## 16. Method
* **Benchmark:** [eval/retrieval_hard.jsonl](eval/retrieval_hard.jsonl), **63 queries** over the three companies, each labelled with the section(s) that answer it: vocabulary mismatch (18: "guard" for escort, "Diwali" for public holiday), paraphrase (14), situational (12), short keyword (10), multi-section (7), typos (2). Labels were checked against the policy text by hand.
* **Dev / test split:** alternate queries go to dev (33) or test (30). Choices are made on dev and confirmed on test, to avoid fooling ourselves. `R@3 dev / test` below shows when the two disagree.
* **Fair comparison:** every technique builds a fresh store from the same three PDFs and answers the same queries; retrieval only (no answer generation), cold cache, `python -m eval.retrieval_lab [--llm]`.
* **Techniques tried:** hybrid (vector + BM25 + RRF), **cross-encoder reranking**, **LLM query rewriting**, **chunking changes** (finer: sentence chunks returning the parent section; coarser: one chunk per heading), **embedding-model swaps** (`minilm-fast`: the same MiniLM on a faster runtime; `bge-small`), **document expansion / doc2query** (LLM writes likely employee questions for each chunk at ingestion), and combinations.
* **Rejected on principle:** a hand-written synonym glossary for query expansion. I wrote the benchmark queries myself, so a glossary I wrote afterwards would leak the answers into the technique and inflate its score.

## 17. Results (all 63 queries)
| Technique | R@1 | R@3 | MRR | nDCG@3 | R@3 dev / test | Retrieval ms (p50) | Extra LLM calls/q | Verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.67 | 0.87 | 0.79 | 0.80 | 0.85 / 0.90 | 145 | 0.0 | reference (Chroma MiniLM, vector search, section chunks) |
| embed-minilm-fast | 0.67 | 0.87 | 0.79 | 0.80 | 0.85 / 0.90 | 5 | 0.0 | **ADOPTED**: identical quality, 30x faster embedding (same model, faster runtime) |
| hybrid | 0.70 | 0.93 | 0.84 | 0.86 | 0.95 / 0.90 | 146 | 0.0 | small MRR gain, below rerank; optional (`RETRIEVAL_MODE=hybrid`) |
| rerank | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 214 | 0.0 | **ADOPTED** (shown with the fast embedder below): biggest no-LLM gain |
| fast+rerank | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 114 | 0.0 | **SHIPPED DEFAULT**: best quality, and faster than the old baseline |
| fast+rerank(pool5) | 0.77 | 0.93 | 0.88 | 0.89 | 0.92 / 0.93 | 67 | 0.0 | cheaper (5 candidates), slightly lower on dev; option |
| hybrid+rerank | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 218 | 0.0 | no gain over rerank alone |
| fast+hybrid+rerank | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 111 | 0.0 | no gain over rerank alone |
| rewrite-llm | 0.77 | 0.94 | 0.88 | 0.89 | 0.92 / 0.97 | 608 | 1.0 | same MRR as rerank for +1 LLM call and +460 ms: not worth it; optional (`QUERY_REWRITE=1`) |
| fast+rewrite | 0.77 | 0.94 | 0.88 | 0.89 | 0.92 / 0.97 | 488 | 1.0 | as above, with the fast embedder |
| doc2query-llm | 0.70 | 0.94 | 0.83 | 0.86 | 0.92 / 0.97 | 142 | 0.0 | good recall at no query-time cost, but lower MRR than rerank; optional (`EXPAND_QUESTIONS=1`) |
| fast+rerank+doc2query | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 107 | 0.0 | no gain over rerank alone (the gains do not stack) |
| chunk-sentence | 0.68 | 0.87 | 0.79 | 0.80 | 0.88 / 0.87 | 134 | 0.0 | **hurts** (small-to-big): sentences lose context |
| chunk-heading | 0.67 | 0.96 | 0.83 | 0.84 | 0.95 / 0.97 | 133 | 0.0 | recall@3 looks high but only because chunks are big; rank-1 accuracy drops and citations get vaguer |
| embed-bge-small | 0.65 | 0.86 | 0.78 | 0.79 | 0.89 / 0.83 | 8 | 0.0 | worse on test; not adopted |
| bge+hybrid | 0.66 | 0.91 | 0.81 | 0.82 | 0.95 / 0.85 | 8 | 0.0 | worse on test |
| bge+rerank | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 125 | 0.0 | same as rerank |
| bge+hybrid+rerank | 0.77 | 0.94 | 0.89 | 0.90 | 0.95 / 0.93 | 123 | 0.0 | same as rerank |

R@k = recall@k (share of the labelled sections found in the top k), MRR = mean reciprocal rank, nDCG@3 = rank-aware gain. "Retrieval ms" is the retrieval stage alone, including the LLM call for rewriting.

## 18. What we learned
1. **Reranking is the best no-LLM improvement:** recall@1 0.67 → 0.77, recall@3 0.87 → 0.94, MRR 0.79 → 0.89, and it helps on both dev and test.
2. **A free 30x speedup:** the same all-MiniLM-L6-v2 model on the fastembed runtime embeds a question in about 5 ms instead of 145 ms, with **identical retrieval results**. Reranking then costs about 70 to 100 ms, so *fast + rerank is both better and faster than the old pipeline* (114 ms vs 145 ms).
3. **Query rewriting works but is not worth it:** about the same quality as reranking (MRR 0.88 vs 0.89), for +1 LLM call and about 460 ms. It also halves free-tier throughput (the Groq limit is tokens per minute) and, under the 2-call cap, leaves the answer step no repair retry. Kept as an option (`QUERY_REWRITE=1`).
4. **Document expansion is the best LLM-based idea:** recall@3 0.94 with **no query-time cost**, since the LLM runs once per chunk at ingestion (cached, and unchanged chunks are never re-expanded, so incremental ingestion still works). Its rank-1 accuracy is lower than reranking and it does not stack with it, so it stays optional (`EXPAND_QUESTIONS=1`).
5. **Chunking changes did not help.** Sentence chunks lose context (0.87). Heading-level chunks reach recall@3 0.96 only because each chunk is big; rank-1 accuracy falls and citations become vaguer ("section 3" instead of "3.1"), so that recall number flatters it.
6. **Hybrid helps slightly** (MRR 0.84) but less than reranking, and adds nothing on top of it. `bge-small` is worse on test than MiniLM.
7. **Dev and test sometimes disagree** (hybrid, bge, heading chunks), which is exactly why we do not trust a single split of 63 queries for small differences.

## 19. Decision: new defaults and what each costs
**Shipped:** `EMBEDDER=minilm-fast` + `RERANK=1` (cross-encoder `ms-marco-MiniLM-L-6-v2`, 10 candidates, return 3). Complexity cost: one extra dependency (`fastembed`), one small module (`app/rerank.py`, 15 lines), two models baked into the Docker image. Optional switches, all off: `QUERY_REWRITE=1`, `EXPAND_QUESTIONS=1`, `CHUNKING=sentence|heading`, `RETRIEVAL_MODE=hybrid`.

**End-to-end regression with the shipped pipeline** (48 cases, real Groq, judge on): 47/48 pass (97.9 %); recall@3 **1.0**; hit@1 0.875; MRR 0.938; nDCG@3 0.954; judge faithful 100 %; numeric hallucination 0 %; leaks 0; 503 errors 0; p50/p95 **502/930 ms**; retrieval stage alone p50 99 ms. Raw: `results/summary_groq_shipped_fast_rerank.json`.
* The one failure is `hard-grade-acme`: the **model** answers "not covered" instead of applying the 8 km rule to "L2, 7 km". It is the known flaky case (it passed in earlier runs and failed in others); retrieval gave it the right section.
* **Honest wrinkle:** on our original, cleaner questions reranking moved the right section from rank 1 to rank 2 for four queries (e.g. "no-show twice": the reranker prefers §7.2 "repeated no-shows" over §7.1), so hit@1 fell from 0.969 to 0.875. recall@3 stayed 1.0 and all four answers were still correct because the model sees all three sections. So the gain is on hard queries (recall@3 and rank-1 there), not on already-easy ones. Over all 95 answerable queries, recall@3 improves from about 0.91 to 0.96 and rank-1 accuracy is about equal.

## 20. Problems found while doing this (and fixed)
* **Silent fallback:** my first run of query rewriting and doc2query showed numbers *identical to the baseline*, which is suspicious. Cause: Groq's JSON mode requires the word "json" in the prompt, my rewrite prompts did not contain it, so every call returned HTTP 400 and the code silently fell back to the plain question. Fixes: free-text LLM calls now omit JSON mode (`json_mode=False`, tested), and the lab counts LLM failures and marks a technique INVALID if more than 5 % of calls fail. Those first numbers were discarded; the table above is from the corrected rerun.
* **Call cap:** a rewrite call counts toward the 2-call cap. A naive "calls > 0 means this is a retry" check would have skipped the first answer attempt after a rewrite; fixed and tested (`test_rewrite_call_counts_toward_the_cap_but_never_blocks_the_first_answer`).
* A version pin for `fastembed` I had not tested (0.8.0) was corrected to the tested 0.9.1.

* **Wrong default, caught late:** I first changed the embedder default with an edit that silently did not apply, so my first "shipped pipeline" numbers (regression p50/p95 774/892 ms, HTTP p50/p95 620/863 ms) were measured with reranking on the OLD slower embedder. I noticed when a scale-test latency (151 ms) contradicted the 5 ms measured earlier. Fixed, a guard test added (`test_default_embedder_is_the_fast_runtime`), and both measurements were redone; the numbers in this report are the corrected ones. Quality was identical (the same model); only speed changed.

## 21. Limits of this experiment
63 queries (30 test) written by the same person who wrote the policies; a one-query difference is 1.6 to 3.3 points, so gaps of 0.03 to 0.05 are within noise, and only the larger effects (reranking, the embedding speedup) should be trusted. No confidence intervals were computed. Policies are short (about 15 chunks per company); techniques such as hybrid search, doc2query and sentence chunking may behave differently on 30-page documents. The LLM techniques ran on one model (`gpt-oss-20b`).

**HTTP benchmark of the shipped pipeline** (Docker container, real Groq, fast MiniLM + rerank, `python -m eval.bench --n 45`): 45 requests, 0 errors (31 answered, 9 not covered, 5 small talk), **p50 507 ms, p95 737 ms, max 1,115 ms** (target p95 < 3 s). Raw: `results/bench.json`.
**Reliability note:** during this work the machine's network to Groq and Docker Hub intermittently reset connections. A connection that never opens is now retried once (safe: nothing reached the model); everything else keeps the "no retry, clear try-again" behaviour.

## 22. Do we cope with 10-30 page policies? (the brief says "10 to 30 pages" in context but "short documents" in requirements)
**How the two statements fit:** the context describes the real-world problem we must be able to handle (long policies); the requirement tells us what to build for the demo (three *short* sample policies, no realistic backend). We did both: the three sample policies are short (about 2 pages each, as required), and the design is meant for long ones: chunking by numbered section, an index (not a scan), per-chunk incremental embedding, a 100-page cap and a 10 MB upload limit (raised from 200 KB after realising real HR PDFs are often 0.5-5 MB).
**What we had not done:** test a long document. `python -m eval.scale_test` builds a synthetic multi-site policy (12 office sites, the same 14 topics repeated per site with different numbers, so many sections look alike: a deliberate distractor test), renders it to a real PDF, ingests it, and asks one targeted question per section (168 questions).

| Measure | Result |
|---|---|
| Size | 16 PDF pages, 168 chunks (27 KB) |
| First ingestion (extract, chunk, embed, store) | 1.31 s |
| Update with ONE sentence changed | 1 chunk embedded, 167 reused, 0.53 s |
| Vector search only | recall@1 0.929, recall@3 1.0, p50 7.5 ms |
| Hybrid (vector + BM25) | recall@1 1.0, recall@3 1.0, p50 5.7 ms |
| **Vector + rerank (shipped)** | recall@1 1.0, recall@3 1.0, p50 150.4 ms |

**Reading it:** nothing breaks at this size: ingestion is seconds, and an update only re-embeds what changed. Search time is dominated by whether we rerank (about 145 ms for 10 candidates of longer text), not by document size. On near-duplicate sections, plain vector search starts to confuse them (rank-1 0.929) while **hybrid and reranking both fix it** (1.0): so the techniques that were marginal on short documents matter more on long, repetitive ones. Hybrid is almost free here, so for a corpus like this `RETRIEVAL_MODE=hybrid` is worth switching on.
**Caveats:** synthetic and templated (not real prose); 16 pages rather than 30 because the sections are short, but 168 chunks is in the range of a real 20-30 page policy; the task is easy in one respect (every query names its site). A real long document would also need real section structure, which our parser requires ("3.1 Title").

---

## 23. Groundedness: explicit measurement (answers the question "are answers grounded in the cited policy?")
**Definition.** A sentence of an answer is **grounded** when (a) at least half of its content words appear in the policy text it cited (or in the question) and (b) every number in it appears in that cited text or the question. Computed with no LLM (`app/guardrails.sentence_groundedness`).
We report four deterministic numbers and one LLM-judged number, and they measure different things:

| Metric | What it tells you | Result (33 answered cases, 40 sentences) |
|---|---|---|
| **Sentence groundedness** | share of answer sentences supported by the cited text | **100 %** |
| Fully grounded answers | share of answers in which every sentence is grounded | **100 %** |
| Citation validity | cited sections really exist in that company's policy | **100 %** |
| Numeric grounding | every number comes from the cited text or the question | **100 %** (also 0 % numeric hallucination in the main evaluation) |
| LLM judge, faithful (`gpt-oss-120b`, independent of the answering model) | every claim backed by the excerpts | **100 %** (35/35 judged) |

Reproduce: `python -m eval.groundedness` (reads the stored results, no LLM, prints every flagged sentence for human review; none were flagged). The same score is now also computed by `python -m eval.run` (`sentence_groundedness` in the summary).
**Can the metric say no?** Yes. `tests/test_groundedness.py` feeds it an invented number, an invented extra claim without numbers ("complimentary breakfast vouchers") and off-topic text, and it flags all three. The judge was likewise validated on 5 deliberately wrong answers (8/8 correct verdicts).
**Limits.** The check is lexical: it cannot see a wrong claim that reuses cited words and no new number (for example "interns are eligible" when the policy says they are not). That is why we also use the LLM judge and show the exact policy text beside every answer. 100 % on 33 answers is evidence of a working pipeline, not proof that it can never hallucinate.
