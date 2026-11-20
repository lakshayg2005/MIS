# Guardrails

Two questions: **(A)** how do we make sure one company's rule never reaches another company's employee, and
**(B)** how do we stop the LLM from making things up. The design rule for both: **never ask the model to behave;
make the bad outcome impossible or detectable in plain code.**

## A. Cross-company isolation: 6 independent layers (any one holding is enough)

| # | Layer | Where | What it guarantees | Proof (test) |
|---|---|---|---|---|
| 1 | **Identity from a signed token only** | `app/auth.py` | The company is a claim in a JWT we verify (HS256 pinned, `exp/iss/aud/sub` required, company must match `^[a-z][a-z0-9-]{1,31}$`). `/ask` has no company field; body, query and headers cannot override it | `test_auth_rejections` (8 attacks incl. `alg=none`, forged, expired), `test_client_cannot_choose_company` |
| 2 | **Physical separation** | `app/store.py` | One Chroma collection per company. A search physically cannot see another company's vectors, there is no `where` clause to forget | `test_retrieval_never_returns_another_companys_chunk` (3 companies × 2 modes) |
| 3 | **Version scoping (metadata filter)** | `store.query` / `get_chunks` | Every search also filters `version == active version` (the pointer lives in SQLite), so a superseded policy can never answer | `test_new_version_replaces_old_everywhere` |
| 4 | **Fail-closed result check** | `app/retrieval.py` | Every returned chunk is re-checked: `company == caller` and `version == active`. One wrong chunk aborts the request with "try again" and logs a critical alert; nothing partial is shown | `test_isolation_violation_fails_closed` |
| 5 | **The model only ever sees the caller's chunks** | `app/service.py` | The prompt is built from layer-4-checked chunks, so even a fully obedient model has nothing of another company to reveal. It has no tools, no memory, no cross-request state | `test_prompt_injection_in_user_question_cannot_cross_tenants` |
| 6 | **Output check** | `parse_model_output` | An answer that names another tenant is rejected before the user sees it; citations are built from our own metadata, not model text | `test_garbage_model_output_never_reaches_user` (case "mentions another company") |

Also: an admin can only ingest into the **token's own** company (`test_admin_rules`); the question-embedding cache
stores only text→vector, never results (`test_cache_never_mixes_company_results`); incremental ingestion reuses vectors only
from the **same company** and the **same embedding model** (`test_other_company_vectors_are_never_reused`,
`test_different_embedding_model_never_reuses_old_vectors`).

## B. Hallucination and misbehaviour: guardrails

| Risk | Guardrail | Where | Proof |
|---|---|---|---|
| Model answers from memory when the policy is silent | **Evidence gate**: below the similarity threshold the LLM is **not called**; reply is the fixed "policy does not say, contact the transport desk" | `service._ask` | `test_no_evidence_skips_llm_entirely` |
| Model invents a number, time or amount | **Numeric grounding**: every number in the answer must appear in the cited excerpt or the question (`8 PM` = `8:00 PM`, `1,500` = `1500`). Otherwise the answer is rejected, one repair attempt, then "try again" | `app/guardrails.py` | `test_runtime_rejects_invented_number`, `test_ungrounded_numbers` |
| Model invents or mis-numbers a citation | The model returns only the **numbers of the excerpts** it used; we validate them (integers, within range, at least one, at most 3) and build document/version/section/page ourselves | `parse_model_output` | parametrised garbage test (12 kinds) |
| Instructions hidden in a policy document | **(1) Quarantine:** sections that look like instructions to an AI are flagged at ingestion, stored, but **never retrieved** (the admin is told which sections). **(2)** If anything slips past, excerpts are fenced as **data**, the fence cannot be closed by the document, and the model has no tools | `build_prompt`, `chunking._INJECTION_RE` | `test_injected_document_text_is_data_not_instructions`, `test_document_cannot_close_the_data_fence`, eval cases `inj-*` |
| Malformed / empty / non-JSON output | Strict parse, one repair try, never more than 2 calls; otherwise a clear "try again" | `service` loop | `test_one_retry_recovers...`, `test_never_more_than_two_llm_calls...` |
| Model writes free-form or off-topic text | **Minimum support**: at least 35 % of the answer's content words must come from the cited text, else rejected (lowest real answer measured: 54.5 %) | `app/guardrails.py` | `test_off_topic_free_text_is_rejected` |
| One user floods the assistant (burns the shared LLM quota) | **Per-user rate limit** (20 questions/min, HTTP 429 with a friendly message). Cheap validation runs before it | `main.RateLimiter` | `test_rate_limit_per_user` |
| Over-long or hostile input | Question ≤ 500 characters; uploads size-, page- and structure-checked | `main.py`, `documents.py` | `test_bad_questions_rejected`, `test_bad_uploads_rejected...` |
| Stale answers after a policy update | Atomic version pointer, old chunks purged | `ingest.py` | `test_new_version_replaces_old_everywhere` |

## Is it strong enough against malicious prompts and malicious documents? An honest assessment

| Attack | Verdict | Why |
|---|---|---|
| User asks for another company's data (any wording, any language) | **Strong** | architectural: the model never receives another company's text, and the company comes from the signed token. Wording cannot matter |
| Forged / replayed / tampered token | **Strong** | verified signature, pinned algorithm, expiry, audience; short test list above |
| User tries to make the model say something false about the user's OWN company's policy | **Moderate** | numeric grounding, minimum support and citation checks help, but a wrong claim that uses no new number and reuses cited words can pass. Mitigated by showing the exact policy text next to every answer |
| Malicious text inside a policy document | **Moderate to strong** | quarantine removes obvious injections before the model sees them; the regex is evadable (paraphrase, other languages), so the second layer (data fence, no tools, output validation) is what matters. Red-team run with quarantine OFF: both injected documents were ignored by the model (6/6 cases) |
| A malicious or compromised **admin** uploading false rules | **Not defended** | an admin is trusted by definition; we limit the blast radius to their own company and keep versions, but there is no approval workflow. This is the real production gap |
| Denial of wallet (spamming questions) | **Basic** | per-user limit; a real deployment also needs per-company and global quotas and abuse alerts |
| Data stored about users | **Minimal by design** | no question text is stored server-side; the chat history lives only in the user's browser tab |

**Bottom line:** isolation is robust by construction; hallucination and injection are reduced and measured, not eliminated. That is why every answer shows its source text.

**What the guardrails do not catch (be upfront):** a wrong *claim that uses no new number* (e.g. "interns are eligible"
when the policy says they are not). That is why we also *measure* faithfulness (below) instead of assuming.

## C. Measuring hallucination and quality (`python -m eval.run --judge openai/gpt-oss-120b`)

Definitions live in [eval/metrics.py](eval/metrics.py) and are unit-tested.

| Metric | Meaning | Kind |
|---|---|---|
| **numeric_hallucination_rate** | answered cases whose answer has a number absent from the cited text and question | deterministic |
| **guardrail_rejected_attempts** | how often the model's raw output was rejected (a direct count of how often it misbehaved) | deterministic |
| **support_ratio** | share of the answer's content words found in the cited text | deterministic |
| **judge_faithful_rate** | every claim supported by the excerpts? (a *different, larger* model judges, to limit self-grading bias) | LLM judge |
| **judge_relevant_rate** | does the answer address the question? | LLM judge |
| **citation_precision / recall** | cited sections that are expected / expected sections that were cited | deterministic |
| **fact_recall** | required facts present in the answer | deterministic |
| **abstention recall (1 − false-answer rate)**, **precision**, **over-abstention** | does it say "not covered" exactly when it should? | deterministic |
| **cross_company_leaks** | citations from another company or forbidden text | deterministic |

Retrieval: recall@1/3, precision@3, hit@k, MRR, nDCG@3. Results: [EVALUATION.md](EVALUATION.md).
