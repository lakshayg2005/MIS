# Commute Policy Assistant

Answers an employee's transport-policy questions **from their own company's current policy only**, cites the
document, version, section and page, says plainly when the policy is silent, and degrades to a clear
"try again" when something fails. Stack: Python, FastAPI, Chroma, MiniLM embeddings, Groq LLM, JWT.

## Run it (one command)

```bash
cp .env.example .env          # then put your free Groq key in LLM_API_KEY  (never committed; optional, see below)
docker compose up             # first build downloads the models into the image
```

Open <http://localhost:8000> (if the port is taken: `PORT=8010 docker compose up`) and sign in with one click as a demo employee
of any company (accounts `employee@acme.example`, `employee@borealis.example`, `employee@cygnus.example`, password `demo-pass-123`).
Without a key the app still starts, using an offline fake LLM; answers are then crude but every safety rule still holds.
The three policies (about 14 pages each) are loaded automatically on first start.

**Zero-setup checks, all inside the image (no Python install needed):**
```bash
docker compose run --rm api python -m eval.run --llm fake --no-write   # the evaluation set, offline, no key
docker compose run --rm api python -m eval.groundedness                # groundedness report from the stored results
```

| Task | Command |
|---|---|
| Tests (fake LLM, no network, ~10 s) | `pip install -r requirements-dev.txt && python -m pytest -q` |
| Evaluation set + score (real LLM, before/after) | `python -m eval.run` (see [EVALUATION.md](EVALUATION.md)) |
| Offline evaluation (retrieval metrics are real; answer text is a crude stand-in) | `python -m eval.run --llm fake --no-write` |
| Latency over HTTP | `python -m eval.bench --base http://localhost:8000` |
| Get a token (mock identity provider) | `python -m scripts.issue_token acme` / `... borealis admin` |
| Upload a new policy version | `python -m scripts.ingest_policy acme data/policies/acme_v2.pdf` |
| Sync a whole folder (uploads only new/changed files; `--watch 30` to poll) | `python -m scripts.sync_policies data/policies` |

Running without Docker: `pip install -r requirements.txt`, then
`set -a; . ./.env; set +a; DEMO_MODE=1 uvicorn app.main:create_app --factory`.

## Which file is the evaluation set?
**`eval/evalset.jsonl` is the case-study evaluation set**: 48 verified cases, one JSON line each with the required fields `id, input, expected, actual, pass, notes` (paraphrases, cross-company traps, "not covered", injection, version-update and small-talk cases). Rerun and re-score it any time with `python -m eval.run` (offline, no key: `docker compose run --rm api python -m eval.run --llm fake --no-write`). How a case is judged (expected status, cited section, required facts, forbidden text, no foreign-company citation) and every metric are defined in [EVALUATION.md](EVALUATION.md), section 1.

| File | What it is | Used by |
|---|---|---|
| `eval/evalset.jsonl` | **the evaluation set** (end to end: retrieval and answers) | `python -m eval.run` |
| `eval/retrieval_hard.jsonl` | 63 harder retrieval-only queries, used only for the bonus before/after comparison (different fields, no answers) | `python -m eval.retrieval_lab` |
| `eval/results/*.json` | raw results behind the numbers in the report | read-only |

## Brief requirements, one by one (where it is, how it is proven)
| Requirement from the brief | Where | Proof |
|---|---|---|
| Grounded answers that cite document and section | `service.py` builds every citation (document, version, section, page, excerpt) from our own metadata | `tests/test_service_robustness.py`; eval: citation precision/recall 1.0 |
| An explicit "not covered" reply when evidence is missing | similarity gate (no LLM call) + the model's own `not_covered`; fixed message pointing to the transport desk | `test_no_evidence_skips_llm_entirely`; eval: not-covered recall 100 % |
| No obedience to instructions found inside retrieved documents | injected sections quarantined at ingestion; excerpts fenced as data; no tools; output validated | `test_injected_document_text_is_data_not_instructions`; eval `inj-*` cases; red-team run with quarantine off |
| A question API behind a JWT that carries the company; retrieval restricted to that company on the server, no client-controlled override | `auth.py` + `store.py` + `retrieval.py` (see "How the JWT works here" below) | `test_auth_rejections`, `test_client_cannot_choose_company`, `test_retrieval_never_returns_another_companys_chunk` |
| Ingestion of a policy, including a new version; only the new one answers; a failed ingestion leaves nothing half-updated | `ingest.py` atomic version switch | `tests/test_ingest_versions.py` |
| Three policy documents with overlapping, contradictory rules | `data/policies/`: full-length versions (13-14 pages each, PDF + Markdown) and `data/policies/short/`: 2-page versions of the same core rules (test fixtures). Both readings of the brief ("short" in the requirements, "10 to 30 pages" in the context) are covered; the contradictory core is identical in both | eval cross-company cases |
| At most 2 LLM calls and at most 3 cited chunks | `service.py` loop + `top_k = 3` | `test_never_more_than_two_llm_calls_and_three_citations` |
| Evaluation set (10+ questions, JSONL with `id, input, expected, actual, pass, notes`), retrieval and answer quality, rerunnable | `eval/evalset.jsonl` (48 cases), `python -m eval.run` | [EVALUATION.md](EVALUATION.md) |
| Bonus: a retrieval improvement, before and after | reranking + faster runtime shipped; many others measured | [EVALUATION.md](EVALUATION.md) sections 16-22 |

## How the JWT works here
The brief says: *"A question API behind a JWT that carries the company. Retrieval is restricted to that company on the server, with no client-controlled override."* In plain words:
1. **JWT** = a signed token. After login the identity provider hands the employee a token containing claims such as `sub` (who), `company` (which company) and `exp` (expiry), signed with a secret only the servers know. Nobody can edit it without breaking the signature.
2. **"Carries the company"**: the company is a claim *inside* that signed token, not something the user types.
3. **"Restricted to that company on the server"**: the server reads the company from the verified token and searches only that company's policy data.
4. **"No client-controlled override"**: there is no `company` field, query parameter or header that can change which company is searched. Sending `{"company": "borealis"}` in the body is simply ignored, and editing the token to say `borealis` invalidates the signature (HTTP 401).

**Where it is used:** `app/auth.py` (`issue_token` = our mock identity provider, used by `POST /auth/login`; `verify_token` checks signature, expiry, issuer, audience, subject and company format, with the algorithm pinned; `current_principal` is the FastAPI dependency). Every endpoint depends on it: `/ask`, `/ask/stream`, `/me` and `/admin/policy` (admin role required, and an admin can only touch their own company). `scripts/issue_token.py` prints a token for testing; the web page signs in through `POST /auth/login`. To add a user: `python -m scripts.create_user alice@acme.example acme employee 'a-long-password'`.

**Login vs JWT:** the login (email + password) is how a user *proves who they are*; the JWT is what the server then trusts on every later request. The brief only asks us to build the issuer, so accounts here are a small SQLite table with scrypt hashes, enough to demonstrate the flow honestly. A production system would delegate login to the company's identity provider (SSO) and only *verify* tokens.

## API

| Endpoint | Who | What |
|---|---|---|
| `POST /ask` `{"question": "..."}` | any employee JWT | grounded answer. **No company field exists**; the company is read from the token |
| `PUT /admin/policy` raw body, `Content-Type: application/pdf` or `text/markdown` | JWT with `role=admin` | ingest a new version of **the token's own company** |
| `POST /ask/stream` | any employee JWT | same answer, delivered as Server-Sent Events: `stage` events (retrieving, drafting with the sections found), then ONE `final` event. Raw model tokens are **not** streamed (see trade-offs) |
| `GET /me` | any JWT | which company/policy version this token answers from |
| `GET /health` | none | liveness |
| `POST /auth/login` `{email, password}` | none (this is the login) | returns a signed JWT carrying the user's company and role. Generic 401 on any failure, 5 attempts/min per client+email, passwords stored only as salted scrypt hashes |
| `GET /`, `GET /demo/accounts` | none, only if `DEMO_MODE=1` | the web page and its one-click demo logins. **Off by default**; demo mode also seeds `employee@<company>.example` and `admin@<company>.example` (password `demo-pass-123`, override with `DEMO_PASSWORD`) |

Responses: `200 answered` (with citations), `200 not_covered`, `200 smalltalk` (greetings, thanks, "are you a dog?", "how do you work?", "do you store my chats?", "can you book a cab?": fixed truthful replies from plain code, no LLM call, no citations; only short messages that are *entirely* such a question match), `503 error` + `Retry-After` (try again), `401/403/400/413/415/422` for bad token/role/input/uploads.

```
curl -s localhost:8000/ask -H "Authorization: Bearer $(python -m scripts.issue_token acme)" \
     -H 'content-type: application/json' -d '{"question":"Am I eligible for a 9:30 PM drop?"}'
```

## Architecture and flow

```
 employee ──JWT──► /ask ──► auth.py        verify signature, exp, iss, aud; company = token claim ONLY
                              │
                              ▼
                          service.py       (every safety decision is plain code)
   1 retrieve ─► Chroma collection of THAT company, ACTIVE version only (SQLite says which),
                 10 candidates, then a cross-encoder reranks them and we keep the top 3
                 hybrid search (vector + BM25) then a cross-encoder reranker over 6 candidates; top 3; timeout 1 s; fail-closed company check on every chunk
   2 no/weak evidence? ─► "not covered" immediately, WITHOUT calling the LLM
   3 ONE LLM call: excerpts fenced as DATA, JSON answer + numbers of the excerpts used
   4 validate: JSON shape, cited numbers ⊆ excerpts given, ≤3, ≥1, no other company named, and every number
     in the answer must appear in the cited text or the question (numeric grounding)
   5 citations built from OUR metadata (document, version, section, page, text), never from model text
                              ▼
          answered | not_covered | error ("try again")

 admin ──JWT(role=admin)──► PUT /admin/policy ──► ingest.py
   parse + embed everything in memory ─► stage chunks (invisible) ─► ONE SQLite transaction flips the
   active version ─► purge old chunks.  Any failure before the flip leaves the old version live.
```

**Principle:** the LLM only phrases an answer from excerpts we choose. Identity, isolation, versioning,
"is there evidence", citations and leak checks are ordinary code the model cannot override.

### How we stay inside the two numbers
* **≤ 2 LLM calls per question.** Normal path = **1** call. The 2nd call exists only to repair a *malformed* reply, and only if ≥ 1 s of the time budget is left. Quota errors, HTTP errors and timeouts are **not** retried (the time or quota is already spent). The one exception: a *connection that never opened* (nothing reached the model, no tokens used) gets one fast retry if the time budget allows; this does not count as an LLM call. `max_llm_calls = 2` is enforced in a loop, and tested.
* **≤ 3 cited chunks.** Retrieval returns only the top 3 (`top_k = 3`), the LLM only ever sees those, and citations are capped at 3 in code. Also tested.
* **p95 < 3 s.** Retrieval ≈ 0.14 s (mostly the CPU embedding of the question); one short LLM call ≈ 0.4–0.9 s; hard budget 2.8 s (retrieval timeout 1 s, LLM timeout = remaining budget). Measured numbers: [EVALUATION.md](EVALUATION.md).

## Incremental ingestion and change detection
* Each chunk gets a **content hash** (SHA-256 of embedding-model name + chunk text). A new version reuses the stored vector of every chunk whose hash is unchanged and **embeds only new or changed chunks**. Measured: changing one sentence of Acme's policy embedded **1 chunk and reused 14**; Acme v1 to v2 embedded 6 and reused 9.
* Each file has a **SHA-256 recorded when it is activated**. Uploading the identical file again is a no-op (nothing embedded, nothing changes).
* `scripts/sync_policies.py` keeps a manifest of file hashes for a policy folder and uploads only new or changed files (`--watch N` polls). Polling is deliberate: no OS-specific dependency, easy to explain.
* The hash includes the **model name**, so switching the embedding model never reuses vectors from another model. Reuse is **per company**.

## Retrieval techniques tried (the bonus), and how to switch them
We compared hybrid, reranking, query rewriting, three chunking strategies, two embedding models and document expansion on a hard 63-query benchmark with a dev/test split (full table in [EVALUATION.md](EVALUATION.md), section 17). **Shipped by default: fast MiniLM + cross-encoder rerank.** Everything else is a switch:

| Setting | Effect | Measured cost / result |
|---|---|---|
| `RERANK=1` (default) / `0`, `RERANK_POOL=6` | cross-encoder reranking and how many candidates it reads | best no-LLM gain; cost grows with the pool (about 40 ms per candidate on 1.5 KB sections) |
| `EMBEDDER=minilm-fast` (default) / `minilm` / `bge-small` | embedding model/runtime | `minilm-fast` is 30x faster; `bge-small` worse |
| `RETRIEVAL_MODE=hybrid` (default) / `vector` | BM25 keyword fusion feeding the reranker | on full-length policies hybrid + rerank(6) has the best MRR and is faster than rerank over 10 vector candidates |
| `QUERY_REWRITE=1` | LLM rewrites the question first | same quality as rerank, +1 LLM call, +~0.5 s |
| `EXPAND_QUESTIONS=1` | LLM writes likely questions per chunk at ingestion | good recall, free at question time, LLM cost at ingestion |
| `CHUNKING=section` (default) / `sentence` / `heading` | chunk granularity | both alternatives measured worse |

Run the comparison yourself: `python -m eval.retrieval_lab` (no LLM) or `python -m eval.retrieval_lab --llm`. If you change `EMBEDDER`, wipe the stored vectors first (`docker compose down -v`).

## Retrieval speed: what we measured, and what we did not build
Per question: embedding it on CPU = **134 ms**, Chroma HNSW search = **1.1 ms**, registry lookup = **0.04 ms**. So 99 % of retrieval time is embedding the question.
Clustering or index tuning could save about 1 ms, so it was **not built** (about 15 chunks per company). What helps is an LRU cache of recent question embeddings (employees repeat the same questions): a repeated question drops from **150 ms to 1.5 ms**. Chroma already indexes with HNSW; we also filter on company (collection) and active version (metadata filter) on every search.

## Safety model (full list with tests: [GUARDRAILS.md](GUARDRAILS.md))

| Threat | Defence (where) |
|---|---|
| Employee of A reads B's policy | company comes only from the signed JWT; one Chroma collection per company; every returned chunk re-checked (company + version) and the request **fails closed** if one is wrong (`retrieval.py`) |
| Client sends `company=B` in body/query/header | no such field exists; ignored (tested) |
| Forged / expired / `alg=none` / wrong-audience token | algorithm pinned to HS256, `exp/iss/aud/sub` required (`auth.py`, tested) |
| Admin of A overwrites B | ingestion always targets the token's own company (tested) |
| Injection text inside a policy ("ignore previous instructions…") | excerpts are fenced as data and the system prompt says so; the model cannot call tools or reach other companies; answers naming another company are rejected; such sections are flagged at ingestion. Two sample policies contain real injections and are in the eval set |
| Injection in the user's question | same: it can change wording at worst, never the company or the data the model sees |
| Model invents a number, time or amount | numeric grounding: every number must appear in the cited excerpt or the question, else the answer is rejected (`app/guardrails.py`, tested) |
| Model invents a section / cites nothing / returns garbage | citations come from our metadata; invalid JSON, out-of-range numbers, missing citations → one repair try → "try again" (parametrised tests) |
| Half-updated policy | atomic version flip, rollback of staged chunks, startup cleanup (tests simulate embedder crash and DB crash) |
| Big or hostile uploads | size limit, page limit, PDF validation, no OCR, section structure validated; failures return 4xx and change nothing |
| Spam / quota burn | per-user rate limit (20/min), 429 with a clear message |
| Injection text inside a policy | sections flagged at ingestion are **quarantined** (stored, never retrieved) |
| Secrets | key only in `.env` (git-ignored); app warns if the dev JWT secret is in use; logs hold company, status and chunk ids, **not** question text |

## Design decisions and trade-offs (what we chose, and what we gave up)

| Decision | Why | Cost / alternative rejected |
|---|---|---|
| One Chroma collection **per company** | leak impossible by construction; a metadata filter is one forgotten `where` from a leak | more collections; fine up to hundreds of tenants, beyond that use a filter plus the same defensive check |
| **SQLite pointer** decides the active version; chunks of a new version are staged first | version switch is one transaction; DB itself forbids two active versions | two stores to keep consistent; startup purge handles a crash in between |
| Chunk **by numbered section** | citations mean "§3.1"; splitting a rule from its exception changes meaning | documents must be structured; unstructured ones are rejected with a clear error instead of guessed at |
| **Local MiniLM** embeddings (fastembed runtime) | free, offline, no quota, ~5 ms per question (the same model was 145 ms on Chroma's default runtime, with identical results) | weaker than hosted embeddings |
| **Cross-encoder reranking** (default) | measured on a 63-query hard benchmark: recall@1 0.67 → 0.77, recall@3 0.87 → 0.94, for ~70-100 ms and no LLM call | extra dependency and model; on easy questions it sometimes puts the right section 2nd (still in the top 3) |
| **One** LLM call, no query rewriting or judge call | latency budget and the 2-call cap | no LLM rewriting or LLM-as-judge; checks are deterministic code instead |
| Similarity **gate before** the LLM | "not covered" is instant, free and cannot hallucinate | a too-high threshold could hide answerable questions; tuned on the eval set (`MIN_SIMILARITY`) |
| Citations **built by code** from numbers the model picks | model cannot fabricate a section or a source | we can verify the cited excerpts exist, not that the model picked the *best* one |
| Top **3** chunks to the LLM | = citation cap; ~40 % fewer tokens; eval shows the right section is always in the top 3 | a rule split across 4+ chunks would be missed |
| Plain `httpx` to Groq's OpenAI-compatible API | ~30 lines, no SDK; the same code works for Gemini | no built-in retries (deliberate: see budget) |
| Hand-written BM25 + RRF (hybrid), **off by default** | explainable in a minute, no dependency; we measured it | it did not beat plain vector search on this data and cost latency, so we do not ship it as default (EVALUATION.md) |
| HS256 shared secret | simplest verifiable JWT for a mock issuer | production would use RS256 with the identity provider's public key (one-line change) |
| PDF in, **no OCR** | wrong text would mean wrong citations, which is worse than a refused upload | scanned PDFs are rejected |
| **Progress + typed reveal, not raw token streaming** | The answer is only shown after the guardrails (JSON, citations, numbers, support) have checked it; streaming raw tokens would show text we might then have to retract. The UI shows real pipeline stages (which sections were found) and reveals the validated answer word by word | perceived speed is lower than true token streaming; total time is already about 0.7 s |
| **No server-side conversation storage** | Privacy (questions can be sensitive), nothing to leak across tenants, no retention/GDPR duty, and every question stays independent so the 2-call / 3-chunk limits hold. The chat lives only in the browser tab (sessionStorage) and is cleared on company switch | no follow-up questions ("and on Sunday?"): each question must stand alone. Follow-ups would need history in the prompt (more tokens, more injection surface, per-user storage). Listed as future work |
| Free tier Groq (8k tokens/min) | zero cost per the rules | ≈ 10 questions/minute; see limitations |

## Policy length: "10 to 30 pages" vs "short documents"
The brief's context says real policies are 10-30 pages; its requirements ask for "three short policy documents" ("short" is not defined). We cover **both** readings:
* `data/policies/` holds three **full-length policies (13-14 pages, about 9,000 words each)**: the verified numbered core rules (sections 1-7/8, the contradictory ones the evaluation checks) followed by realistic chapters (definitions, roles, rosters, vehicles and drivers, tracking, safety procedures, contingency, grievances, audit, governance). The extra chapters were drafted by an LLM under two checked constraints: no numbers or amounts (so they cannot compete with a core rule) and no mention of another company (`scripts/gen_long_policies.py`). These are what the app loads and what the evaluation runs on.
* `data/policies/short/` holds the **2-page** versions of the same core rules; the unit tests use them because exact chunk counts are asserted.
Also stress-tested with a synthetic multi-site policy (168 near-duplicate sections): ingests in about 1.3 s; a one-sentence update re-embeds 1 chunk ([EVALUATION.md](EVALUATION.md)). Limits: 10 MB upload, 100 pages, numbered sections (`3.1 Title`).

## Limitations and known failures
* The LLM can still mis-phrase an answer inside valid citations; the citation card lets the user verify, and the eval set measures how often.
* Groq's free tier limits to ~8,000 tokens/minute: under heavier load users get "try again". A paid tier, a second provider as fallback, or a local model would fix it.
* Policies must follow the numbered-section structure (`3.1 Title`); free-form PDFs are rejected.
* Basic per-user rate limiting only (20 questions/min), no audit database, no server-side question history (by choice: privacy and scope).
* The injection regex only *reports* suspicious text; it is not the defence.
* Latency numbers are from one dev machine with a free API, not production figures.

## Project layout
```
app/      main.py (API)  auth.py  service.py (pipeline)  retrieval.py  store.py  ingest.py
          chunking.py  documents.py (PDF/MD)  embeddings.py  llm.py  guardrails.py  config.py
data/policies/   3 companies + Acme v2, as .md source and .pdf (made by scripts/make_pdfs.py)
tests/    171 tests, fake LLM, no network      eval/   evalset.jsonl, run.py, bench.py, results/
web/      demo page    scripts/   token issuer, policy uploader, folder sync, PDF generator
```
