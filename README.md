# Commute Policy Assistant

Answers an employee's transport-policy questions **from their own company's current policy only**, cites the
document, version, section and page, says plainly when the policy is silent, and degrades to a clear
"try again" when something fails. Stack: Python, FastAPI, Chroma, MiniLM embeddings, Groq LLM, JWT.

## Run it (one command)

```bash
cp .env.example .env          # then put your free Groq key in LLM_API_KEY  (never committed)
docker compose up             # first build downloads the embedding model into the image
```

Open <http://localhost:8000> (if the port is taken: `PORT=8010 docker compose up`).
Without a key the app still starts, using an offline fake LLM; answers are then crude but safety rules still hold.

| Task | Command |
|---|---|
| Tests (fake LLM, no network, ~10 s) | `pip install -r requirements-dev.txt && python -m pytest -q` |
| Evaluation set + score (real LLM, before/after) | `python -m eval.run` (see [EVALUATION.md](EVALUATION.md)) |
| Offline evaluation | `python -m eval.run --llm fake` |
| Latency over HTTP | `python -m eval.bench --base http://localhost:8000` |
| Get a token (mock identity provider) | `python -m scripts.issue_token acme` / `... borealis admin` |
| Upload a new policy version | `python -m scripts.ingest_policy acme data/policies/acme_v2.pdf` |
| Sync a whole folder (uploads only new/changed files; `--watch 30` to poll) | `python -m scripts.sync_policies data/policies` |

Running without Docker: `pip install -r requirements.txt`, then
`set -a; . ./.env; set +a; DEMO_MODE=1 uvicorn app.main:create_app --factory`.

## API

| Endpoint | Who | What |
|---|---|---|
| `POST /ask` `{"question": "..."}` | any employee JWT | grounded answer. **No company field exists**; the company is read from the token |
| `PUT /admin/policy` raw body, `Content-Type: application/pdf` or `text/markdown` | JWT with `role=admin` | ingest a new version of **the token's own company** |
| `POST /ask/stream` | any employee JWT | same answer, delivered as Server-Sent Events: `stage` events (retrieving, drafting with the sections found), then ONE `final` event. Raw model tokens are **not** streamed (see trade-offs) |
| `GET /me` | any JWT | which company/policy version this token answers from |
| `GET /health` | none | liveness |
| `POST /demo/token`, `GET /` | none, only if `DEMO_MODE=1` | demo web page and a stand-in login. **Off by default** |

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
                 vector search (hybrid = vector + BM25 is optional, see EVALUATION.md); top 3; timeout 1 s; fail-closed company check on every chunk
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
| `RERANK=1` (default) / `0` | cross-encoder reranking | best no-LLM gain |
| `EMBEDDER=minilm-fast` (default) / `minilm` / `bge-small` | embedding model/runtime | `minilm-fast` is 30x faster; `bge-small` worse |
| `RETRIEVAL_MODE=vector` (default) / `hybrid` | adds BM25 keyword fusion | small gain, none on top of rerank |
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
The brief's context says real policies are 10-30 pages; its requirements ask for three *short* sample documents. We did both: the samples are short, and the system is built and **stress-tested for long ones** (synthetic 16-page, 168-section policy: ingests in about 1.3 s, a one-sentence update re-embeds 1 chunk, search stays fast; see [EVALUATION.md](EVALUATION.md) section 22). Limits: 10 MB upload, 100 pages, sections must be numbered (`3.1 Title`). On long, repetitive policies plain vector search confuses near-duplicate sections, so consider `RETRIEVAL_MODE=hybrid`.

## Limitations and known failures
* The LLM can still mis-phrase an answer inside valid citations; the citation card lets the user verify, and the eval set measures how often.
* Groq's free tier limits to ~8,000 tokens/minute: under heavier load users get "try again". A paid tier, a second provider as fallback, or a local model would fix it.
* Policies must follow the numbered-section structure (`3.1 Title`); free-form PDFs are rejected.
* No per-user rate limiting, no audit database, no question history (by choice: privacy and scope).
* The injection regex only *reports* suspicious text; it is not the defence.
* Latency numbers are from one dev machine with a free API, not production figures.

## Project layout
```
app/      main.py (API)  auth.py  service.py (pipeline)  retrieval.py  store.py  ingest.py
          chunking.py  documents.py (PDF/MD)  embeddings.py  llm.py  guardrails.py  config.py
data/policies/   3 companies + Acme v2, as .md source and .pdf (made by scripts/make_pdfs.py)
tests/    97 tests, fake LLM, no network      eval/   evalset.jsonl, run.py, bench.py, results/
web/      demo page    scripts/   token issuer, policy uploader, folder sync, PDF generator
```
