# 2-minute screen recording: script

Prep (before recording): `docker compose up` is running, browser at http://localhost:8000 (use `PORT=8010 docker compose up` if 8000 is busy), terminal visible next to it.

| Time | Show | Say |
|---|---|---|
| 0:00 | Terminal: `docker compose up` already healthy; `git log --oneline` | "One command starts everything. Commits show steady work." |
| 0:15 | Web page, click **Acme employee** under 'Demo accounts' (or type employee@acme.example / demo-pass-123), click "After what time is a security escort mandatory...?" | "Answer comes from Acme's policy only. Each answer shows document, version, section, page. Click 'Show the policy text'." |
| 0:35 | Sign out, sign in as **Borealis**, same question | "Same question, different company, different rule: 10 PM, not 8 PM. The company comes from the JWT, never from the request." |
| 0:55 | Ask "Are pets allowed in the cab?" | "When the policy is silent it says so and points to the transport desk. No guess." |
| 1:10 | Terminal: `python -m scripts.ingest_policy acme data/policies/acme_v2.pdf`, then ask Acme the escort question again | "Upload version 2 as the Acme admin. Only the new version answers now: 9 PM. A failed upload would leave v1 untouched." |
| 1:30 | Terminal: `python -m pytest -q` (62 passed) then `python -m eval.run --llm fake --no-write` or show `EVALUATION.md` table | "62 tests with a fake LLM; a fixed 42-case set; 0 cross-company leaks; p95 under 1 s against a 3 s target." |
| 1:50 | Show `docker compose stop` is NOT needed; end on README trade-offs table | "Hybrid search was measured and did not help, so the simpler vector search ships. Trade-offs are in the README." |
