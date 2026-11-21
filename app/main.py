import json
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .auth import Principal, current_principal, issue_token
from .config import Settings
from .embeddings import make_embedder
from .ingest import Ingestor, IngestError
from .llm import make_llm
from .retrieval import Retriever
from .service import ERROR_MSG, QuestionService
from .store import PolicyStore
from .users import UserStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


class AskBody(BaseModel):
    # NOTE: deliberately no company field. Company is taken from the JWT only.
    question: str = Field(min_length=1)


class RateLimiter:
    """Sliding window per user (JWT subject). Stops one user from burning the shared LLM quota (the free tier
    allows ~10 questions/min in total). In memory: fine for one process; use Redis if you run several."""

    def __init__(self, per_min: int):
        self.per_min, self._hits, self._lock = per_min, defaultdict(deque), threading.Lock()

    def allow(self, who: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[who]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True


def seed_if_empty(settings, store, ingestor):
    """First start only: load each company's lowest-version file so `docker compose up` works with no setup."""
    for path in sorted(Path(settings.seed_dir).glob("*_v1.*")):
        company = path.name.split("_")[0]
        kind = path.suffix.lstrip(".")
        if kind in ("pdf", "md") and store.active(company) is None and not (kind == "md" and path.with_suffix(".pdf").exists()):
            try:
                ingestor.ingest(company, path.read_bytes(), kind)
                log.info("seeded %s", company)
            except IngestError as e:
                log.error("seed failed for %s: %s", company, e)


def create_app(settings: Settings | None = None, llm=None, embedder=None) -> FastAPI:
    settings = settings or Settings()
    if settings.jwt_secret.startswith("dev-only-secret"):
        log.warning("JWT_SECRET is the built-in dev default: set JWT_SECRET before any real use")

    @asynccontextmanager
    async def lifespan(app):
        emb = embedder or make_embedder(settings.embedder)
        store = PolicyStore(settings.data_dir, include_flagged=not settings.quarantine_flagged)
        for c in store.companies():
            store.purge_inactive(c)  # clear chunks left by a crash mid-ingest
        the_llm = llm or make_llm(settings.llm_provider, settings.llm_api_key, settings.llm_model)
        expander = None
        if settings.expand_questions:
            from .expansion import Doc2Query
            expander = Doc2Query(the_llm, str(Path(settings.data_dir) / "doc2query_cache.json"))
        ingestor = Ingestor(store, emb, settings.max_policy_bytes, settings.embedder, settings.chunking, expander)
        seed_if_empty(settings, store, ingestor)
        users = UserStore(str(Path(settings.data_dir) / "registry.db"))
        if settings.demo_mode and users.count() == 0:  # demo accounts only in demo mode
            for c in store.companies():
                users.create(f"employee@{c}.example", c, "employee", settings.demo_password)
                users.create(f"admin@{c}.example", c, "admin", settings.demo_password)
        app.state.users = users
        emb.embed(["warm up"])  # load the model before the first user request
        reranker = None
        if settings.rerank:
            from .rerank import CrossEncoderReranker
            reranker = CrossEncoderReranker()  # loaded once at startup, not on the first user request
            reranker.rerank("warm up", [{"title": "a", "text": "b"}, {"title": "c", "text": "d"}])
        transform = None
        if settings.query_rewrite:
            from .query_transforms import make_rewriter
            transform = make_rewriter(the_llm)
        retriever = Retriever(store, emb, settings.retrieval_mode, settings.top_k, reranker=reranker, transform=transform, pool=settings.rerank_pool)
        app.state.service = QuestionService(settings, store, retriever, the_llm)
        app.state.ingestor = ingestor
        yield

    app = FastAPI(title="Commute Policy Assistant", lifespan=lifespan)
    app.state.settings = settings
    app.state.limiter = RateLimiter(settings.rate_limit_per_min)
    app.state.login_limiter = RateLimiter(5)  # 5 login attempts per minute per (client, email): slows password guessing

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/me")
    def me(request: Request, who: Principal = Depends(current_principal)):
        """Which company and which policy version this token answers from (shown in the UI)."""
        act = request.app.state.service.store.active(who.company)
        return {"company": who.company, "role": who.role,
                "policy": {"title": act[1], "version": act[0]} if act else None}

    class LoginBody(BaseModel):
        email: str = Field(min_length=3, max_length=200)
        password: str = Field(min_length=1, max_length=200)

    @app.post("/auth/login")
    def login(body: LoginBody, request: Request):
        """email + password -> signed JWT carrying the company. Generic error, per-(client, email) rate limit."""
        who = f"{request.client.host if request.client else '?'}:{body.email.strip().lower()}"
        if not request.app.state.login_limiter.allow(who):
            return JSONResponse({"detail": "Too many attempts. Please wait a minute and try again."}, status_code=429, headers={"Retry-After": "60"})
        found = request.app.state.users.authenticate(body.email, body.password)
        if found is None:
            raise HTTPException(401, "Invalid email or password")
        company, role = found
        return {"token": issue_token(settings, body.email.strip().lower(), company, role, ttl_s=3600),
                "company": company, "role": role, "expires_in": 3600}

    if settings.demo_mode:
        @app.get("/demo/accounts")
        def demo_accounts(request: Request):
            """DEMO ONLY: lists the seeded demo logins so the web page can offer one-click sign-in."""
            return {"password": settings.demo_password,
                    "accounts": [{"email": f"employee@{c}.example", "company": c} for c in request.app.state.service.store.companies()]}

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")

    def _validated_question(request: Request, body: AskBody, who: Principal):
        """Shared by /ask and /ask/stream: rate limit + length check. Returns (question, error_response)."""
        q = body.question.strip()
        if not q or len(q) > settings.max_question_chars:
            raise HTTPException(400, f"question must be 1-{settings.max_question_chars} characters")
        if not request.app.state.limiter.allow(f"{who.company}:{who.subject}"):
            return None, JSONResponse({"status": "error", "reason": "rate_limited", "citations": [],
                                       "answer": "You are asking too quickly. Please wait a minute and try again."},
                                      status_code=429, headers={"Retry-After": "30"})
        return q, None

    def _log(who, ans):
        log.info("ask company=%s sub=%s status=%s reason=%s ms=%d calls=%d chunks=%s", who.company, who.subject,
                 ans.status, ans.reason, ans.latency_ms, ans.llm_calls, ans.retrieved)

    @app.post("/ask")
    def ask(body: AskBody, request: Request, who: Principal = Depends(current_principal)):
        q, err = _validated_question(request, body, who)
        if err:
            return err
        ans = request.app.state.service.ask(who.company, q)
        _log(who, ans)
        payload = ans.to_dict()
        if ans.status == "error":
            return JSONResponse(payload, status_code=503, headers={"Retry-After": "5"})
        return payload

    @app.post("/ask/stream")
    def ask_stream(body: AskBody, request: Request, who: Principal = Depends(current_principal)):
        """Server-Sent Events: `stage` events while the pipeline runs, then ONE `final` event with the validated answer.
        We deliberately do NOT stream raw model tokens: the answer is only shown after the guardrails have checked it."""
        q, err = _validated_question(request, body, who)
        if err:
            return err
        events: queue.Queue = queue.Queue()

        def worker():
            try:
                ans = request.app.state.service.ask(who.company, q, on_stage=lambda name, info: events.put(("stage", {"step": name, **info})))
                _log(who, ans)
                events.put(("final", ans.to_dict()))
            except Exception:
                log.exception("stream worker failed")
                events.put(("final", {"status": "error", "answer": ERROR_MSG, "citations": [], "reason": "internal"}))
            events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        def gen():
            while (item := events.get()) is not None:
                yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.put("/admin/policy")
    async def put_policy(request: Request, who: Principal = Depends(current_principal)):
        """Body = the raw policy file. Content-Type: application/pdf or text/markdown."""
        if who.role != "admin":
            raise HTTPException(403, "admin role required")
        if int(request.headers.get("content-length") or 0) > settings.max_policy_bytes:
            raise HTTPException(413, "document too large")
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        kind = {"application/pdf": "pdf", "text/markdown": "md", "text/plain": "md"}.get(ctype)
        if kind is None:
            raise HTTPException(415, "send Content-Type application/pdf or text/markdown")
        data = await request.body()
        try:  # an admin can only update THEIR OWN company's policy
            r = await run_in_threadpool(request.app.state.ingestor.ingest, who.company, data, kind)
        except IngestError as e:
            raise HTTPException(e.http_status, str(e)) from None
        return {"company": r.company, "version": r.version, "chunks": r.chunks,
                "previous_version": r.previous_version, "flagged_sections": r.suspicious_sections,
                "embedded_chunks": r.embedded, "reused_chunks": r.reused, "unchanged": r.unchanged}

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        log.exception("unhandled")
        return JSONResponse({"status": "error", "answer": ERROR_MSG, "reason": "internal"}, status_code=500)

    return app
