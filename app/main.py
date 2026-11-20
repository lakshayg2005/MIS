import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .auth import Principal, current_principal, issue_token
from .config import Settings
from .embeddings import make_embedder
from .ingest import Ingestor, IngestError
from .llm import make_llm
from .retrieval import Retriever
from .service import ERROR_MSG, QuestionService
from .store import PolicyStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


class AskBody(BaseModel):
    # NOTE: deliberately no company field. Company is taken from the JWT only.
    question: str = Field(min_length=1)


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
        store = PolicyStore(settings.data_dir)
        for c in store.companies():
            store.purge_inactive(c)  # clear chunks left by a crash mid-ingest
        ingestor = Ingestor(store, emb, settings.max_policy_bytes, settings.embedder)
        seed_if_empty(settings, store, ingestor)
        emb.embed(["warm up"])  # load the model before the first user request
        retriever = Retriever(store, emb, settings.retrieval_mode, settings.top_k)
        app.state.service = QuestionService(settings, store, retriever,
                                            llm or make_llm(settings.llm_provider, settings.llm_api_key, settings.llm_model))
        app.state.ingestor = ingestor
        yield

    app = FastAPI(title="Commute Policy Assistant", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/me")
    def me(request: Request, who: Principal = Depends(current_principal)):
        """Which company and which policy version this token answers from (shown in the UI)."""
        act = request.app.state.service.store.active(who.company)
        return {"company": who.company, "role": who.role,
                "policy": {"title": act[1], "version": act[0]} if act else None}

    if settings.demo_mode:
        class DemoTokenBody(BaseModel):
            company: str

        @app.post("/demo/token")
        def demo_token(body: DemoTokenBody, request: Request):
            """DEMO ONLY (DEMO_MODE=1): stands in for the company's login so the web page works.
            Hands out EMPLOYEE tokens for known companies only; never an admin token."""
            if body.company not in request.app.state.service.store.companies():
                raise HTTPException(404, "unknown company")
            return {"token": issue_token(settings, f"demo-employee@{body.company}", body.company, "employee", ttl_s=1800)}

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")

    @app.post("/ask")
    def ask(body: AskBody, request: Request, who: Principal = Depends(current_principal)):
        q = body.question.strip()
        if not q or len(q) > settings.max_question_chars:
            raise HTTPException(400, f"question must be 1-{settings.max_question_chars} characters")
        ans = request.app.state.service.ask(who.company, q)
        log.info("ask company=%s sub=%s status=%s reason=%s ms=%d calls=%d chunks=%s", who.company, who.subject,
                 ans.status, ans.reason, ans.latency_ms, ans.llm_calls, ans.retrieved)
        payload = ans.to_dict()
        if ans.status == "error":
            return JSONResponse(payload, status_code=503, headers={"Retry-After": "5"})
        return payload

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
