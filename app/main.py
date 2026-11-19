import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import Principal, current_principal
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


class IngestBody(BaseModel):
    markdown: str


def seed_if_empty(settings, store, ingestor):
    """First start only: load each company's lowest-version file so `docker compose up` works with no setup."""
    for path in sorted(Path(settings.seed_dir).glob("*_v1.md")):
        company = path.name.split("_")[0]
        if store.active(company) is None:
            try:
                ingestor.ingest(company, path.read_text())
                log.info("seeded %s", company)
            except IngestError as e:
                log.error("seed failed for %s: %s", company, e)


def create_app(settings: Settings | None = None, llm=None, embedder=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        emb = embedder or make_embedder(settings.embedder)
        store = PolicyStore(settings.data_dir)
        for c in store.companies():
            store.purge_inactive(c)  # clear chunks left by a crash mid-ingest
        ingestor = Ingestor(store, emb, settings.max_policy_bytes)
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
    def put_policy(body: IngestBody, request: Request, who: Principal = Depends(current_principal)):
        if who.role != "admin":
            raise HTTPException(403, "admin role required")
        try:  # an admin can only update THEIR OWN company's policy
            r = request.app.state.ingestor.ingest(who.company, body.markdown)
        except IngestError as e:
            raise HTTPException(e.http_status, str(e)) from None
        return {"company": r.company, "version": r.version, "chunks": r.chunks,
                "previous_version": r.previous_version, "flagged_sections": r.suspicious_sections}

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        log.exception("unhandled")
        return JSONResponse({"status": "error", "answer": ERROR_MSG, "reason": "internal"}, status_code=500)

    return app
