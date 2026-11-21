"""All tunables in one place. Every value can be overridden by an environment variable."""
import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    # --- auth ---
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "dev-only-secret-change-me-32-bytes!!"))
    jwt_algorithm: str = "HS256"  # pinned: never read the algorithm from the token header
    jwt_issuer: str = "commute-demo-issuer"
    jwt_audience: str = "commute-policy-assistant"
    demo_mode: bool = field(default_factory=lambda: os.getenv("DEMO_MODE", "0") == "1")  # exposes /demo/token for the web page. NEVER on in production
    quarantine_flagged: bool = field(default_factory=lambda: os.getenv("QUARANTINE_FLAGGED", "1") == "1")  # sections with instruction-like text are stored but never retrieved
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN", "20")))  # per user
    # --- storage ---
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./var"))
    seed_dir: str = field(default_factory=lambda: os.getenv("SEED_DIR", "./data/policies"))
    # --- retrieval ---
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "minilm-fast"))  # minilm-fast (default: same MiniLM weights, ~30x faster runtime) | minilm | bge-small | hash
    retrieval_mode: str = field(default_factory=lambda: os.getenv("RETRIEVAL_MODE", "vector"))  # vector | hybrid
    rerank: bool = field(default_factory=lambda: os.getenv("RERANK", "1") == "1")  # cross-encoder reranking of the top candidates (measured: +0.10 recall@1)
    query_rewrite: bool = field(default_factory=lambda: os.getenv("QUERY_REWRITE", "0") == "1")  # LLM rewrites the question first: +1 LLM call, +~0.5 s (off: measured no better than reranking)
    expand_questions: bool = field(default_factory=lambda: os.getenv("EXPAND_QUESTIONS", "0") == "1")  # doc2query at ingestion (LLM writes likely questions per chunk)
    chunking: str = field(default_factory=lambda: os.getenv("CHUNKING", "section"))  # section (default) | sentence | heading (both measured worse)
    top_k: int = 3  # chunks retrieved AND shown to the LLM: also the brief's cap of 3 cited chunks, and fewer tokens per call
    min_similarity: float = field(default_factory=lambda: _f("MIN_SIMILARITY", 0.25))  # below this: "not covered", no LLM call
    # --- llm ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "fake"))  # groq | gemini | fake
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    max_llm_calls: int = 2  # hard cap required by the brief (we normally use 1)
    # --- budgets (seconds) ---
    total_budget_s: float = field(default_factory=lambda: _f("TOTAL_BUDGET_S", 2.8))  # p95 target is 3 s
    retrieval_timeout_s: float = field(default_factory=lambda: _f("RETRIEVAL_TIMEOUT_S", 1.0))
    retry_min_remaining_s: float = 1.0  # only retry a malformed LLM answer if this much budget is left
    # --- input limits ---
    max_question_chars: int = 500
    max_policy_bytes: int = 200_000
