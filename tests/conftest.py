import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.config import Settings
from app.embeddings import HashEmbedder
from app.llm import LLMError
from app.main import create_app

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"


class ScriptedLLM:
    """Fake LLM: returns/raises the scripted items in order (the last one repeats). Records calls."""

    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    def complete(self, system, user, timeout_s, **kw):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        if "__ECHO__" in item:  # a valid, grounded answer: the first sentence of excerpt [1], citing [1]
            m = re.search(r"\[1\] .*?\n(.*?)(?=\n\[2\] |\n</policy_excerpts>)", user, re.S)
            first = m.group(1).strip().split(". ")[0].rstrip(".") + "." if m else "x"
            item = item.replace("__ECHO__", json.dumps({"status": "answered", "answer": first, "citations": [1]}))
        return item


ANSWER_1 = "__ECHO__"


@pytest.fixture(scope="session")
def seeded_dir(tmp_path_factory):
    """Seed the three policies once per test session; each test gets a private copy (fast, isolated)."""
    from app.ingest import Ingestor
    from app.store import PolicyStore
    d = tmp_path_factory.mktemp("seed") / "var"
    store, ing = PolicyStore(str(d)), Ingestor(PolicyStore(str(d)), HashEmbedder(), 10**6, "hash")
    for name in ("acme", "borealis", "cygnus"):
        ing.ingest(name, (POLICIES / f"{name}_v1.md").read_bytes())
    del store, ing
    return d


@pytest.fixture
def settings(tmp_path, seeded_dir):
    shutil.copytree(seeded_dir, tmp_path / "var")
    return Settings(data_dir=str(tmp_path / "var"), seed_dir=str(POLICIES), embedder="hash",
                    min_similarity=0.05, rerank=False, jwt_secret="test-secret-test-secret-test-secret!")


@pytest.fixture
def make_client(settings):
    clients = []

    def _make(llm=None, **overrides):
        for k, v in overrides.items():
            setattr(settings, k, v)
        c = TestClient(create_app(settings, llm=llm or ScriptedLLM(ANSWER_1), embedder=HashEmbedder()))
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def tok(settings):
    return lambda company, role="employee": {"Authorization": "Bearer " + issue_token(settings, f"u@{company}", company, role)}


@pytest.fixture
def policy_text():
    return lambda name: (POLICIES / name).read_bytes()
