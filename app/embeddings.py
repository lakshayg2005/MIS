"""Embedders. The app only needs `embed(list[str]) -> list[list[float]]`."""
import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list) -> list: ...


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 (ONNX, runs on CPU, no API, no cost). Chroma ships the runner;
    the model file is downloaded once (baked into the Docker image at build time)."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts):
        return [list(map(float, v)) for v in self._fn(texts)]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Tests only: fast, offline, reproducible.
    It has no semantics beyond shared words, so never use it to judge retrieval quality."""

    DIM = 512

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FastEmbedder:
    """fastembed (ONNX) models. Same idea as MiniLMEmbedder but a different runtime / model choice.
    Some models (bge) want a special prefix on QUERIES, which fastembed applies in query_embed."""

    MODELS = {"minilm-fast": "sentence-transformers/all-MiniLM-L6-v2", "bge-small": "BAAI/bge-small-en-v1.5"}

    def __init__(self, key: str):
        from fastembed import TextEmbedding

        self._m = TextEmbedding(self.MODELS[key])

    def embed(self, texts):
        return [list(map(float, v)) for v in self._m.embed(list(texts))]

    def embed_query(self, text):
        return [list(map(float, v)) for v in self._m.query_embed([text])][0]


def make_embedder(name: str) -> Embedder:
    if name in FastEmbedder.MODELS:
        return FastEmbedder(name)
    if name == "hash":
        return HashEmbedder()
    if name == "minilm":
        return MiniLMEmbedder()
    raise ValueError(f"unknown embedder {name!r}")
