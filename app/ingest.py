"""Policy ingestion with all-or-nothing semantics.

Order of operations (each step can fail without harming what users currently see):
  1. validate + parse + embed EVERYTHING in memory      (nothing written yet)
  2. stage chunks of the new version in Chroma           (invisible: queries filter on the active version)
  3. one SQLite transaction flips active -> new version  (the only visible change)
  4. best-effort purge of old versions                   (cosmetic: they are already unreachable)
A failure in 1-3 deletes the staged chunks and leaves the previous version active.
"""
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass

from .chunking import PolicyFormatError, parse_policy
from .documents import extract_lines
from .store import PolicyStore, valid_company

log = logging.getLogger("ingest")


class IngestError(Exception):
    def __init__(self, message, http_status=422):
        super().__init__(message)
        self.http_status = http_status


@dataclass
class IngestResult:
    company: str
    version: int
    chunks: int
    previous_version: int | None
    suspicious_sections: list


class Ingestor:
    def __init__(self, store: PolicyStore, embedder, max_bytes: int):
        self._store, self._embedder, self._max_bytes = store, embedder, max_bytes
        self._locks = defaultdict(threading.Lock)  # one writer per company at a time

    def ingest(self, company: str, data: bytes, kind: str = "md") -> IngestResult:
        if not valid_company(company):
            raise IngestError("invalid company id")
        if len(data) > self._max_bytes:
            raise IngestError("document too large", 413)
        try:
            parsed = parse_policy(extract_lines(data, kind))
        except PolicyFormatError as e:
            raise IngestError(f"invalid policy document: {e}") from e

        with self._locks[company]:
            active = self._store.active(company)
            if active and parsed.version <= active[0]:
                raise IngestError(
                    f"version {parsed.version} is not newer than the active version {active[0]}", 409)
            try:
                texts = [f"{c.heading} / {c.title}: {c.text}" for c in parsed.chunks]
                vectors = self._embedder.embed(texts)  # step 1
                if len(vectors) != len(parsed.chunks):
                    raise RuntimeError("embedder returned wrong number of vectors")
            except Exception as e:
                log.exception("embedding failed; nothing written")
                raise IngestError("embedding failed; policy unchanged", 503) from e
            try:
                self._store.drop_version(company, parsed.version)  # clear leftovers of an earlier crashed attempt
                self._store.stage(company, parsed.version, parsed.title, parsed.chunks, vectors)  # step 2
                self._store.activate(company, parsed.version, parsed.title, len(parsed.chunks))  # step 3
            except Exception as e:
                log.exception("ingest failed; rolling back staged chunks")
                try:
                    self._store.drop_version(company, parsed.version)
                except Exception:
                    log.exception("rollback cleanup failed (staged chunks are unreachable anyway)")
                raise IngestError("storage failed; policy unchanged", 503) from e
            try:
                self._store.purge_inactive(company)  # step 4
            except Exception:
                log.warning("purge of old versions failed; they remain unreachable")
            sus = [c.section for c in parsed.chunks if c.suspicious]
            if sus:
                log.warning("company=%s v%s: sections %s contain instruction-like text", company, parsed.version, sus)
            return IngestResult(company, parsed.version, len(parsed.chunks), active[0] if active else None, sus)
