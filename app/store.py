"""Persistence: SQLite = which version of each company's policy is ACTIVE; Chroma = the chunks.

Isolation: one Chroma collection per company (separate index, nothing to filter wrongly).
Versioning: chunks of a new version are staged next to the old ones; a single SQLite transaction
flips the 'active' pointer. Queries always filter on the active version, so a half-written
version is invisible and the flip is the only moment anything changes.
"""
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import chromadb

COMPANY_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")


def valid_company(c) -> bool:
    return isinstance(c, str) and bool(COMPANY_RE.match(c))


class PolicyStore:
    def __init__(self, data_dir: str, include_flagged: bool = False):
        self.include_flagged = include_flagged  # False = quarantine: injection-flagged sections are never served
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._db_path = str(root / "registry.db")
        self._chroma = chromadb.PersistentClient(path=str(root / "chroma"))
        with closing(self._conn()) as c, c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS policies(
                     company TEXT NOT NULL, version INTEGER NOT NULL, title TEXT NOT NULL,
                     status TEXT NOT NULL CHECK(status IN ('active','superseded')),
                     chunk_count INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                     file_sha256 TEXT, PRIMARY KEY(company, version))"""
            )
            cols = [r[1] for r in c.execute("PRAGMA table_info(policies)")]
            if "file_sha256" not in cols:  # databases created before change detection existed
                c.execute("ALTER TABLE policies ADD COLUMN file_sha256 TEXT")
            # DB-level guarantee: at most one active version per company.
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_active ON policies(company) WHERE status='active'")

    def _conn(self):
        return sqlite3.connect(self._db_path, timeout=5)

    def _collection(self, company: str):
        assert valid_company(company), "invalid company"
        return self._chroma.get_or_create_collection(f"policy_{company}", metadata={"hnsw:space": "cosine"})

    # ---- registry -------------------------------------------------------------------------
    def active(self, company: str):
        """-> (version, title) or None"""
        with closing(self._conn()) as c:
            row = c.execute("SELECT version, title FROM policies WHERE company=? AND status='active'", (company,)).fetchone()
        return row

    def companies(self) -> list:
        with closing(self._conn()) as c:
            return [r[0] for r in c.execute("SELECT company FROM policies WHERE status='active' ORDER BY company")]

    # ---- write path (used by ingestion only) ---------------------------------------------
    def active_file_hash(self, company):
        with closing(self._conn()) as c:
            row = c.execute("SELECT file_sha256 FROM policies WHERE company=? AND status='active'", (company,)).fetchone()
        return row[0] if row else None

    def vectors_by_hash(self, company) -> dict:
        """content-hash -> stored vector, for the ACTIVE version. Lets a new version reuse vectors of unchanged chunks."""
        act = self.active(company)
        if act is None:
            return {}
        res = self._collection(company).get(where={"version": act[0]}, include=["embeddings", "metadatas"])
        return {m["hash"]: [float(x) for x in v] for m, v in zip(res["metadatas"], res["embeddings"]) if m.get("hash")}

    def stage(self, company, version, doc_title, chunks, embeddings, hashes):
        ids = [f"{company}:v{version}:{ch.key}" for ch in chunks]
        metas = [
            {"company": company, "version": version, "section": ch.section, "title": ch.title,
             "heading": ch.heading, "page": ch.page, "doc_title": doc_title, "suspicious": ch.suspicious, "hash": h}
            for ch, h in zip(chunks, hashes)
        ]
        self._collection(company).add(ids=ids, documents=[ch.text for ch in chunks], embeddings=embeddings, metadatas=metas)

    def drop_version(self, company, version):
        self._collection(company).delete(where={"version": version})

    def activate(self, company, version, title, chunk_count, file_sha256=None):
        """Atomic switch. Either the new version is active and the old one superseded, or nothing changed."""
        with closing(self._conn()) as c, c:  # `with c` = one transaction, rolled back on error
            c.execute("UPDATE policies SET status='superseded' WHERE company=? AND status='active'", (company,))
            c.execute("INSERT INTO policies(company, version, title, status, chunk_count, file_sha256) VALUES (?,?,?,'active',?,?)",
                      (company, version, title, chunk_count, file_sha256))

    def purge_inactive(self, company):
        """Delete chunks of every non-active version ('old one is dead'). Safe to retry."""
        act = self.active(company)
        if act is None:
            return
        self._collection(company).delete(where={"version": {"$ne": act[0]}})

    # ---- read path ------------------------------------------------------------------------
    def _where(self, version):
        return {"version": version} if self.include_flagged else {"$and": [{"version": version}, {"suspicious": False}]}

    def get_chunks(self, company, version) -> list:
        res = self._collection(company).get(where=self._where(version), include=["documents", "metadatas"])
        return [{"id": i, "text": d, **m} for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def query(self, company, version, embedding, k) -> list:
        col = self._collection(company)
        n = col.count()
        if n == 0:
            return []
        res = col.query(query_embeddings=[embedding], n_results=min(k, n), where=self._where(version),
                        include=["documents", "metadatas", "distances"])
        return [
            {"id": i, "text": d, "sim": 1.0 - dist, **m}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
