"""Users and password login. The brief only asks us to BUILD a JWT issuer (any mock will do); this is the small, honest
version of one: email + password -> signed JWT carrying the company. Passwords are never stored: only scrypt hashes
(salted, memory-hard). Login does not reveal whether the email or the password was wrong, and takes the same time either way."""
import hashlib
import hmac
import os
import sqlite3
from contextlib import closing

from .store import valid_company

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def _check(password: str, stored: str) -> bool:
    try:
        _, salt, expect = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=_N, r=_R, p=_P, dklen=32)
        return hmac.compare_digest(got, bytes.fromhex(expect))
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password("dummy-password-for-timing")  # so an unknown email costs the same as a wrong password


class UserStore:
    def __init__(self, db_path: str):
        self._db = db_path
        with closing(sqlite3.connect(self._db)) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS users(
                           email TEXT PRIMARY KEY, company TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('employee','admin')),
                           pw_hash TEXT NOT NULL)""")

    def create(self, email: str, company: str, role: str, password: str) -> None:
        if not valid_company(company) or role not in ("employee", "admin") or len(password) < 8:
            raise ValueError("invalid user (company id, role, or password shorter than 8 characters)")
        with closing(sqlite3.connect(self._db)) as c, c:
            c.execute("INSERT OR REPLACE INTO users(email, company, role, pw_hash) VALUES (?,?,?,?)",
                      (email.strip().lower(), company, role, hash_password(password)))

    def authenticate(self, email: str, password: str):
        """-> (company, role) or None. Constant-ish time, no hint about which part was wrong."""
        with closing(sqlite3.connect(self._db)) as c:
            row = c.execute("SELECT company, role, pw_hash FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
        ok = _check(password, row[2] if row else _DUMMY)
        return (row[0], row[1]) if row and ok else None

    def count(self) -> int:
        with closing(sqlite3.connect(self._db)) as c:
            return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
