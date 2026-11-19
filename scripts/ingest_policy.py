"""Upload a policy as the company's admin.  Usage: python -m scripts.ingest_policy <company> <file.md> [base_url]"""
import sys

import httpx

from app.auth import issue_token
from app.config import Settings

if __name__ == "__main__":
    company, path = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
    tok = issue_token(Settings(), f"admin@{company}.example", company, "admin")
    ctype = "application/pdf" if path.endswith(".pdf") else "text/markdown"
    r = httpx.put(f"{base}/admin/policy", content=open(path, "rb").read(),
                  headers={"Authorization": f"Bearer {tok}", "Content-Type": ctype}, timeout=60)
    print(r.status_code, r.text)
