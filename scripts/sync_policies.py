"""Change detection for policy files. Uploads only NEW or CHANGED files, then (inside the server) only the
new or changed chunks are embedded.

  python -m scripts.sync_policies data/policies                 # one pass
  python -m scripts.sync_policies data/policies --watch 30      # poll every 30 s

Files must be named  <company>_v<version>.pdf|md   (e.g. acme_v2.pdf).
How it knows something changed: SHA-256 of each file is kept in <folder>/.sync_manifest.json. A file whose
hash is unchanged is skipped without any network call. (Polling is deliberate: no OS-specific watcher
dependency, trivial to explain. Swap in `watchdog` if you need instant reaction.)
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import httpx

NAME_RE = re.compile(r"^([a-z][a-z0-9-]{1,31})_v(\d+)\.(pdf|md)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_uploads(folder: Path, manifest: dict) -> list:
    """Pure function: which files need uploading, oldest version first (so v2 lands before v3)."""
    todo = []
    for f in folder.iterdir():
        m = NAME_RE.match(f.name)
        if m and manifest.get(f.name) != sha256(f):
            todo.append((m.group(1), int(m.group(2)), f))
    return sorted(todo, key=lambda t: (t[0], t[1]))


def sync_once(folder: Path, base: str, token_for) -> None:
    mf = folder / ".sync_manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    for company, version, f in plan_uploads(folder, manifest):
        ctype = "application/pdf" if f.suffix == ".pdf" else "text/markdown"
        r = httpx.put(f"{base}/admin/policy", content=f.read_bytes(), timeout=120,
                      headers={"Authorization": f"Bearer {token_for(company)}", "Content-Type": ctype})
        if r.status_code == 200:
            j = r.json()
            print(f"{f.name}: {'unchanged' if j['unchanged'] else 'ingested'} v{j['version']} "
                  f"embedded={j['embedded_chunks']} reused={j['reused_chunks']}")
            manifest[f.name] = sha256(f)
        elif r.status_code == 409:  # older than what is live, or edited without a version bump: report, do not retry forever
            print(f"{f.name}: skipped ({r.json().get('detail')})")
            manifest[f.name] = sha256(f)
        else:
            print(f"{f.name}: FAILED {r.status_code} {r.text[:120]}  (will retry next pass)")
    mf.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    from app.auth import issue_token
    from app.config import Settings

    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--watch", type=int, default=0, help="poll every N seconds (0 = one pass)")
    a = ap.parse_args()
    cfg = Settings()
    tok = lambda company: issue_token(cfg, f"sync@{company}", company, "admin")  # noqa: E731
    while True:
        sync_once(Path(a.folder), a.base, tok)
        if not a.watch:
            break
        time.sleep(a.watch)
