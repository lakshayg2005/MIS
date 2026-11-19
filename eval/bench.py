"""End-to-end latency over real HTTP (JWT, FastAPI, retrieval, REAL LLM call).

    docker compose up -d            # DEMO_MODE=1 provides tokens
    python -m eval.bench --base http://localhost:8000 --n 60

Cycles through the answerable eval questions. Sleeps between requests to respect the free-tier rate
limit; the sleep is NOT counted in the latency (we time each request on its own).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=2.2)
    a = ap.parse_args()
    cases = [json.loads(line) for line in (Path(__file__).parent / "evalset.jsonl").read_text().splitlines() if line.strip()]
    cases = [c for c in cases if not c["input"].get("setup_ingest")]
    tokens = {}
    with httpx.Client(base_url=a.base, timeout=10) as http:
        lat, statuses = [], {}
        for i in range(a.n):
            c = cases[i % len(cases)]["input"]
            co = c["company"]
            if co not in tokens:
                tokens[co] = http.post("/demo/token", json={"company": co}).json()["token"]
            t = time.perf_counter()
            r = http.post("/ask", json={"question": c["question"]}, headers={"Authorization": "Bearer " + tokens[co]})
            ms = (time.perf_counter() - t) * 1000
            lat.append(ms)
            key = r.json().get("status", "?") + ("/" + r.json().get("reason", "") if r.json().get("status") == "error" else "")
            statuses[key] = statuses.get(key, 0) + 1
            time.sleep(a.sleep)
    lat.sort()
    out = {"requests": len(lat), "statuses": statuses, "p50_ms": int(statistics.median(lat)),
           "p95_ms": int(lat[min(len(lat) - 1, int(0.95 * len(lat)))]), "max_ms": int(lat[-1]), "target_p95_ms": 3000}
    print(json.dumps(out, indent=1))
    Path(__file__).parent.joinpath("results", "bench.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
