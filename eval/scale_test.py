"""Scale check: does the pipeline still work on a 10-30 page policy (hundreds of chunks), where many sections look alike?

    python -m eval.scale_test

Builds a synthetic multi-site policy (one numbered heading per office site, the SAME 14 topics repeated for each site, with
different values), renders it to a real PDF, ingests it, and asks one targeted question per (site, topic) pair. Because the
sections are near-duplicates of each other (different only in site and numbers), this is a deliberately hard distractor test.
Reports: pages, chunks, ingestion time, and recall@1/@3 + latency for vector, hybrid and vector+rerank.
"""
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from scripts.make_pdfs import render

ROOT = Path(__file__).resolve().parent.parent
SITES = ["Bengaluru", "Pune", "Hyderabad", "Chennai", "Gurugram", "Noida", "Mumbai", "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Indore"]
# (topic title, policy sentence template, question template)
TOPICS = [
    ("Security escort", "At the {site} office a security escort is mandatory for women employees on drops after {t} PM. The desk assigns the escort when the drop is booked.", "after what time is an escort needed for women on drops at the {site} office"),
    ("Booking cut-off", "At {site}, a cab must be booked at least {n} hours before the shift. Later requests are ad-hoc trips and need approval.", "how many hours before the shift must I book a cab at {site}"),
    ("Cost cap", "At {site}, an ad-hoc cab trip is capped at Rs {c} per trip. Anything above needs written approval from the finance controller.", "what is the cab trip cost limit at {site}"),
    ("Weekend transport", "At {site}, weekend pickups need approval from the reporting manager raised {n} hours in advance.", "weekend cab rules for the {site} office"),
    ("Airport trips", "At {site}, airport pickups and drops are approved by the {who}. The reporting manager cannot approve them.", "who approves airport trips at {site}"),
    ("No-show penalty", "At {site}, the second no-show in a month carries a fee of Rs {c}. After {n} no-shows transport is suspended for the month.", "what is the no-show fee at {site}"),
    ("Guest travel", "At {site}, guests may travel only with written approval from the {who}. Guests are not allowed in night cabs.", "can a guest ride the cab at {site}"),
    ("Cancellation", "At {site}, a booking can be cancelled free of charge up to {n} hours before the scheduled time.", "free cancellation deadline at {site}"),
    ("Eligibility by distance", "At {site}, employees living more than {n} km from the office are eligible for transport regardless of grade.", "how far must I live to be eligible for transport at {site}"),
    ("Night drop timings", "At {site}, the last drop of the night leaves the office at {t} AM on fixed routes.", "when does the last night drop leave the {site} office"),
    ("Safety reporting", "At {site}, safety incidents must be reported to the {who} within {n} hours.", "how soon must a safety incident be reported at {site}"),
    ("Public holidays", "At {site}, transport on public holidays is available only for critical rosters listed by the {who}.", "is there transport on public holidays at {site}"),
    ("Shift-based eligibility", "At {site}, employees on shifts ending after {t} PM are eligible for transport irrespective of distance.", "am I eligible for transport on a late shift at {site}"),
    ("Ad-hoc approval", "At {site}, ad-hoc trips are approved by the {who} and must be recorded before the trip starts.", "who approves ad-hoc trips at {site}"),
]
WHO = ["department head", "transport administrator", "facilities head", "site security lead", "transport desk lead"]


def build(seed=7):
    rnd = random.Random(seed)
    lines, queries = ["Multi-Site Transport Policy", "Version: 1", ""], []
    for si, site in enumerate(SITES, 1):
        lines += [f"{si}. {site} office", ""]
        for ti, (title, tmpl, qtmpl) in enumerate(TOPICS, 1):
            vals = dict(site=site, t=rnd.choice([7, 8, 9, 10, 11]), n=rnd.choice([2, 3, 4, 6, 12, 24, 48]), c=rnd.choice([150, 200, 250, 300, 1000, 1200, 1500]), who=rnd.choice(WHO))
            body = tmpl.format(**vals)
            body += " " + " ".join(f"This rule applies to all {site} roster groups and is reviewed each quarter by HR; exceptions need a written request." for _ in range(2))
            lines += [f"{si}.{ti} {title}", body, ""]
            queries.append({"company": "bigco", "query": qtmpl.format(**vals), "relevant": [f"{si}.{ti}"]})
    return "\n".join(lines), queries


def main():
    from fpdf import FPDF  # noqa: F401  (render needs it)
    from app.config import Settings
    from app.documents import extract_lines  # noqa: F401
    from app.embeddings import make_embedder
    from app.ingest import Ingestor
    from app.rerank import CrossEncoderReranker
    from app.retrieval import Retriever
    from app.store import PolicyStore

    md, queries = build()
    d = Path(tempfile.mkdtemp())
    pdf = d / "bigco_v1.pdf"
    render(md, pdf)
    from pypdf import PdfReader
    pages = len(PdfReader(str(pdf)).pages)
    s = Settings(data_dir=str(d / "var"))
    emb, store = make_embedder(s.embedder), PolicyStore(s.data_dir)
    emb.embed(["warm"])
    t = time.perf_counter()
    res = Ingestor(store, emb, 10**7, s.embedder).ingest("bigco", pdf.read_bytes(), "pdf")
    ingest_s = time.perf_counter() - t
    t = time.perf_counter()
    store.vectors_by_hash("bigco")
    out = {"pages": pages, "chunks": res.chunks, "pdf_kb": round(pdf.stat().st_size / 1024), "queries": len(queries),
           "ingest_s_first_time": round(ingest_s, 2)}
    rer = CrossEncoderReranker()
    rer.rerank("warm", [{"title": "a", "text": "b"}, {"title": "c", "text": "d"}])
    for name, kw in (("vector", {}), ("hybrid", {"mode": "hybrid"}), ("vector+rerank (shipped)", {"reranker": rer})):
        r = Retriever(store, emb, kw.pop("mode", "vector"), 3, **kw)
        h1 = h3 = 0
        lat = []
        for q in queries:
            r._qcache.clear()
            t = time.perf_counter()
            hits = r.search("bigco", q["query"]).hits
            lat.append((time.perf_counter() - t) * 1000)
            got = [h["section"] for h in hits]
            h1 += got[:1] == q["relevant"]
            h3 += q["relevant"][0] in got[:3]
        lat.sort()
        out[name] = {"recall@1": round(h1 / len(queries), 3), "recall@3": round(h3 / len(queries), 3),
                     "latency_ms_p50": round(statistics.median(lat), 1), "latency_ms_p95": round(lat[int(0.95 * len(lat))], 1)}
    # incremental update on a big document: change ONE sentence and re-ingest as v2
    md2 = md.replace("Version: 1", "Version: 2").replace("is capped at Rs", "is capped at Rs 9", 1)
    pdf2 = d / "bigco_v2.pdf"
    render(md2, pdf2)  # v2 as a PDF too: same extraction as v1, so unchanged sections have identical text
    t = time.perf_counter()
    r2 = Ingestor(store, emb, 10**7, s.embedder).ingest("bigco", pdf2.read_bytes(), "pdf")
    out["update_one_sentence"] = {"embedded": r2.embedded, "reused": r2.reused, "seconds": round(time.perf_counter() - t, 2)}
    print(json.dumps(out, indent=1))
    (ROOT / "eval" / "results" / "scale_test.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
