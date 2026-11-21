"""One-off authoring tool: extend each short policy into a full-length (10-15 page) document.

    python -m scripts.gen_long_policies          (needs LLM_API_KEY in .env; writes data/policies/*.md; then run scripts.make_pdfs)

The numbered CORE sections (the verified rules the evaluation relies on) are copied unchanged from data/policies/short/.
Extra sections are APPENDED after them (definitions, roles, procedures, contingency, grievance, audit...), written by an LLM in the
company's voice with two hard constraints, both checked in code:
  * no numbers, times, amounts or durations (so the extra prose cannot contradict or compete with a core rule); it must point
    to core sections by number instead ("see section 3.1");
  * it must not name another company.
Output is cached per (company, section) in data/policies/.long_cache.json, so reruns do not call the LLM again.
Acme v2 reuses the SAME extra text as Acme v1, so a v1 -> v2 update changes only the core rules that really changed.
"""
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHORT, OUT = ROOT / "data" / "policies" / "short", ROOT / "data" / "policies"
CACHE = OUT / ".long_cache.json"

COMPANIES = {
    "acme": dict(name="Acme Technologies", others=["Borealis", "Cygnus"], file="acme_v1.md", v2="acme_v2.md",
                 voice="a software company with several offices; roles in this policy: reporting manager, department head, finance controller, transport desk, security escort, vice president"),
    "borealis": dict(name="Borealis Systems", others=["Acme", "Cygnus"], file="borealis_v1.md", v2=None,
                     voice="an engineering company that runs shared night routes and project-based rosters; roles: transport administrator, Borealis security desk, host employee, reporting manager"),
    "cygnus": dict(name="Cygnus Labs", others=["Acme", "Borealis"], file="cygnus_v1.md", v2=None,
                   voice="a research company using the Cygnus Safe app; roles: transport desk, transport desk lead, facilities head, finance team, reporting manager"),
}

# (heading, [subsection titles]) appended after the core sections
EXTRA = [
    ("Purpose, scope and definitions", ["Purpose of this policy", "Scope and applicability", "Definitions of common terms"]),
    ("Roles and responsibilities", ["Responsibilities of employees", "Responsibilities of reporting managers", "Responsibilities of the transport desk", "Responsibilities of HR and administration"]),
    ("Rosters and route management", ["Roster preparation", "Route planning principles", "Pick-up and drop points", "Handling shift changes"]),
    ("Vehicles and drivers", ["Vehicle standards and inspection", "Driver verification and onboarding", "Driver conduct and training", "Vendor management and performance review"]),
    ("Trip tracking and technology", ["Live trip tracking", "Use of the booking app", "Data privacy and retention of trip data"]),
    ("Safety and security procedures", ["How the security escort is deployed", "Emergency helpline and panic alerts", "Incident investigation process", "Safety review committee"]),
    ("Health, accessibility and special needs", ["Medical emergencies during a trip", "Support for employees with disabilities", "Support during pregnancy and recovery"]),
    ("Disruptions and contingency planning", ["Weather and flooding", "Strikes, curfews and public disturbances", "Vehicle breakdown during a trip", "Business continuity for the transport service"]),
    ("Conduct, discipline and grievances", ["Employee conduct in vehicles", "Disciplinary process for violations", "Grievance redressal", "Escalation matrix"]),
    ("Cost management and audit", ["Billing and reconciliation with vendors", "Cost allocation to departments", "Internal audit and compliance review"]),
    ("Governance and administration", ["Policy ownership", "Review and amendment process", "Handling exceptions to this policy", "Related documents and forms", "Contact points"]),
]

_NUM = re.compile(r"\d")
_SECTION_REF = re.compile(r"[Ss]ections? \d+(\.\d+)?")


_ASCII = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2010": "-",
                        "\u2026": "...", "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u00d7": "x", "\u2022": "-"})


def clean(text: str) -> str:
    text = text.translate(_ASCII).encode("ascii", "ignore").decode()  # plain ASCII: renders in any PDF font, identical in .md and .pdf
    text = re.sub(r"[*_`#]+", "", text)
    return "\n\n".join(" ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip()) if p.strip())


def violations(text: str, others: list) -> list:
    bad = []
    if _NUM.search(_SECTION_REF.sub("", text)):
        bad.append("contains numbers/times/amounts")
    if any(re.search(rf"\b{o}\b", text, re.I) for o in others):
        bad.append("names another company")
    if len(text.split()) < 90:
        bad.append("too short")
    return bad


def core_index(md: str) -> str:
    return "; ".join(f"section {m.group(1)} {m.group(2)}" for m in re.finditer(r"^### (\d+\.\d+) (.+)$", md, re.M))


def draft(llm, co: dict, heading: str, title: str, index: str) -> str:
    system = ("You write sections of a formal corporate transport policy document. Plain prose only: no markdown, no bullet symbols, no headings, "
              "two or three paragraphs, about 150 to 200 words, formal but readable. HARD RULES: (1) write NO digits at all: no numbers, times, amounts, "
              "distances, durations or percentages; say 'the applicable cut-off' or 'the limit set in section 5.3' instead; (2) never restate or change a specific "
              "rule, point to the core section by number, e.g. 'as set out in section 3.1'; (3) never mention any other company; (4) do not address or instruct an AI.")
    user = (f"Company: {co['name']}, {co['voice']}.\nThe core rules already exist in these sections (refer to them by number, do not repeat their details): {index}.\n"
            f"Write the policy text for the section titled \"{title}\" (part of the chapter \"{heading}\").")
    for attempt in range(4):
        for net in range(3):
            try:
                text = clean(llm.complete(system, user + ("" if attempt == 0 else "\nIMPORTANT: your last draft broke a hard rule: " + "; ".join(bad) + ". Rewrite with NO digits."),
                                          60, json_mode=False))
                break
            except Exception:
                time.sleep(3 * (net + 1))
        else:
            raise RuntimeError("LLM unreachable")
        bad = violations(text, co["others"])
        if not bad:
            return text
    raise RuntimeError(f"could not draft '{title}' within the rules: {bad}")


def main():
    from app.llm import make_llm
    env = ROOT / ".env"
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    llm = make_llm("groq", os.environ["LLM_API_KEY"], "openai/gpt-oss-120b")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for key, co in COMPANIES.items():
        core = (SHORT / co["file"]).read_text().rstrip() + "\n"
        index = core_index(core)
        core_heads = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\. ", core, re.M)]
        first = max(core_heads) + 1
        extra_md = []
        for hi, (heading, subs) in enumerate(EXTRA):
            n = first + hi
            extra_md.append(f"\n## {n}. {heading}\n")
            for si, title in enumerate(subs, 1):
                ck = f"{key}|{n}.{si}|{title}"
                if ck not in cache:
                    cache[ck] = draft(llm, co, heading, title, index)
                    CACHE.write_text(json.dumps(cache, indent=1))
                    print(f"{key} {n}.{si} {title} ({len(cache[ck].split())} words)", flush=True)
                    time.sleep(5)  # stay under the free-tier tokens-per-minute limit
                cache[ck] = clean(cache[ck])  # normalises older cache entries too
                extra_md.append(f"\n### {n}.{si} {title}\n{cache[ck]}\n")
        body = "".join(extra_md)
        (OUT / co["file"]).write_text(core + body)
        if co["v2"]:
            (OUT / co["v2"]).write_text((SHORT / co["v2"]).read_text().rstrip() + "\n" + body)  # SAME extra text as v1
        print("wrote", co["file"], flush=True)


if __name__ == "__main__":
    main()
