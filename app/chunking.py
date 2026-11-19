"""Split a markdown policy into citeable chunks.

Format expected (what HR documents are converted to before ingestion):
    # Title
    Version: 2
    ## 3. Night shift and safety
    ### 3.1 Security escort
    text...

One chunk = one ### subsection (the unit people cite: "section 3.1"). Rules in a subsection are
kept together because splitting "escort after 8 PM" from "unless..." would change the meaning.
Very long subsections are split on paragraph boundaries but keep the same section id.
"""
import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1200

_VERSION_RE = re.compile(r"^Version:\s*(\d+)\s*$", re.MULTILINE)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^###\s+(\d+(?:\.\d+)*)\s+(.+?)\s*$")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")

# Heuristic only, used to *report* suspicious text at ingestion. The real defence is architectural
# (company comes from the JWT, model output is validated), not this regex.
_INJECTION_RE = re.compile(
    r"ignore (all )?(previous|prior|above) instructions|disregard (your|all|the) (rules|instructions)"
    r"|note to (ai|assistants?)|^\s*system\s*:|reveal .{0,40}polic(y|ies)|you are now",
    re.IGNORECASE,
)


class PolicyFormatError(ValueError):
    """The document cannot be turned into a valid set of chunks. Ingestion must stop."""


@dataclass(frozen=True)
class Chunk:
    section: str  # "3.1"
    title: str  # "Security escort"
    heading: str  # "3. Night shift and safety"
    text: str
    part: int = 1  # >1 only when a long subsection was split
    suspicious: bool = False

    @property
    def key(self) -> str:
        return f"{self.section}" if self.part == 1 else f"{self.section}#{self.part}"


@dataclass(frozen=True)
class ParsedPolicy:
    title: str
    version: int
    chunks: list


def _split_long(text: str) -> list:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > MAX_CHUNK_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_policy(markdown: str) -> ParsedPolicy:
    if not markdown or not markdown.strip():
        raise PolicyFormatError("document is empty")
    h1 = _H1_RE.search(markdown)
    ver = _VERSION_RE.search(markdown)
    if not h1:
        raise PolicyFormatError("missing '# Title' line")
    if not ver:
        raise PolicyFormatError("missing 'Version: N' line")

    chunks, seen = [], set()
    heading, cur = "", None  # cur = [section, title, heading, [lines]]

    def flush():
        if cur is None:
            return
        body = "\n".join(cur[3]).strip()
        if not body:
            raise PolicyFormatError(f"section {cur[0]} has no text")
        for i, part in enumerate(_split_long(body), start=1):
            chunks.append(Chunk(cur[0], cur[1], cur[2], part, i, bool(_INJECTION_RE.search(part))))

    for line in markdown.splitlines():
        m3, m2 = _SECTION_RE.match(line), _H2_RE.match(line)
        if m3:
            flush()
            if m3.group(1) in seen:
                raise PolicyFormatError(f"duplicate section number {m3.group(1)}")
            seen.add(m3.group(1))
            cur = [m3.group(1), m3.group(2), heading, []]
        elif m2:
            flush()
            cur, heading = None, m2.group(1)
        elif cur is not None:
            cur[3].append(line)
    flush()
    if not chunks:
        raise PolicyFormatError("no '### N.N Title' sections found")
    return ParsedPolicy(h1.group(1), int(ver.group(1)), chunks)
