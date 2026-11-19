"""Split a policy into citeable chunks. Works on lines tagged with their page number, so the
same parser serves Markdown (page 0 = unknown) and text extracted from a PDF.

Expected structure (numbers are the document's own, and are what we cite):
    Title                      <- first non-empty line (a leading '# ' is allowed)
    Version: 2
    3. Night shift and safety  <- heading   (optional '## ' prefix in Markdown)
    3.1 Security escort        <- section   (optional '### ' prefix); its number must start with its heading's number
    rule text...

One chunk = one section (the unit people cite). Rules inside a section stay together because
splitting "escort after 8 PM" from its exceptions would change the meaning. Very long sections are
split on paragraph boundaries but keep the same section id.
"""
import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1200

_VERSION_RE = re.compile(r"^Version:\s*(\d+)\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(?:#{1,6}\s+)?(\d+\.\d+)\s+(\S.*?)\s*$")
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+)?(\d+)\.\s+(\S.*?)\s*$")

# Heuristic used only to REPORT suspicious text at ingestion. The real defences are architectural
# (company comes from the JWT, answers are validated), not this regex.
_INJECTION_RE = re.compile(
    r"ignore (all )?(previous|prior|above) instructions|disregard (your|all|the) (rules|instructions)"
    r"|note to (ai|assistants?)|^\s*system\s*:|reveal .{0,40}polic(y|ies)|you are now",
    re.IGNORECASE | re.MULTILINE,
)


class PolicyFormatError(ValueError):
    """The document cannot be turned into a valid set of chunks. Ingestion must stop."""


@dataclass(frozen=True)
class Chunk:
    section: str  # "3.1"
    title: str  # "Security escort"
    heading: str  # "3. Night shift and safety"
    text: str
    page: int = 0  # page where the section starts (0 = unknown, e.g. Markdown)
    part: int = 1  # >1 only when a long section was split
    suspicious: bool = False

    @property
    def key(self) -> str:
        return self.section if self.part == 1 else f"{self.section}#{self.part}"


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


def parse_policy(lines: list) -> ParsedPolicy:
    """lines: list of (page, text). Raises PolicyFormatError on anything we cannot cite reliably."""
    lines = [(p, t.rstrip()) for p, t in lines]
    nonblank = [t.strip() for _, t in lines if t.strip()]
    if not nonblank:
        raise PolicyFormatError("document is empty")
    title = nonblank[0].lstrip("# ").strip()
    version = next((int(m.group(1)) for t in nonblank[:6] if (m := _VERSION_RE.match(t))), None)
    if version is None:
        raise PolicyFormatError("missing 'Version: N' line near the top")

    chunks, seen = [], set()
    heading, head_no, cur = "", None, None  # cur = [section, title, heading, page, [body lines]]

    def flush():
        if cur is None:
            return
        body = "\n".join(cur[4]).strip()
        if not body:
            raise PolicyFormatError(f"section {cur[0]} has no text")
        for i, part in enumerate(_split_long(body), start=1):
            chunks.append(Chunk(cur[0], cur[1], cur[2], part, cur[3], i, bool(_INJECTION_RE.search(part))))

    for page, line in lines:
        s = line.strip()
        m_sec, m_head = _SECTION_RE.match(s), _HEADING_RE.match(s)
        if m_sec and (head_no is None or m_sec.group(1).split(".")[0] == head_no):
            flush()
            if m_sec.group(1) in seen:
                raise PolicyFormatError(f"duplicate section number {m_sec.group(1)}")
            seen.add(m_sec.group(1))
            cur = [m_sec.group(1), m_sec.group(2), heading, page, []]
        elif m_head and not _VERSION_RE.match(s):
            flush()
            cur, head_no, heading = None, m_head.group(1), f"{m_head.group(1)}. {m_head.group(2)}"
        elif cur is not None:
            cur[4].append(s)
    flush()
    if not chunks:
        raise PolicyFormatError("no numbered sections (like '3.1 Security escort') found")
    return ParsedPolicy(title, version, chunks)
