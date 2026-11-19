"""Turn uploaded bytes (PDF or Markdown) into page-tagged text lines for the chunker."""
import io
import re

from .chunking import PolicyFormatError

MAX_PAGES = 100
_PAGE_NO_RE = re.compile(r"^(page\s+)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)


def extract_lines(data: bytes, kind: str) -> list:
    if kind == "md":
        try:
            return [(0, line) for line in data.decode("utf-8").splitlines()]
        except UnicodeDecodeError:
            raise PolicyFormatError("markdown must be UTF-8 text") from None
    if kind == "pdf":
        return _pdf_lines(data)
    raise PolicyFormatError("unsupported file type (use PDF or Markdown)")


def _pdf_lines(data: bytes) -> list:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    if not data.startswith(b"%PDF"):
        raise PolicyFormatError("file is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PolicyFormatError("PDF is password-protected")
        if len(reader.pages) > MAX_PAGES:
            raise PolicyFormatError(f"PDF has more than {MAX_PAGES} pages")
        pages = [(p.extract_text() or "").splitlines() for p in reader.pages]
    except PolicyFormatError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError) as e:
        raise PolicyFormatError(f"PDF could not be read ({type(e).__name__})") from None
    if not any(t.strip() for ls in pages for t in ls):
        raise PolicyFormatError("PDF has no extractable text (scanned images are not supported: no OCR)")
    # Drop running headers/footers: page-number lines, and any line repeated at the top/bottom of every page.
    if len(pages) > 1:
        edge = lambda ls: {t.strip() for t in (ls[:1] + ls[-1:]) if t.strip()}  # noqa: E731
        repeated = set.intersection(*(edge(ls) for ls in pages))
    else:
        repeated = set()
    out = []
    for n, ls in enumerate(pages, start=1):
        for t in ls:
            if t.strip() in repeated or _PAGE_NO_RE.match(t.strip()):
                continue
            out.append((n, t))
    return out
