"""Dev tool: render the Markdown policy sources into the PDFs HR would actually hand us.
Usage: python -m scripts.make_pdfs     (writes data/policies/*.pdf next to the .md sources)
Each page gets a running header and 'Page N' footer, like a real document, so that ingestion
has to cope with them."""
import re
from pathlib import Path

from fpdf import FPDF

DIR = Path(__file__).resolve().parent.parent / "data" / "policies"  # full-length (about 15 pages); short/ holds the 2-page test fixtures


class Doc(FPDF):
    def __init__(self, header):
        super().__init__()
        self.hdr = header

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self.hdr, new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def render(md: str, out: Path):
    lines = md.splitlines()
    pdf = Doc("Confidential - HR Transport Policy")
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    for line in lines:
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 15); text = line[2:]
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 12); text = line[3:]; pdf.ln(2)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 10.5); text = line[4:]
        else:
            pdf.set_font("Helvetica", "", 10); text = line
        if text.strip():
            pdf.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(out))


if __name__ == "__main__":
    for md in sorted(list(DIR.glob("*.md")) + list((DIR / "short").glob("*.md"))):
        render(md.read_text(), md.with_suffix(".pdf"))
        print("wrote", md.with_suffix(".pdf").relative_to(DIR))
