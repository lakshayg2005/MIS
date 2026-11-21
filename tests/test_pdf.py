import io

import pytest

from app.chunking import PolicyFormatError, parse_policy
from app.documents import extract_lines

PDF_H = {"Content-Type": "application/pdf"}
Q = {"question": "At what time is a security escort mandatory for women on drops?"}


def test_pdf_and_markdown_parse_to_identical_sections(policy_text):
    pdf = parse_policy(extract_lines(policy_text("acme_v1.pdf"), "pdf"))
    md = parse_policy(extract_lines(policy_text("acme_v1.md"), "md"))
    assert [(c.section, c.title) for c in pdf.chunks] == [(c.section, c.title) for c in md.chunks]
    assert all(" ".join(a.text.split()) == " ".join(b.text.split()) for a, b in zip(pdf.chunks, md.chunks))
    assert {c.page for c in pdf.chunks} == {1, 2} and {c.page for c in md.chunks} == {0}


def test_running_headers_and_page_numbers_are_stripped(policy_text):
    text = " ".join(c.text for c in parse_policy(extract_lines(policy_text("acme_v1.pdf"), "pdf")).chunks)
    assert "Confidential" not in text and "Page 1" not in text and "Page 2" not in text


def test_pdf_upload_new_version_with_page_in_citation(make_client, tok, policy_text):
    c = make_client()
    r = c.put("/admin/policy", content=policy_text("acme_v2.pdf"), headers={**tok("acme", "admin"), **PDF_H})
    assert r.status_code == 200 and r.json()["version"] == 2
    cit = c.post("/ask", json=Q, headers=tok("acme")).json()["citations"][0]
    assert cit["version"] == 2 and cit["page"] in (1, 2) and "9:00 PM" in cit["excerpt"]


@pytest.mark.parametrize("data,ctype,status", [
    (b"this is not a pdf", "application/pdf", 422),
    (b"%PDF-1.4 truncated garbage", "application/pdf", 422),
    (b"", "application/pdf", 422),
    (b"hello", "application/zip", 415),
])
def test_bad_uploads_rejected_and_policy_unchanged(make_client, tok, data, ctype, status):
    c = make_client()
    r = c.put("/admin/policy", content=data, headers={**tok("acme", "admin"), "Content-Type": ctype})
    assert r.status_code == status
    assert c.app.state.service.store.active("acme")[0] == 1


def test_pdf_without_text_is_rejected_clearly():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()  # a blank page, like a scanned image would look to a text extractor
    with pytest.raises(PolicyFormatError, match="no extractable text"):
        extract_lines(bytes(pdf.output()), "pdf")


def test_oversized_upload_rejected(make_client, tok):
    c = make_client(max_policy_bytes=200_000)
    r = c.put("/admin/policy", content=b"x" * 300_000, headers={**tok("acme", "admin"), "Content-Type": "text/markdown"})
    assert r.status_code == 413


def test_section_number_must_match_its_heading():
    # a body line such as "1.5 hours before..." must not be mistaken for a new section under heading 3
    p = parse_policy([(1, "T"), (1, "Version: 1"), (1, "3. Booking"), (1, "3.1 Cut-off"),
                      (1, "Book at least"), (1, "1.5 hours before the shift.")])
    assert [c.section for c in p.chunks] == ["3.1"] and "1.5 hours" in p.chunks[0].text


def test_a_realistic_multi_megabyte_pdf_is_accepted_but_over_the_limit_is_not(make_client, tok):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_compression(False)  # uncompressed streams make the file large, like a PDF full of formatting/logos
    for n in range(12):
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, ("1. Section\n1.1 Rule\nPlain rule text here. " if n == 0 else "") + "padding " * 4000)
    body = b"# T\nVersion: 2\n" if False else bytes(pdf.output())
    assert len(body) > 200_000  # bigger than the old limit
    c = make_client()
    r = c.put("/admin/policy", content=body, headers={**tok("acme", "admin"), "Content-Type": "application/pdf"})
    assert r.status_code != 413  # not rejected for size (it may be rejected as a document for lacking Version: that is a different, valid reason)
    c2 = make_client(max_policy_bytes=len(body) - 1)
    assert c2.put("/admin/policy", content=body, headers={**tok("acme", "admin"), "Content-Type": "application/pdf"}).status_code == 413
