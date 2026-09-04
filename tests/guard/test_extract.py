"""Tests for pulling text out of files the model cannot be shown directly."""

from __future__ import annotations

import shutil
import zipfile

import pytest

from rezunate_guard import extract
from rezunate_guard.extract import ExtractionError, can_extract, extract_text, strip_markup

needs_pdftotext = pytest.mark.skipif(
    shutil.which("pdftotext") is None, reason="poppler's pdftotext is not installed"
)


class TestStrippingMarkup:
    def test_a_name_in_a_tag_comes_out_as_text(self):
        assert strip_markup("<p>Ali Hassan</p>") == "Ali Hassan"

    def test_paragraphs_do_not_run_together(self):
        """Without a break the two names would read as one, which the models then miss."""
        stripped = strip_markup("<w:p><w:t>Ali Hassan</w:t></w:p><w:p><w:t>Sara Khan</w:t></w:p>")
        assert stripped.splitlines() == ["Ali Hassan", "Sara Khan"]

    def test_entities_are_turned_back_into_characters(self):
        assert strip_markup("<t>Ali &amp; Sara &lt;here&gt;</t>") == "Ali & Sara <here>"

    def test_attributes_do_not_survive(self):
        assert "width" not in strip_markup('<svg width="10"><text>Ali</text></svg>')


class TestOfficeFiles:
    def test_text_comes_out_of_a_docx(self, tmp_path, make_docx):
        assert extract_text(make_docx(tmp_path / "a.docx")) == "Ali Hassan"

    def test_a_docx_is_recognised_by_its_bytes(self, tmp_path, make_docx):
        renamed = make_docx(tmp_path / "notes.txt")
        assert can_extract(renamed) and extract_text(renamed) == "Ali Hassan"

    def test_an_archive_with_no_document_is_refused(self, tmp_path):
        path = tmp_path / "b.docx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("random.bin", "x")
        with pytest.raises(ExtractionError, match="no document"):
            extract_text(path)

    def test_a_broken_archive_is_refused(self, tmp_path):
        path = tmp_path / "c.docx"
        path.write_bytes(b"PK\x03\x04broken")
        with pytest.raises(ExtractionError):
            extract_text(path)


@needs_pdftotext
class TestPdfFiles:
    def test_text_comes_out_of_a_pdf(self, tmp_path, make_pdf):
        text = extract_text(make_pdf(tmp_path / "a.pdf"))
        assert "Ali Hassan" in text and "+971 50 123 4567" in text

    def test_a_pdf_with_no_text_layer_is_refused(self, tmp_path, make_pdf):
        """A scan. Reading it needs OCR, so an empty answer must not pass for success."""
        path = make_pdf(tmp_path / "b.pdf", lines=())
        with pytest.raises(ExtractionError, match="no text"):
            extract_text(path)


class TestWhatCannotBeRead:
    def test_an_unknown_format_is_refused(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
        assert not can_extract(path)
        with pytest.raises(ExtractionError, match="no text extractor"):
            extract_text(path)

    def test_a_missing_file_is_refused(self, tmp_path):
        assert not can_extract(tmp_path / "gone.pdf")
        with pytest.raises(ExtractionError):
            extract_text(tmp_path / "gone.pdf")

    def test_a_pdf_without_the_tool_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(extract.shutil, "which", lambda name: None)
        path = tmp_path / "a.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        assert not can_extract(path)
        with pytest.raises(ExtractionError, match="no PDF text extractor"):
            extract_text(path)
