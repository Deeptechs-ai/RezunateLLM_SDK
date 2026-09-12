"""Keep the test run off the developer's machine and off the network."""

from __future__ import annotations

import zipfile

import pytest

from rezunate_guard import constants


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("rezunate-home")
    monkeypatch.setenv(constants.HOME_ENV, str(home))
    monkeypatch.delenv(constants.API_KEY_ENV, raising=False)
    return home


@pytest.fixture
def make_docx():
    """Build a real Word file: a zip with a document part inside."""

    def build(path, body="Ali Hassan"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "word/document.xml",
                f"<w:document><w:p><w:t>{body}</w:t></w:p></w:document>",
            )
        return path

    return build


@pytest.fixture
def make_pdf():
    """Build a real PDF with a text layer, so `pdftotext` has something to find."""

    def build(path, lines=("Patient Ali Hassan", "Phone +971 50 123 4567")):
        drawn = "\n".join(f"({line}) Tj T*" for line in lines)
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td\n{drawn}\nET".encode()
        objects = [
            b"<</Type/Catalog/Pages 2 0 R>>",
            b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>",
            b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
            b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        ]
        # The binary marker every real PDF writer emits. Without it this file is pure
        # ASCII, sniffs as text, and never takes the path these tests are about.
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj".encode() + body + b" endobj\n"

        start = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode()
        out += f"trailer<</Size {len(objects) + 1}/Root 1 0 R>>\nstartxref\n{start}\n%%EOF".encode()
        path.write_bytes(bytes(out))
        return path

    return build
