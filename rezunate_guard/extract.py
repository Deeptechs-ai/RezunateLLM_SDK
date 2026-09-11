"""Pull readable text out of PDF or Word etc, the model would otherwise have to be refused."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

#: How long an external extractor may run before we give up and refuse the read.
EXTRACT_TIMEOUT_SECONDS = 60

#: Parts of an Office file worth reading, in the order they should appear.
_OFFICE_PARTS = (
    "word/document.xml",
    "xl/sharedStrings.xml",
    "ppt/slides/slide",
)

#: Where one block of text ends and the next begins. Without this every paragraph in a
#: Word file would run into the one after it.
_BLOCK_END = re.compile(r"</(w:p|w:tr|a:p|si|text:p)>", re.IGNORECASE)

_TAG = re.compile(r"<[^>]+>")
_BLANK_RUN = re.compile(r"\n{3,}")


class ExtractionError(Exception):
    """Text could not be pulled out of a file. Callers must fall back to refusing it."""


def strip_markup(markup: str) -> str:
    """Turn a lump of XML or HTML into the text a reader would see.

    Worth doing rather than scanning the markup as-is: the detection models return
    nothing at all for a name inside a tag, so `<p>Ali Hassan</p>` scans clean while the
    same words in prose do not.

    Args:
        markup: XML or HTML.

    Returns:
        The text, with block elements separated by newlines.
    """
    text = _BLOCK_END.sub("\n", markup)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    # Trailing spaces are what is left where a tag used to be.
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_RUN.sub("\n\n", text).strip()


def _leading_bytes(path: Path, count: int = 8) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(count)
    except OSError as exc:
        raise ExtractionError(f"could not open the file: {exc}") from exc


def _from_pdf(path: Path) -> str:
    """Extract a PDF's text layer using poppler's `pdftotext`.

    Returns:
        The text. Empty if the PDF is a scan, which has no text layer to find.

    Raises:
        ExtractionError: If `pdftotext` is missing, fails, or takes too long.
    """
    tool = shutil.which("pdftotext")
    if tool is None:
        raise ExtractionError("no PDF text extractor is installed (poppler's pdftotext)")

    try:
        finished = subprocess.run(
            [tool, "-layout", "-q", str(path), "-"],
            capture_output=True,
            timeout=EXTRACT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExtractionError(f"pdftotext could not run: {exc}") from exc

    if finished.returncode != 0:
        raise ExtractionError("pdftotext could not read this PDF")
    return finished.stdout.decode("utf-8", errors="replace")


def _from_office(path: Path) -> str:
    """Extract text from a zip-packaged Office file.

    Reads the document parts straight out of the archive, so nothing needs installing.

    Raises:
        ExtractionError: If the archive cannot be opened or holds no document part.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in sorted(archive.namelist())
                if name.startswith(_OFFICE_PARTS) and name.endswith(".xml")
            ]
            if not names:
                raise ExtractionError("this archive holds no document to read")
            parts = [archive.read(name).decode("utf-8", errors="replace") for name in names]
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        raise ExtractionError(f"could not read the document: {exc}") from exc

    return strip_markup("\n".join(parts))


def can_extract(path: Path | str) -> bool:
    """Return True if `extract_text` has a way to read this file.

    Cheap enough to ask before deciding whether to refuse a read.
    """
    try:
        head = _leading_bytes(Path(path))
    except ExtractionError:
        head = b""  # a file we cannot open is one we cannot read

    is_pdf = head.startswith(b"%PDF")
    is_office = head.startswith(b"PK\x03\x04")

    return is_office or (is_pdf and shutil.which("pdftotext") is not None)


def extract_text(path: Path | str) -> str:
    """Return the readable text of a file the model cannot be shown directly.

    Args:
        path: The file to read.

    Returns:
        Its text. Never empty; a file with nothing to find raises instead, since an
        empty redacted_copy would tell the model the file was blank.

    Raises:
        ExtractionError: If the format is unsupported, unreadable, or holds no text.
    """
    path = Path(path)
    head = _leading_bytes(path)

    if head.startswith(b"%PDF"):
        text = _from_pdf(path)
    elif head.startswith(b"PK\x03\x04"):
        text = _from_office(path)
    else:
        raise ExtractionError("there is no text extractor for this kind of file")

    if not text.strip():
        # A scanned PDF lands here: pages of pictures and no text layer. Reading it needs
        # OCR, so until that exists the honest answer is that we cannot.
        raise ExtractionError("no text could be found in this file; it may be a scan")
    return text
