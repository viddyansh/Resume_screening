"""
parser.py
---------
Handles extraction of raw text from PDF and TXT resume files.

Extraction strategy (tried in order):
  1. PyMuPDF  (fitz)   — fast native text layer extraction
  2. PyPDF2            — fallback for text-layer PDFs
  3. pdfminer.six      — deeper text-layer extraction
  4. pytesseract + PyMuPDF pixmap  — OCR for scanned/image PDFs (no poppler needed)
  5. pytesseract + pdf2image       — final OCR fallback

Why OCR fallback?
  Many real-world resumes are scanned documents saved as PDF.
  They contain NO embedded text — only rasterised images of text.
  Standard PDF parsers return empty strings for these files.
  OCR (Optical Character Recognition) reads the image pixels and
  converts them back to machine-readable text.

Requirements for OCR (only needed for scanned PDFs):
  pip install pytesseract pillow
  # Also install Tesseract binary:
  #   Windows : https://github.com/UB-Mannheim/tesseract/wiki
  #             (tick "Add to PATH" during install)
  #   macOS   : brew install tesseract
  #   Linux   : sudo apt install tesseract-ocr
"""

import os
import io
import logging
import tempfile

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH = 30  # chars required to count as a successful extraction


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_meaningful(text: str) -> bool:
    """True if text has enough real alphabetic content."""
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) >= _MIN_TEXT_LENGTH and any(c.isalpha() for c in stripped)


def _extract_via_pymupdf(file_path: str) -> str:
    """Stage 1 — PyMuPDF native text layer."""
    import fitz
    text = ""
    doc = fitz.open(file_path)
    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF has zero pages.")
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text


def _extract_via_pypdf2(file_path: str) -> str:
    """Stage 2 — PyPDF2 native text layer."""
    import PyPDF2
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        if len(reader.pages) == 0:
            raise ValueError("PDF has zero pages.")
        for page in reader.pages:
            chunk = page.extract_text()
            if chunk:
                text += chunk
    return text


def _extract_via_pdfminer(file_path: str) -> str:
    """Stage 3 — pdfminer.six (deeper layout analysis)."""
    from pdfminer.high_level import extract_text as pm_extract
    return pm_extract(file_path)


def _extract_via_ocr_pymupdf_pixmap(file_path: str) -> str:
    """
    Stage 4 — OCR using PyMuPDF page pixmaps + Tesseract.
    No poppler required — just pytesseract + Pillow.
    """
    import fitz
    import pytesseract
    from PIL import Image

    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        mat = fitz.Matrix(3.0, 3.0)   # 3× zoom ≈ 216 DPI — good OCR quality
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text += pytesseract.image_to_string(img, lang="eng")
    doc.close()
    return text


def _extract_via_ocr_pdf2image(file_path: str) -> str:
    """Stage 5 — OCR via pdf2image + Tesseract (requires poppler)."""
    import pytesseract
    from pdf2image import convert_from_path
    # Windows with poppler not on PATH:
    # images = convert_from_path(file_path, dpi=300, poppler_path=r"C:\poppler\bin")
    images = convert_from_path(file_path, dpi=300)
    return "".join(pytesseract.image_to_string(img, lang="eng") for img in images)


# ──────────────────────────────────────────────────────────────────────────────
# Main public extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using a 5-stage strategy.
    Falls through to OCR automatically for scanned/image-based PDFs.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text string.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError:        All extraction stages failed — see message for fix.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")

    stages = [
        ("PyMuPDF (native text)",        _extract_via_pymupdf),
        ("PyPDF2 (native text)",         _extract_via_pypdf2),
        ("pdfminer.six (native text)",   _extract_via_pdfminer),
        ("OCR — PyMuPDF+Tesseract",      _extract_via_ocr_pymupdf_pixmap),
        ("OCR — pdf2image+Tesseract",    _extract_via_ocr_pdf2image),
    ]

    last_error = None
    for stage_name, extractor in stages:
        try:
            text = extractor(file_path)
            if _is_meaningful(text):
                logger.info(
                    f"[parser] '{os.path.basename(file_path)}' "
                    f"extracted via {stage_name} ({len(text.strip())} chars)."
                )
                return text
            logger.warning(
                f"[parser] {stage_name} → only {len(text.strip())} chars. "
                "Trying next stage."
            )
        except ImportError as e:
            logger.warning(f"[parser] {stage_name} not available — {e}")
        except Exception as e:
            logger.warning(f"[parser] {stage_name} error — {e}")
            last_error = e

    raise ValueError(
        f"Could not extract text from '{os.path.basename(file_path)}'.\n\n"
        "📋 This PDF appears to be SCANNED / IMAGE-BASED (no embedded text).\n\n"
        "✅ FIX OPTION 1 — Install OCR:\n"
        "   pip install pytesseract pillow\n"
        "   Then install Tesseract:\n"
        "     Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
        "               (tick 'Add to PATH' during install, then restart terminal)\n"
        "     macOS   : brew install tesseract\n"
        "     Linux   : sudo apt install tesseract-ocr\n\n"
        "✅ FIX OPTION 2 — Convert to text first:\n"
        "   Open in MS Word → File → Save As → Plain Text (.txt)\n"
        "   Then re-upload the .txt file.\n"
        + (f"\nLast error: {last_error}" if last_error else "")
    )


# ──────────────────────────────────────────────────────────────────────────────
# TXT extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a .txt file.
    Tries UTF-8, latin-1, and cp1252 encodings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")

    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Could not decode '{os.path.basename(file_path)}'. "
        "Please re-save as UTF-8 and re-upload."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bytes entry point (Streamlit)
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from in-memory bytes (Streamlit uploaded files).
    Writes to a temp file, delegates to the appropriate extractor,
    then deletes the temp file.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".pdf", ".txt"):
        raise ValueError(
            f"Unsupported format '{ext}'. Only .pdf and .txt are accepted."
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return extract_text_from_pdf(tmp_path) if ext == ".pdf" else extract_text_from_txt(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_resume(source, filename: str = None) -> str:
    """
    Unified entry point: accepts a file path (str) OR raw bytes.

    Args:
        source:   str path  OR  bytes object.
        filename: Required when source is bytes.
    """
    if isinstance(source, bytes):
        if filename is None:
            raise ValueError("'filename' must be provided when source is bytes.")
        return extract_text_from_bytes(source, filename)

    path = str(source)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Accepted: .pdf, .txt")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helper
# ──────────────────────────────────────────────────────────────────────────────

def diagnose_pdf(file_path: str) -> dict:
    """
    Run each native extractor and report results — useful for debugging.

    Usage:
        from parser import diagnose_pdf
        import pprint; pprint.pprint(diagnose_pdf("resume.pdf"))
    """
    results = {}
    for name, fn in [
        ("PyMuPDF",      _extract_via_pymupdf),
        ("PyPDF2",       _extract_via_pypdf2),
        ("pdfminer.six", _extract_via_pdfminer),
    ]:
        try:
            text = fn(file_path)
            results[name] = {
                "chars":      len(text.strip()),
                "meaningful": _is_meaningful(text),
                "preview":    text.strip()[:300] or "(empty)",
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
