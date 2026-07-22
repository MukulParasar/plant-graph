"""
Universal document ingestion: extracts raw text from heterogeneous formats
(plain text, PDF text-layer, scanned/image PDFs and images via OCR).
"""
import os
from pathlib import Path

import pdfplumber
from PIL import Image
import pytesseract


def extract_text_from_pdf(path: str) -> str:
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if len(page_text.strip()) < 20:
                # likely a scanned page with no text layer -> OCR fallback
                im = page.to_image(resolution=200).original
                page_text = pytesseract.image_to_string(im)
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_image(path: str) -> str:
    return pytesseract.image_to_string(Image.open(path))


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return extract_text_from_image(path)
    if ext in (".txt", ".md"):
        return Path(path).read_text(errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100):
    """Simple sliding-window chunking for search/citation purposes."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({"start": start, "end": end, "text": text[start:end]})
        if end == n:
            break
        start = end - overlap
    return chunks
