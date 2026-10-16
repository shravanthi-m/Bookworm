"""Turn a scan folder into a searchable PDF: the colour page image (text, illustrations,
folds and all) with an invisible OCR text layer on top, so the PDF can be searched,
copied from, and read by other tools.

    python -m bookworm pdf scans/mybook            -> scans/mybook/book.pdf

The text layer comes from Tesseract run on our preprocessed page (lighting-flattened +
sharpened — far better than OCR on the raw colour image), rendered as a text-only PDF and
overlaid on the colour image page at the same size.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np


def _pages(scan_dir: Path) -> list[Path]:
    manifest = scan_dir / "manifest.json"
    if manifest.exists():
        return [scan_dir / p["file"] for p in json.loads(manifest.read_text())["pages"]]
    return sorted((scan_dir / "pages").glob("*.png"))


def _dpi(img: np.ndarray, page_width_mm: float | None) -> int:
    """Physical-ish page size in the PDF: page pixels / page width. Falls back to 150 dpi."""
    if not page_width_mm:
        return 150
    return int(np.clip(img.shape[1] / (page_width_mm / 25.4), 50, 1200))


def _image_page(img_bgr: np.ndarray, dpi: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(buf, "PDF", resolution=dpi)
    return buf.getvalue()


def _text_layer(ocr_img: np.ndarray, dpi: int, lang: str, psm: int) -> bytes:
    import pytesseract

    return pytesseract.image_to_pdf_or_hocr(
        ocr_img, extension="pdf", lang=lang, config=f"--psm {psm} --dpi {dpi} -c textonly_pdf=1")


def build_pdf(scan_dir: str | Path, out: str | Path | None = None, searchable: bool = True,
              lang: str = "eng", psm: int = 3, page_width_mm: float | None = None,
              ocr_input: str = "gray") -> Path:
    """Write a (searchable) PDF of every page in ``scan_dir``. Returns the PDF path."""
    from pypdf import PdfReader, PdfWriter

    from .vision.camera import read_image
    from .vision.preprocess import prepare_for_ocr

    scan_dir = Path(scan_dir)
    out = Path(out) if out else scan_dir / "book.pdf"
    pages = _pages(scan_dir)
    if not pages:
        raise FileNotFoundError(f"no pages in {scan_dir}/pages")

    writer = PdfWriter()
    for i, path in enumerate(pages, 1):
        img = read_image(path)
        dpi = _dpi(img, page_width_mm)
        page = writer.add_page(PdfReader(io.BytesIO(_image_page(img, dpi))).pages[0])
        if searchable:
            text = PdfReader(io.BytesIO(_text_layer(prepare_for_ocr(img, ocr_input), dpi, lang, psm))).pages[0]
            page.merge_page(text)                    # same pixels + same dpi -> same page size
        print(f"\r[pdf] page {i}/{len(pages)}", end="", flush=True)
    writer.add_metadata({"/Producer": "Bookworm", "/Title": scan_dir.name})
    with open(out, "wb") as f:
        writer.write(f)
    print(f"\n[pdf] {'searchable ' if searchable else ''}PDF -> {out}")
    return out
