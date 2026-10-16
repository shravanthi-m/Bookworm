"""Turn a scan folder into a searchable PDF: the colour page image (text, illustrations,
folds and all) with an invisible OCR text layer on top, so the PDF can be searched,
copied from, and read by other tools.

    python -m bookworm pdf scans/mybook            -> scans/mybook/book.pdf

The text layer comes from Apple Vision on macOS (line boxes -> invisible text via
reportlab), else Tesseract on our preprocessed page; either is rendered as a text-only
PDF page and overlaid on the colour image page at the same size.
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


def _apple_text_layer(img: np.ndarray, dpi: int, apple) -> bytes:
    """Text-only PDF page from Apple Vision: each recognised line drawn invisibly (render
    mode 3) over its box, stretched to the box width so selection lines up with the print."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    w_pt, h_pt = img.shape[1] / dpi * 72, img.shape[0] / dpi * 72
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w_pt, h_pt))
    for text, _conf, (x, y, w, h) in apple._lines(img):     # normalised, origin bottom-left = PDF's
        size = max(1.0, h * h_pt * 0.9)
        width = stringWidth(text, "Helvetica", size)
        t = c.beginText(x * w_pt, y * h_pt + h * h_pt * 0.15)
        t.setTextRenderMode(3)                               # invisible, but searchable/selectable
        t.setFont("Helvetica", size)
        if width > 0:
            t.setHorizScale(100.0 * w * w_pt / width)
        t.textOut(text)
        c.drawText(t)
    c.showPage()
    c.save()
    return buf.getvalue()


def build_pdf(scan_dir: str | Path, out: str | Path | None = None, searchable: bool = True,
              lang: str = "eng", psm: int = 3, page_width_mm: float | None = None,
              ocr_input: str = "gray", engine: str = "auto") -> Path:
    """Write a (searchable) PDF of every page in ``scan_dir``. Returns the PDF path."""
    from pypdf import PdfReader, PdfWriter

    from .vision.camera import read_image
    from .vision.preprocess import prepare_for_ocr

    scan_dir = Path(scan_dir)
    out = Path(out) if out else scan_dir / "book.pdf"
    pages = _pages(scan_dir)
    if not pages:
        raise FileNotFoundError(f"no pages in {scan_dir}/pages")

    apple = None
    if searchable and engine in ("auto", "apple"):
        try:                                                 # better text layer on macOS
            import reportlab  # noqa: F401

            from .vision.ocr import AppleVisionOcr

            apple = AppleVisionOcr()
        except ImportError:
            apple = None

    writer = PdfWriter()
    for i, path in enumerate(pages, 1):
        img = read_image(path)
        dpi = _dpi(img, page_width_mm)
        page = writer.add_page(PdfReader(io.BytesIO(_image_page(img, dpi))).pages[0])
        if searchable:
            layer = (_apple_text_layer(img, dpi, apple) if apple is not None
                     else _text_layer(prepare_for_ocr(img, ocr_input), dpi, lang, psm))
            page.merge_page(PdfReader(io.BytesIO(layer)).pages[0])   # same pixels + dpi -> same size
        print(f"\r[pdf] page {i}/{len(pages)}", end="", flush=True)
    writer.add_metadata({"/Producer": "Bookworm", "/Title": scan_dir.name})
    with open(out, "wb") as f:
        writer.write(f)
    kind = "searchable (Apple Vision) " if apple else ("searchable (Tesseract) " if searchable else "")
    print(f"\n[pdf] {kind}PDF -> {out}")
    return out
