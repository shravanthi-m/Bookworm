"""OCR with Tesseract, plus text cleanup and page-number extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


def ocr_image(ocr, page):
    """The image a given engine reads best: colour for Apple Vision, preprocessed gray for Tesseract."""
    return page.color if getattr(ocr, "wants_color", False) else page.ocr


def make_ocr(ocfg: dict):
    """ocr.engine: auto (Apple Vision on macOS if available, else Tesseract) | apple | tesseract."""
    engine = ocfg.get("engine", "auto")
    if engine in ("auto", "apple"):
        try:
            return AppleVisionOcr(**ocfg)
        except ImportError:
            if engine == "apple":
                raise
    return TesseractOcr(**ocfg)


@dataclass
class OcrResult:
    text: str
    mean_conf: float
    words: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.words)


class TesseractOcr:
    def __init__(self, lang: str = "eng", psm: int = 3, min_word_conf: float = 50, **_):
        import pytesseract  # imported lazily so tests/mocks don't need it

        self._tess = pytesseract
        self.lang = lang
        self.psm = psm
        self.min_word_conf = min_word_conf

    def read(self, img: np.ndarray) -> OcrResult:
        data = self._tess.image_to_data(
            img, lang=self.lang, config=f"--psm {self.psm}", output_type=self._tess.Output.DICT
        )
        lines: dict[tuple, list[str]] = {}
        confs, words = [], []
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not word.strip() or conf < self.min_word_conf:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(conf)
            words.append(word)
        raw = _lines_to_text(lines)
        return OcrResult(clean_text(raw), float(np.mean(confs)) if confs else 0.0, words)

    def page_number(self, page: np.ndarray, frac: float = 0.12) -> int | None:
        """Page number from the header or footer strip (top/bottom ``frac`` of the page).

        Read as normal text (a digits-only whitelist turns header words like "JOSTEIN
        GAARDER" into junk numbers). Pass 1 uses line layout ("204 : JOSTEIN GAARDER");
        pass 2 uses sparse mode, which still isolates the number when the header is curled.
        """
        h = page.shape[0]
        strips = (page[: int(h * frac)], page[int(h * (1 - frac)):])        # header, then footer
        for psm in (6, 11):
            for strip in strips:
                n = self._number_in(strip, psm)
                if n is not None:
                    return n
        return None

    def _number_in(self, strip: np.ndarray, psm: int, min_conf: float = 70) -> int | None:
        d = self._tess.image_to_data(strip, config=f"--psm {psm}", output_type=self._tess.Output.DICT)
        lines: dict[tuple, list[tuple[str, float, int]]] = {}
        for i, w in enumerate(d["text"]):
            t = w.strip().strip(PUNCT)
            if t:
                key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
                lines.setdefault(key, []).append((t, float(d["conf"][i]), d["height"][i]))
        heights = [hh for ws in lines.values() for _, _, hh in ws]
        if not heights:
            return None
        min_h = 0.6 * float(np.median(heights))                  # ignore specks read as digits
        best = None
        for ws in lines.values():
            if len(ws) > 8:                                       # a body-text line, not a header
                continue
            for t, conf, hh in (ws[0], ws[-1]):
                if _is_page_number(t) and conf >= min_conf and hh >= min_h and (best is None or conf > best[0]):
                    best = (conf, int(t))
        return best[1] if best else None


PUNCT = ".,:;|()[]{}-–—'\"*·•‘’“”"


def _is_page_number(token: str) -> bool:
    # 2-4 digits: pages 1-9 are almost always unnumbered front matter, and single digits are
    # the most common OCR misreads ("o" -> 0, "l" -> 1).
    return token.isdigit() and 2 <= len(token) <= 4


class AppleVisionOcr:
    """macOS Vision framework text recognition (the engine behind Live Text), via `ocrmac`.

    Free, offline, ~1 s per page. On the live phone feed it read ~50% more words than
    Tesseract with far fewer garbled ones, because it is trained on photos, not scans.
    """

    wants_color = True

    def __init__(self, min_word_conf: float = 50, level: str = "accurate", **_):
        import sys

        if sys.platform != "darwin":
            raise ImportError("Apple Vision OCR needs macOS")
        from ocrmac import ocrmac  # pip install ocrmac (macOS only)
        from PIL import Image

        self._ocrmac, self._Image = ocrmac, Image
        self.min_conf = min_word_conf / 100.0
        self.level = level
        self._cache: tuple = (None, [])

    def _lines(self, img: np.ndarray) -> list:
        """[(text, conf 0-1, (x, y, w, h) normalised, y from the bottom)], top to bottom."""
        key = (img.__array_interface__["data"][0], img.shape)
        if self._cache[0] == key:          # read() and page_number() on the same image: OCR once
            return self._cache[1]
        import cv2

        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)
        res = self._ocrmac.OCR(self._Image.fromarray(rgb), recognition_level=self.level,
                               language_preference=["en-US"]).recognize()
        lines = sorted(res, key=lambda r: (-(r[2][1] + r[2][3]), r[2][0]))
        self._cache = (key, lines)
        return lines

    def read(self, img: np.ndarray) -> OcrResult:
        lines = [ln for ln in self._lines(img) if ln[1] >= self.min_conf]
        if not lines:
            return OcrResult("", 0.0, [])
        med_h = float(np.median([b[3] for _, _, b in lines]))
        out, prev_y = [], None
        for text, _, (x, y, w, h) in lines:
            if prev_y is not None and prev_y - (y + h) > 0.8 * med_h:
                out.append("")             # a vertical gap = new paragraph
            out.append(text)
            prev_y = y
        raw = "\n".join(out)
        return OcrResult(clean_text(raw), 100.0 * float(np.mean([c for _, c, _ in lines])), raw.split())

    def page_number(self, img: np.ndarray, frac: float = 0.12) -> int | None:
        for text, conf, (x, y, w, h) in self._lines(img):
            cy = y + h / 2
            if (cy >= 1 - frac or cy <= frac) and conf >= self.min_conf:
                n = parse_page_number(text)
                if n is not None:
                    return n
        return None


def _lines_to_text(lines: dict[tuple, list[str]]) -> str:
    out, last_par = [], None
    for (block, par, _), ws in sorted(lines.items()):
        if last_par is not None and (block, par) != last_par:
            out.append("")  # blank line between paragraphs
        out.append(" ".join(ws))
        last_par = (block, par)
    return "\n".join(out)


def clean_text(text: str) -> str:
    """Make OCR output pleasant to listen to."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if not re.fullmatch(r"\d{1,4}", p)]  # drop lone page numbers
    return "\n\n".join(p for p in paras if p)


def parse_page_number(text: str, max_tokens: int = 8) -> int | None:
    """A 1-4 digit number at the start or end of a short line ("204 : JOSTEIN GAARDER",
    "SOPHIE'S WORLD : 201", "- 17 -"). Long lines are body text and are ignored."""
    for line in text.splitlines():
        tokens = [t.strip(PUNCT) for t in line.split()]
        tokens = [t for t in tokens if t]
        if not tokens or len(tokens) > max_tokens:
            continue
        for t in (tokens[0], tokens[-1]):
            if _is_page_number(t):
                return int(t)
    return None
