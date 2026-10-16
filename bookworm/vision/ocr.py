"""OCR with Tesseract, plus text cleanup and page-number extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


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
