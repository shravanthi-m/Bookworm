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

    def page_number(self, page: np.ndarray, bottom_frac: float = 0.08) -> int | None:
        h = page.shape[0]
        strips = [page[int(h * (1 - bottom_frac)):], page[: int(h * bottom_frac)]]  # footer, then header
        for strip in strips:
            txt = self._tess.image_to_string(strip, config="--psm 6 -c tessedit_char_whitelist=0123456789")
            n = parse_page_number(txt)
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


def parse_page_number(text: str) -> int | None:
    m = re.search(r"\b(\d{1,4})\b", text)
    return int(m.group(1)) if m else None
