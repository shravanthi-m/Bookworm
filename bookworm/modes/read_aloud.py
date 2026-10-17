"""Read-aloud mode: speak each page as the robot turns through the book.

Speech is queued on a background thread, so on_spread returns immediately and the
arm flips the next page while this one is still being read aloud.
"""
from __future__ import annotations

import re

MESSAGES = {
    "flip_failed": "I couldn't turn the page. Can you help me?",
    "multi": "Oops, I think I turned more than one page.",
    "backward": "Hmm, the page went backwards.",
}


_WORD_LISTS = ("/usr/share/dict/words", "/usr/share/dict/american-english", "/usr/share/dict/british-english")
_lexicon: set[str] | None = None


def lexicon() -> set[str]:
    """System word list (macOS ships 235k words; Linux: apt install wamerican). Empty if none."""
    global _lexicon
    if _lexicon is None:
        _lexicon = set()
        for path in _WORD_LISTS:
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    _lexicon.update(w.strip().lower() for w in f if w.strip())
            except OSError:
                continue
    return _lexicon


def _known(word: str, lex: set[str]) -> bool:
    w = word.lower().strip("'’")
    if w in lex or len(w) <= 1:
        return w in lex or w in ("a", "i")
    # Word lists often lack inflections: try removing common endings (and restoring an 'e').
    for suffix in ("'s", "’s", "s", "es", "ed", "d", "ing", "ly", "er", "est", "n't", "’t"):
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            stem = w[: -len(suffix)]
            if stem in lex or stem + "e" in lex or (len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in lex):
                return True
    return False


def sentence_quality(sentence: str, lex: set[str] | None = None) -> float:
    """0..1: how much a sentence looks like clean English rather than OCR noise."""
    lex = lexicon() if lex is None else lex
    tokens = sentence.split()
    if not tokens:
        return 0.0
    words = [re.sub(r"^[^\w']+|[^\w']+$", "", t) for t in tokens]
    words = [w for w in words if w]
    junk = sum(1 for t in tokens if not re.search(r"[A-Za-z]", t)) / len(tokens)   # "|", "¢", "4)"
    if not words:
        return 0.0
    if lex:
        good = sum(_known(w, lex) or (w[0].isupper() and w[1:].islower()) for w in words)  # names count
    else:  # no word list: letters-only, has a vowel, plausible length
        good = sum(bool(re.fullmatch(r"[A-Za-z']{1,15}", w) and re.search(r"[aeiouyAEIOUY]", w)) for w in words)
    return max(0.0, good / len(words) - junk)


def split_sentences(text: str) -> list[str]:
    """Sentence split on paragraph breaks and sentence ends, incl. closing quotes ("…all.' Then")."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        flat = " ".join(para.split())
        out += [s for s in re.split(r"(?<=[.!?])[’'\"”]?\s+(?=[A-Z‘'\"“])", flat) if s]
    return out


def _is_header(sentence: str) -> bool:
    """Running heads like '200 : JOSTEIN GAARDER' — mostly capitals and/or digits."""
    letters = [c for c in sentence if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6


def _caps_word(w: str) -> bool:
    letters = [c for c in w if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def strip_running_head(text: str) -> str:
    """Drop a page's running head ('200 : JOSTEIN GAARDER', "SOPHIE'S WORLD : 201"): leading
    page numbers, separators and ALL-CAPS words, up to the first ordinary word."""
    words = text.split()
    i = 0
    while i < len(words):
        letters = [c for c in words[i] if c.isalpha()]
        if not letters:                               # '200', ':', '|', '•'
            i += 1
        elif all(c.isupper() for c in letters) and (
                len(letters) >= 2 or (i + 1 < len(words) and _caps_word(words[i + 1]))):
            i += 1                                    # GAARDER, SOPHIE'S, the "A" of "A BOOK"
        else:
            break
    return " ".join(words[i:])


def from_top(texts: list[str], max_words: int = 60, overrun: int = 30) -> tuple[str, str]:
    """Read in order from the top of the first page: about ``max_words`` words, finishing the
    sentence in progress (at most ``overrun`` extra words). Returns (this chunk, the rest)."""
    text = " ".join(strip_running_head(t) for t in texts if t.strip())
    words = re.sub(r"[•«»|■□▪●◆~_]", " ", text).split()        # OCR specks the voice would trip on
    if len(words) <= max_words:
        return " ".join(words), ""
    end = max_words
    while end < min(len(words), max_words + overrun) and not re.search(r"[.!?][’'\"”]?$", words[end - 1]):
        end += 1
    return " ".join(words[:end]), " ".join(words[end:])


def excerpt(text: str, max_words: int | None, min_quality: float = 0.95, max_sentence_words: int = 35) -> str:
    """Clean whole sentences, in reading order, up to ``max_words`` (None = no limit).

    OCR of small print garbles some sentences; only near-perfect ones (``min_quality``
    share of recognised words) of sensible length are read. Long run-on "sentences" are
    usually several lines glued together where OCR lost the punctuation, so they're
    skipped. If no sentence is clean, the best one is used only if it is close (>= 0.85);
    otherwise returns "" (better silent than reading noise). A single over-long sentence
    is cut with '...'.
    """
    sentences = [s for s in split_sentences(text) if not _is_header(s)]
    scored = [(s, sentence_quality(s)) for s in sentences if 3 <= len(s.split()) <= max_sentence_words]
    clean = [s for s, q in scored if q >= min_quality]
    if not clean and scored:
        best, q = max(scored, key=lambda sq: sq[1])
        clean = [best] if q >= 0.85 else []
    out, n = [], 0
    for s in clean:
        w = len(s.split())
        if max_words and out and n + w > max_words:
            break
        out.append(s)
        n += w
    result = " ".join(out)
    words = result.split()
    return " ".join(words[:max_words]) + "..." if max_words and len(words) > max_words else result


class ReadAloud:
    def __init__(self, speaker, cfg: dict):
        self.speaker, self.cfg = speaker, cfg

    def on_spread(self, index, spread, results):
        o = self.cfg["ocr"]
        for side in self.cfg["modes"]["read"]["read_order"]:
            res = results.get(side)
            if res is None or res.word_count < o["min_words_for_text"]:
                continue  # blank page or a picture
            if res.mean_conf < o["min_page_conf"]:
                self.speaker.say(f"I can't read the {side} page clearly.")
                continue
            r = self.cfg["modes"]["read"]
            text = excerpt(res.text, r.get("max_words_per_page"), r.get("min_sentence_quality", 0.95))
            if text:                          # "" = nothing clean enough to read on this page
                self.speaker.say(text)

    def on_event(self, event):
        self.speaker.say(MESSAGES.get(event, event))

    def finish(self, summary):
        self.speaker.say("The end." if summary.stopped_because == "end of book" else "Stopping here.")
        self.speaker.wait()


class Combined:
    """Fan one session out to several modes, e.g. read aloud AND store the scan."""

    def __init__(self, *handlers):
        self.handlers = handlers

    def on_spread(self, index, spread, results):
        for h in self.handlers:
            h.on_spread(index, spread, results)

    def on_event(self, event):
        for h in self.handlers:
            h.on_event(event)

    def finish(self, summary):
        for h in self.handlers:
            h.finish(summary)
