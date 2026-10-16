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


def excerpt(text: str, max_words: int | None) -> str:
    """Whole sentences up to ``max_words`` (at least one sentence, cut with '...' if it alone is too long)."""
    if not max_words or len(text.split()) <= max_words:
        return text
    out, n = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        w = len(sentence.split())
        if out and n + w > max_words:
            break
        out.append(sentence)
        n += w
    result = " ".join(out)
    words = result.split()
    return " ".join(words[:max_words]) + "..." if len(words) > max_words else result


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
            else:
                self.speaker.say(excerpt(res.text, self.cfg["modes"]["read"].get("max_words_per_page")))

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
