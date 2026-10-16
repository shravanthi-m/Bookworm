"""Read-aloud mode: speak each page as the robot turns through the book."""
from __future__ import annotations

MESSAGES = {
    "flip_failed": "I couldn't turn the page. Can you help me?",
    "multi": "Oops, I think I turned more than one page.",
    "backward": "Hmm, the page went backwards.",
}


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
                self.speaker.say(res.text)

    def on_event(self, event):
        self.speaker.say(MESSAGES.get(event, event))

    def finish(self, summary):
        self.speaker.say("The end." if summary.stopped_because == "end of book" else "Stopping here.")
