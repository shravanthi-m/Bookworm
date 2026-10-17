import time

from bookworm.speech import BackgroundSpeaker


class SlowSay:
    def __init__(self):
        self.spoken = []

    def __call__(self, text):
        self.spoken.append(text)
        time.sleep(0.2)

    def stop(self):
        pass


def test_stop_then_resume_continues_from_the_interrupted_sentence():
    say = SlowSay()
    s = BackgroundSpeaker(say)
    s.say("One. Two. Three.")
    time.sleep(0.3)                     # cut off during "Two."
    s.stop()
    s.wait()
    assert say.spoken == ["One.", "Two."]
    assert s.resume()
    s.wait()
    assert say.spoken[2:] == ["Two.", "Three."]
    assert not s.resume()               # nothing left to continue


def test_short_replies_are_not_kept_for_resume():
    say = SlowSay()
    s = BackgroundSpeaker(say)
    s.say("Okay, reading.", resumable=False)
    s.say("Page one. Page two.")
    time.sleep(0.3)                     # cut off during "Page one."
    s.stop()
    s.say("Sorry.", resumable=False)    # a reply after the stop must not replace the saved place
    s.wait()
    s.stop()                            # a second stop with nothing playing keeps it too
    assert s.resume()
    s.wait()
    assert say.spoken[-2:] == ["Page one.", "Page two."]


# ---- which voice is active (shown on the control page) -------------------------------------
import io
import urllib.error

import bookworm.speech.elevenlabs as el
from bookworm.speech import make_speaker


def test_no_key_says_mac_voice_and_why(monkeypatch):
    monkeypatch.setattr(el, "api_key", lambda: "")
    v = make_speaker({"engine": "auto"}).voice
    assert v["fallback"] and v["why"] == "no ElevenLabs key in .env"
    assert v["name"] in ("Mac voice", "system voice")


def test_system_engine_is_not_flagged():
    v = make_speaker({"engine": "system"}).voice
    assert not v["fallback"] and v["why"] == ""


class Fallback:
    name = "Mac voice"

    def __init__(self):
        self.spoken = []

    def __call__(self, text):
        self.spoken.append(text)


def _eleven(tmp_path, monkeypatch, fetch):
    monkeypatch.setattr(el, "CACHE", tmp_path)
    say = el.ElevenLabsSay("k", Fallback(), timeout_s=0.5)
    monkeypatch.setattr(say, "_fetch", fetch)
    monkeypatch.setattr(el.subprocess, "Popen", lambda *a, **k: type("P", (), {"wait": lambda s: 0, "poll": lambda s: 0})())
    return say


def test_elevenlabs_ok_then_key_rejected_flags_the_fallback(tmp_path, monkeypatch):
    def ok(text):
        p = tmp_path / "clip.mp3"; p.write_bytes(b"x"); return p
    say = _eleven(tmp_path, monkeypatch, ok)
    say("Hello.")
    assert say.status() == {"name": "ElevenLabs", "fallback": False, "why": ""}

    def rejected(text):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO())
    monkeypatch.setattr(say, "_fetch", rejected)
    say("Next sentence.")
    assert say.fallback.spoken == ["Next sentence."]          # never silent
    assert say.status() == {"name": "Mac voice", "fallback": True, "why": "ElevenLabs key rejected"}


def test_slow_elevenlabs_reads_too_slow(tmp_path, monkeypatch):
    def slow(text):
        time.sleep(2)
    say = _eleven(tmp_path, monkeypatch, slow)
    say("Hello.")
    assert say.status()["why"] == "ElevenLabs too slow"
