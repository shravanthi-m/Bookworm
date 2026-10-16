"""Text-to-speech backends."""
from __future__ import annotations


class PrintSpeaker:
    """Prints instead of speaking — for tests and quiet development."""

    def __init__(self, **_):
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
        print(f"[speak] {text}")


class Pyttsx3Speaker:
    """Offline TTS (espeak on Linux/Pi, NSSpeech on macOS, SAPI on Windows)."""

    def __init__(self, rate: int = 170, **_):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def say(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()


def make_speaker(scfg: dict):
    if scfg.get("engine") == "pyttsx3":
        return Pyttsx3Speaker(**scfg)
    return PrintSpeaker(**scfg)
