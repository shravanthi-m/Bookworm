"""Text-to-speech backends.

``BackgroundSpeaker`` speaks on a worker thread so the arm can flip the next page while
the current one is being read — that overlap hides most of the per-page latency.
"""
from __future__ import annotations

import platform
import queue
import re
import shutil
import subprocess
import threading


class PrintSpeaker:
    """Prints instead of speaking — for tests and quiet development."""

    def __init__(self, **_):
        self.spoken: list[str] = []

    def say(self, text: str, resumable: bool = True) -> None:
        self.spoken.append(text)
        print(f"[speak] {text}")

    def wait(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def resume(self) -> bool:
        return False


class _SystemSay:
    """Blocking say() via a subprocess, so stop() can cut it off mid-sentence."""

    def __init__(self, cmd: list[str]):
        self.cmd, self.proc = cmd, None

    def __call__(self, text: str) -> None:
        self.proc = subprocess.Popen(self.cmd + [text])
        self.proc.wait()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def _system_say(rate: int = 170, voice: str | None = None, **_):
    """Blocking say() using the OS voice: macOS `say`, Linux `espeak-ng`/`espeak`."""
    if platform.system() == "Darwin":
        base = ["say", "-r", str(rate)] + (["-v", voice] if voice else [])
    else:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not exe:
            raise RuntimeError("no TTS found: install espeak-ng (apt install espeak-ng)")
        base = [exe, "-s", str(rate)] + (["-v", voice] if voice else [])
    return _SystemSay(base)


class BackgroundSpeaker:
    """Queues utterances and speaks them in order on a worker thread. ``say`` returns at once.

    Text is spoken a sentence at a time, so ``stop()`` can remember where it was cut off and
    ``resume()`` can carry on from the interrupted sentence.
    """

    SENTENCE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, say_fn):
        self._say = say_fn
        self._q: queue.Queue = queue.Queue()
        self._current: str | None = None
        self._paused: list[str] = []
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text, resumable = self._q.get()
            self._current = text if resumable else None
            try:
                self._say(text)
            finally:
                self._current = None
                self._q.task_done()

    def say(self, text: str, resumable: bool = True) -> None:
        """``resumable=False`` for short replies ("Okay, reading."): stop() won't keep them for resume()."""
        print(f"[speak] {text[:80]}{'...' if len(text) > 80 else ''}")
        if resumable:
            self._paused = []                       # new reading: forget an old pause
        for sentence in self.SENTENCE.split(text.strip()):
            if sentence:
                self._q.put((sentence, resumable))

    def wait(self) -> None:
        """Block until everything queued has been spoken."""
        self._q.join()

    def stop(self) -> None:
        """Drop everything queued and cut off the current sentence; remember it for resume()."""
        left = [self._current] if self._current else []
        try:
            while True:
                text, resumable = self._q.get_nowait()
                if resumable:
                    left.append(text)
                self._q.task_done()
        except queue.Empty:
            pass
        if hasattr(self._say, "stop"):
            self._say.stop()
        if left:                                    # a second stop with nothing playing keeps the first
            self._paused = left

    def resume(self) -> bool:
        """Carry on from the sentence stop() interrupted. False if there's nothing to resume."""
        left, self._paused = self._paused, []
        for sentence in left:
            self._q.put((sentence, True))
        return bool(left)

    @property
    def speaking(self) -> bool:
        return self._q.unfinished_tasks > 0


def make_speaker(scfg: dict):
    engine = scfg.get("engine", "system")
    if engine == "print":
        return PrintSpeaker()
    if engine == "system":
        return BackgroundSpeaker(_system_say(**scfg))
    raise ValueError(f"unknown speech engine {engine!r} (system | print)")
