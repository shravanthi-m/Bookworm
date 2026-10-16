"""Text-to-speech backends.

``BackgroundSpeaker`` speaks on a worker thread so the arm can flip the next page while
the current one is being read — that overlap hides most of the per-page latency.
"""
from __future__ import annotations

import platform
import queue
import shutil
import subprocess
import threading


class PrintSpeaker:
    """Prints instead of speaking — for tests and quiet development."""

    def __init__(self, **_):
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
        print(f"[speak] {text}")

    def wait(self) -> None:
        pass


def _system_say(rate: int = 170, voice: str | None = None, **_):
    """Blocking say() using the OS voice: macOS `say`, Linux `espeak-ng`/`espeak`."""
    if platform.system() == "Darwin":
        base = ["say", "-r", str(rate)] + (["-v", voice] if voice else [])
    else:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not exe:
            raise RuntimeError("no TTS found: install espeak-ng (apt install espeak-ng)")
        base = [exe, "-s", str(rate)] + (["-v", voice] if voice else [])
    return lambda text: subprocess.run(base + [text], check=False)


class BackgroundSpeaker:
    """Queues utterances and speaks them in order on a worker thread. ``say`` returns at once."""

    def __init__(self, say_fn):
        self._say = say_fn
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self._q.get()
            try:
                self._say(text)
            finally:
                self._q.task_done()

    def say(self, text: str) -> None:
        print(f"[speak] {text[:80]}{'...' if len(text) > 80 else ''}")
        self._q.put(text)

    def wait(self) -> None:
        """Block until everything queued has been spoken."""
        self._q.join()


def make_speaker(scfg: dict):
    engine = scfg.get("engine", "system")
    if engine == "print":
        return PrintSpeaker()
    if engine == "system":
        return BackgroundSpeaker(_system_say(**scfg))
    raise ValueError(f"unknown speech engine {engine!r} (system | print)")
