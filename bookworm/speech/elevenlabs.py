"""ElevenLabs voice for BackgroundSpeaker, with the Mac voice as an automatic fallback.

Key: ELEVENLABS_API_KEY in .env. Settings: `speech.elevenlabs` in the config.

Latency tricks:
  - each sentence is requested as soon as it is queued (prefetch), so sentence 2 downloads
    while sentence 1 plays and only the first sentence's wait is ever heard;
  - every clip is cached on disk by (voice, model, text): the fixed replies ("Okay, reading.")
    cost one request ever and then play instantly, and re-reading a page costs nothing;
  - if a clip isn't back within `timeout_s`, that sentence is spoken by the fallback voice
    instead (and after repeated failures ElevenLabs is skipped for a minute), so the robot
    never goes silent because of the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128"
CACHE = Path(__file__).resolve().parents[2] / "runs" / "tts_cache"


def api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    for name in ("ELEVENLABS_API_KEY", "ELEVEN_LABS_KEY", "ELEVENLABS_KEY", "ELEVEN_LABS_API_KEY"):
        if (os.getenv(name) or "").strip():
            return os.getenv(name).strip()
    return ""


class ElevenLabsSay:
    """Blocking say(text) that plays an ElevenLabs clip (fetched ahead via prefetch())."""

    def __init__(self, key: str, fallback, voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
                 model: str = "eleven_flash_v2_5", timeout_s: float = 4.0, **_):
        self.key, self.fallback = key, fallback
        self.voice, self.model, self.timeout = voice_id, model, timeout_s
        self._pool = ThreadPoolExecutor(2)
        self._pending: dict[str, Future] = {}
        self._fails, self._skip_until = 0, 0.0
        self._last_ok, self._last_fail, self._why = 0.0, 0.0, ""   # for status(): is the Mac voice standing in?
        self.proc = None
        CACHE.mkdir(parents=True, exist_ok=True)

    # ---- fetching ------------------------------------------------------------------
    def _path(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.voice}|{self.model}|{text}".encode()).hexdigest()[:20]
        return CACHE / f"{h}.mp3"

    def _fetch(self, text: str) -> Path:
        path = self._path(text)
        if path.exists():
            return path
        req = urllib.request.Request(
            URL.format(voice=self.voice), method="POST",
            data=json.dumps({"text": text, "model_id": self.model}).encode(),
            headers={"xi-api-key": self.key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout + 6) as r:
            audio = r.read()
        print(f"[tts] elevenlabs {time.perf_counter() - t0:.2f}s for {len(text)} chars", flush=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(audio)
        tmp.replace(path)
        return path

    def prefetch(self, text: str) -> None:
        if time.time() < self._skip_until or text in self._pending or self._path(text).exists():
            return
        self._pending[text] = self._pool.submit(self._fetch, text)

    # ---- speaking ------------------------------------------------------------------
    def __call__(self, text: str) -> None:
        path = self._path(text)
        if not path.exists():
            if time.time() < self._skip_until:
                return self.fallback(text)
            self.prefetch(text)
            fut = self._pending.get(text)
            try:
                path = fut.result(timeout=self.timeout) if fut else self._fetch(text)
                self._fails = 0
            except Exception as e:  # noqa: BLE001 - timeout, HTTP error, no network: never go silent
                self._fails += 1
                self._last_fail, self._why = time.time(), _reason(e)
                if self._fails >= 2:
                    self._skip_until = time.time() + 60
                print(f"[tts] elevenlabs failed ({str(e)[:120]}); using the Mac voice", flush=True)
                return self.fallback(text)
            finally:
                self._pending.pop(text, None)
        self._last_ok = time.time()
        self.proc = subprocess.Popen(["afplay", str(path)])
        self.proc.wait()

    def status(self) -> dict:
        """ElevenLabs, unless its last request failed (or it's being skipped): then the Mac voice is speaking."""
        if self._last_fail > self._last_ok or time.time() < self._skip_until:
            fallback = getattr(self.fallback, "name", "Mac voice")
            return {"name": fallback, "fallback": True, "why": f"ElevenLabs {self._why}"}
        return {"name": "ElevenLabs", "fallback": False, "why": ""}

    def stop(self) -> None:
        for fut in self._pending.values():
            fut.cancel()
        self._pending.clear()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        if hasattr(self.fallback, "stop"):
            self.fallback.stop()

    def warm(self, phrases) -> None:
        """Fetch fixed replies in the background so they play instantly from the first use."""
        for p in phrases:
            self.prefetch(p)


def _reason(e: Exception) -> str:
    """Short, human reason an ElevenLabs request failed (shown on the control page)."""
    if isinstance(e, urllib.error.HTTPError):
        return {401: "key rejected", 402: "out of credits", 404: "voice not on this account",
                429: "rate-limited"}.get(e.code, f"error {e.code}")
    if isinstance(e, urllib.error.URLError):
        return "unreachable (no internet?)"
    if isinstance(e, TimeoutError) or type(e).__name__ == "TimeoutError":
        return "too slow"
    return "failed"


def list_voices(key: str) -> list[tuple[str, str, str]]:
    """[(name, voice_id, description)] for the voices on this ElevenLabs account."""
    req = urllib.request.Request("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key})
    with urllib.request.urlopen(req, timeout=15) as r:
        voices = json.load(r)["voices"]
    out = []
    for v in voices:
        labels = v.get("labels") or {}
        desc = ", ".join(str(labels[k]) for k in ("gender", "age", "accent", "descriptive", "use_case") if labels.get(k))
        out.append((v["name"], v["voice_id"], desc))
    return sorted(out)
