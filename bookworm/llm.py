"""OpenRouter client (one OpenAI-compatible endpoint, any model) for summaries and image checks.

Key: OPENROUTER_API_KEY in .env. Model and timeout: `llm:` in the config.
Stdlib only. Every call either returns or raises LlmError, so callers can always fall back.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

URL = "https://openrouter.ai/api/v1/chat/completions"


class LlmError(RuntimeError):
    pass


def _image_part(img: np.ndarray, max_side: int = 1280) -> dict:
    """JPEG data URL, downscaled: the checks need shapes, not letters, and smaller is cheaper and faster."""
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise LlmError("could not encode image")
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}}


class OpenRouter:
    def __init__(self, model: str = "google/gemini-2.5-flash", timeout_s: float = 20.0, **_):
        from dotenv import load_dotenv

        load_dotenv()
        self.key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY") or "").strip()
        self.model, self.timeout = model, timeout_s

    @property
    def ready(self) -> bool:
        return bool(self.key)

    def ask(self, prompt: str, images: list[np.ndarray] = (), system: str | None = None,
            json_reply: bool = False, max_tokens: int = 300) -> str:
        if not self.key:
            raise LlmError("OPENROUTER_API_KEY not set in .env")
        content = [{"type": "text", "text": prompt}] + [_image_part(im) for im in images]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if json_reply:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "X-Title": "Bookworm"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise LlmError(f"OpenRouter {e.code}: {e.read().decode(errors='replace')[:200]}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LlmError(f"OpenRouter unreachable: {e}") from e
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LlmError(f"unexpected reply: {str(data)[:200]}") from e

    def ask_json(self, prompt: str, images: list[np.ndarray] = (), **kw) -> dict:
        text = self.ask(prompt, images, json_reply=True, **kw)
        m = re.search(r"\{.*\}", text, re.S)          # tolerate ```json fences or chatter around it
        try:
            return json.loads(m.group() if m else text)
        except json.JSONDecodeError as e:
            raise LlmError(f"reply was not JSON: {text[:200]}") from e


def make_llm(lcfg: dict | None) -> OpenRouter:
    return OpenRouter(**(lcfg or {}))


SUMMARY_SYSTEM = ("You summarise pages of a book being read aloud by a robot. The text comes from OCR "
                  "and may contain misread words; ignore page numbers, headers and garbage. Reply with "
                  "plain spoken sentences only: no lists, markdown, or quotation of OCR errors.")


def summarize(llm: OpenRouter, text: str, sentences: int = 2) -> str:
    prompt = (f"Summarise what happens on these pages in {sentences} short, simple sentences a child "
              f"would understand.\n\nPAGES:\n{text[:6000]}")
    return llm.ask(prompt, system=SUMMARY_SYSTEM, max_tokens=200)
