import io
import json

import numpy as np
import pytest

from bookworm import llm as llm_mod
from bookworm.llm import LlmError, OpenRouter, summarize
from bookworm.vision.vlm_check import check_frame


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def fake_openrouter(monkeypatch, reply: str, sent: list):
    def urlopen(req, timeout):
        sent.append(json.loads(req.data))
        return FakeResponse(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", urlopen)
    client = OpenRouter()
    client.key = "test"
    return client


def test_summary_sends_text_and_returns_reply(monkeypatch):
    sent = []
    client = fake_openrouter(monkeypatch, "A fox finds a hat.", sent)
    assert summarize(client, "The fox saw a red hat.") == "A fox finds a hat."
    assert "The fox saw a red hat." in sent[0]["messages"][-1]["content"][0]["text"]


def test_image_is_sent_as_data_url(monkeypatch):
    sent = []
    client = fake_openrouter(monkeypatch, '{"ok": true}', sent)
    client.ask_json("q", [np.zeros((2000, 3000, 3), np.uint8)])
    part = sent[0]["messages"][-1]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_two_leaves_is_a_failure_even_if_model_says_ok(monkeypatch):
    client = fake_openrouter(monkeypatch, '```json\n{"ok": true, "leaves_lifted": 2, "reason": "two"}\n```', [])
    v = check_frame(client, np.zeros((10, 10, 3), np.uint8), "lift")
    assert v.ok is False and v.details["leaves_lifted"] == 2


def test_api_failure_gives_undecided_verdict(monkeypatch):
    def boom(req, timeout):
        raise TimeoutError("slow")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", boom)
    client = OpenRouter()
    client.key = "test"
    assert check_frame(client, np.zeros((10, 10, 3), np.uint8), "holder").ok is None


def test_missing_key_raises():
    client = OpenRouter()
    client.key = ""
    with pytest.raises(LlmError):
        client.ask("hi")
