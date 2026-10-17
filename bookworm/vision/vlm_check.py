"""`bookworm vlm-check`: ask a vision model (via OpenRouter) yes/no questions about a flip.
Standalone like watch-flip: it only looks at a camera and never moves the arm.

    vlm-check lift   --image side.jpg          side camera mid-flip: is exactly one leaf lifted?
    vlm-check holder --live                    overhead after the flip: page flat and in the holder?
    vlm-check lift   --live --every 2          keep checking (camera from llm.cameras: lift -> side webcam)

The fold detector (watch-flip) stays the fast, offline check. This one is slower (~1–3 s per
call) but needs no reference recordings and can say *what* went wrong, so use the two to
cross-check each other. Every checked frame + verdict is saved to runs/vlm-<time>/.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ..llm import LlmError, OpenRouter

CHECKS = {
    "lift": (
        "This photo is from a camera beside a robot that turns book pages with the eraser end of a "
        "pencil. Look at the page edges. Answer about the moment shown:\n"
        "- leaves_lifted: how many separate paper leaves are lifted off the stack (0, 1, 2, or 3 for 3+)\n"
        "- pencil_touching: is the eraser touching the lifted leaf or the page under it\n"
        "- folding: is a lifted leaf creasing, folding over on itself, or crumpling\n"
        "- ok: true only if exactly one leaf is lifted cleanly and nothing is folding"),
    "holder": (
        "This photo shows an open book from above (it may be upside down), just after a robot turned "
        "a page. Dark 3D-printed holders at the book's outer edge keep the turned pages down. Ignore "
        "any pencil or robot parts lying beside the book. Answer:\n"
        "- page_flat: are both visible pages lying flat (no page standing up, curled over, or half-turned)\n"
        "- in_holder: are the page edges next to each holder tucked under/behind it, not sticking "
        "up over it (true/false; null if no holder visible)\n"
        "- folding: is any page creased, dog-eared, or folded over on itself\n"
        "- ok: true only if the pages are flat, the turned page is in the holder, and nothing is folded"),
}
SCHEMA_HINT = ('Reply with JSON only: {"ok": bool, "leaves_lifted": int|null, "pencil_touching": bool|null, '
               '"page_flat": bool|null, "in_holder": bool|null, "folding": bool|null, '
               '"confidence": 0-1, "reason": "one short sentence"}')


@dataclass
class Verdict:
    check: str
    ok: bool | None             # None = could not decide (API error) -> caller falls back
    reason: str
    confidence: float | None = None
    details: dict | None = None
    seconds: float = 0.0


def check_frame(llm: OpenRouter, frame: np.ndarray, check: str) -> Verdict:
    t0 = time.perf_counter()
    try:
        r = llm.ask_json(f"{CHECKS[check]}\n\n{SCHEMA_HINT}", [frame], max_tokens=200)
    except LlmError as e:
        return Verdict(check, None, f"no answer: {e}", seconds=time.perf_counter() - t0)
    # Decide from the individual answers, not the model's own "ok" (it fails pages for
    # irrelevant things, e.g. a pencil lying beside the book).
    if r.get("folding") is True:
        ok = False
    elif check == "lift" and isinstance(r.get("leaves_lifted"), int):
        ok = r["leaves_lifted"] == 1
    elif check == "holder" and r.get("page_flat") is not None:
        ok = r["page_flat"] is True and r.get("in_holder") is not False
    else:
        ok = r.get("ok")
    conf = r.get("confidence")
    return Verdict(check, bool(ok) if ok is not None else None, str(r.get("reason", "")),
                   float(conf) if isinstance(conf, (int, float)) else None,
                   {k: v for k, v in r.items() if k not in ("ok", "reason", "confidence")},
                   time.perf_counter() - t0)


def run(cfg: dict, args) -> None:
    from .watch import _Folder, _Viam, _Webcam

    from ..llm import make_llm

    llm = make_llm(cfg.get("llm"))
    if not llm.ready:
        raise SystemExit("Set OPENROUTER_API_KEY in .env first (openrouter.ai -> Keys).")
    if args.image:
        from .camera import read_image

        frames, src = iter([read_image(args.image)]), None
    else:
        if args.live:
            name = args.camera or (cfg.get("llm") or {}).get("cameras", {}).get(args.check) or cfg["robot"]["camera"]
            c = dict(cfg, robot=dict(cfg["robot"], camera=name))
            print(f"camera: {name}")
            src = _Viam(c)
        elif args.images:
            src = _Folder(args.images)
        else:
            src = _Webcam(args.webcam)
        frames = src.frames()

    out = Path(f"runs/vlm-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.check} check with {llm.model}  -> {out}")
    camera = args.live or args.webcam is not None      # vs. a fixed set of images
    n = 0
    try:
        with open(out / "verdicts.jsonl", "w") as log:
            for frame in frames:
                v = check_frame(llm, frame, args.check)
                cv2.imwrite(str(out / f"{n:04d}.jpg"), frame)
                log.write(json.dumps(dict(asdict(v), frame=f"{n:04d}.jpg")) + "\n")
                log.flush()
                mark = {True: "OK  ", False: "FAIL", None: "??  "}[v.ok]
                conf = f" ({v.confidence:.0%})" if v.confidence is not None else ""
                print(f"[{n:04d}] {mark}{conf} {v.reason}   {v.details or ''}  {v.seconds:.1f}s", flush=True)
                n += 1
                if camera:
                    if not args.every:
                        break                          # one check of the current view
                    time.sleep(max(0.0, args.every - v.seconds))
    except KeyboardInterrupt:
        pass
    finally:
        if src is not None:
            src.close()
