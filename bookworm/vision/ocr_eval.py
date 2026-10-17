"""OCR test set: score methods against the true text, and multi-frame word voting.

Layout (data/ocr_eval/, git-ignored — book pages are copyrighted):
    spread_01/frame_1.jpg … frame_5.jpg      live-feed frames  (bookworm capture-set)
    spread_01/truth.left.txt, truth.right.txt the correct text (draft from refs/, then fixed by hand)
    refs/IMG_*.HEIC                           one sharp phone photo per spread, in spread order

    ./bookworm.sh --config config/phone_mount.yaml ocr-eval --make-truth   # drafts from refs/
    ./bookworm.sh --config config/phone_mount.yaml ocr-eval                # score every method
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from pathlib import Path

from ..session import best_of_frames, clean_words
from .camera import read_image
from .ocr import OcrResult, ocr_image
from .preprocess import process_shot

WORD = re.compile(r"[\w’']+")


def words(text: str) -> list[str]:
    """Lowercase words without punctuation — what accuracy is measured on."""
    return [w.replace("’", "'") for w in WORD.findall(text.lower())]


def word_accuracy(truth: str, hyp: str) -> float:
    t, h = words(truth), words(hyp)
    if not t:
        return 1.0 if not h else 0.0
    sm = difflib.SequenceMatcher(a=t, b=h, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return max(0.0, 1 - errors / len(t))


def confusions(truth: str, hyp: str) -> Counter:
    """(true word, OCR output) pairs where OCR got a word wrong."""
    t, h = words(truth), words(hyp)
    out: Counter = Counter()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=t, b=h, autojunk=False).get_opcodes():
        if op == "replace":
            out[(" ".join(t[i1:i2]), " ".join(h[j1:j2]))] += 1
        elif op == "delete":
            out[(" ".join(t[i1:i2]), "∅")] += 1
    return out


def vote(texts: list[str], backbone: int = 0) -> str:
    """Word-level majority vote across OCR outputs of the same page.

    Every other hypothesis is aligned to the backbone; each backbone word takes the most
    common reading among the words aligned to it (ties keep the backbone). The backbone's
    layout (line breaks, punctuation) is kept, only word tokens are swapped.
    """
    base = texts[backbone]
    base_tokens = list(WORD.finditer(base))
    base_words = [m.group().lower() for m in base_tokens]
    ballots: list[Counter] = [Counter([m.group()]) for m in base_tokens]
    for k, other in enumerate(texts):
        if k == backbone:
            continue
        ow = [m.group() for m in WORD.finditer(other)]
        sm = difflib.SequenceMatcher(a=base_words, b=[w.lower() for w in ow], autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ("equal", "replace") and (i2 - i1) == (j2 - j1):
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    ballots[i][ow[j]] += 1
    out, pos = [], 0
    for m, ballot in zip(base_tokens, ballots):
        top, n = ballot.most_common(1)[0]
        winner = m.group() if ballot[m.group()] == n else top
        out.append(base[pos:m.start()] + winner)
        pos = m.end()
    return "".join(out) + base[pos:]


def read_frames(frames, vcfg, ocr) -> dict[str, list[OcrResult]]:
    """OCR every frame: {side: [OcrResult per frame]}."""
    per_side: dict[str, list[OcrResult]] = {}
    for f in frames:
        for side, page in process_shot(f, "spread", vcfg).items():
            per_side.setdefault(side, []).append(ocr.read(ocr_image(ocr, page)))
    return per_side


METHODS = {
    "single frame": lambda rs: rs[0].text,
    "best of 3": lambda rs: max(rs[:3], key=lambda r: (clean_words(r), r.word_count)).text,
    "vote 3": lambda rs: vote([r.text for r in rs[:3]], _best(rs[:3])),
    "vote 5": lambda rs: vote([r.text for r in rs], _best(rs)),
}


def _best(rs) -> int:
    return max(range(len(rs)), key=lambda i: (clean_words(rs[i]), rs[i].word_count))


def spreads(root: Path) -> list[Path]:
    return sorted(d for d in root.glob("spread_*") if d.is_dir())


def make_truth(root: Path, vcfg: dict, ocr) -> None:
    """Draft truth.left/right.txt per spread from the reference photos in refs/ (in order)."""
    refs = sorted((root / "refs").glob("*"))
    refs = [p for p in refs if p.suffix.lower() in (".heic", ".heif", ".jpg", ".jpeg", ".png")]
    ss = spreads(root)
    if len(refs) != len(ss):
        print(f"!! {len(refs)} photos in {root}/refs for {len(ss)} spreads — pairing in order anyway")
    for d, ref in zip(ss, refs):
        pages = process_shot(read_image(ref), "spread", vcfg)
        for side, page in pages.items():
            path = d / f"truth.{side}.txt"
            if path.exists():
                print(f"  {path} exists, kept (delete it to redraft)")
                continue
            path.write_text(ocr.read(ocr_image(ocr, page)).text + "\n")
        print(f"  {d.name} <- {ref.name}")
    print("Drafts written. Skim each truth.*.txt against the photo and fix mistakes, then run ocr-eval.")


def evaluate(root: Path, vcfg: dict, ocr, show_confusions: int = 15) -> dict:
    totals: dict[str, list[float]] = {m: [] for m in METHODS}
    conf: Counter = Counter()
    for d in spreads(root):
        frames = [read_image(p) for p in sorted(d.glob("frame_*.jpg"))]
        truths = {s: (d / f"truth.{s}.txt") for s in ("left", "right")}
        if not frames or not all(p.exists() for p in truths.values()):
            print(f"{d.name}: skipped (needs frames + truth.left/right.txt)")
            continue
        per_side = read_frames(frames, vcfg, ocr)
        row = []
        for m, fn in METHODS.items():
            accs = []
            for side, results in per_side.items():
                truth = truths[side].read_text()
                hyp = fn(results)
                accs.append(word_accuracy(truth, hyp))
                if m == "vote 5":
                    conf += confusions(truth, hyp)
            totals[m] += accs
            row.append(f"{m} {sum(accs) / len(accs):.1%}")
        print(f"{d.name}: " + "  ".join(row))
    summary = {m: sum(v) / len(v) for m, v in totals.items() if v}
    if summary:
        print("\nWORD ACCURACY (mean over pages)")
        for m, a in summary.items():
            print(f"  {m:13s} {a:.1%}")
        print(f"\nMost common errors (vote 5), true -> OCR:")
        for (t, h), n in conf.most_common(show_confusions):
            print(f"  {n:3d}x  {t!r} -> {h!r}")
    return summary
