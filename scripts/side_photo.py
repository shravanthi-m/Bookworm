"""Take a photo with the side webcam: py scripts/side_photo.py [--index 1] [--label touch]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookworm.hardware.sidecam import take_photo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=1)
ap.add_argument("--label", default="")
a = ap.parse_args()
_, path = take_photo(a.index, label=a.label)
print(path)
