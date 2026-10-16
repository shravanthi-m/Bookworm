# 📖 Bookworm

**A robot arm that turns real book pages with a pencil eraser. It can read the book aloud or scan it without damaging it.**

A uFactory arm, controlled through [Viam](https://www.viam.com/), holds a pencil in its gripper. Its calibrated wrist camera photographs each spread. Computer vision straightens and cleans the pages, and Tesseract OCR reads them. Then the arm presses the eraser onto the page, buckles the leaf, and carries it over the spine. Before going on, vision checks that exactly one page turned.

## Two modes

| Mode | Command | For | Output |
|------|---------|-----|--------|
| **Read aloud** | `python -m bookworm read` | reading to children, or anyone who can't hold or see a book | speech, page by page |
| **Scan** | `python -m bookworm scan --out scans/mybook` | non-destructive digitisation: no cutting the spine, no flatbed | `pages/0001.png…`, `book.txt`, `book.pdf`, `manifest.json` |

Both share one loop:

```
 move to capture pose ─▶ wrist camera ─▶ warp to page (camera geometry) ─▶ mask holders ─▶ OCR
          ▲                                                                                 │
          │                                                                  read aloud  OR  save scan
          │                                                                                 │
   verify flip ◀── capture next spread ◀── pencil-eraser flip (book-frame waypoints) ◀──────┘
 (page # +2? image changed?)
```

## Hardware

Fixed setup, already configured in Viam: **uFactory arm + gripper + calibrated wrist camera**, plus a pencil (eraser down) in the gripper. 3D-printed parts: a compliant pencil holder, a book cradle, and passive page holders in matte green. There are no servos or microcontrollers. Details: [docs/HARDWARE.md](docs/HARDWARE.md).

## Repo layout

```
bookworm/
  vision/       geometry.py (pixel↔world) · preprocess.py · page_detect.py · ocr.py · verify.py
  hardware/     base.py (Robot interface) · viam_robot.py · book_frame.py · flip.py · mock.py
  modes/        read_aloud.py · scan.py
  speech/       text-to-speech
  session.py    shared capture → OCR → handler → flip → verify loop
  __main__.py   CLI
config/default.yaml   every tunable: component names, poses, flip waypoints, thresholds
docs/PLAN.md          sprint plan · docs/DIVISION.md who owns what · docs/HARDWARE.md setup + 3D prints
tests/                run without the arm, Viam, or a camera
```

## Setup

```bash
git clone https://github.com/shravanthi-m/Bookworm.git && cd Bookworm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract          # macOS   |  sudo apt install tesseract-ocr espeak  (Linux)
cp .env.example .env            # then paste in the Viam API key (below)
pytest                          # should pass with no hardware
```

### Viam API key

1. In [app.viam.com](https://app.viam.com), open the machine → **CONNECT** → **API keys**.
2. Fill in `.env`: `VIAM_ADDRESS`, `VIAM_API_KEY_ID`, `VIAM_API_KEY`.
3. `.env` is git-ignored. **Never commit keys.** If one leaks, delete it in the Viam app and make a new one.

Component names match the configured machine: `arm`, `gripper`, `cam`. If yours are different, override them in `config/local.yaml`:

```yaml
robot: {arm: my-xarm, gripper: my-gripper, camera: my-cam}
```

## First-time calibration (with the book in its cradle)

```bash
python -m bookworm check                 # connection, components, intrinsics, camera pose
python -m bookworm pencil                # open gripper → insert pencil → grab
python -m bookworm calibrate-book        # touch eraser to spine bottom, spine top, right page corner
python -m bookworm save-pose home        # jog the arm (Viam CONTROL tab / freedrive) to each pose first
python -m bookworm save-pose overview    #   whole spread in view   (read mode)
python -m bookworm save-pose scan_left   #   left page fills frame  (scan mode)
python -m bookworm save-pose scan_right  #   right page fills frame (scan mode)
python -m bookworm overlay overview      # runs/overlay.jpg — outline must sit on the pages
python -m bookworm flip --times 5        # tune flip.sequence in config until flips are reliable
```

Calibration is written to `config/local.yaml` (git-ignored, specific to your rig).

## Development without the arm

```bash
python -m bookworm read --images photos/ --quiet     # replay spread photos, print instead of speak
python -m bookworm scan --images photos/ --out scans/test
python -m bookworm ocr photo.jpg                     # one image → text + runs/debug/*.png
python -m bookworm flip --mock                       # print the flip waypoints
```

Every run saves its captures and OCR text to `runs/<timestamp>-<mode>/`. Use these as vision test data.

## Where vision runs

Vision is **plain Python on the laptop** (the Viam *client*), not a Viam module or a separate service. Viam provides the **inputs**: colour frames from the wrist camera (`cam`, an RGB-D camera), its intrinsics, and the camera pose from the Viam frame system (`cam` is a child of `arm`). Everything after that (OpenCV preprocessing, Tesseract OCR, flip checks) runs locally.

- **No training, no API keys for models.** The pipeline is classical OpenCV plus Tesseract, which comes with its own pretrained English model. Printed book text is the easy case for OCR.
- **No OpenRouter or LLM in the loop.** If Tesseract struggles (decorative fonts, children's books with text over pictures), a vision-LLM OCR backend can be added behind the same `ocr.read()` interface without touching anything else. It costs network latency and money per page.
- **Could become a Viam vision-service module later** if we wanted it on the robot computer, but for a hackathon a local client is faster to iterate on.

## Testing the vision pipeline on its own (no arm motion)

```bash
python -m bookworm vision --images photos/                  # any photos of open books (phone is fine)
python -m bookworm vision --live                            # one frame from the wrist camera, where the arm is now
python -m bookworm vision --live --frames 10 --manual       # turn pages by hand between frames → tests flip checking
```

Each run writes `runs/vision-<timestamp>/`, containing:
- `*.panel.jpg` debug images: raw frame with the detected outline, then the colour pages, then the binarised pages.
- OCR text per page.
- `results.json`: sharpness, OCR confidence, page numbers, flip outcome for each consecutive pair, and timing.

Ground truth is optional. Put `photo.left.txt` / `photo.right.txt` next to `photo.jpg` and the report adds word accuracy.

## How the vision works

| Step | Code | Why |
|------|------|-----|
| Pixel ↔ world from the wrist camera's intrinsics + Viam frame system | `vision/geometry.py` | the page outline is computed, not guessed, at any arm pose |
| Perspective warp to the projected page quad at a set px/mm | `preprocess.process_shot` | upright pages; about 300 dpi in scan mode |
| Sharpness gate after arm motion | `preprocess.sharpness` | waits out vibration |
| Holder masking by colour | `mask_color` | green holders would otherwise show up as junk text |
| Spine split at the darkest column | `find_gutter` | for single-shot spreads |
| CLAHE + adaptive threshold → Tesseract with confidence filtering | `prepare_for_ocr`, `ocr.py` | uneven light; low-confidence fragments get dropped |
| Right-page corner from the book outline, in book mm | `page_detect.locate_right_corner` | aims the eraser even if the book has shifted slightly |
| Flip verification: page number +2 = OK, +0 = retry, >+2 = skipped pages, image difference as a fallback | `verify.classify_flip` | catches stuck and double flips |

## Team

| Area | Owner | Tasks |
|------|-------|-------|
| Vision | Shrav | [DIVISION.md § Shrav](docs/DIVISION.md#-shrav--vision) |
| CAD + motion software | _Teammate B_ | [DIVISION.md § Teammate B](docs/DIVISION.md#-teammate-b--cad--motion-software) |
| Platform, modes & demo | _Teammate C_ | [DIVISION.md § Teammate C](docs/DIVISION.md#-teammate-c--platform-modes--demo) |

Sprint plan: [docs/PLAN.md](docs/PLAN.md).

## Results

_Fill in during Sprints 2–5: OCR accuracy, page-number hit rate, flip success rate, scan samples, demo video._
