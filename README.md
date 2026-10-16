# 📖 Bookworm

**A robot that reads physical books aloud and turns the pages itself.**

An overhead camera photographs the open book. Computer vision splits and cleans each page, Tesseract OCR reads it, and text-to-speech reads it aloud. Then a pen-eraser arm turns the page, using friction to lift it, while 3D-printed finger clamps hold the book open. The robot checks that exactly one page turned before reading on. Built on [Viam](https://www.viam.com/).

```
 ┌────────┐   ┌──────────────────────── vision ─────────────────────────┐   ┌──────┐
 │ camera │──▶│ warp to book → mask clamps → split at spine → binarise  │──▶│ OCR  │──▶ 🔊 speak
 └────────┘   └─────────────────────────────────────────────────────────┘   └──────┘
      ▲                                                                         │
      │        ┌──────── verify flip ────────┐       ┌──── flip (Viam servos) ──┴─┐
      └────────│ page # +2? image changed?   │◀──────│ clamps · eraser lift · arm │
               └─────────────────────────────┘       └────────────────────────────┘
```

## Repo layout

```
bookworm/
  vision/        camera.py · preprocess.py · ocr.py · verify.py   ← the vision pipeline
  hardware/      base.py (interface + flip runner) · viam_link.py · mock.py
  speech/        text-to-speech (pyttsx3, or print for quiet dev)
  controller.py  main loop: capture → OCR → speak → flip → verify
  rig.py         builds camera/servos from config
  __main__.py    CLI
config/default.yaml   every tunable: servo angles, flip motion, thresholds
docs/PLAN.md          sprint plan, roles, risks
docs/HARDWARE_INTERFACE.md   contract with the mechanical/3D-print side
tests/                runs without hardware, Viam, or a camera
```

## Setup

```bash
git clone https://github.com/shravanthi-m/Bookworm.git && cd Bookworm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract          # macOS   |  sudo apt install tesseract-ocr espeak  (Pi/Linux)
cp .env.example .env            # then fill in your Viam keys (see below)
pytest                          # should pass with no hardware attached
```

### Viam credentials

1. In [app.viam.com](https://app.viam.com), open the machine → **CONNECT** tab → **API keys**.
2. Put the values in `.env`:
   ```
   VIAM_ADDRESS=your-machine-main.xxxxxxxx.viam.cloud
   VIAM_API_KEY_ID=...
   VIAM_API_KEY=...
   ```
3. `.env` is git-ignored. **Never commit keys.** If a key leaks, delete it in the Viam app and make a new one.

### Viam machine config

Add these components on the machine's **CONFIGURE** tab. The names must match `config/default.yaml → hardware.components` and `camera.viam_name`:

| Name | Type | Notes |
|------|------|-------|
| `cam` | camera (`webcam`) | overhead, pointing straight down |
| `clamp-left`, `clamp-right` | servo | finger clamps |
| `eraser-lift` | servo | raises/lowers the pen eraser |
| `flip-arm` | servo | swings the eraser across the book |

Then run `python -m bookworm check`. It connects, lists the components, flags any missing names, and saves a test frame.

## Usage

```bash
python -m bookworm check                         # test the Viam connection + camera
python -m bookworm calibrate                     # click the 4 book corners → config/local.yaml
python -m bookworm ocr photo.jpg                 # vision on one image; debug images in runs/debug/
python -m bookworm flip --times 10               # tune the flip motion on the rig
python -m bookworm run                           # read the book! (Ctrl+C to stop)

# No hardware? Develop against photos with fake servos:
python -m bookworm run --images photos/ --quiet
```

Every run saves each captured spread and its OCR text to `runs/<timestamp>/`. Use these to tune the vision.

## How the vision works

| Step | Code | Why |
|------|------|-----|
| Perspective warp to the calibrated corners | `preprocess.warp_to_book` | flattens a slightly angled camera view |
| Sharpness gate (variance of the Laplacian) | `preprocess.sharpness` | re-captures if the page is still moving after a flip |
| Clamp masking by colour and by rectangles | `mask_color`, `mask_rects` | clamp fingers would otherwise come out as junk characters |
| Split at the spine (darkest column near the centre) | `find_gutter` | the gutter shadow is a strong, cheap cue |
| CLAHE + adaptive threshold | `prepare_for_ocr` | handles uneven lighting and curl shadows |
| Tesseract with word-confidence filtering | `ocr.TesseractOcr` | low-confidence fragments get dropped instead of spoken |
| Text cleanup | `ocr.clean_text` | re-joins hyphenated words and drops page-number lines |
| Flip verification | `verify.classify_flip` | page number +2 = OK, +0 = retry, >+2 = skipped pages; image difference as a fallback |

## Tuning cheat-sheet (`config/local.yaml`)

| Symptom | Knob |
|---------|------|
| OCR spells out the clamp as junk | `vision.clamp_masks`, `vision.clamp_color_hsv` |
| Rejects sharp images as blurry | lower `vision.min_sharpness` |
| Says "can't read" too often | lower `ocr.min_page_conf`, try `ocr.psm: 6` |
| Retries a flip that worked | lower `verify.change_threshold` |
| Lifts two pages at once | less pressure (`hardware.angles.lift_down`), a shorter drag in `flip_sequence` |

## Team & plan

See **[docs/PLAN.md](docs/PLAN.md)** for the sprint plan and **[docs/HARDWARE_INTERFACE.md](docs/HARDWARE_INTERFACE.md)** for the mechanical contract.

| Area | Owner |
|------|-------|
| Vision | _TBD_ |
| Software / Viam integration | _TBD_ |
| Mechanical / 3D printing | _TBD_ |

## Results

_Fill in during Sprints 2–5: OCR accuracy, page-number hit rate, flip success rate, demo video._
