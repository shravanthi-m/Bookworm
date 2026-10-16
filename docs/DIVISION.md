# Work Division

Three owners, three parts of the codebase. Each person owns their folder and opens PRs into `main`. The **contracts** at the bottom are the agreed interfaces between parts; if you need to change one, tell the others first.

> Replace **Teammate B** and **Teammate C** with real names.

---

## 🟦 Shrav — Vision

**Owns:** `bookworm/vision/`, `tests/test_vision.py`, the `vision` and `ocr` config sections

| # | Task | Where | Done when |
|---|------|-------|-----------|
| V1 | Check the camera geometry from Viam: intrinsics and camera pose in world. Confirm the frame convention | `vision/geometry.py`, `bookworm check`, `bookworm overlay` | the overlay outline sits on the real pages to within about 3 mm |
| V2 | Page extraction: perspective warp from the projected book quad, spine split, holder masking (tune HSV against the printed parts) | `vision/preprocess.py` | clean, upright page images at all three capture poses |
| V3 | Capture quality: settle time after arm moves, sharpness gate, exposure / glare | `vision.settle_s`, `min_sharpness` | no blurry or glare-washed captures in 20 tries |
| V4 | OCR tuning: psm, confidence thresholds, text cleanup for speech | `vision/ocr.py` | ≥ 95 % word accuracy on the demo book |
| V5 | Page-number detection (header/footer) | `TesseractOcr.page_number` | ≥ 90 % hit rate on the demo book |
| V6 | **Corner detection to guide the eraser**: book outline → right-page corner in book mm | `vision/page_detect.py` | the corner lands within 5 mm of a ruler measurement |
| V7 | Flip verification thresholds from real flip / no-flip / double-flip images | `vision/verify.py` | catches every double flip in a 20-flip test |
| V9 | **Depth stream** (the `cam` is RGB-D): measure the page surface height before each flip (replaces the `z_drop_per_flip_mm` guess) and the curl near the spine | `hardware/viam_robot.py` → new `capture_depth()`, `vision/` | the measured page height is within 1 mm of a touch-off |
| V10 | **Pencil/gripper mask** (`calibrate-tool-mask`) and **lighting**: illumination flattening, glare % in the bench, pick a lamp position with the team | `vision/tool_mask.py`, `preprocess.py` | no pencil pixels in page crops; glare < 1 % at the capture poses |
| V8 | Scan quality for archival mode: 300 dpi warp, white balance, optional curl dewarping near the spine (stretch) | `preprocess.py` | pages look like a flatbed scan |

## 🟩 Teammate B — CAD + motion software

**Owns:** CAD files (`cad/`), `bookworm/hardware/flip.py`, `bookworm/hardware/book_frame.py`, the `flip`, `book` and `robot.tool_length_mm` config

| # | Task | Where | Done when |
|---|------|-------|-----------|
| M1 | **Pencil holder** for the gripper, with a compliant / spring-loaded tip | CAD | pencil sits vertical and repeatable; measure `tool_length_mm` |
| M2 | **Book cradle** with corner stops | CAD | book goes back in the same spot to within 2 mm |
| M3 | **Passive page holders** in matte green (see [HARDWARE.md](HARDWARE.md)) | CAD | pages stay flat and a lifted leaf can pass |
| M4 | Book-frame calibration (`calibrate-book`) and checking it against `overlay` with Shrav | `book_frame.py`, CLI | the eraser touches the requested book coordinate to within 3 mm |
| M5 | **Tune the flip waypoints**: press depth, drag length, slip-under, carry arc | `flip.sequence` in YAML, `bookworm flip --times 10` | ≥ 80 % single-leaf success |
| M6 | Stack height compensation, safe travel heights, arm speed limits in the Viam config | `z_drop_per_flip_mm`, Viam app | no page tears across a 20-flip run |

## 🟧 Teammate C — Platform, modes & demo

**Owns:** `bookworm/hardware/viam_robot.py`, `session.py`, `modes/`, `speech/`, `__main__.py`, `tests/test_session.py`, README

| # | Task | Where | Done when |
|---|------|-------|-----------|
| P1 | Viam connection: confirm component names, wrist camera in the frame system, `.env` on every laptop | `viam_robot.py`, `bookworm check` | `check` is clean on every machine |
| P2 | Record the capture poses: `home`, `overview`, `scan_left`, `scan_right` | `bookworm save-pose` | each pose frames its region with margin (check with `overlay`) |
| P3 | **Read-aloud mode**: a child-friendly voice, pacing, pause/resume/stop control, friendly error messages | `modes/read_aloud.py`, `speech/` | reads 5 spreads hands-off |
| P4 | **Scan mode**: searchable PDF (Tesseract PDF output + merge), resume an interrupted scan, manifest | `modes/scan.py` | 20-page scan produces a searchable PDF |
| P7 | Consider the `builtin` motion service (obstacles are already configured) for collision-aware moves to the capture poses | `viam_robot.py` | moves plan around the table and walls |
| P5 | Robustness: reconnect on dropped Wi-Fi, timeouts, Ctrl+C sends the arm home | `viam_robot.py`, `session.py` | pulling Wi-Fi for 5 s doesn't crash the run |
| P6 | Demo: script, backup demo (`--images` with recorded photos), README results, video | README | rehearsal runs end to end |

---

## Contracts between parts

| Contract | Defined in | Between |
|----------|-----------|---------|
| `Page(color, binary)` per side, from `process_shot()` | `vision/preprocess.py` | Vision → Session/Modes |
| `CameraGeometry(K, dist, T_world_cam)`, from `Robot.camera_geometry()` | `hardware/base.py` | Platform → Vision |
| `locate_right_corner()` returns `(x, y)` in book mm, or `None` | `vision/page_detect.py` | Vision → Motion |
| Book frame: origin at the bottom of the spine, x toward the right edge, y up the spine, z = 0 on the right page | `hardware/book_frame.py` | Motion ↔ Vision |
| Flip waypoints in book mm; the pencil tip pose is converted to a flange pose via `tool_length_mm` | `hardware/flip.py` | Motion → Platform |
| Mode handler: `on_spread(i, spread, ocr_results)`, `on_event(e)`, `finish(summary)` | `session.py` | Platform ↔ Modes |
| `FlipOutcome` (ok / no_flip / multi / backward / unknown) | `vision/verify.py` | Vision → Session |

## Workflow
- Branch per task: `shrav/v6-corner-detect`, `b/m5-flip-tuning`, `c/p4-searchable-pdf`.
- `pytest` must pass before merging. Add a test for anything you can test without hardware.
- Rig time is shared: book slots in the team chat. Save rig captures to `runs/` and share good ones as test data.
- Never commit `.env` or Viam keys.
