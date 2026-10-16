# Bookworm — Hackathon Sprint Plan

Sprints are time-boxed blocks (about 3–4 h each) for a roughly 24–36 h hackathon. Squeeze or stretch them to fit the real schedule. Each sprint ends with a **demoable checkpoint**. If a checkpoint slips, cut scope from the "stretch" items. Don't push back the next sprint.

## Roles

| Role | Owns | Lives in |
|------|------|----------|
| **Vision** | camera → page images → text → flip verification | `bookworm/vision/` |
| **Software / Viam** | Viam machine config, servo wiring, control loop, TTS, CLI | `bookworm/hardware/`, `controller.py`, `speech/` |
| **Mechanical (3D print)** | finger clamps, eraser arm, camera mount | CAD files; contract in [HARDWARE_INTERFACE.md](HARDWARE_INTERFACE.md) |

Vision and software can work in parallel from the start. `--images` and `--mock` let you run the whole pipeline without the rig.

---

## Sprint 0 — Setup (≈1 h, everyone)
- [ ] Clone the repo, create a venv, `pip install -r requirements.txt`, install Tesseract.
- [ ] Create the machine in the Viam app; install `viam-server` on the Pi.
- [ ] Copy `.env.example` → `.env` and paste in the API key. **Never commit `.env`.**
- [ ] Agree on Viam component names (`cam`, `clamp-left`, `clamp-right`, `eraser-lift`, `flip-arm`).
- [ ] `pytest` passes on every laptop.

**Checkpoint:** `python -m bookworm check` lists every component and saves `check_frame.jpg`.

## Sprint 1 — See the book (Vision)
- [ ] Fix the camera mount (temporary is fine). Take 20+ photos of real spreads under the venue's lighting. Save them to a shared folder.
- [ ] `python -m bookworm calibrate` → book corners land in `config/local.yaml`.
- [ ] `python -m bookworm ocr photo.jpg` → check `runs/debug/left.png` and `right.png` are cleanly split and binarised.
- [ ] Tune `min_sharpness`, the adaptive threshold, and the gutter search band.

**In parallel (Software):** wire the servos in Viam and check each one moves from the Viam app's CONTROL tab. Then run `python -m bookworm flip --mock`.

**Checkpoint:** readable OCR text from a real photo of the target book.

## Sprint 2 — Read aloud end to end, no flipping
- [ ] TTS works on the demo machine (`speech.engine: pyttsx3`).
- [ ] Mask the clamps: measure the finger rectangles (`clamp_masks`) **or** use the marker colour (`clamp_color_hsv`).
- [ ] Check page-number extraction on 20 photos. Record the hit rate in the README's "Results" section.
- [ ] `python -m bookworm run --images photos/ --mock` reads the whole folder aloud.

**Checkpoint:** the robot reads one spread aloud from the live Viam camera.

## Sprint 3 — Flip the page (Software + Mechanical)
- [ ] Mount the printed clamps and the eraser arm. Tune `hardware.angles` until the clamps hold and release cleanly.
- [ ] Tune `flip_sequence` in YAML with `python -m bookworm flip --times 10`. Log the success rate.
- [ ] Find what goes wrong: double pages, page doesn't lift, page falls back. Adjust drag angle, waits, and eraser pressure (`lift_down`).

**Checkpoint:** 8 of 10 flips succeed on their own.

## Sprint 4 — Close the loop
- [ ] Full `python -m bookworm run` on the rig: read → flip → verify → read.
- [ ] Check verification on real failures: a stuck page → retry; a double flip → announced.
- [ ] Tune `verify.change_threshold` using saved `runs/` images (flip vs. no-flip pairs).
- [ ] End-of-book detection on blank pages.

**Checkpoint:** reads 5 consecutive spreads hands-off.

## Sprint 5 — Demo polish
- [ ] Run a 10-spread rehearsal and fix the top failure.
- [ ] Prepare a backup demo: `--images` mode with recorded photos in case the rig fails on stage.
- [ ] Update the README with results, photos, and a video.
- [ ] Stretch: a "Start/Stop" button, a Viam data-capture dashboard, or speaking while the next flip runs (thread the TTS).

**Checkpoint:** a demo-ready run plus the backup.

---

## Risks & mitigations
| Risk | Mitigation |
|------|------------|
| Glare or uneven lighting ruins OCR | Diffuse light (paper over an LED), angle the camera off-axis, CLAHE + adaptive threshold already in the pipeline |
| Clamps hide text | Narrow fingers near the page edge, a marker colour for auto-masking, and `clamp_masks` |
| Eraser lifts 2+ pages | Lighter `lift_down` pressure, a shorter drag; the verifier announces multi-flips |
| Page curl near the spine distorts OCR | Hold the book open flat (clamps), keep the camera high, try `ocr.psm: 6` |
| Venue Wi-Fi drops the Viam connection | Run the client on the same LAN as the Pi; keep the `--images` backup demo |
| No page numbers (front matter) | The verifier falls back to image difference (`UNKNOWN` = assume OK) |
