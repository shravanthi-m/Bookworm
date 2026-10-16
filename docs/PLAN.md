# Bookworm — Hackathon Sprint Plan

Sprints are time-boxed blocks (about 3–4 h) for a roughly 24–36 h hackathon. Each ends with a **demoable checkpoint**. If one slips, cut stretch items. Don't push back the next sprint. Who does what: [DIVISION.md](DIVISION.md). Hardware details: [HARDWARE.md](HARDWARE.md).

Owners: **S** = Shrav (vision), **B** = CAD + motion, **C** = platform + modes.

## Sprint 0 — Setup (≈1 h, everyone)
- [ ] Clone, venv, `pip install -r requirements.txt`, install Tesseract. `pytest` is green.
- [ ] `.env` with the Viam API key on every laptop (**never commit it**).
- [ ] **C:** `python -m bookworm check` lists the arm, gripper and wrist camera, and prints the intrinsics and camera pose.
- [ ] **B:** start the pencil holder and cradle CAD. Print a first pencil holder ASAP.

**Checkpoint:** everyone gets a `runs/check_frame.jpg` from the wrist camera.

## Sprint 1 — Calibrate the world
- [ ] **B:** the pencil goes in its holder (`bookworm pencil`); measure `tool_length_mm`; the book sits in the (temporary) cradle.
- [ ] **B:** `bookworm calibrate-book` (touch 3 points).
- [ ] **C:** save the poses `home`, `overview`, `scan_left`, `scan_right` (`bookworm save-pose`).
- [ ] **S:** `bookworm overlay overview` → the outline matches the pages. If not, debug the camera frame convention or calibration (V1).
- [ ] **S:** collect 30+ captures of real spreads from each pose for offline vision work.

**Checkpoint:** the overlay is accurate, and the eraser goes to a requested book coordinate within 3 mm.

## Sprint 2 — See and read (no flipping)
- [ ] **S:** page warp, spine split, holder masking, OCR tuning, page numbers (V2–V5) on the saved captures.
- [ ] **C:** read-aloud mode and voice on one spread; scan mode writes pages + PDF from `--images`.
- [ ] **B:** first flip attempts: `bookworm flip` with a single press-and-drag. Observe how the leaf buckles.

**Checkpoint:** the robot reads one live spread aloud **and** scans one spread to PNG/PDF.

## Sprint 3 — Flip the page
- [ ] **B:** tune the `flip.sequence` waypoints with `bookworm flip --times 10` and log the success rate (M5).
- [ ] **B:** printed page holders mounted and the cradle finalised (M2, M3).
- [ ] **S:** corner detection feeding the flip (V6); compare it with the calibrated corner.
- [ ] **C:** Ctrl+C sends the arm home safely; reconnect handling (P5).

**Checkpoint:** 8 of 10 single-leaf flips succeed.

## Sprint 4 — Close the loop, both modes
- [ ] **All:** `bookworm read` runs read → flip → verify → read on the rig.
- [ ] **S:** tune flip verification on real flip / no-flip / double-flip captures (V7).
- [ ] **C:** `bookworm scan` over 10 spreads → searchable PDF (P4).
- [ ] **B:** stack-height compensation over a longer run (M6).

**Checkpoint:** reads 5 spreads hands-off, and scans 10 spreads to a PDF.

## Sprint 5 — Demo polish (see [DEMO.md](DEMO.md) for the 2-minute script and time budget)
- [ ] 10-spread rehearsal of each mode; fix the top failure.
- [ ] **C:** backup demo with `--images` and recorded photos; README results; video.
- [ ] **S (stretch):** curl dewarping for scan quality (V8).
- [ ] Stretch: a pause/resume voice command, or a kid-friendly "point at a word" interaction.

**Checkpoint:** demo-ready, with a backup.

---

## Risks & mitigations
| Risk | Mitigation |
|------|------------|
| The eraser tears or lifts several leaves | compliant pencil holder, gentle press depth, stack-height compensation; the verifier announces multi-flips |
| The book shifts between flips | cradle with stops; vision corner refinement; re-run `calibrate-book` if it's bumped |
| Wrong camera-frame convention or calibration | check with `overlay` in Sprint 1 before building on it |
| Glare under venue lights | matte prints, angle capture poses slightly off-normal, diffuse light |
| Page holders hide text | narrow green fingers at the margins, colour masking |
| Venue Wi-Fi drops the Viam connection | run the client on the same LAN as the machine; `--images` backup demo |
| No page numbers (front matter) | the verifier falls back to image difference |
