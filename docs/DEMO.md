# 2-Minute Demo Plan

## Where the time goes

Every `read` / `scan` run prints a per-stage timing table (move, settle, capture, preprocess, OCR, page number, flip). **Fill in the "measured" column from real runs.** The estimates below are only for planning.

| Stage (per spread) | Estimate | Measured | Notes |
|--------------------|---------:|---------:|-------|
| Move to capture pose + settle | 1.5–2.5 s | | skip the move if the arm is already there |
| Capture (Viam, colour frame) | 0.3–1 s | | slower over venue Wi-Fi; use wired / same LAN |
| Preprocess (warp, masks, lighting) | < 0.1 s | | |
| OCR, both pages in parallel | 0.3–1 s | | |
| Page-number OCR | 0.1–0.4 s | | |
| **Flip** (about 10 waypoints) | **8–15 s** | | the big one; fewer or faster waypoints help most |
| **Speaking one spread** | **text-dependent** | | 170 wpm ≈ 2.8 words/s |

**The key trick is to speak while flipping.** Speech runs in the background, so the arm turns page *n + 1* while page *n* is still being read. A spread's latency is then roughly `max(speech, flip + capture + OCR)` instead of their sum.

| Book type | Words per spread | Speech | What the audience sees |
|-----------|-----------------:|-------:|------------------------|
| Novel | 500+ | 3+ min | one spread, then the demo is over ❌ |
| Chapter book | 150 | ~55 s | 2 spreads |
| **Picture book** | **20–60** | **7–20 s** | **4–5 spreads; flipping is fully hidden behind speech** ✅ |

## Recommended demo: about 2:00

| Time | What happens |
|-----:|--------------|
| 0:00 | Pitch, one line each: accessible reading for kids and anyone who can't hold a book, plus non-destructive scanning. Book is already in the cradle, pencil in the gripper, arm at `overview`. |
| 0:10 | `python -m bookworm read`. The first spread is captured and OCR'd (about 2–3 s), then it starts reading. |
| 0:15–1:30 | 4–5 spreads. Each flip happens *while* it reads. A flip check is visible (a double flip gets announced). |
| 1:30 | Ctrl+C, or end of book: "The end." |
| 1:35 | **Scan mode:** show `book.pdf` and `book.txt` from a scan recorded earlier (or scan one spread live if time allows). |
| 1:55 | Close. |

## Choosing the book
- A children's picture book: **large, plain font**, short text, **matte paper** (glossy pages = glare), and a flexible binding that lies flat.
- Text not printed over busy pictures. That's where OCR struggles most.
- Rehearse with **the exact copy** used on stage. Run `python -m bookworm vision --live --manual` over every spread you plan to show.

## Lighting & camera (biggest quality wins, no code)
- Bring a **desk lamp or LED panel, diffused** (paper over it), shining from the side at a low angle. Avoid overhead spotlights straight onto the page: they cause glare. The bench reports `glare %` per frame.
- Wrap the pencil in **green tape**. The existing colour mask then removes it, in addition to `calibrate-tool-mask`.
- Re-run `calibrate-tool-mask` if the pencil is re-seated in the gripper.

## Robustness on stage
- Run the client on a wired connection or the same LAN as the robot. Close other heavy apps.
- **Backup demo:** `python -m bookworm read --images rehearsal_frames/`, using frames recorded at rehearsal. It runs without the arm or network and says the same words.
- Pre-flight: `check` → `overlay` → `flip --times 3` → `vision --live` on the first spread.
