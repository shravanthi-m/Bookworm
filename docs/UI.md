# Running the Bookworm control page

The control page is a local website. It shows the book, has buttons for every action, and can be driven by voice.

## Before you start

| Check | Why |
|---|---|
| `.env` has `VIAM_ADDRESS`, `VIAM_API_KEY_ID`, `VIAM_API_KEY` | connects to the arm and cameras (copy `.env.example`) |
| `.env` has `OPENROUTER_API_KEY=…` (or `OPEN_ROUTER_KEY`) | only for **Summarise** and `vlm-check`; everything else works without it |
| The phone camera is up: on Shrav's Mac, `viam-server -config ~/viam-shrav-mac.json` is running | the overhead iPhone feed (`phone-cam`) only exists while that runs |
| The iPhone is on its mount, unlocked, Center Stage **off** | Continuity Camera; Center Stage switches to the ultra-wide lens |
| You use **Chrome** | voice commands use Chrome's speech recognition |
| The arm is parked (`out_of_the_way`) and someone can reach the e-stop | Flip / Move away / Come back move the real arm |

You can run the page on **any laptop**. It reaches the arm and cameras through Viam, so only the phone itself has to be on Shrav's Mac. On a Mac, OCR uses Apple Vision and speech uses the built-in voice. On Linux, OCR falls back to Tesseract and speech to `espeak-ng`.

## Start it

```bash
./bookworm.sh --config config/phone_mount.yaml ui          # then open http://localhost:8000 in Chrome
```

- Status says "Connecting to camera…" for up to a minute, then **Ready**.
- Stop the server with `Ctrl+C` in its terminal.
- `--port 8001`: use a different port.
- `--quiet`: print instead of speaking.
- `--images DIR`: no robot or camera; it uses a folder of photos, and **Flip** moves to the next photo. Good for trying the page anywhere.

## Buttons

| Button | What happens | Time |
|---|---|---|
| **Next page** (or → key) | turns the page, then refreshes the pages on screen | ~30 s (the flip) |
| **Flip** | runs Midhat's recorded flip (`scripts/flip_path.py`), then parks the arm out of view | ~30 s |
| **Read** | reads a few clean sentences from the current pages aloud | ~3 s, then speaking |
| **Scan** | saves both pages (image + text) and rebuilds the searchable PDF | ~8 s |
| **Summarise** | 2-sentence, child-friendly summary of the pages, spoken (needs the OpenRouter key) | ~5 s |
| **Flip + Read / Flip + Scan / Flip + Summarise** | the flip, then that action | flip + action |
| **Start listening** | turns voice commands on/off (red dot = listening) | |
| **Stop** | stops everything now: the arm (halts where it is), the flip script, and speech mid-sentence | instant |
| **Show camera** | live overhead feed in the corner | |
| **Open the scanned book (PDF)** | appears after the first scan | |

Buttons grey out while something is running. Summarise stays grey until an OpenRouter key is in `.env`.

## Voice commands

1. Click **Start listening** and allow the microphone when Chrome asks.
   - The Mac's built-in mic works.
   - For the iPhone mic, pick it in System Settings → Sound → Input.
2. Start every command with **"Bookworm"**.
3. Bookworm answers out loud ("Okay, reading.") so you know it heard you, then does it.

| Say "Bookworm, …" | Does | Bookworm says |
|---|---|---|
| **stop** / pause / be quiet | Stop button (works even mid-sentence) | (goes quiet) |
| **continue** / keep reading / keep going / resume / go on | carries on reading from the sentence where it was stopped | (just continues) |
| **flip** | Flip | "Turning the page." |
| **next page** / turn the page | Next page | "Turning the page." |
| **read** | Read | "Okay, reading." |
| **scan** | Scan | "Scanning." |
| **summarise** | Summarise | "Let me summarise." |
| **flip and read** / flip and scan / flip and summarise | the combined buttons | "Turning the page, then …" |
| **move away** / go away / back off / park | arm to its park spot, clear of the book | "Moving away." |
| **come back** / get ready | arm to just above the page, ready to flip | "Coming back." |

- **What it heard:** the line under the status shows text live while you talk, then the final phrase. If it stays empty while you speak, the mic isn't reaching Chrome.
- **Pausing for the command:** you can pause after "Bookworm". It shows "Yes?" and waits 6 s for the command.
- **Misheard words that still count:** Chrome often mishears these, so they're accepted too: "Brooklyn" for Bookworm, and "movie" / "move" for move away.
- **Busy:** if Bookworm is busy it says "Hang on, I'm still busy". Say "stop" first.
- **Misheard:** if it hears "Bookworm" followed by something it doesn't know, it says "Sorry, I didn't catch that."
- **Its own voice:** while Bookworm is talking it ignores everything except **stop**, so it can't trigger itself.
- **Logging:** every phrase heard is printed in the server terminal as `[heard] …`, which is useful for adding phrases it often mishears.
- Chrome sends the audio to Google for recognition, so it needs internet.

### Arm safety for "move away" / "come back"

`scripts/arm_goto.py` only makes the single move Midhat's flip already makes, between `out_of_the_way` and the spot above the page (`config/waypoints.yaml`).
- If the arm is anywhere else, for example stopped mid-flip, it **refuses** and says so in the status. Move it with the Viam CONTROL tab.
- To check where the arm is without moving it: `.venv/bin/python scripts/arm_goto.py where`.

## What happens behind each action

```
Read / Summarise:  1 camera frame → find the book → rotate (vision.rotate: 180) → split pages at the fold
                   → Apple Vision OCR → Read: keep clean sentences (~30 words/page) and speak
                                       → Summarise: send the text to OpenRouter (Gemini 2.5 Flash) and speak the reply
Scan:              3 frames → same as above, but words are voted across frames (more accurate)
                   → scans/<time>/pages/*.png + *.txt + book.txt + book.pdf (searchable)
Flip:              ui.flip_command in config/phone_mount.yaml (Midhat's scripts/flip_path.py)
```

Settings live in `config/default.yaml`, overridden by `config/phone_mount.yaml`:
- `modes.read.max_words_per_page`, `modes.read.frames`
- `ocr.frames`, `ocr.combine`
- `llm.model`, `llm.summary_sentences`
- `speech.rate`, `speech.voice` (`say -v '?'` lists voices)

## Vision checks (standalone, never move the arm)

```bash
./bookworm.sh --config config/phone_mount.yaml vlm-check lift   --live           # side webcam: exactly one page lifted?
./bookworm.sh --config config/phone_mount.yaml vlm-check holder --live           # overhead: pages flat, turned page in its holder?
./bookworm.sh --config config/phone_mount.yaml vlm-check lift   --live --every 2 # keep checking every 2 s during a flip
./bookworm.sh --config config/phone_mount.yaml vlm-check holder --image photo.jpg
```

- Each check takes about 2 s and costs about $0.001 (OpenRouter).
- It prints OK / FAIL / ?? with a reason, and saves frames and answers to `runs/vlm-<time>/`.
- Cameras are set in `llm.cameras`: lift uses `webcam`, holder uses `phone-cam`.
- The offline fold detector is `watch-flip` (see the README).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Stuck on "Connecting to camera…" | `viam-server` isn't running on Shrav's Mac, so `phone-cam` is missing. Start it. The terminal prints `[ui] camera not ready: …` every 10 s with the reason. |
| `could not connect to …viam.cloud` | Machine offline or Wi-Fi blip. Check the dot at app.viam.com and retry. |
| Pages upside down / text is garbage | Phone was remounted: set `vision.rotate` (0 / 90 / 180 / 270) in `config/phone_mount.yaml`. |
| Read says nothing | Check the Mac's sound output (System Settings → Sound → Output). The text it tried to read is shown under the status. |
| Voice does nothing | Use Chrome. Allow the mic (icon in the address bar). Watch the "heard" line. Say "Bookworm" first. |
| "Microphone blocked" | Click the mic icon in Chrome's address bar → Allow, then reload. |
| Summarise is grey | No OpenRouter key in `.env`. Add it and restart the page server. |
| Everything is slow | Close Zoom, extra Chrome tabs, and other big apps. On an 8 GB Mac the Viam connection and OCR slow down badly when memory is full. |
| `ModuleNotFoundError` / numpy errors | Run through `./bookworm.sh` (not plain `python`), which uses `.venv` and clears a conda/ROS `PYTHONPATH`. |
