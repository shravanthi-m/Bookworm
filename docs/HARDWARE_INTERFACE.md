# Hardware ↔ Software Contract

What the software expects from the mechanism. The mechanical teammate can design freely within these limits. If anything here has to change, update this file and `config/default.yaml` together.

## Actuators (all Viam `servo` components, 0–180°)

| Viam name | Role | Software calls it with |
|-----------|------|------------------------|
| `clamp-left` | finger clamp holding the left page stack | `clamp_engaged` / `clamp_released` angles |
| `clamp-right` | finger clamp holding the right page stack | same (can be mirrored, set per side) |
| `eraser-lift` | lowers the pen eraser onto / off the page | `lift_down` / `lift_up` angles |
| `flip-arm` | rotates the eraser across the book | 0° = over the right page's outer edge, 180° = over the left page |

All angles live in `config/default.yaml → hardware.angles`, and the motion lives in `flip_sequence`. **Re-tuning a print never needs a code change.** Edit the YAML and run `python -m bookworm flip --times 5`.

## Default flip motion (edit in YAML)
1. Arm parks over the right page's outer edge. The right clamp releases.
2. The eraser lowers and the arm drags a few degrees inward. Friction buckles and lifts the top page.
3. The eraser lifts. The left clamp releases. The arm sweeps to 180°, carrying the page over.
4. The left clamp re-engages on the turned page. The arm returns. The right clamp re-engages.

## Requests for the 3D-printed parts (help vision a lot)
- **Clamp fingers in a solid, bright colour that won't appear in the book.** Default is green (HSV 40–85). The software then masks them out automatically. If you use another colour, tell the vision owner the HSV range.
- **Matte finish** on the fingers and arm. Glossy plastic causes glare.
- **Keep the fingers narrow and near the outer edges or corners.** Every millimetre over the text is lost words.
- **Rigid camera mount** above the book centre, ideally 35–50 cm high, lens parallel to the page. Once mounted, don't move it (or re-run `calibrate`).
- **Fixed book position:** a cradle or corner stops, so the calibrated corners stay valid.
- Nothing the arm moves should stay in the camera view when it's at 0° (parked).
