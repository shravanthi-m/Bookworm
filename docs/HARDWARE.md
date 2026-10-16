# Hardware

## Fixed setup (already configured in Viam)

| Part | Viam component (config key) | Used for |
|------|-----------------------------|----------|
| uFactory xArm 850 (`viam:ufactory:xArm850`) | `arm` (`robot.arm`) | moving the camera to capture poses and moving the pencil to flip pages |
| Gripper | `gripper` (`robot.gripper`) | holding the pencil, eraser end down |
| Intel RealSense wrist camera (`viam:camera:realsense`) | `cam` (`robot.camera`) | colour 1280×720 + aligned depth; intrinsics (fx≈fy≈906 px, ~70° HFOV) and frame (parent = `arm`) map pixels to world mm. We request the colour stream only (31 ms vs 485 ms with depth) |

Everything is reached through Viam API calls, using the API key in `.env`. Nothing else is wired up: **no servos, no microcontroller**. If the component names in the Viam app are different, set them in `config/local.yaml`. Then run `python -m bookworm check` to confirm the connection, camera intrinsics and camera pose.

Arm, gripper and camera come from the shared fragment `xarm850-realsense-original` (camera attributes: `width_px: 1280, height_px: 720, sensors: [color, depth], align_color_depth: true`). **Don't edit the shared fragment** — it's used by other machines. To change camera settings for this machine only, use a *fragment mod* on this machine in the Viam app.

Also on the machine: the `builtin` motion service with obstacle geometries (`table`, `ceiling`, `wall-front`, `wall-side`), a `gripper` frame 105 mm out from the flange, a `pencil` generic component and a `home-pose` switch. *Verified 2026-10-16 with `bookworm check`.*

> **Assumption to verify early:** the wrist camera's frame follows the pinhole convention (x right, y down, z out of the lens). `python -m bookworm overlay` draws the calibrated book outline on a live frame. If the outline sits on the pages, the geometry is right.

## How a page gets turned

The gripper holds a pencil with the eraser pointing down. The motion is a list of waypoints in **book coordinates** (`flip.sequence` in `config/default.yaml`):

1. Hover just inside the bottom-outer corner of the right page (vision can refine where that corner is).
2. Press the eraser onto the top leaf and drag it toward the spine. Friction buckles the leaf, so its corner lifts.
3. Slip the eraser under the raised corner, lift it, carry it over the spine, and lay it on the left.

Book coordinates come from `calibrate-book`: you touch the eraser to the bottom of the spine, the top of the spine, and the right page's bottom-outer corner. After that, the book frame stays valid as long as the book doesn't move, which is why the cradle matters. The right stack gets thinner as you go, so each flip lowers the touch height by `book.z_drop_per_flip_mm`.

## 3D-printed parts (Mechanical owner)

| Part | Requirements | Why software cares |
|------|--------------|--------------------|
| **Pencil holder** for the gripper fingers | holds the pencil **vertical and repeatable**, with the eraser sticking out a fixed length. Ideally **spring-loaded / compliant along the pencil axis**, about 5–10 mm of travel | the arm has no force sensing; a compliant tip forgives small z errors and changing stack height instead of tearing pages. Measure flange → eraser tip and put it in `robot.tool_length_mm` |
| **Book cradle** | fixed to the table, with corner stops so the book sits in the same place every time; holds the spine flat | the book-frame calibration and capture poses are only valid while the book doesn't move |
| **Page holders** (passive, no motors) | keep both stacks flat while still letting one leaf be lifted out and laid down. Options: thin flexible fingers at the bottom edge only, weighted hinged fingers, or a clear acrylic bar across the bottom margin | flat pages = better OCR; a lifted leaf must get past them |
| Colour of anything visible to the camera | **bright matte green** (or tell vision the HSV range); matte, not glossy | vision masks this colour out automatically (`vision.clamp_color_hsv`); gloss causes glare |

**Open design question for the page holders:** a passive holder has to stop the stacks from springing up but still let a leaf through. If the arm needs to nudge a holder during the flip, add those waypoints to `flip.sequence`. No code change is needed.

## Option: phone as a fixed overhead camera (Mac sub-part)

The RealSense has **fixed focus and 1280×720**. Small book text sits right at its limit: close up it's out of focus, further away the letters are too few pixels. A phone gives **autofocus + higher resolution**, and when mounted above the book it **never sees the pencil**.

**1. Test quality first, without Viam** (2 minutes). Use iPhone Continuity Camera: iOS 16+, macOS 13+, same Apple ID, Wi-Fi and Bluetooth on, phone locked and held still near the Mac.
```bash
python -m bookworm webcams                      # find the phone's index; thumbnails in runs/webcams/
python -m bookworm vision --webcam N            # compare words/conf with the wrist camera
```
macOS asks for camera permission the first time (System Settings → Privacy & Security → Camera → allow your terminal / VS Code).

**Status (2026-10-16):** sub-part `shrav-mac` with camera `phone-cam` is created on the machine, and `viam-server` is installed on Shrav's Mac (`viam-server -config ~/viam-shrav-mac.json`; the file holds the part secret and is not in the repo). The feed is available to the team while that runs. Final mount settings: [`config/phone_mount.yaml`](../config/phone_mount.yaml) (live feed is upright: `rotate: 0`; measured book corners). **The Viam `webcam` model defaults to 640×480** — `phone-cam` has `width_px: 1920, height_px: 1080` set on the sub-part; without it the text is unreadable. Keep Center Stage / Portrait off (Control Center → Video Effects) or the view goes ultra-wide. Installing `viam-server` with Homebrew failed with "already locked …/openssl@3" until `ffmpeg` was upgraded first: `brew upgrade ffmpeg`, then `HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1 brew install viamrobotics/brews/viam-server`.

**2. Put it in Viam as a sub-part of the machine (already done, for reference):**
1. app.viam.com → the machine → **CONFIGURE** → **+** → **Sub-part**, named for example `shrav-mac`.
2. Install and start `viam-server` on the Mac using the setup instructions the app shows for the sub-part (Homebrew `viam-server`, run with the downloaded config). Allow it camera access.
3. On the sub-part, add a **camera** component, model **`webcam`**. Pick the iPhone as the video device and name it `phone-cam`.
4. `python -m bookworm check` should now list `phone-cam`.

**3. Point Bookworm at it** in `config/local.yaml`:
```yaml
robot: {camera: phone-cam, camera_calibrated: false}
vision: {rotate: 270}              # whatever `vision --webcam N` suggests for your mount
modes:
  scan: {shots: [{pose: overview, region: spread}]}   # fixed camera: one shot per spread, arm parked
```
Then:
- `python -m bookworm calibrate-view`: click the spread's four corners once. The phone and book must not move after this.
- `python -m bookworm save-pose overview`: an arm pose **out of the phone's view** (the arm parks there for each capture).

**Mounting:** directly above the book, about 30–40 cm up, outside the arm's reach (tripod or desk-lamp arm with a phone clamp). Ask Teammate C to add it as an obstacle in Viam so motion planning avoids it.

**Trade-off:** with an uncalibrated fixed camera, vision can't aim the eraser at a detected corner, so the flip uses the calibrated corner from `calibrate-book`. Everything else works the same: OCR, flip checking, scan storage, reading.
