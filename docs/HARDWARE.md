# Hardware

## Fixed setup (already configured in Viam)

| Part | Viam component (config key) | Used for |
|------|-----------------------------|----------|
| uFactory arm | `arm` (`robot.arm`) | moving the camera to capture poses and moving the pencil to flip pages |
| Gripper | `gripper` (`robot.gripper`) | holding the pencil, eraser end down |
| Calibrated wrist camera | `wrist-cam` (`robot.camera`) | page images; its intrinsics and frame (parent = arm) map pixels to world mm |

Everything is reached through Viam API calls, using the API key in `.env`. Nothing else is wired up: **no servos, no microcontroller**. If the component names in the Viam app are different, set them in `config/local.yaml`. Then run `python -m bookworm check` to confirm the connection, camera intrinsics and camera pose.

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
