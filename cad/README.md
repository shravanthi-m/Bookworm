# CAD

3D-printed parts. Requirements are in [../docs/HARDWARE.md](../docs/HARDWARE.md#3d-printed-parts-mechanical-owner).

| Part | File | Status |
|------|------|--------|
| Pencil holder (compliant tip) | | not started |
| Book cradle | | not started |
| Page rack (left outer edge, matte green) | | printing: C-wall with 50 mm × 1 mm shelves at ~20 and ~41 mm. The flip lays each leaf short, then drags the eraser outward so the flattening leaf's edge slides under a shelf (`flip.sequence`, `surface: left`) |

Commit the source CAD (`.f3d` / `.step`) and the printable `.stl` for each part. After printing the pencil holder, measure flange → eraser tip and update `robot.tool_length_mm`.
