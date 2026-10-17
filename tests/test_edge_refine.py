import cv2
import numpy as np

from bookworm.vision.edge_refine import refine_edge, to_world
from bookworm.hardware import CameraGeometry


def scene(angle_deg=-8.0):
    """Tan table, a pale rotated page with a dark text block, one clean free edge along the top."""
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:] = cv2.cvtColor(np.uint8([[[12, 90, 190]]]), cv2.COLOR_HSV2BGR)[0, 0]
    page = cv2.cvtColor(np.uint8([[[100, 12, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    quad = np.array([[400, 200], [1000, 200], [1000, 700], [400, 700]], np.float32)
    R = cv2.getRotationMatrix2D((700, 450), angle_deg, 1.0)
    quad = (np.hstack([quad, np.ones((4, 1))]) @ R.T).astype(np.int32)
    cv2.fillConvexPoly(img, quad, page)
    cv2.putText(img, "TEXT TEXT TEXT", (520, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 30), 5)
    return img, quad[0].astype(float), quad[1].astype(float)


def perp(p, a, b):
    u = (b - a) / np.linalg.norm(b - a)
    d = np.asarray(p) - a
    return abs(d[0] * u[1] - d[1] * u[0])


def test_snaps_off_hint_to_edge():
    img, a, b = scene()
    mid = (a + b) / 2
    r = refine_edge(img, mid + [30, 22])           # hint 22px below the edge, inside the page
    assert r is not None and r.score > 0.9
    assert perp(r.grasp, a, b) < 3
    assert perp(r.edge[0], a, b) < 3 and perp(r.edge[1], a, b) < 3
    assert min(np.linalg.norm(np.array(r.corner) - a), np.linalg.norm(np.array(r.corner) - b)) < 8


def test_far_hint_finds_nothing():
    img, a, b = scene()
    assert refine_edge(img, (60, 60)) is None


def test_world_lift_overhead_camera():
    img, a, b = scene(0.0)
    r = refine_edge(img, (700, 215))
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1.0]])
    T = np.eye(4); T[:3, :3] = np.diag([1, -1, -1]); T[:3, 3] = [0, 0, 400]   # looking straight down from 400 mm
    w = to_world(r, CameraGeometry(K, None, T), plane_z=0.0)
    gx, gy, gz = w["grasp"]
    assert abs(gz) < 1e-6
    assert abs(gx - (r.grasp[0] - 640) * 0.4) < 0.1 and abs(gy + (r.grasp[1] - 360) * 0.4) < 0.1
