"""Step drop test: lower the tool tip a fixed step at a time, read where it really is after each step, and treat a
step that does not go down as contact.

The arm has no contact sensing of its own here (collision_sensitivity 0), but a hard surface stops it: the commanded
2 mm goes unfinished and the pose read back (forward kinematics of the real joint angles) is higher than asked.
This module is only the decision; the caller moves the arm, reads the pose and the joint load, and backs up. It can be
switched off (``enabled=False``, config ``gui.depth.step_drop.enabled``): ``check`` then never reports a stall and the
person's "It's touching" press is the only way a search ends.

Two rules (``mode``):
  fixed       a step that went down less than ``min_drop_mm`` stalled.
  regression  a line is fitted through the steps so far (measured vs commanded tip z, and each joint's load vs step).
              A step stalls when it lands higher than that line predicts by more than ``k`` standard deviations of
              the earlier steps (and by at least ``pos_floor_mm``), or when a joint load leaves its trend by more than
              ``load_k`` standard deviations (before there is a trend: a load change of ``load_k`` x ``load_floor``
              from the steps so far). Catches a partly landed step and the rising push before any stall.
              The fixed rule stays on as a backstop.
Both rules are evaluated on every step and written to ``trace`` whichever mode is active, so a chart can show what the
other rule would have done.

A paper page gives way instead of stopping the tip, so this finds hard surfaces (the table) reliably and the page
only if it is stiff; the person's press remains the fallback for both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stall:
    step: int                 # 0-based index of the step (after the baseline) that did not go down
    asked_mm: float           # how far that step was supposed to go down
    dropped_mm: float         # how far it did
    tip_z: float              # where the tip really is (the surface, to within the arm's compliance)
    previous_tip_z: float     # where it was before: the position to go back to
    reason: str = "position"  # which rule fired: position | regression | load

    def text(self) -> str:
        return (f"step {self.step + 1}: the tip went down {self.dropped_mm:.1f} of {self.asked_mm:.1f} mm "
                f"(now at z {self.tip_z:.1f}; {self.reason})")


class StepDrop:
    """Feed it, after every step, the measured tip z (the first call is the baseline at the start height), optionally
    the commanded tip z and the per-joint load."""

    def __init__(self, enabled: bool = True, step_mm: float = 2.0, min_drop_mm: float | None = None, mode: str = "fixed",
                 k: float = 5.0, pos_floor_mm: float = 0.3, load_k: float = 8.0, min_points: int = 6, use_load: bool = True,
                 load_joints=(1, 2), load_floor: float = 0.3):
        if step_mm <= 0:
            raise ValueError("step_mm must be positive")
        if mode not in ("fixed", "regression"):
            raise ValueError("mode must be 'fixed' or 'regression'")
        self.enabled = bool(enabled)
        self.step_mm = float(step_mm)
        self.min_drop_mm = float(step_mm / 2 if min_drop_mm is None else min_drop_mm)
        if not 0 < self.min_drop_mm <= self.step_mm:
            raise ValueError("min_drop_mm must be between 0 and the step size")
        self.mode, self.k, self.pos_floor_mm = mode, float(k), float(pos_floor_mm)
        self.load_k, self.min_points, self.use_load = float(load_k), max(3, int(min_points)), bool(use_load)
        self.load_joints = None if load_joints is None else [int(j) for j in load_joints]   # 0-based; None = all joints
        self.load_floor = float(load_floor)       # the load's own noise floor, in its units (Nm on the xArm)
        self.prev: float | None = None
        self.n = 0
        self.history: list[float] = []
        self.trace: list[dict] = []
        self._x: list[float] = []                 # commanded tip z of the steps that landed
        self._y: list[float] = []                 # measured tip z of those
        self._loads: list[list[float]] = []       # joint loads of those (same rows)

    # ---- regression pieces ---------------------------------------------------------------------------------
    @staticmethod
    def _fit(x, y, x0, floor):
        """Line through (x, y) and what it says about a new point at x0: (prediction, prediction std).
        The spread is the unbiased one (n - 2) widened by the usual prediction-interval term, so a short history
        does not look more certain than it is."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        sxx = float(((x - x.mean()) ** 2).sum())
        if sxx < 1e-9:
            return float(y.mean()), max(float(y.std()), floor)
        b, a = np.polyfit(x, y, 1)
        ssr = float(((y - (a + b * x)) ** 2).sum())
        sd = max((ssr / max(n - 2, 1)) ** 0.5, floor)
        return float(a + b * x0), sd * (1 + 1 / n + (x0 - x.mean()) ** 2 / sxx) ** 0.5

    def predict(self, commanded_tip_z: float) -> float | None:
        """Where the tip should read if the next step lands, from the steps so far; None until there are enough."""
        if len(self._x) < self.min_points:
            return None
        return self._fit(self._x, self._y, float(commanded_tip_z), 0.05)[0]

    def load_score(self, load) -> float | None:
        """Largest standardised deviation of a watched joint's load from its trend over the steps so far. Only the
        joints that carry a downward push are watched (on the xArm6: joints 2 and 3); the wrist joints' readings
        jump by ~2 Nm in free air and would raise false alarms. Also used after backing off a contact: still above
        ``load_k`` there means the arm is still pushing on something.

        Before there are ``min_points`` steps (no trend yet) the score is the change from the last step and from the
        mean so far, in units of ``load_floor``: a contact in the first few steps is caught too, before it can become
        part of the trend (free-air steps change by ~0.05 Nm; the real hit changed by 11 Nm)."""
        if load is None or not self.use_load or not self._loads:
            return None
        L = np.asarray(self._loads, float)
        if L.shape[1] != len(load):
            return None
        joints = range(L.shape[1]) if self.load_joints is None else [j for j in self.load_joints if j < L.shape[1]]
        worst = 0.0
        if len(L) < self.min_points:
            for j in joints:
                d = max(abs(float(load[j]) - L[-1, j]), abs(float(load[j]) - float(L[:, j].mean())))
                worst = max(worst, d / self.load_floor)
            return float(worst)
        idx = np.arange(len(L), dtype=float)
        for j in joints:
            floor = max(self.load_floor, 1e-3 * (np.abs(L[:, j]).max() + 1e-6))   # a perfectly smooth trend still has some noise
            pred, sd = self._fit(idx, L[:, j], float(len(L)), floor)
            worst = max(worst, abs(float(load[j]) - pred) / sd)
        return float(worst)

    # ---- the test ------------------------------------------------------------------------------------------
    def check(self, measured_tip_z: float, commanded_tip_z: float | None = None, load=None) -> Stall | None:
        """Record one step. Returns a Stall if this step did not go down (under the active mode), else None."""
        z = float(measured_tip_z)
        i, self.n = self.n, self.n + 1
        self.history.append(z)
        cmd = float(commanded_tip_z) if commanded_tip_z is not None else (z if self.prev is None else self._x[-1] - self.step_mm)
        load = None if load is None else [float(v) for v in load]
        entry = {"i": i, "cmd": cmd, "meas": z, "load": load, "pred": None, "resid": None, "band": None,
                 "pos_z": None, "load_z": None, "flags": {"fixed": False, "pos": False, "load": False}}
        self.trace.append(entry)
        if self.prev is None:                                          # baseline: the start height
            self.prev = z
            self._x.append(cmd), self._y.append(z)
            if load is not None:
                self._loads.append(load)
            return None
        dropped = self.prev - z
        before, self.prev = self.prev, z

        fixed = bool(dropped < self.min_drop_mm)
        pos = False
        if len(self._x) >= self.min_points:
            pred, sd = self._fit(self._x, self._y, cmd, 0.05)
            resid = z - pred                                           # positive: landed higher than the trend says
            band = max(self.k * sd, self.pos_floor_mm)
            entry.update(pred=pred, resid=resid, band=band, pos_z=resid / sd)
            pos = bool(resid > band)
        score = self.load_score(load)
        lflag = bool(score is not None and score > self.load_k)
        entry.update(load_z=score, flags={"fixed": fixed, "pos": pos, "load": lflag})

        if self.mode == "regression":
            hit = "regression" if pos else "load" if lflag else "position" if fixed else None
        else:
            hit = "position" if fixed else None
        if hit is None or not self.enabled:
            self._x.append(cmd), self._y.append(z)                     # a step that landed joins the fit
            if load is not None:
                self._loads.append(load)
            return None
        return Stall(step=i - 1, asked_mm=self.step_mm, dropped_mm=dropped, tip_z=z, previous_tip_z=before, reason=hit)
