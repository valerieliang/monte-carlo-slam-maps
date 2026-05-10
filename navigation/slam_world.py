"""
navigation/slam_world.py
------------------------
Builds a lightweight collision-query object from the current SLAM map so
the controller can avoid walls it has actually *learned about*, not just
the ground-truth geometry.

This addresses the case where the proximity fan misses a narrow corner:
by querying both the ground-truth world AND the SLAM-derived walls, the
robot gets a denser wall representation and is less likely to slip through.

Usage
-----
    slam_w = SLAMWorld.from_state(slam_state)
    # use exactly like env.World for ray_intersect:
    result = slam_w.ray_intersect(origin, angle, max_range)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from slam.state import SLAMState


class SLAMWorld:
    """
    Minimal world-like object built from line features in the SLAM state.

    Each line feature whose segment endpoints are known contributes one
    wall segment.  Corner features contribute a tiny virtual square obstacle
    so the proximity fan detects them even at angles between rays.

    Parameters
    ----------
    segments : list of (p0, p1) pairs  — wall segment endpoints (2,) arrays
    corners  : list of (x, y) positions — corner obstacles
    corner_r : radius of virtual corner obstacle (m)
    """

    def __init__(self,
                 segments: list,
                 corners:  list,
                 corner_r: float = 0.15):
        self._segments  = segments   # list of (p0_array, p1_array)
        self._corners   = corners    # list of (2,) arrays
        self._corner_r  = corner_r

    @classmethod
    def from_state(cls, state: 'SLAMState',
                   corner_r: float = 0.15) -> 'SLAMWorld':
        """Build a SLAMWorld from the current EKF state."""
        segments = []
        corners  = []

        for feat in state.features:
            mean = state.feature_mean(feat.idx)

            if feat.kind == 'line':
                if feat.seg_p0 is not None and feat.seg_p1 is not None:
                    segments.append((feat.seg_p0.copy(), feat.seg_p1.copy()))

            else:  # corner
                corners.append(mean.copy())

        return cls(segments, corners, corner_r)

    def ray_intersect(self,
                      origin:    np.ndarray,
                      angle:     float,
                      max_range: float = 30.0
                      ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Find the closest intersection with any SLAM-mapped wall or corner.
        Same API as env.World.ray_intersect.
        """
        origin = np.asarray(origin, dtype=float)
        ray_d  = np.array([np.cos(angle), np.sin(angle)])

        best_t  = max_range
        best_pt = None

        # -- Line segments ----------------------------------------------------
        for p0, p1 in self._segments:
            d  = p1 - p0
            b  = p0 - origin
            A  = np.column_stack([ray_d, -d])
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) < 1e-12:
                continue
            ts  = np.linalg.solve(A, b)
            t, s = ts
            if t > 1e-6 and t < best_t and 0.0 <= s <= 1.0:
                best_t  = t
                best_pt = origin + t * ray_d

        # -- Corner obstacles (modelled as tiny squares) ----------------------
        for cpos in self._corners:
            # Ray-sphere intersection (approximate corner as circle)
            oc  = origin - cpos
            b2  = float(np.dot(ray_d, oc))
            c   = float(np.dot(oc, oc)) - self._corner_r ** 2
            disc = b2 * b2 - c
            if disc < 0:
                continue
            t = -b2 - np.sqrt(disc)
            if t > 1e-6 and t < best_t:
                best_t  = t
                best_pt = origin + t * ray_d

        if best_pt is not None:
            return float(best_t), best_pt
        return None