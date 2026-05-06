"""
montecarlo/virtual_features.py
-------------------------------
Construct the four virtual line features that circumscribe the current SLAM
map (Section III of the paper).

The virtual lines form an axis-aligned rectangle just outside the outermost
mapped features.  Each line:
  - Is represented in the same polar (rho, alpha) format as real line features
  - Has a fixed covariance diag(0.32, 0.32) as specified in the paper
  - Ensures MC points near unexplored frontiers score near 0.5

The offset from the nearest feature to the virtual line equals the robot
radius (0.25 m), matching the paper.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from slam.state import SLAMState


VIRTUAL_COV_DIAG = 0.32   # paper Section III — fixed covariance for virtual lines


@dataclass
class VirtualLine:
    """
    A virtual boundary line in polar form.

    rho   : perpendicular distance from world origin (m)
    alpha : angle of line's outward normal (rad)
    cov   : (2,2) fixed covariance matrix
    seg_p0, seg_p1 : (2,) endpoints of the bounding segment (for scoring region)
    """
    rho:   float
    alpha: float
    cov:   np.ndarray   # (2,2)
    seg_p0: np.ndarray  # (2,)
    seg_p1: np.ndarray  # (2,)


def build_virtual_features(state: SLAMState,
                            robot_radius: float = 0.25,
                            virtual_cov:  float = VIRTUAL_COV_DIAG
                            ) -> Tuple[List[VirtualLine],
                                       Tuple[float, float, float, float]]:
    """
    Compute the four virtual boundary lines from the current SLAM map.

    Parameters
    ----------
    state        : current SLAMState — used to find map extents
    robot_radius : offset from outermost feature to virtual line (m)
    virtual_cov  : diagonal covariance value for virtual lines

    Returns
    -------
    (virtual_lines, bounds)
      virtual_lines : list of 4 VirtualLine objects
      bounds        : (xmin, xmax, ymin, ymax) of the bounding rectangle
    """
    if state.n_features == 0:
        # No map yet — return a tiny box around the robot
        xv, yv, _ = state.pose
        r = robot_radius + 1.0
        xmin, xmax = xv - r, xv + r
        ymin, ymax = yv - r, yv + r
    else:
        xmin, xmax, ymin, ymax = _map_extents(state)

    # Expand by robot_radius on each side (paper: virtual line offset = robot radius)
    xmin -= robot_radius
    xmax += robot_radius
    ymin -= robot_radius
    ymax += robot_radius

    cov = np.diag([virtual_cov, virtual_cov])

    # Four axis-aligned boundary lines, normals pointing inward
    # South wall:  y = ymin,  normal points up   (alpha = pi/2)
    # North wall:  y = ymax,  normal points down  (alpha = -pi/2, rho = -ymax → normalise)
    # West  wall:  x = xmin,  normal points right (alpha = 0)
    # East  wall:  x = xmax,  normal points left  (alpha = pi, rho = -xmax → normalise)

    virtual_lines = [
        # South: rho = ymin (measured along y), alpha = pi/2
        VirtualLine(
            rho    = ymin,
            alpha  = np.pi / 2,
            cov    = cov.copy(),
            seg_p0 = np.array([xmin, ymin]),
            seg_p1 = np.array([xmax, ymin]),
        ),
        # North: rho = ymax, alpha = -pi/2 → normalise: rho = ymax, alpha = pi/2 flipped
        # Use rho = ymax, alpha = np.pi/2 with sign flip convention:
        # rho >= 0 convention: rho = ymax, alpha = pi/2 (same as south but positive rho)
        # The north wall normal points in -y direction from outside → alpha = -pi/2
        # but rho must be positive: rho = ymax, alpha = pi/2 works when ymax > 0
        VirtualLine(
            rho    = ymax,
            alpha  = np.pi / 2,
            cov    = cov.copy(),
            seg_p0 = np.array([xmin, ymax]),
            seg_p1 = np.array([xmax, ymax]),
        ),
        # West: rho = xmin, alpha = 0
        VirtualLine(
            rho    = xmin,
            alpha  = 0.0,
            cov    = cov.copy(),
            seg_p0 = np.array([xmin, ymin]),
            seg_p1 = np.array([xmin, ymax]),
        ),
        # East: rho = xmax, alpha = 0
        VirtualLine(
            rho    = xmax,
            alpha  = 0.0,
            cov    = cov.copy(),
            seg_p0 = np.array([xmax, ymin]),
            seg_p1 = np.array([xmax, ymax]),
        ),
    ]

    return virtual_lines, (xmin, xmax, ymin, ymax)


def _map_extents(state: SLAMState) -> Tuple[float, float, float, float]:
    """
    Return (xmin, xmax, ymin, ymax) across all features in the SLAM map.

    For corner features: use the (x, y) mean directly.
    For line features: use the foot-of-perpendicular point as a proxy,
    plus the stored segment endpoints if available.
    """
    xs, ys = [], []

    # Always include vehicle pose
    xv, yv, _ = state.pose
    xs.append(xv); ys.append(yv)

    for feat in state.features:
        mean = state.feature_mean(feat.idx)
        if feat.kind == 'corner':
            xs.append(mean[0]); ys.append(mean[1])
        else:
            # Line: include foot-of-perpendicular
            rho, alpha = mean
            xs.append(rho * np.cos(alpha))
            ys.append(rho * np.sin(alpha))
            # Also include stored segment endpoints
            if feat.seg_p0 is not None:
                xs.append(feat.seg_p0[0]); ys.append(feat.seg_p0[1])
            if feat.seg_p1 is not None:
                xs.append(feat.seg_p1[0]); ys.append(feat.seg_p1[1])

    return min(xs), max(xs), min(ys), max(ys)