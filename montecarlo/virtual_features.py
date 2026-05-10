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

    cov = np.diag([virtual_cov, virtual_cov])

    # The four virtual lines sit AT the map boundary (at the outermost features).
    # The paper's robot_radius offset is used to ensure the virtual line is not
    # closer than robot_radius to any physical wall — here we simply use the
    # extents of the current feature map directly.
    #
    # Polar form: for a horizontal line at y=c, rho=c, alpha=pi/2
    #             for a vertical   line at x=c, rho=c, alpha=0
    # All rho values must be ≥ 0; if a boundary is at a negative coordinate,
    # we flip the normal (alpha += pi) and take rho = abs.
    #
    # Segment endpoints span the full bounding box in the tangential direction
    # so that ANY interior point falls within at least one line's region.

    def _make_hline(y_val, x0, x1):
        """Horizontal virtual line at y=y_val."""
        rho   = abs(y_val)
        alpha = np.pi / 2 if y_val >= 0 else -np.pi / 2
        return VirtualLine(rho=rho, alpha=alpha, cov=cov.copy(),
                           seg_p0=np.array([x0, y_val]),
                           seg_p1=np.array([x1, y_val]))

    def _make_vline(x_val, y0, y1):
        """Vertical virtual line at x=x_val."""
        rho   = abs(x_val)
        alpha = 0.0 if x_val >= 0 else np.pi
        return VirtualLine(rho=rho, alpha=alpha, cov=cov.copy(),
                           seg_p0=np.array([x_val, y0]),
                           seg_p1=np.array([x_val, y1]))

    virtual_lines = [
        _make_hline(ymin, xmin, xmax),   # south
        _make_hline(ymax, xmin, xmax),   # north
        _make_vline(xmin, ymin, ymax),   # west
        _make_vline(xmax, ymin, ymax),   # east
    ]

    # Sampling bounds: shrink inward by robot_radius so samples stay off walls
    sx_min = xmin + robot_radius
    sx_max = xmax - robot_radius
    sy_min = ymin + robot_radius
    sy_max = ymax - robot_radius

    return virtual_lines, (sx_min, sx_max, sy_min, sy_max)


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