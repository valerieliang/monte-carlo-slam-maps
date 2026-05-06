"""
montecarlo/uncertainty_map.py
------------------------------
Orchestrates the full MC uncertainty map pipeline (Section III of paper):

  1. Build virtual boundary features from current SLAM map
  2. Sample N uniform points inside the bounding box (Eqs. 6-7)
  3. Filter to navigable points (Section III-B)
  4. Score each navigable point via sum-of-Gaussians (Eq. 10)
  5. Return scored points for navigation and rendering

Points scoring near 0.5 are "uncertain" — good navigation targets.
Points near 0   are clearly free space.
Points near 1   are clearly occupied (near a well-known feature).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

from montecarlo.virtual_features import build_virtual_features
from montecarlo.sampler           import sample_points
from montecarlo.navigability      import navigable_mask
from montecarlo.probability       import score_points_vectorised

if TYPE_CHECKING:
    from slam.state import SLAMState


@dataclass
class UncertaintyMap:
    """
    Result of one MC uncertainty map computation.

    Attributes
    ----------
    points   : (M, 2) navigable sample points
    scores   : (M,)   uncertainty score in [0, 1] for each point
    bounds   : (xmin, xmax, ymin, ymax) of the bounding box used
    n_total  : total points sampled before navigability filter
    """
    points:  np.ndarray
    scores:  np.ndarray
    bounds:  Tuple[float, float, float, float]
    n_total: int

    @property
    def uncertain_mask(self) -> np.ndarray:
        """Boolean mask: True for points in the uncertainty band [lo, hi]."""
        return self._band_mask

    def set_band(self, lo: float, hi: float) -> None:
        self._band_mask = (self.scores >= lo) & (self.scores <= hi)
        self._lo = lo
        self._hi = hi

    @property
    def uncertain_points(self) -> np.ndarray:
        """Points scoring in the uncertainty band."""
        return self.points[self._band_mask]

    @property
    def uncertain_scores(self) -> np.ndarray:
        return self.scores[self._band_mask]


def build_uncertainty_map(state:        'SLAMState',
                           world,
                           robot_pos:   np.ndarray,
                           n_samples:   int   = 300,
                           robot_radius: float = 0.25,
                           virtual_cov: float  = 0.32,
                           uncertainty_lo: float = 0.40,
                           uncertainty_hi: float = 0.60,
                           rng: np.random.Generator | None = None
                           ) -> UncertaintyMap:
    """
    Run the full MC uncertainty map pipeline.

    Parameters
    ----------
    state          : current SLAMState
    world          : env.World (for navigability ray-casting)
    robot_pos      : (2,) current robot position
    n_samples      : number of MC points to draw
    robot_radius   : robot radius for virtual feature offset
    virtual_cov    : diagonal covariance for virtual boundary lines
    uncertainty_lo : lower threshold for "uncertain" band
    uncertainty_hi : upper threshold for "uncertain" band
    rng            : numpy Generator

    Returns
    -------
    UncertaintyMap
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Virtual features + bounding box
    virtual_lines, bounds = build_virtual_features(
        state, robot_radius=robot_radius, virtual_cov=virtual_cov)

    # 2. Sample
    candidates = sample_points(bounds, n_samples, rng=rng)

    # 3. Navigability filter
    nav = navigable_mask(candidates, robot_pos, world)
    navigable_pts = candidates[nav]

    # 4. Score
    if len(navigable_pts) == 0:
        result = UncertaintyMap(
            points  = np.empty((0, 2)),
            scores  = np.array([]),
            bounds  = bounds,
            n_total = n_samples,
        )
        result.set_band(uncertainty_lo, uncertainty_hi)
        return result

    scores = score_points_vectorised(navigable_pts, state, virtual_lines)

    result = UncertaintyMap(
        points  = navigable_pts,
        scores  = scores,
        bounds  = bounds,
        n_total = n_samples,
    )
    result.set_band(uncertainty_lo, uncertainty_hi)
    return result