"""
montecarlo/uncertainty_map.py
------------------------------
MC uncertainty map pipeline (Section III of paper).

Scoring model
-------------
Each MC point receives an uncertainty score in [0, 1]:

  score(p) = max over k of:  w_k * spatial_kernel(p, pos_k, sigma_k)

where:
  w_k    = exp(-obs_count_k / tau)
    -- freshly seen feature (obs=1):  w ≈ exp(-1/tau) ≈ high
    -- well-observed feature:         w ≈ 0             → suppressed
  
  spatial_kernel = exp(-dist(p, pos_k)^2 / (2 * sigma^2))
    -- FIXED width sigma (not EKF covariance)
    -- ensures a well-mapped wall still has a spatial footprint

Taking the max rather than sum avoids double-counting when many
features cluster near the same wall section.

Score interpretation:
  ~0   = open floor, well away from any feature (free space)
  ~0.5 = near a freshly-seen / uncertain feature (explore here)
  ~1   = right on top of a new feature (uncertain boundary)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

from montecarlo.virtual_features import build_virtual_features
from montecarlo.sampler           import sample_points
from montecarlo.navigability      import navigable_mask

if TYPE_CHECKING:
    from slam.state import SLAMState


# ---------------------------------------------------------------------------
# UncertaintyMap result container
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyMap:
    """
    Result of one MC uncertainty map computation.

    points   : (M, 2) navigable sample points
    scores   : (M,)   uncertainty score in [0, 1]
    bounds   : (xmin, xmax, ymin, ymax) sampling bounding box
    n_total  : total points sampled before navigability filter
    """
    points:  np.ndarray
    scores:  np.ndarray
    bounds:  Tuple[float, float, float, float]
    n_total: int

    @property
    def uncertain_mask(self) -> np.ndarray:
        return self._band_mask

    def set_band(self, lo: float, hi: float) -> None:
        self._band_mask = (self.scores >= lo) & (self.scores <= hi)

    @property
    def uncertain_points(self) -> np.ndarray:
        return self.points[self._band_mask]

    @property
    def uncertain_scores(self) -> np.ndarray:
        return self.scores[self._band_mask]


# ---------------------------------------------------------------------------
# Feature position helper
# ---------------------------------------------------------------------------

def _feature_anchors(feat, state: 'SLAMState',
                      spacing: float = 2.0) -> np.ndarray:
    """
    Return one or more XY anchor positions for a feature's spatial kernel.

    Corners: single point at the corner position.

    Lines: a series of points sampled along the segment at `spacing` intervals.
    Using multiple anchors ensures the full wall length influences nearby
    floor points — a single midpoint only covers the centre of long walls.
    """
    mean = state.feature_mean(feat.idx)

    if feat.kind == 'corner':
        return mean.reshape(1, 2)

    # Line feature — sample anchors along segment
    if feat.seg_p0 is not None and feat.seg_p1 is not None:
        p0, p1 = feat.seg_p0, feat.seg_p1
        length = np.linalg.norm(p1 - p0)
        if length < 1e-6:
            return (0.5 * (p0 + p1)).reshape(1, 2)
        n = max(1, int(np.ceil(length / spacing)))
        ts = np.linspace(0.0, 1.0, n + 1)
        return np.array([p0 + t * (p1 - p0) for t in ts])

    # Fallback: foot-of-perpendicular from origin
    rho, alpha = mean
    pt = np.array([rho * np.cos(alpha), rho * np.sin(alpha)])
    return pt.reshape(1, 2)


# keep the old name as a compat alias returning the midpoint
def _feature_xy(feat, state: 'SLAMState') -> np.ndarray:
    """Return the midpoint XY position of a feature (legacy, use _feature_anchors)."""
    mean = state.feature_mean(feat.idx)
    if feat.kind == 'corner':
        return mean
    if feat.seg_p0 is not None and feat.seg_p1 is not None:
        return 0.5 * (feat.seg_p0 + feat.seg_p1)
    rho, alpha = mean
    return np.array([rho * np.cos(alpha), rho * np.sin(alpha)])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_uncertainty(points:       np.ndarray,
                       state:         'SLAMState',
                       spatial_sigma: float = 2.5,
                       ) -> np.ndarray:
    """
    Score each point by proximity to freshly-seen (uncertain) features.

    Parameters
    ----------
    points        : (N, 2)
    state         : SLAMState
    spatial_sigma : fixed spatial reach of each feature's influence (metres).
                    Points within ~sigma metres of a new feature get high scores.
                    Typical room width ~3m → sigma=1.5 covers half the corridor.
    obs_tau       : decay constant for obs_count weighting.
                    w = exp(-obs_count / tau).
                    tau=8 → feature seen 8 times has w=exp(-1)≈0.37,
                            feature seen 24 times has w=exp(-3)≈0.05.

    Returns
    -------
    scores : (N,) in [0, 1].
             High near features that haven't been well-observed yet.
             Low in open space far from any feature, or near confident features.
    """
    N = len(points)
    if N == 0 or state.n_features == 0:
        return np.zeros(N)

    scores = np.zeros(N)
    two_s2 = 2.0 * spatial_sigma ** 2

    for feat in state.features:
        obs_count = max(feat.obs_count, 1)

        # Harmonic decay: w = 1/(1+obs) so fresh features score high,
        # well-mapped ones fade toward zero.
        w = 1.0 / (1.0 + obs_count)
        if w < 0.01:
            continue   # fully mapped, skip

        # Anchor points along the feature (multiple for long walls)
        anchors = _feature_anchors(feat, state, spacing=spatial_sigma)

        # For each anchor, compute Gaussian kernel and take max over anchors
        feat_contrib = np.zeros(N)
        for anchor in anchors:
            dists_sq = np.sum((points - anchor) ** 2, axis=1)
            kernel   = np.exp(-dists_sq / two_s2)
            feat_contrib = np.maximum(feat_contrib, kernel)

        # Take max across features so dense clusters don't double-count
        scores = np.maximum(scores, w * feat_contrib)

    # Normalise: w_max = 1/(1+1) = 0.5 at obs=1, kernel=1 at dist=0
    # Scale so that a brand-new feature at dist=0 gives score=1.0
    scores /= 0.5

    np.clip(scores, 0.0, 1.0, out=scores)
    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_uncertainty_map(state:           'SLAMState',
                           world,
                           robot_pos:       np.ndarray,
                           n_samples:       int   = 300,
                           robot_radius:    float = 0.25,
                           virtual_cov:     float = 0.32,
                           uncertainty_lo:  float = 0.25,
                           uncertainty_hi:  float = 0.75,
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
    robot_radius   : for bounding box expansion
    uncertainty_lo : lower threshold for the uncertain band
    uncertainty_hi : upper threshold for the uncertain band
    rng            : numpy Generator
    """
    if rng is None:
        rng = np.random.default_rng()

    # Bounding box from current map extents
    _, bounds = build_virtual_features(
        state, robot_radius=robot_radius, virtual_cov=virtual_cov)

    # Sample uniform points
    candidates = sample_points(bounds, n_samples, rng=rng)

    # Navigability filter
    nav           = navigable_mask(candidates, robot_pos, world)
    navigable_pts = candidates[nav]

    if len(navigable_pts) == 0:
        result = UncertaintyMap(
            points  = np.empty((0, 2)),
            scores  = np.array([]),
            bounds  = bounds,
            n_total = n_samples,
        )
        result.set_band(uncertainty_lo, uncertainty_hi)
        return result

    scores = _score_uncertainty(navigable_pts, state)

    result = UncertaintyMap(
        points  = navigable_pts,
        scores  = scores,
        bounds  = bounds,
        n_total = n_samples,
    )
    result.set_band(uncertainty_lo, uncertainty_hi)
    return result