"""
montecarlo/probability.py
-------------------------
Compute the probability of each MC point under the sum-of-Gaussians model
(Eqs. 8, 9, 10 of the paper).

Each SLAM feature is treated as a Gaussian distribution:
  - Corner (Eq. 8): 2-D Gaussian in (x, y) with mean and covariance from EKF
  - Line   (Eq. 9): 1-D Gaussian on perpendicular distance, evaluated only
                    for points within the segment's region (Fig. 3)

Virtual boundary lines use the same formula with fixed covariance diag(0.32, 0.32).

Sum of Gaussians (Eq. 10):
  P(p_i) = sum_k  sigma_k * N(p_i ; mu_k, Psi_k)

where sigma_k = 1/L (uniform weighting, L = total number of features).
"""

from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from slam.state import SLAMState
    from montecarlo.virtual_features import VirtualLine


# ── Corner probability (Eq. 8) ────────────────────────────────────────────────

def corner_prob(point: np.ndarray,
                mean:  np.ndarray,
                cov:   np.ndarray) -> float:
    """
    2-D Gaussian probability of a point given a corner feature (Eq. 8).

    P(p) = 1 / (2π√|Ψ|) · exp(-½ (p-μ)ᵀ Ψ⁻¹ (p-μ))

    Parameters
    ----------
    point : (2,) [px, py]
    mean  : (2,) [μx, μy]
    cov   : (2,2) Ψ

    Returns
    -------
    probability value (float >= 0)
    """
    det = np.linalg.det(cov)
    if det <= 0:
        return 0.0
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0

    diff     = point - mean
    exponent = -0.5 * diff @ cov_inv @ diff
    norm     = 1.0 / (2.0 * np.pi * np.sqrt(det))
    return float(norm * np.exp(exponent))


# ── Line probability (Eq. 9) ──────────────────────────────────────────────────

def _point_in_segment_region(point:  np.ndarray,
                              seg_p0: np.ndarray,
                              seg_p1: np.ndarray) -> bool:
    """
    True if point falls within the region of the segment (Fig. 3 of paper).

    The region is the infinite strip bounded by normals through p0 and p1 —
    i.e. the projection of the point onto the segment direction lies in [0, L].
    """
    d   = seg_p1 - seg_p0
    L   = np.linalg.norm(d)
    if L < 1e-9:
        return False
    t   = np.dot(point - seg_p0, d / L)
    return 0.0 <= t <= L


def line_prob(point:  np.ndarray,
              rho:    float,
              alpha:  float,
              cov:    np.ndarray,
              seg_p0: np.ndarray,
              seg_p1: np.ndarray) -> float:
    """
    1-D Gaussian probability of a point w.r.t. a line feature (Eq. 9).

    Only non-zero when the point falls within the segment's region (Fig. 3).

    The point is assumed to lie on a line parallel to the feature line —
    angle difference is zero, only the perpendicular distance Δρ matters.

    P(p) = 1/√(2π|Ψ|) · exp(-½ [Δρ, 0] Ψ⁻¹ [Δρ, 0]ᵀ)

    Parameters
    ----------
    point  : (2,) [px, py]
    rho    : line feature rho (global polar)
    alpha  : line feature alpha (global polar)
    cov    : (2,2) Ψ — covariance of (rho, alpha)
    seg_p0 : (2,) segment start (for region check)
    seg_p1 : (2,) segment end

    Returns
    -------
    probability value (float >= 0)
    """
    if not _point_in_segment_region(point, seg_p0, seg_p1):
        return 0.0

    # Perpendicular distance from point to the line's parallel through origin
    # rho_point = px*cos(alpha) + py*sin(alpha)
    # Δρ = rho_point - rho  (signed distance along normal)
    rho_point = point[0] * np.cos(alpha) + point[1] * np.sin(alpha)
    delta_rho = rho_point - rho

    # Evaluate Gaussian: Υ = [rho + Δρ, alpha],  Γ = [rho, alpha]
    # The angle difference is 0, so innovation = [Δρ, 0]
    nu = np.array([delta_rho, 0.0])

    det = np.linalg.det(cov)
    if det <= 0:
        return 0.0
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0

    exponent = -0.5 * nu @ cov_inv @ nu
    norm     = 1.0 / np.sqrt((2.0 * np.pi)**2 * det)
    return float(norm * np.exp(exponent))


# ── Sum of Gaussians (Eq. 10) ─────────────────────────────────────────────────

def score_points(points:        np.ndarray,
                 state:         'SLAMState',
                 virtual_lines: List['VirtualLine']
                 ) -> np.ndarray:
    """
    Score each MC point using the sum-of-Gaussians formula (Eq. 10).

    Parameters
    ----------
    points        : (N, 2) navigable MC points
    state         : current SLAMState
    virtual_lines : list of VirtualLine boundary features

    Returns
    -------
    scores : (N,) float array, values in [0, 1]
             ~0   = clearly free space
             ~0.5 = uncertain (target for navigation)
             ~1   = clearly occupied
    """
    N = len(points)
    if N == 0:
        return np.array([])

    # Total number of feature contributions
    n_real     = state.n_features
    n_virtual  = len(virtual_lines)
    L          = n_real + n_virtual
    if L == 0:
        return np.full(N, 0.5)

    sigma = 1.0 / L   # uniform weighting

    scores = np.zeros(N)

    # ── Real features ──────────────────────────────────────────────────────
    for feat in state.features:
        mean = state.feature_mean(feat.idx)
        cov  = state.feature_cov(feat.idx)

        # Clamp covariance eigenvalues to avoid degenerate Gaussians
        cov = _regularise_cov(cov)

        if feat.kind == 'corner':
            for i, pt in enumerate(points):
                scores[i] += sigma * corner_prob(pt, mean, cov)
        else:
            rho, alpha = mean
            p0 = feat.seg_p0 if feat.seg_p0 is not None else mean
            p1 = feat.seg_p1 if feat.seg_p1 is not None else mean
            for i, pt in enumerate(points):
                scores[i] += sigma * line_prob(pt, rho, alpha, cov, p0, p1)

    # ── Virtual boundary lines ─────────────────────────────────────────────
    for vl in virtual_lines:
        cov = _regularise_cov(vl.cov)
        for i, pt in enumerate(points):
            scores[i] += sigma * line_prob(
                pt, vl.rho, vl.alpha, cov, vl.seg_p0, vl.seg_p1)

    return scores


def score_points_vectorised(points:        np.ndarray,
                             state:         'SLAMState',
                             virtual_lines: List['VirtualLine']
                             ) -> np.ndarray:
    """
    Vectorised version of score_points for better performance with N > 500.

    Same semantics as score_points but operates on the full point array at
    once using numpy broadcasting.
    """
    N = len(points)
    if N == 0:
        return np.array([])

    n_real    = state.n_features
    n_virtual = len(virtual_lines)
    L         = n_real + n_virtual
    if L == 0:
        return np.full(N, 0.5)

    # Uniform weighting across ALL features (real + virtual): sigma = 1/L
    # The paper's virtual-line covariance diag(0.32≈1/π, 0.32) is derived
    # so that with 4 virtual lines at sigma=1/4, the sum peaks at 0.5
    # at any boundary point — exactly the uncertainty threshold.
    sigma  = 1.0 / L
    scores = np.zeros(N)

    # ── Real features ──────────────────────────────────────────────────────
    for feat in state.features:
        mean = state.feature_mean(feat.idx)
        cov  = _regularise_cov(state.feature_cov(feat.idx))

        if feat.kind == 'corner':
            scores += sigma * _corner_prob_batch(points, mean, cov)
        else:
            rho, alpha = mean
            p0 = feat.seg_p0 if feat.seg_p0 is not None else mean
            p1 = feat.seg_p1 if feat.seg_p1 is not None else mean
            scores += sigma * _line_prob_batch(points, rho, alpha, cov, p0, p1)

    # ── Virtual boundary lines ─────────────────────────────────────────────
    for vl in virtual_lines:
        cov = _regularise_cov(vl.cov)
        scores += sigma * _line_prob_batch(
            points, vl.rho, vl.alpha, cov, vl.seg_p0, vl.seg_p1)

    # Clamp to [0, 1] — scores can exceed 1.0 near dense feature clusters
    np.clip(scores, 0.0, 1.0, out=scores)

    return scores


# ── Vectorised helpers ────────────────────────────────────────────────────────

def _corner_prob_batch(points: np.ndarray,
                       mean:   np.ndarray,
                       cov:    np.ndarray) -> np.ndarray:
    """(N,) corner probabilities for all points at once."""
    det = np.linalg.det(cov)
    if det <= 0:
        return np.zeros(len(points))
    cov_inv  = np.linalg.inv(cov)
    diff     = points - mean          # (N, 2)
    exponent = -0.5 * np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
    norm     = 1.0 / (2.0 * np.pi * np.sqrt(det))
    return norm * np.exp(exponent)


def _line_prob_batch(points:  np.ndarray,
                     rho:     float,
                     alpha:   float,
                     cov:     np.ndarray,
                     seg_p0:  np.ndarray,
                     seg_p1:  np.ndarray) -> np.ndarray:
    """(N,) line probabilities for all points at once."""
    N   = len(points)
    det = np.linalg.det(cov)
    if det <= 0:
        return np.zeros(N)
    cov_inv = np.linalg.inv(cov)
    norm    = 1.0 / np.sqrt((2.0 * np.pi)**2 * det)

    # Region mask
    d = seg_p1 - seg_p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return np.zeros(N)
    d_hat = d / L
    t     = (points - seg_p0) @ d_hat      # (N,)
    in_region = (t >= 0.0) & (t <= L)

    # Perpendicular distance
    rho_pts   = points[:, 0] * np.cos(alpha) + points[:, 1] * np.sin(alpha)
    delta_rho = rho_pts - rho             # (N,)

    # Build innovation [Δρ, 0] for each point
    nu        = np.zeros((N, 2))
    nu[:, 0]  = delta_rho
    exponent  = -0.5 * np.einsum('ni,ij,nj->n', nu, cov_inv, nu)

    result    = norm * np.exp(exponent)
    result[~in_region] = 0.0
    return result


# ── Boundary score (replaces virtual-line Gaussian for large rooms) ───────────

def boundary_scores(points:       np.ndarray,
                    bounds:        tuple,
                    decay_length:  float | None = None) -> np.ndarray:
    """
    Score each point by proximity to the unexplored boundary.

    Returns values in [0, 0.5]:
      0.5  at the boundary
      decaying inward with length scale decay_length

    This replaces the virtual-line Gaussian model for environments larger
    than a few metres, where the narrow paper covariance (0.32) would give
    near-zero scores in the room interior.

    Parameters
    ----------
    points       : (N, 2)
    bounds       : (xmin, xmax, ymin, ymax) — sampling bounds
    decay_length : length scale in metres.  Defaults to
                   max(width, height) / 4.0 so the score drops to ~0.07
                   at the room centre.
    """
    xmin, xmax, ymin, ymax = bounds
    w = xmax - xmin
    h = ymax - ymin

    if decay_length is None:
        decay_length = max(w, h) / 4.0

    # Distance from each point to each of the 4 walls
    d_south = points[:, 1] - ymin   # dist to south wall
    d_north = ymax - points[:, 1]   # dist to north wall
    d_west  = points[:, 0] - xmin   # dist to west wall
    d_east  = xmax - points[:, 0]   # dist to east wall

    # Minimum distance to any boundary
    d_min = np.minimum(np.minimum(d_south, d_north),
                       np.minimum(d_west,  d_east))
    d_min = np.maximum(d_min, 0.0)

    return 0.5 * np.exp(-d_min / decay_length)


# ── Utility ───────────────────────────────────────────────────────────────────

def _regularise_cov(cov: np.ndarray, min_eig: float = 1e-4) -> np.ndarray:
    """Clamp minimum eigenvalue to avoid singular covariances."""
    vals, vecs = np.linalg.eigh(cov)
    vals       = np.maximum(vals, min_eig)
    return (vecs * vals) @ vecs.T