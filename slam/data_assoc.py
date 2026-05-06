"""
slam/data_assoc.py
------------------
Data association: match incoming feature observations to known map features
using Mahalanobis-distance gating (nearest-neighbour).

Algorithm (per observation):
  1. Compute expected measurement h(x̂, feat_i) for each existing feature
     of the same kind.
  2. Compute innovation  ν_i = z - h_i  and covariance S_i = H_i P H_i^T + R
  3. Compute Mahalanobis distance  d_i = ν_i^T S_i^{-1} ν_i
  4. If min(d_i) < gate_chi2 threshold → matched (return feat_idx)
     Else → new feature (return -1)
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

from slam.state import SLAMState, _wrap
from slam.update import (
    _h_corner, _h_corner_jacobian,
    _h_line,   _h_line_jacobian,
)


# ---------------------------------------------------------------------------
# Observation containers (produced by features.py)
# ---------------------------------------------------------------------------

# These mirror the dataclasses in slam/features.py; imported here to keep
# data_assoc self-contained.  We duck-type on .z and .kind.


# ---------------------------------------------------------------------------
# Core gating
# ---------------------------------------------------------------------------

def mahalanobis(nu: np.ndarray, S: np.ndarray) -> float:
    """
    Mahalanobis distance squared: ν^T S^{-1} ν.
    Returns np.inf on singular S.
    """
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.inf
    return float(nu @ S_inv @ nu)


def associate(state: SLAMState,
              z: np.ndarray,
              kind: str,
              R: np.ndarray,
              gate_chi2: float = 9.21
              ) -> Tuple[int, float]:
    """
    Find the best matching existing feature for observation z.

    Parameters
    ----------
    state     : current SLAMState
    z         : (2,) observation ([r, β] for corner, [ρ, α] for line)
    kind      : 'corner' | 'line'
    R         : (2,2) measurement noise covariance
    gate_chi2 : chi-squared threshold (chi2(2, 0.99) ≈ 9.21)

    Returns
    -------
    (feat_idx, distance)
      feat_idx = -1  if no match (new feature)
      feat_idx >= 0  index into state.features
    """
    best_idx  = -1
    best_dist = gate_chi2   # only accept if below threshold

    P = state.P

    for i, feat in enumerate(state.features):
        if feat.kind != kind:
            continue

        # expected measurement
        if kind == 'corner':
            z_hat = _h_corner(state, i)
            H     = _h_corner_jacobian(state, i)
        else:
            z_hat = _h_line(state, i)
            H     = _h_line_jacobian(state, i)

        # innovation
        nu    = z - z_hat
        nu[1] = _wrap(nu[1])   # wrap angular component

        # innovation covariance
        S = H @ P @ H.T + R

        dist = mahalanobis(nu, S)
        if dist < best_dist:
            best_dist = dist
            best_idx  = i

    return best_idx, best_dist


# ---------------------------------------------------------------------------
# Batch association for a full observation list
# ---------------------------------------------------------------------------

def associate_observations(state: SLAMState,
                            observations: list,
                            R_corner: np.ndarray,
                            R_line:   np.ndarray,
                            gate_chi2: float = 9.21
                            ) -> List[Tuple[object, int]]:
    """
    Associate every observation in `observations` to either an existing
    feature or flag it as new.

    Parameters
    ----------
    observations : list of CornerObs | LineObs  (from slam/features.py)
                   Each must have .z (2,) and .feature_kind ('corner'|'line')
    R_corner     : (2,2) corner measurement noise
    R_line       : (2,2) line measurement noise

    Returns
    -------
    List of (obs, feat_idx) where feat_idx == -1 means new feature.
    """
    results = []
    for obs in observations:
        R = R_corner if obs.feature_kind == 'corner' else R_line
        feat_idx, _ = associate(state, obs.z, obs.feature_kind, R, gate_chi2)
        results.append((obs, feat_idx))
    return results