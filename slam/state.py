"""
slam/state.py
-------------
EKF-SLAM state vector and covariance matrix management.

State vector layout (Eq. 1):
    x̂ = [x_v, y_v, θ_v | x_c1, y_c1, x_c2, y_c2, ..., ρ_l1, α_l1, ...]

Covariance matrix layout (Eq. 2):
    P = | Pv,v   Pv,m |
        | Pm,v   Pm,m |

Feature types
-------------
  'corner' : (x, y) in Cartesian — 2 state elements
  'line'   : (ρ, α) in polar    — 2 state elements

All features are size-2, so every feature occupies exactly 2 slots.
The feature index (0-based) maps to state slice [3 + 2*i : 3 + 2*i + 2].
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Feature descriptor (metadata alongside the state vector)
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """Metadata for one EKF feature."""
    idx:   int          # index into the features list (0-based)
    kind:  str          # 'corner' | 'line'
    # For lines: store the segment endpoints for region-check & rendering
    seg_p0: Optional[np.ndarray] = None   # (2,) world frame
    seg_p1: Optional[np.ndarray] = None   # (2,) world frame
    # observation count — used as confidence proxy
    obs_count: int = 1

    @property
    def state_slice(self) -> slice:
        """Slice into the full state vector for this feature's 2 elements."""
        start = 3 + 2 * self.idx
        return slice(start, start + 2)

    def __repr__(self) -> str:
        return f"Feature(idx={self.idx}, kind={self.kind}, obs={self.obs_count})"


# ---------------------------------------------------------------------------
# SLAMState
# ---------------------------------------------------------------------------

class SLAMState:
    """
    Holds the EKF-SLAM state vector x̂ and covariance P.

    Parameters
    ----------
    init_pose : (3,) [x, y, theta]  initial vehicle pose
    init_pose_cov : (3,3) initial vehicle pose covariance
    """

    POSE_DIM = 3      # vehicle pose dimensionality
    FEAT_DIM = 2      # every feature is 2D

    def __init__(self,
                 init_pose: np.ndarray,
                 init_pose_cov: Optional[np.ndarray] = None):

        self._x: np.ndarray = np.array(init_pose, dtype=float)   # (3,)
        if init_pose_cov is None:
            init_pose_cov = np.zeros((3, 3))
        self._P: np.ndarray = np.array(init_pose_cov, dtype=float)

        self.features: List[Feature] = []   # metadata list

    # ---------------------------------------------------------------- accessors

    @property
    def x(self) -> np.ndarray:
        """Full state vector (3 + 2*n_features,)."""
        return self._x

    @property
    def P(self) -> np.ndarray:
        """Full covariance matrix (dim x dim)."""
        return self._P

    @property
    def dim(self) -> int:
        return len(self._x)

    @property
    def n_features(self) -> int:
        return len(self.features)

    # -- vehicle pose ----------------------------------------------------------

    @property
    def pose(self) -> np.ndarray:
        """(x_v, y_v, θ_v) — copy."""
        return self._x[:3].copy()

    @pose.setter
    def pose(self, value: np.ndarray) -> None:
        self._x[:3] = value

    @property
    def Pvv(self) -> np.ndarray:
        """3×3 vehicle pose covariance block."""
        return self._P[:3, :3]

    # -- feature access --------------------------------------------------------

    def feature_mean(self, feat_idx: int) -> np.ndarray:
        """Return 2-element mean for feature feat_idx."""
        sl = self.features[feat_idx].state_slice
        return self._x[sl].copy()

    def feature_cov(self, feat_idx: int) -> np.ndarray:
        """Return 2×2 covariance for feature feat_idx."""
        sl = self.features[feat_idx].state_slice
        return self._P[np.ix_(range(sl.start, sl.stop),
                               range(sl.start, sl.stop))].copy()

    def feature_cross_cov_with_vehicle(self, feat_idx: int) -> np.ndarray:
        """Return 3×2 cross-covariance Pv,fi."""
        sl = self.features[feat_idx].state_slice
        cols = list(range(sl.start, sl.stop))
        return self._P[:3, cols].copy()

    # ---------------------------------------------------------------- mutation

    def set_pose(self, pose: np.ndarray) -> None:
        self._x[:3] = pose

    def set_Pvv(self, Pvv: np.ndarray) -> None:
        self._P[:3, :3] = Pvv

    def update_x(self, delta: np.ndarray) -> None:
        """Apply additive correction to full state."""
        self._x += delta
        # wrap vehicle heading
        self._x[2] = _wrap(self._x[2])

    def update_P(self, new_P: np.ndarray) -> None:
        self._P = new_P

    # ---------------------------------------------------------------- augment

    def add_feature(self,
                    mean: np.ndarray,
                    cov:  np.ndarray,
                    kind: str,
                    cross_cov_Pvi: np.ndarray,
                    seg_p0: Optional[np.ndarray] = None,
                    seg_p1: Optional[np.ndarray] = None
                    ) -> int:
        """
        Augment state and covariance with a new feature.

        Parameters
        ----------
        mean        : (2,)  initial feature mean
        cov         : (2,2) initial feature covariance
        kind        : 'corner' | 'line'
        cross_cov_Pvi : (dim, 2) cross-covariance between existing state
                        and new feature (computed in update.py during init)
        seg_p0/p1   : optional segment endpoints for line features

        Returns
        -------
        New feature index (0-based).
        """
        n   = self.dim       # current state dimension
        n2  = n + 2          # new dimension

        # -- extend state vector
        new_x = np.empty(n2)
        new_x[:n] = self._x
        new_x[n:] = mean
        self._x = new_x

        # -- extend covariance
        new_P = np.zeros((n2, n2))
        new_P[:n, :n] = self._P
        new_P[:n, n:] = cross_cov_Pvi          # (n, 2)
        new_P[n:, :n] = cross_cov_Pvi.T        # (2, n)
        new_P[n:, n:] = cov                    # (2, 2)
        self._P = new_P

        # -- register metadata
        feat_idx = len(self.features)
        self.features.append(Feature(
            idx       = feat_idx,
            kind      = kind,
            seg_p0    = seg_p0,
            seg_p1    = seg_p1,
            obs_count = 1,
        ))

        return feat_idx

    # ---------------------------------------------------------------- helpers

    def symmetrize_P(self) -> None:
        """Force P to be exactly symmetric (numerical drift guard)."""
        self._P = 0.5 * (self._P + self._P.T)

    def __repr__(self) -> str:
        return (f"SLAMState(pose={self._x[:3].round(3)}, "
                f"n_features={self.n_features}, dim={self.dim})")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)