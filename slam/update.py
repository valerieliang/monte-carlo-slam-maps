"""
slam/update.py
--------------
EKF-SLAM update step (sequential, one feature at a time).

For each pre-associated observation we:
  1. Compute the expected measurement h(x̂) and its Jacobian H
  2. Form the innovation   ν = z - h(x̂)
  3. Compute innovation covariance S = H P H^T + R
  4. Compute Kalman gain   K = P H^T S^{-1}
  5. Update state           x̂ ← x̂ + K ν
  6. Update covariance      P ← (I - K H) P

Feature initialisation (new features):
  Augments the state by inverting the measurement model and computing
  the initial covariance via the Jacobian of the inverse model.
"""

from __future__ import annotations
import numpy as np
from slam.state import SLAMState, _wrap


# ---------------------------------------------------------------------------
# Measurement Jacobians
# ---------------------------------------------------------------------------

def _h_corner_jacobian(state: SLAMState, feat_idx: int) -> np.ndarray:
    """
    Jacobian H of the corner measurement model (Eq. 4) w.r.t. the full
    state vector.  Shape: (2, state.dim).

    h = [r, β] where
        r = sqrt((x_v - x_c)^2 + (y_v - y_c)^2)
        β = atan2(y_c - y_v, x_c - x_v) - θ_v
    """
    xv, yv, th = state.pose
    feat        = state.feature_mean(feat_idx)
    xc, yc      = feat

    dx  = xc - xv
    dy  = yc - yv
    r2  = dx**2 + dy**2
    r   = np.sqrt(r2)

    if r < 1e-6:
        r = 1e-6

    H = np.zeros((2, state.dim))

    # ∂h/∂[xv, yv, θv]
    H[0, 0] = -dx / r          # ∂r/∂x_v
    H[0, 1] = -dy / r          # ∂r/∂y_v
    H[0, 2] =  0.0             # ∂r/∂θ_v

    H[1, 0] =  dy / r2         # ∂β/∂x_v
    H[1, 1] = -dx / r2         # ∂β/∂y_v
    H[1, 2] = -1.0             # ∂β/∂θ_v

    # ∂h/∂[xc, yc]
    sl = state.features[feat_idx].state_slice
    H[0, sl.start]     =  dx / r     # ∂r/∂x_c
    H[0, sl.start + 1] =  dy / r     # ∂r/∂y_c

    H[1, sl.start]     = -dy / r2    # ∂β/∂x_c
    H[1, sl.start + 1] =  dx / r2    # ∂β/∂y_c

    return H


def _h_line_jacobian(state: SLAMState, feat_idx: int) -> np.ndarray:
    """
    Jacobian H of the line measurement model (Eq. 5) w.r.t. full state.
    Shape: (2, state.dim).

    h = [ρ - x_v cos(α) - y_v sin(α),  α - θ_v]
    """
    xv, yv, th = state.pose
    feat        = state.feature_mean(feat_idx)
    rho, alpha  = feat

    H = np.zeros((2, state.dim))

    # ∂h/∂[xv, yv, θv]
    H[0, 0] = -np.cos(alpha)   # ∂z_ρ/∂x_v
    H[0, 1] = -np.sin(alpha)   # ∂z_ρ/∂y_v
    H[0, 2] =  0.0

    H[1, 0] =  0.0             # ∂z_α/∂x_v
    H[1, 1] =  0.0
    H[1, 2] = -1.0             # ∂z_α/∂θ_v

    # ∂h/∂[ρ, α]
    sl = state.features[feat_idx].state_slice
    H[0, sl.start]     = 1.0   # ∂z_ρ/∂ρ
    H[0, sl.start + 1] = (xv * np.sin(alpha)
                           - yv * np.cos(alpha))  # ∂z_ρ/∂α

    H[1, sl.start]     = 0.0   # ∂z_α/∂ρ
    H[1, sl.start + 1] = 1.0   # ∂z_α/∂α

    return H


# ---------------------------------------------------------------------------
# Expected measurements
# ---------------------------------------------------------------------------

def _h_corner(state: SLAMState, feat_idx: int) -> np.ndarray:
    """Expected corner observation [r, β] (Eq. 4, noiseless)."""
    xv, yv, th = state.pose
    xc, yc     = state.feature_mean(feat_idx)
    dx, dy     = xc - xv, yc - yv
    r          = np.sqrt(dx**2 + dy**2)
    beta       = _wrap(np.arctan2(dy, dx) - th)
    return np.array([r, beta])


def _h_line(state: SLAMState, feat_idx: int) -> np.ndarray:
    """Expected line observation [ρ_obs, α_obs] (Eq. 5, noiseless)."""
    xv, yv, th = state.pose
    rho, alpha = state.feature_mean(feat_idx)
    rho_obs    = rho - xv * np.cos(alpha) - yv * np.sin(alpha)
    alpha_obs  = _wrap(alpha - th)
    return np.array([rho_obs, alpha_obs])


# ---------------------------------------------------------------------------
# Single-feature update
# ---------------------------------------------------------------------------

def update_single(state: SLAMState,
                  feat_idx: int,
                  z: np.ndarray,
                  R: np.ndarray) -> None:
    """
    Apply one EKF update for a pre-associated observation.

    Parameters
    ----------
    state    : SLAMState  (modified in-place)
    feat_idx : index of the matched feature in state.features
    z        : (2,) observed measurement [r, β] or [ρ, α]
    R        : (2,2) measurement noise covariance
    """
    feat = state.features[feat_idx]

    # expected measurement and Jacobian
    if feat.kind == 'corner':
        z_hat = _h_corner(state, feat_idx)
        H     = _h_corner_jacobian(state, feat_idx)
    else:
        z_hat = _h_line(state, feat_idx)
        H     = _h_line_jacobian(state, feat_idx)

    # innovation
    nu = z - z_hat
    if feat.kind == 'corner':
        nu[1] = _wrap(nu[1])   # wrap bearing innovation
    else:
        nu[1] = _wrap(nu[1])   # wrap angle innovation

    # innovation covariance
    P = state.P
    S = H @ P @ H.T + R        # (2, 2)

    # Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return                 # degenerate, skip

    K = P @ H.T @ S_inv        # (n, 2)

    # state update
    state.update_x(K @ nu)

    # covariance update  (Joseph form for numerical stability)
    n   = state.dim
    IKH = np.eye(n) - K @ H
    new_P = IKH @ P @ IKH.T + K @ R @ K.T
    state.update_P(new_P)
    state.symmetrize_P()

    feat.obs_count += 1


# ---------------------------------------------------------------------------
# Feature initialisation (augment state with a new feature)
# ---------------------------------------------------------------------------

def init_corner(state: SLAMState,
                z: np.ndarray,
                R: np.ndarray,
                init_cov_scale: float = 0.5) -> int:
    """
    Initialise a new corner feature from observation z = [r, β].

    Inverse measurement model (world frame position):
        x_c = x_v + r · cos(θ_v + β)
        y_c = y_v + r · sin(θ_v + β)

    Initial covariance via linearisation of the inverse model.

    Returns the new feature index.
    """
    xv, yv, th = state.pose
    r, beta    = z
    angle      = th + beta

    # mean
    xc = xv + r * np.cos(angle)
    yc = yv + r * np.sin(angle)
    mean = np.array([xc, yc])

    # Jacobian of inverse model w.r.t. [x_v, y_v, θ_v, r, β]
    # g(xv, z) = [xv + r cos(θ+β), yv + r sin(θ+β)]
    Jv = np.array([
        [1.0, 0.0, -r * np.sin(angle)],
        [0.0, 1.0,  r * np.cos(angle)],
    ])   # (2, 3)  ∂g/∂x_v

    Jz = np.array([
        [np.cos(angle), -r * np.sin(angle)],
        [np.sin(angle),  r * np.cos(angle)],
    ])   # (2, 2)  ∂g/∂z

    Pvv = state.Pvv   # (3, 3)

    # Initial feature covariance
    cov = Jv @ Pvv @ Jv.T + Jz @ R @ Jz.T
    cov += np.eye(2) * init_cov_scale   # inflate slightly

    # Cross-covariance between existing state and new feature: (n, 2)
    # P_{existing, new} = P_{existing, v} · Jv^T
    n   = state.dim
    Pxv = state.P[:, :3]           # (n, 3)  all rows, vehicle cols
    cross = Pxv @ Jv.T             # (n, 2)

    return state.add_feature(mean, cov, 'corner', cross,
                              seg_p0=None, seg_p1=None)


def init_line(state: SLAMState,
              z: np.ndarray,
              R: np.ndarray,
              init_cov_scale: float = 0.5,
              seg_p0: np.ndarray | None = None,
              seg_p1: np.ndarray | None = None) -> int:
    """
    Initialise a new line feature from observation z = [ρ_obs, α_obs].

    Inverse model (global polar parameters):
        ρ     = ρ_obs + x_v cos(α) + y_v sin(α)
        α     = α_obs + θ_v

    Returns the new feature index.
    """
    xv, yv, th = state.pose
    rho_obs, alpha_obs = z

    alpha = _wrap(alpha_obs + th)
    rho   = rho_obs + xv * np.cos(alpha) + yv * np.sin(alpha)
    if rho < 0:
        rho   = -rho
        alpha = _wrap(alpha + np.pi)

    mean = np.array([rho, alpha])

    # Jacobian of inverse model w.r.t. [x_v, y_v, θ_v]
    Jv = np.array([
        [np.cos(alpha),  np.sin(alpha),
         -xv * np.sin(alpha) + yv * np.cos(alpha)],
        [0.0,            0.0,           1.0       ],
    ])   # (2, 3)

    # Jacobian w.r.t. [ρ_obs, α_obs]
    Jz = np.array([
        [1.0, xv * np.sin(alpha) - yv * np.cos(alpha)],
        [0.0, 1.0                                     ],
    ])   # (2, 2)

    Pvv = state.Pvv
    cov = Jv @ Pvv @ Jv.T + Jz @ R @ Jz.T
    cov += np.eye(2) * init_cov_scale

    n     = state.dim
    Pxv   = state.P[:, :3]
    cross = Pxv @ Jv.T   # (n, 2)

    return state.add_feature(mean, cov, 'line', cross,
                              seg_p0=seg_p0, seg_p1=seg_p1)