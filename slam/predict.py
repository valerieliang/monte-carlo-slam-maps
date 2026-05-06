"""
slam/predict.py
---------------
EKF-SLAM prediction step (Eq. 3 + Jacobian propagation).

Unicycle motion model:
    x_v' = x_v + v·cos(θ)·dt
    y_v' = y_v + v·sin(θ)·dt
    θ'   = θ + ω·dt

Process noise Q acts on the velocity inputs (v, ω).
The Jacobian F maps Q into state space via the linearised motion model.

Only the 3×3 vehicle block of the covariance is touched directly; the
cross-correlations Pv,m and Pm,v are updated via the same Jacobian.
"""

from __future__ import annotations
import numpy as np
from slam.state import SLAMState, _wrap


def predict(state: SLAMState,
            v:     float,
            omega: float,
            dt:    float,
            Q_v:   float,
            Q_w:   float) -> None:
    """
    In-place EKF prediction step.

    Parameters
    ----------
    state : SLAMState  (modified in-place)
    v     : linear  velocity command (m/s)
    omega : angular velocity command (rad/s)
    dt    : timestep (s)
    Q_v   : process noise variance for linear  velocity
    Q_w   : process noise variance for angular velocity
    """
    xv, yv, th = state.pose

    # -- propagate mean -------------------------------------------------------
    new_x  = xv + v * np.cos(th) * dt
    new_y  = yv + v * np.sin(th) * dt
    new_th = _wrap(th + omega * dt)
    state.set_pose(np.array([new_x, new_y, new_th]))

    # -- Jacobian of motion model w.r.t. vehicle pose (3×3) -------------------
    #   F_v = d f(xv, u) / d xv
    Fv = np.array([
        [1.0, 0.0, -v * np.sin(th) * dt],
        [0.0, 1.0,  v * np.cos(th) * dt],
        [0.0, 0.0,  1.0               ],
    ])

    # -- Jacobian of motion model w.r.t. noise inputs (3×2) -------------------
    #   G = d f / d [v, omega]
    G = np.array([
        [np.cos(th) * dt,  0.0],
        [np.sin(th) * dt,  0.0],
        [0.0,              dt ],
    ])

    # Process noise in velocity space
    Q = np.diag([Q_v, Q_w])

    # Additive process noise in state space
    Q_state = G @ Q @ G.T   # (3×3)

    # -- propagate covariance --------------------------------------------------
    # Full-state Jacobian F is block-diagonal:
    #   F = | Fv   0 |
    #       | 0    I |
    # We exploit structure to avoid building the full (n×n) matrix.

    P   = state.P
    n   = state.dim
    nf  = n - 3   # number of feature state elements

    # Pv,v block
    Pvv_new = Fv @ P[:3, :3] @ Fv.T + Q_state

    # Pv,m  and  Pm,v  cross-correlation blocks
    if nf > 0:
        Pvm_new = Fv @ P[:3, 3:]          # (3, nf)
        Pmv_new = Pvm_new.T               # (nf, 3)

        # Build updated P
        new_P = P.copy()
        new_P[:3, :3]  = Pvv_new
        new_P[:3, 3:]  = Pvm_new
        new_P[3:, :3]  = Pmv_new
        # Pm,m block unchanged (features not moved by prediction)
    else:
        new_P = Pvv_new

    state.update_P(new_P)
    state.symmetrize_P()