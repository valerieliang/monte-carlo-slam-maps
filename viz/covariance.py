"""
viz/covariance.py
-----------------
Utilities for drawing 2-D covariance ellipses on matplotlib axes.

Used by renderer.py (Phase 4) to visualise EKF-SLAM feature uncertainty.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional


def cov_ellipse(mean: np.ndarray,
                cov:  np.ndarray,
                ax:   plt.Axes,
                n_std: float = 2.0,
                **kwargs) -> mpatches.Ellipse:
    """
    Draw a covariance ellipse on ax.

    Parameters
    ----------
    mean  : (2,) centre
    cov   : (2,2) covariance matrix
    ax    : matplotlib Axes
    n_std : number of standard deviations (default 2σ ≈ 95%)
    **kwargs : forwarded to Ellipse (color, alpha, zorder, …)

    Returns
    -------
    The Ellipse artist (already added to ax).
    """
    # eigendecomposition
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    # clip to avoid negative eigenvalues from numerical drift
    vals = np.maximum(vals, 1e-10)

    width  = 2.0 * n_std * np.sqrt(vals[1])
    height = 2.0 * n_std * np.sqrt(vals[0])

    # angle of the major axis (degrees, matplotlib convention)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))

    defaults = dict(
        fill      = False,
        linewidth = 1.0,
        alpha     = 0.5,
        zorder    = 6,
    )
    defaults.update(kwargs)

    ellipse = mpatches.Ellipse(
        xy     = (mean[0], mean[1]),
        width  = width,
        height = height,
        angle  = angle,
        **defaults,
    )
    ax.add_patch(ellipse)
    return ellipse


def draw_slam_features(state,
                       ax: plt.Axes,
                       corner_color: str = '#f87171',
                       line_color:   str = '#34d399',
                       ellipse_n_std: float = 2.0) -> None:
    """
    Draw all EKF-SLAM features and their covariance ellipses.

    Parameters
    ----------
    state : SLAMState
    ax    : matplotlib Axes
    """
    for feat in state.features:
        mean = state.feature_mean(feat.idx)
        cov  = state.feature_cov(feat.idx)

        if feat.kind == 'corner':
            color = corner_color
            ax.plot(mean[0], mean[1], '+',
                    color=color, markersize=8, markeredgewidth=2.0,
                    zorder=7)
        else:
            # line feature: draw from seg endpoints if known, else a small tick
            color = line_color
            ax.plot(mean[0] if feat.seg_p0 is None else
                    0.5*(feat.seg_p0[0]+feat.seg_p1[0]),
                    mean[1] if feat.seg_p0 is None else
                    0.5*(feat.seg_p0[1]+feat.seg_p1[1]),
                    'x', color=color, markersize=8, markeredgewidth=2.0,
                    zorder=7)
            if feat.seg_p0 is not None and feat.seg_p1 is not None:
                ax.plot([feat.seg_p0[0], feat.seg_p1[0]],
                        [feat.seg_p0[1], feat.seg_p1[1]],
                        '-', color=color, linewidth=1.5,
                        alpha=0.6, zorder=6)

        # draw covariance ellipse (only 2×2 block for position/rho)
        cov_ellipse(
            mean[:2] if feat.kind == 'corner'
                     else _polar_to_xy(mean),
            cov,
            ax,
            n_std    = ellipse_n_std,
            edgecolor = color,
            linewidth = 0.8,
            alpha    = 0.45,
        )


def _polar_to_xy(polar: np.ndarray) -> np.ndarray:
    """Convert polar (ρ, α) line params to the closest point to origin."""
    rho, alpha = polar
    return np.array([rho * np.cos(alpha), rho * np.sin(alpha)])