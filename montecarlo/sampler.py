"""
montecarlo/sampler.py
---------------------
Generate uniformly distributed Monte Carlo sample points inside the map
bounding box (Eqs. 6-7 of the paper).

    mu_i ~ U(0, 1),  i = 1..N
    m_i  = X_min + (X_max - X_min) * mu_i
    n_i  = Y_min + (Y_max - Y_min) * mu_i

The pair (m_i, n_i) is one MC point.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def sample_points(bounds: Tuple[float, float, float, float],
                  n:      int,
                  rng:    np.random.Generator | None = None
                  ) -> np.ndarray:
    """
    Draw N uniform random points inside the bounding box.

    Parameters
    ----------
    bounds : (xmin, xmax, ymin, ymax)
    n      : number of points
    rng    : numpy Generator (created if None)

    Returns
    -------
    pts : (N, 2) array of [x, y] sample points
    """
    if rng is None:
        rng = np.random.default_rng()

    xmin, xmax, ymin, ymax = bounds

    mu_x = rng.uniform(0.0, 1.0, n)
    mu_y = rng.uniform(0.0, 1.0, n)

    xs = xmin + (xmax - xmin) * mu_x   # Eq. 6
    ys = ymin + (ymax - ymin) * mu_y   # Eq. 7

    return np.column_stack([xs, ys])