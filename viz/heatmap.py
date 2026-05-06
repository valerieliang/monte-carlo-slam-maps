"""
viz/heatmap.py
--------------
Renders the Monte Carlo uncertainty heatmap on a matplotlib axes.

Points are coloured by their uncertainty score:
  score ~0   → dark blue   (free space — well known)
  score ~0.5 → bright amber (uncertain — navigation target)
  score ~1   → dark red    (occupied — near a feature)

The uncertainty band [lo, hi] is highlighted with larger markers so
the navigation targets stand out visually.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from montecarlo.uncertainty_map import UncertaintyMap


# Custom colormap: dark-blue → teal → amber  (emphasises the 0.5 region)
_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'uncertainty',
    [
        (0.0,  '#0f2a47'),   # dark blue  — free
        (0.35, '#1e6b8a'),   # teal
        (0.5,  '#f59e0b'),   # amber      — uncertain
        (0.65, '#c2410c'),   # orange-red
        (1.0,  '#450a0a'),   # dark red   — occupied
    ]
)


class HeatmapRenderer:
    """
    Manages the heatmap scatter artists on a matplotlib Axes.

    Usage
    -----
    hm = HeatmapRenderer(ax)
    hm.update(uncertainty_map)   # call each frame
    hm.clear()                   # hide the heatmap
    """

    def __init__(self, ax: plt.Axes,
                 base_size:  float = 8.0,
                 band_size:  float = 18.0,
                 base_alpha: float = 0.55,
                 band_alpha: float = 0.85,
                 zorder:     int   = 5):
        self._ax         = ax
        self._base_size  = base_size
        self._band_size  = band_size
        self._base_alpha = base_alpha
        self._band_alpha = band_alpha
        self._zorder     = zorder

        self._scatter_all  = None
        self._scatter_band = None
        self._colorbar     = None
        self._norm         = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def update(self, umap: 'UncertaintyMap') -> None:
        """Redraw the heatmap from a fresh UncertaintyMap."""
        self.clear()

        pts    = umap.points
        scores = umap.scores

        if len(pts) == 0:
            return

        # All navigable points — small, semi-transparent
        self._scatter_all = self._ax.scatter(
            pts[:, 0], pts[:, 1],
            c       = scores,
            cmap    = _CMAP,
            norm    = self._norm,
            s       = self._base_size,
            alpha   = self._base_alpha,
            zorder  = self._zorder,
            linewidths = 0,
        )

        # Uncertain band — larger, more opaque, stand out as nav targets
        band = umap.uncertain_mask
        if band.any():
            self._scatter_band = self._ax.scatter(
                pts[band, 0], pts[band, 1],
                c       = scores[band],
                cmap    = _CMAP,
                norm    = self._norm,
                s       = self._band_size,
                alpha   = self._band_alpha,
                zorder  = self._zorder + 1,
                linewidths = 0,
            )

    def clear(self) -> None:
        """Remove all heatmap artists from the axes."""
        for attr in ('_scatter_all', '_scatter_band'):
            artist = getattr(self, attr)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

    def add_colorbar(self, fig: plt.Figure,
                     label: str = 'Uncertainty score') -> None:
        """Attach a colorbar to the figure (call once after init)."""
        if self._scatter_all is None:
            return
        if self._colorbar is not None:
            self._colorbar.remove()
        self._colorbar = fig.colorbar(
            self._scatter_all, ax=self._ax,
            label=label, shrink=0.6, pad=0.02,
            ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        self._colorbar.ax.tick_params(labelsize=7)


def draw_goal(ax: plt.Axes,
              goal: Optional[np.ndarray],
              artists: list) -> list:
    """
    Draw or update the navigation goal marker.
    Clears previous goal artists first.

    Parameters
    ----------
    ax      : matplotlib Axes
    goal    : (2,) goal position, or None to just clear
    artists : list of previous goal artists to remove

    Returns
    -------
    new list of artists
    """
    for a in artists:
        try:
            a.remove()
        except Exception:
            pass

    if goal is None:
        return []

    new_artists = []

    # Outer ring
    ring, = ax.plot(goal[0], goal[1], 'o',
                    color='#f59e0b', markersize=16,
                    markerfacecolor='none', markeredgewidth=2.5,
                    zorder=15)
    # Inner dot
    dot, = ax.plot(goal[0], goal[1], 'o',
                   color='#f59e0b', markersize=5,
                   markerfacecolor='#f59e0b', markeredgewidth=0,
                   zorder=16)
    new_artists = [ring, dot]
    return new_artists