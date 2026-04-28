"""
viz/renderer.py
---------------
Bird's-eye 2-D renderer using matplotlib.

Draws:
  • Ground-truth walls  (solid dark lines)
  • Ground-truth corners (hollow squares -- convex=blue, concave=orange)
  • Robot pose arrow
  • Robot travel trail
  • World bounds / grid

Later phases will add:
  • Laser point cloud  (Phase 2 – sensor.py)
  • SLAM map overlay   (Phase 4 – ekf)
  • Covariance ellipses (Phase 4)
  • MC uncertainty heatmap (Phase 5)
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from typing import Optional, List, Tuple

# Use a non-interactive backend when running headless; override at call-site
# for interactive sessions.
matplotlib.rcParams['toolbar'] = 'None'


# -----------------------------------------------------------------------------
# Colour palette  (dark-terminal-friendly)
# -----------------------------------------------------------------------------
PAL = dict(
    bg          = '#0f1117',   # near-black background
    grid        = '#1e2130',   # subtle grid
    wall        = '#e2e8f0',   # bright white-grey walls
    wall_alpha  = 0.90,
    corner_cvx  = '#38bdf8',   # sky-blue  -- convex corners
    corner_ccv  = '#fb923c',   # orange    -- concave corners
    corner_alpha= 0.85,
    robot_body  = '#a78bfa',   # violet
    robot_arrow = '#f0abfc',   # pink-violet
    trail       = '#6366f1',   # indigo
    trail_alpha = 0.35,
    # placeholders for later phases
    laser       = '#fde68a',   # amber   (Phase 2)
    slam_wall   = '#34d399',   # emerald (Phase 4)
    slam_corner = '#f87171',   # red     (Phase 4)
    mc_low      = '#1e3a5f',   # dark blue  (Phase 5)
    mc_high     = '#f59e0b',   # amber      (Phase 5)
)


class Renderer:
    """
    Maintains a persistent matplotlib figure for real-time bird's-eye display.

    Usage (manual drive loop)
    -------------------------
    >>> r = Renderer(world)
    >>> r.init()
    >>> while running:
    ...     robot.step(v, omega, dt)
    ...     r.update(robot)
    ...     plt.pause(0.02)
    >>> r.close()
    """

    def __init__(self,
                 world,                   # env.World
                 figsize: Tuple[int, int] = (10, 7),
                 dpi:     int             = 110,
                 title:   str             = 'Active SLAM -- Phase 1'):
        self.world   = world
        self.figsize = figsize
        self.dpi     = dpi
        self.title   = title

        self.fig: Optional[plt.Figure]  = None
        self.ax:  Optional[plt.Axes]    = None

        # artist handles updated each frame
        self._robot_arrow: Optional[FancyArrowPatch] = None
        self._robot_dot:   Optional[plt.Artist]       = None
        self._trail_line:  Optional[plt.Line2D]        = None

    # -- public API ------------------------------------------------------------

    def init(self) -> None:
        """Create figure and draw all static geometry."""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.fig.patch.set_facecolor(PAL['bg'])
        self.ax.set_facecolor(PAL['bg'])

        self._style_axes()
        self._draw_static()
        self._init_dynamic()

        self.fig.tight_layout(pad=1.2)
        plt.title(self.title, color='#94a3b8', fontsize=10, pad=8,
                  fontfamily='monospace')

    def update(self, robot, laser_pts=None, slam_segments=None,
               slam_corners=None, mc_points=None) -> None:
        """
        Refresh all dynamic artists.  Call this every sim tick, then
        follow with plt.pause(dt) or fig.canvas.flush_events().

        Parameters
        ----------
        robot          : env.Robot
        laser_pts      : list of (x,y) -- Phase 2 raw laser returns
        slam_segments  : list of Segment -- Phase 4 SLAM map walls
        slam_corners   : list of Corner  -- Phase 4 SLAM map corners
        mc_points      : list of (x, y, prob) -- Phase 5 uncertainty map
        """
        self._update_robot(robot)
        self._update_trail(robot)

        # Phase 2+ overlays (no-op if None)
        if laser_pts is not None:
            self._draw_laser(laser_pts)
        if slam_segments is not None:
            self._draw_slam_map(slam_segments, slam_corners or [])
        if mc_points is not None:
            self._draw_mc(mc_points)

        self.fig.canvas.draw_idle()

    def close(self) -> None:
        plt.close(self.fig)

    # -- static geometry -------------------------------------------------------

    def _style_axes(self) -> None:
        ax = self.ax
        xmin, xmax, ymin, ymax = self.world.bounds
        pad = 1.2
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('x  (m)', color='#475569', fontsize=8, labelpad=4)
        ax.set_ylabel('y  (m)', color='#475569', fontsize=8, labelpad=4)
        ax.tick_params(colors='#475569', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#1e2130')
        ax.grid(True, color=PAL['grid'], linewidth=0.4, linestyle='--',
                alpha=0.6)

        # metre tick grid aligned to integers
        ax.set_xticks(range(int(xmin) - 1, int(xmax) + 2))
        ax.set_yticks(range(int(ymin) - 1, int(ymax) + 2))

    def _draw_static(self) -> None:
        """Draw ground-truth walls and corners once."""
        # walls
        for seg in self.world.segments:
            xs = [seg.p0[0], seg.p1[0]]
            ys = [seg.p0[1], seg.p1[1]]
            self.ax.plot(xs, ys,
                         color=PAL['wall'],
                         linewidth=2.0,
                         alpha=PAL['wall_alpha'],
                         solid_capstyle='round',
                         zorder=3)

        # corners -- hollow squares, colour-coded by type
        for c in self.world.corners:
            col = (PAL['corner_cvx'] if c.kind == 'convex'
                   else PAL['corner_ccv'])
            self.ax.plot(c.pos[0], c.pos[1],
                         marker='s',
                         markersize=7,
                         markerfacecolor='none',
                         markeredgecolor=col,
                         markeredgewidth=1.6,
                         alpha=PAL['corner_alpha'],
                         zorder=4)

        # legend
        leg_items = [
            mlines.Line2D([], [], color=PAL['wall'],       linewidth=2,
                          label='wall'),
            mlines.Line2D([], [], color=PAL['corner_cvx'], marker='s',
                          linestyle='none', markerfacecolor='none',
                          markeredgewidth=1.5, markersize=6,
                          label='convex corner'),
            mlines.Line2D([], [], color=PAL['corner_ccv'], marker='s',
                          linestyle='none', markerfacecolor='none',
                          markeredgewidth=1.5, markersize=6,
                          label='concave corner'),
            mlines.Line2D([], [], color=PAL['robot_body'], marker='o',
                          linestyle='none', markersize=6,
                          label='robot'),
            mlines.Line2D([], [], color=PAL['trail'],
                          linewidth=1.2, alpha=0.6, label='trail'),
        ]
        self.ax.legend(handles=leg_items,
                       loc='upper right',
                       fontsize=7,
                       framealpha=0.25,
                       facecolor='#1e2130',
                       edgecolor='#334155',
                       labelcolor='#94a3b8')

    # -- dynamic artists init --------------------------------------------------

    def _init_dynamic(self) -> None:
        # trail -- initialise with an empty line
        self._trail_line, = self.ax.plot([], [],
                                          color=PAL['trail'],
                                          linewidth=1.0,
                                          alpha=PAL['trail_alpha'],
                                          zorder=5)
        # robot dot
        self._robot_dot, = self.ax.plot([], [],
                                         'o',
                                         color=PAL['robot_body'],
                                         markersize=9,
                                         zorder=7)
        # heading arrow -- placeholder; rebuilt each frame
        self._robot_arrow = None

    # -- dynamic update helpers ------------------------------------------------

    def _update_robot(self, robot) -> None:
        # dot
        self._robot_dot.set_data([robot.x], [robot.y])

        # remove old arrow
        if self._robot_arrow is not None:
            self._robot_arrow.remove()

        arrow_len = 0.55
        dx = arrow_len * np.cos(robot.theta)
        dy = arrow_len * np.sin(robot.theta)

        self._robot_arrow = self.ax.annotate(
            '', xy=(robot.x + dx, robot.y + dy),
            xytext=(robot.x, robot.y),
            arrowprops=dict(
                arrowstyle='->', color=PAL['robot_arrow'],
                lw=2.0, mutation_scale=14),
            zorder=8)

        # update status text in title
        self.ax.set_title(
            f'x={robot.x:6.2f} m   y={robot.y:6.2f} m   '
            f'\u03b8={np.degrees(robot.theta):6.1f}\u00b0',
            color='#64748b', fontsize=8, pad=4, fontfamily='monospace')

    def _update_trail(self, robot) -> None:
        if len(robot.trail) < 2:
            return
        trail = np.array(robot.trail)
        self._trail_line.set_data(trail[:, 0], trail[:, 1])

    # -- Phase 2+ stubs (no-op in Phase 1) ------------------------------------

    def _draw_laser(self, laser_pts) -> None:
        """Overwritten / extended in Phase 2."""
        pass

    def _draw_slam_map(self, segments, corners) -> None:
        """Overwritten / extended in Phase 4."""
        pass

    def _draw_mc(self, mc_points) -> None:
        """Overwritten / extended in Phase 5."""
        pass