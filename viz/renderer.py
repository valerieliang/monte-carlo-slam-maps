"""
viz/renderer.py
---------------
Bird's-eye 2-D renderer using matplotlib.

Phase 1: walls, corners, robot pose + trail
Phase 2: laser beam fan + hit-point cloud  (ACTIVE)
Phase 4: SLAM map overlay + covariance ellipses (stub)
Phase 5: MC uncertainty heatmap (stub)
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from typing import Optional, Tuple, List

matplotlib.rcParams['toolbar'] = 'None'

PAL = dict(
    bg           = '#0f1117',
    grid         = '#1e2130',
    wall         = '#e2e8f0',
    wall_alpha   = 0.90,
    corner_cvx   = '#38bdf8',
    corner_ccv   = '#fb923c',
    corner_alpha = 0.85,
    robot_body   = '#a78bfa',
    robot_arrow  = '#f0abfc',
    trail        = '#6366f1',
    trail_alpha  = 0.35,
    laser        = '#fde68a',
    laser_alpha  = 0.75,
    beam_alpha   = 0.10,
    slam_wall    = '#34d399',
    slam_corner  = '#f87171',
    mc_low       = '#1e3a5f',
    mc_high      = '#f59e0b',
)


class Renderer:
    """
    Persistent matplotlib figure for real-time bird's-eye display.

    update() accepts optional laser_scan (ScanResult) from Phase 2 onward.
    """

    def __init__(self,
                 world,
                 figsize: Tuple[int, int] = (10, 7),
                 dpi:     int             = 110,
                 title:   str             = 'Active SLAM'):
        self.world   = world
        self.figsize = figsize
        self.dpi     = dpi
        self.title   = title

        self.fig: Optional[plt.Figure] = None
        self.ax:  Optional[plt.Axes]   = None

        self._robot_arrow = None
        self._robot_dot   = None
        self._trail_line  = None
        self._laser_pts   = None
        self._beam_fan    = None

    # ------------------------------------------------------------------ init

    def init(self) -> None:
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

    # ----------------------------------------------------------------- update

    def update(self, robot,
               laser_scan=None,
               slam_segments=None,
               slam_corners=None,
               mc_points=None) -> None:
        self._update_robot(robot)
        self._update_trail(robot)

        if laser_scan is not None:
            self._update_laser(robot, laser_scan)
        else:
            self._laser_pts.set_data([], [])
            self._beam_fan.set_segments([])

        if slam_segments is not None:
            self._draw_slam_map(slam_segments, slam_corners or [])

        if mc_points is not None:
            self._draw_mc(mc_points)

        self.fig.canvas.draw_idle()

    def close(self) -> None:
        plt.close(self.fig)

    # --------------------------------------------------------- static geometry

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
        ax.grid(True, color=PAL['grid'], linewidth=0.4,
                linestyle='--', alpha=0.6)
        ax.set_xticks(range(int(xmin) - 1, int(xmax) + 2))
        ax.set_yticks(range(int(ymin) - 1, int(ymax) + 2))

    def _draw_static(self) -> None:
        for seg in self.world.segments:
            self.ax.plot([seg.p0[0], seg.p1[0]],
                         [seg.p0[1], seg.p1[1]],
                         color=PAL['wall'],
                         linewidth=2.0,
                         alpha=PAL['wall_alpha'],
                         solid_capstyle='round',
                         zorder=3)

        for c in self.world.corners:
            col = PAL['corner_cvx'] if c.kind == 'convex' else PAL['corner_ccv']
            self.ax.plot(c.pos[0], c.pos[1],
                         marker='s', markersize=7,
                         markerfacecolor='none',
                         markeredgecolor=col,
                         markeredgewidth=1.6,
                         alpha=PAL['corner_alpha'],
                         zorder=4)

        leg = [
            mlines.Line2D([], [], color=PAL['wall'], linewidth=2,
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
            mlines.Line2D([], [], color=PAL['laser'], marker='.',
                          linestyle='none', markersize=5,
                          label='laser returns'),
        ]
        self.ax.legend(handles=leg, loc='upper right', fontsize=7,
                       framealpha=0.25, facecolor='#1e2130',
                       edgecolor='#334155', labelcolor='#94a3b8')

    # -------------------------------------------------------- dynamic artists

    def _init_dynamic(self) -> None:
        self._trail_line, = self.ax.plot(
            [], [], color=PAL['trail'], linewidth=1.0,
            alpha=PAL['trail_alpha'], zorder=5)

        self._robot_dot, = self.ax.plot(
            [], [], 'o', color=PAL['robot_body'], markersize=9, zorder=7)

        self._robot_arrow = None

        # laser hit points
        self._laser_pts, = self.ax.plot(
            [], [], '.', color=PAL['laser'],
            markersize=2.5, alpha=PAL['laser_alpha'], zorder=6)

        # beam fan
        self._beam_fan = LineCollection(
            [], colors=PAL['laser'], linewidths=0.3,
            alpha=PAL['beam_alpha'], zorder=2)
        self.ax.add_collection(self._beam_fan)

    # ----------------------------------------------------- per-frame updates

    def _update_robot(self, robot) -> None:
        self._robot_dot.set_data([robot.x], [robot.y])

        if self._robot_arrow is not None:
            self._robot_arrow.remove()

        arrow_len = 0.55
        dx = arrow_len * np.cos(robot.theta)
        dy = arrow_len * np.sin(robot.theta)
        self._robot_arrow = self.ax.annotate(
            '', xy=(robot.x + dx, robot.y + dy),
            xytext=(robot.x, robot.y),
            arrowprops=dict(arrowstyle='->', color=PAL['robot_arrow'],
                            lw=2.0, mutation_scale=14),
            zorder=8)

        self.ax.set_title(
            f'x={robot.x:6.2f} m   y={robot.y:6.2f} m   '
            f'theta={np.degrees(robot.theta):6.1f} deg',
            color='#64748b', fontsize=8, pad=4, fontfamily='monospace')

    def _update_trail(self, robot) -> None:
        if len(robot.trail) < 2:
            return
        trail = np.array(robot.trail)
        self._trail_line.set_data(trail[:, 0], trail[:, 1])

    def _update_laser(self, robot, scan) -> None:
        """
        Draw laser returns and beam fan from a ScanResult.

        scan.valid_hits  -> scatter of hit points (world frame)
        beam fan         -> lines from robot origin to each hit point
        """
        hits = scan.valid_hits          # (M, 2)

        if hits.shape[0] > 0:
            self._laser_pts.set_data(hits[:, 0], hits[:, 1])
        else:
            self._laser_pts.set_data([], [])

        # build beam segments: [[origin, hit], ...]  shape (M, 2, 2)
        origin = np.array([robot.x, robot.y])
        if hits.shape[0] > 0:
            origins  = np.tile(origin, (hits.shape[0], 1))   # (M, 2)
            segments = np.stack([origins, hits], axis=1)      # (M, 2, 2)
            self._beam_fan.set_segments(segments)
        else:
            self._beam_fan.set_segments([])

    # --------------------------------------------------- Phase 4/5 stubs

    def _draw_slam_map(self, segments, corners) -> None:
        """Filled in Phase 4."""
        pass

    def _draw_mc(self, mc_points) -> None:
        """Filled in Phase 5."""
        pass