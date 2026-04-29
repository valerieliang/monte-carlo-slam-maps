"""
viz/renderer.py
---------------
Bird's-eye 2-D renderer using matplotlib.

Phase 1: walls, corners, robot pose + trail
Phase 2: laser beam fan + hit-point cloud
Phase 3: extracted feature overlays  (ACTIVE)
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
    # Phase 3 — extracted features
    feat_corner  = '#4ade80',   # green  — observed corner hit
    feat_line    = '#f472b6',   # pink   — observed line midpoint tick
    feat_alpha   = 0.85,
    # Phase 4 stubs
    slam_wall    = '#34d399',
    slam_corner  = '#f87171',
    # Phase 5 stub
    mc_low       = '#1e3a5f',
    mc_high      = '#f59e0b',
)


class Renderer:
    """
    Persistent matplotlib figure for real-time bird's-eye display.

    update() accepts:
      laser_scan    : ScanResult        (Phase 2+)
      corner_obs    : list[CornerObs]   (Phase 3+)
      line_obs      : list[LineObs]     (Phase 3+)
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

        self._robot_arrow       = None
        self._robot_dot         = None
        self._trail_line        = None
        self._laser_pts         = None
        self._beam_fan          = None
        self._feat_corner_pts   = None   # Phase 3
        self._feat_line_pts     = None   # Phase 3

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
               corner_obs=None,
               line_obs=None,
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

        self._update_features(robot, corner_obs or [], line_obs or [])

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
                         color=PAL['wall'], linewidth=2.0,
                         alpha=PAL['wall_alpha'], solid_capstyle='round',
                         zorder=3)

        for c in self.world.corners:
            col = PAL['corner_cvx'] if c.kind == 'convex' else PAL['corner_ccv']
            self.ax.plot(c.pos[0], c.pos[1],
                         marker='s', markersize=7,
                         markerfacecolor='none', markeredgecolor=col,
                         markeredgewidth=1.6, alpha=PAL['corner_alpha'],
                         zorder=4)

        leg = [
            mlines.Line2D([], [], color=PAL['wall'], linewidth=2,
                          label='wall (GT)'),
            mlines.Line2D([], [], color=PAL['corner_cvx'], marker='s',
                          linestyle='none', markerfacecolor='none',
                          markeredgewidth=1.5, markersize=6,
                          label='convex corner (GT)'),
            mlines.Line2D([], [], color=PAL['corner_ccv'], marker='s',
                          linestyle='none', markerfacecolor='none',
                          markeredgewidth=1.5, markersize=6,
                          label='concave corner (GT)'),
            mlines.Line2D([], [], color=PAL['robot_body'], marker='o',
                          linestyle='none', markersize=6, label='robot'),
            mlines.Line2D([], [], color=PAL['trail'],
                          linewidth=1.2, alpha=0.6, label='trail'),
            mlines.Line2D([], [], color=PAL['laser'], marker='.',
                          linestyle='none', markersize=5,
                          label='laser returns'),
            mlines.Line2D([], [], color=PAL['feat_corner'], marker='o',
                          linestyle='none', markersize=7,
                          label='observed corner'),
            mlines.Line2D([], [], color=PAL['feat_line'], marker='P',
                          linestyle='none', markersize=7,
                          label='observed line'),
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

        # Phase 2: laser
        self._laser_pts, = self.ax.plot(
            [], [], '.', color=PAL['laser'],
            markersize=2.5, alpha=PAL['laser_alpha'], zorder=6)

        self._beam_fan = LineCollection(
            [], colors=PAL['laser'], linewidths=0.3,
            alpha=PAL['beam_alpha'], zorder=2)
        self.ax.add_collection(self._beam_fan)

        # Phase 3: observed corner positions (world frame, back-projected)
        self._feat_corner_pts, = self.ax.plot(
            [], [], 'o',
            color=PAL['feat_corner'],
            markersize=8,
            markerfacecolor='none',
            markeredgewidth=2.0,
            alpha=PAL['feat_alpha'],
            zorder=9)

        # Phase 3: observed line representative points
        self._feat_line_pts, = self.ax.plot(
            [], [], 'P',
            color=PAL['feat_line'],
            markersize=7,
            markerfacecolor=PAL['feat_line'],
            markeredgewidth=0,
            alpha=PAL['feat_alpha'],
            zorder=9)

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
        hits   = scan.valid_hits
        origin = np.array([robot.x, robot.y])

        if hits.shape[0] > 0:
            self._laser_pts.set_data(hits[:, 0], hits[:, 1])
            origins  = np.tile(origin, (hits.shape[0], 1))
            segments = np.stack([origins, hits], axis=1)
            self._beam_fan.set_segments(segments)
        else:
            self._laser_pts.set_data([], [])
            self._beam_fan.set_segments([])

    def _update_features(self, robot, corner_obs, line_obs) -> None:
        """
        Back-project extracted feature observations into world frame for display.

        Corner: world_pos = robot_pos + r * [cos(theta+beta), sin(theta+beta)]
        Line:   closest point on observed line to robot (the foot of perpendicular)
                = rho_obs * [cos(alpha_obs + theta), sin(alpha_obs + theta)]
                  offset from robot
        """
        rx, ry, rt = robot.x, robot.y, robot.theta

        # ── corners ──────────────────────────────────────────────────────────
        if corner_obs:
            cxs, cys = [], []
            for obs in corner_obs:
                r, beta   = obs.z
                world_ang = rt + beta
                cxs.append(rx + r * np.cos(world_ang))
                cys.append(ry + r * np.sin(world_ang))
            self._feat_corner_pts.set_data(cxs, cys)
        else:
            self._feat_corner_pts.set_data([], [])

        # ── lines ─────────────────────────────────────────────────────────────
        if line_obs:
            lxs, lys = [], []
            for obs in line_obs:
                rho_obs, alpha_obs = obs.z
                # convert robot-frame alpha_obs back to world-frame line normal
                alpha_world = _wrap(alpha_obs + rt)
                # foot of perpendicular from world origin to the observed line,
                # shifted by rho_obs along the normal
                foot_x = rho_obs * np.cos(alpha_world)
                foot_y = rho_obs * np.sin(alpha_world)
                # convert from line-origin-relative to world (add robot offset)
                world_x = rx + foot_x
                world_y = ry + foot_y
                lxs.append(world_x)
                lys.append(world_y)
            self._feat_line_pts.set_data(lxs, lys)
        else:
            self._feat_line_pts.set_data([], [])

    # --------------------------------------------------- Phase 4/5 stubs

    def _draw_slam_map(self, segments, corners) -> None:
        pass

    def _draw_mc(self, mc_points) -> None:
        pass


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi