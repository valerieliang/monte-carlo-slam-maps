"""
viz/renderer.py
---------------
Bird's-eye 2-D renderer using matplotlib.

Phase 1: static world + manual keyboard drive.
Phase 2: live laser scan overlay.
Phase 3: extracted feature overlays.
Phase 4: SLAM map overlay + covariance ellipses  (ACTIVE)
Phase 5: MC uncertainty heatmap (stub)

Controls (set up in main.py)
-----------------------------
  F  : toggle extracted-feature overlay (observed corners / wall midpoints)
  M  : toggle SLAM map overlay (estimated corners / walls + uncertainty)
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
    # Phase 3 — extracted features (raw per-frame observations)
    feat_corner  = '#4ade80',   # green  — observed corner hit
    feat_line    = '#f472b6',   # pink   — observed wall midpoint tick
    feat_alpha   = 0.85,
    # Phase 4 — SLAM map
    slam_corner  = '#f87171',   # red    — EKF corner estimate
    slam_line    = '#34d399',   # teal   — EKF wall estimate
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
        self._feat_corner_pts   = None
        self._feat_line_pts     = None

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
        # Ground-truth walls
        for seg in self.world.segments:
            self.ax.plot([seg.p0[0], seg.p1[0]],
                         [seg.p0[1], seg.p1[1]],
                         color=PAL['wall'], linewidth=2.0,
                         alpha=PAL['wall_alpha'], solid_capstyle='round',
                         zorder=3)

        # Ground-truth corners
        for c in self.world.corners:
            col = PAL['corner_cvx'] if c.kind == 'convex' else PAL['corner_ccv']
            self.ax.plot(c.pos[0], c.pos[1],
                         marker='s', markersize=7,
                         markerfacecolor='none', markeredgecolor=col,
                         markeredgewidth=1.6, alpha=PAL['corner_alpha'],
                         zorder=4)

        # ── Legend ───────────────────────────────────────────────────────────
        # Ground truth
        gt_entries = [
            mlines.Line2D([], [], color=PAL['wall'], linewidth=2,
                          label='wall — ground truth'),
            mlines.Line2D([], [], color=PAL['corner_cvx'], marker='s',
                          linestyle='none', markerfacecolor='none',
                          markeredgewidth=1.5, markersize=6,
                          label='convex corner — GT'),
            mlines.Line2D([], [], color=PAL['corner_ccv'], marker='s',
                          linestyle='none', markerfacecolor='none',
                          markeredgewidth=1.5, markersize=6,
                          label='concave corner — GT'),
        ]

        # Robot & sensor
        sensor_entries = [
            mlines.Line2D([], [], color=PAL['robot_body'], marker='o',
                          linestyle='none', markersize=6,
                          label='robot pose'),
            mlines.Line2D([], [], color=PAL['trail'],
                          linewidth=1.2, alpha=0.6,
                          label='robot trail'),
            mlines.Line2D([], [], color=PAL['laser'], marker='.',
                          linestyle='none', markersize=5,
                          label='laser returns'),
        ]

        # Raw per-frame feature observations  (F to toggle)
        feat_entries = [
            mlines.Line2D([], [], color=PAL['feat_corner'], marker='o',
                          linestyle='none', markersize=7,
                          markerfacecolor='none', markeredgewidth=2.0,
                          label='observed corner — this frame'),
            mlines.Line2D([], [], color=PAL['feat_line'], marker='P',
                          linestyle='none', markersize=7,
                          label='observed wall midpoint — this frame'),
        ]

        # SLAM map overlay  (M to toggle)
        slam_entries = [
            mlines.Line2D([], [], color=PAL['slam_corner'], marker='x',
                          linestyle='none', markersize=9,
                          markeredgewidth=2.5,
                          label='SLAM corner estimate  ✕'),
            mlines.Line2D([], [], color=PAL['slam_corner'],
                          marker='o', markersize=8, linestyle='none',
                          markerfacecolor='none', markeredgewidth=1.0,
                          alpha=0.5,
                          label='  └ 2σ position uncertainty'),
            mlines.Line2D([], [], color=PAL['slam_line'],
                          linestyle='--', linewidth=1.8, alpha=0.8,
                          label='SLAM wall estimate  (dashed)'),
            mlines.Line2D([], [], color=PAL['slam_line'], marker='D',
                          linestyle='none', markersize=6,
                          label='  └ wall midpoint  ◆'),
            mlines.Line2D([], [], color=PAL['slam_line'],
                          marker='o', markersize=8, linestyle='none',
                          markerfacecolor='none', markeredgewidth=1.0,
                          alpha=0.4,
                          label='  └ 2σ wall uncertainty'),
        ]

        leg = gt_entries + sensor_entries + feat_entries + slam_entries
        self.ax.legend(
            handles        = leg,
            loc            = 'upper right',
            fontsize       = 7,
            framealpha     = 0.30,
            facecolor      = '#1e2130',
            edgecolor      = '#334155',
            labelcolor     = '#94a3b8',
            title          = 'Legend    F = features    M = SLAM map',
            title_fontsize = 7,
        )

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
        Back-project extracted feature observations into world frame.

        Corner: world_pos = robot_pos + r * [cos(theta+beta), sin(theta+beta)]
        Line:   closest point on observed line to robot (foot of perpendicular)
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
        # Use cluster_midpoint when available (marks centre of observed wall
        # section).  Fall back to foot-of-perpendicular from the robot.
        if line_obs:
            lxs, lys = [], []
            for obs in line_obs:
                if obs.cluster_midpoint is not None:
                    lxs.append(obs.cluster_midpoint[0])
                    lys.append(obs.cluster_midpoint[1])
                else:
                    rho_obs, alpha_obs = obs.z
                    alpha_world = _wrap(alpha_obs + rt)
                    lxs.append(rx + rho_obs * np.cos(alpha_world))
                    lys.append(ry + rho_obs * np.sin(alpha_world))
            self._feat_line_pts.set_data(lxs, lys)
        else:
            self._feat_line_pts.set_data([], [])

    # --------------------------------------------------- Phase 5 stub

    def _draw_mc(self, mc_points) -> None:
        pass


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi