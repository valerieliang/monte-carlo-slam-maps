"""
main.py
-------
Simulation entry point.

Phase 1: static world + manual keyboard drive.
Phase 2: live laser scan overlay.
Phase 3: synthetic feature extraction overlay.
Phase 4: EKF-SLAM predict + update + covariance ellipses
Phase 5: Monte Carlo uncertainty heatmap  (ACTIVE)

Controls
--------
  W / up    : forward        A / left  : turn left
  S / down  : backward       D / right : turn right
  Q         : quit           R         : reset robot
  1/2/3     : switch preset  (lab / corridor / open)
  F         : toggle feature overlay on/off
  M         : toggle SLAM map overlay on/off
  U         : toggle MC uncertainty heatmap on/off
  H         : print help

Run
---
  python main.py
  python main.py --world corridor
"""

from __future__ import annotations
import argparse, sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config            import Config
from env.world         import World
from env.robot         import Robot
from env.sensor        import Sensor
from slam.features     import FeatureExtractor
from slam.state        import SLAMState
from slam.predict      import predict
from slam.update       import update_single, init_corner, init_line
from slam.data_assoc   import associate_observations
from viz.renderer      import Renderer
from viz.covariance    import draw_slam_features
from viz.heatmap       import HeatmapRenderer, draw_goal
from montecarlo.uncertainty_map import build_uncertainty_map


# ----------------------------------------------------------------- key state

class KeyState:
    def __init__(self):
        self._held: set[str] = set()
        self.quit            = False
        self.reset           = False
        self.preset: str | None = None
        self.toggle_features = False
        self.toggle_slam     = False
        self.toggle_heatmap  = False

    def on_press(self, event) -> None:
        k = (event.key or '').lower()
        self._held.add(k)
        if k == 'q':
            self.quit = True
        elif k == 'r':
            self.reset = True
        elif k in ('1', '2', '3'):
            self.preset = {'1': 'lab', '2': 'corridor', '3': 'open'}[k]
        elif k == 'f':
            self.toggle_features = True
        elif k == 'm':
            self.toggle_slam = True
        elif k == 'u':
            self.toggle_heatmap = True
        elif k == 'h':
            print(__doc__)

    def on_release(self, event) -> None:
        self._held.discard((event.key or '').lower())

    def compute_command(self, max_v, max_omega):
        v = omega = 0.0
        if 'w' in self._held or 'up'    in self._held: v     += max_v
        if 's' in self._held or 'down'  in self._held: v     -= max_v
        if 'a' in self._held or 'left'  in self._held: omega += max_omega
        if 'd' in self._held or 'right' in self._held: omega -= max_omega
        return v, omega


# ---------------------------------------------------------------- sim factory

def build_sim(cfg: Config, preset: str | None = None):
    p         = preset or cfg.world.preset
    world     = World.from_preset(p)
    robot     = Robot(cfg.robot.start_x, cfg.robot.start_y,
                      cfg.robot.start_theta, cfg.robot.radius)
    sensor    = Sensor.from_cfg(cfg.sensor)
    extractor = FeatureExtractor.from_cfg(cfg)
    renderer  = Renderer(world,
                         figsize=tuple(cfg.renderer.figsize),
                         dpi=cfg.renderer.dpi,
                         title=f'Active SLAM — Phase 4  [{p}]')
    # EKF-SLAM state: start with zero initial pose covariance (known start)
    slam_state = SLAMState(
        init_pose     = robot.pose,
        init_pose_cov = np.zeros((3, 3)),
    )
    return world, robot, sensor, extractor, renderer, slam_state


def _build_R(cfg) -> tuple:
    R_corner = np.diag([cfg.ekf.R_range**2, cfg.ekf.R_bearing**2])
    R_line   = np.diag([cfg.ekf.R_range**2, cfg.ekf.R_bearing**2])
    return R_corner, R_line


# ----------------------------------------------------------------- main loop

def run(cfg: Config, preset: str | None = None) -> None:
    world, robot, sensor, extractor, renderer, slam_state = build_sim(cfg, preset)
    renderer.init()

    keys            = KeyState()
    show_features   = True
    show_slam       = True
    show_heatmap    = False   # U to toggle on

    renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
    renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)

    plt.ion()
    plt.show(block=False)

    dt         = cfg.sim.dt
    frame_dur  = 1.0 / cfg.sim.render_fps
    last_frame = time.perf_counter()

    R_corner, R_line = _build_R(cfg)

    last_scan                 = sensor.scan(robot, world)
    last_corners, last_lines  = extractor.extract(robot.pose, world, last_scan)

    # SLAM feature artists (re-drawn each frame on the axes)
    slam_artists  = []

    # Phase 5 — heatmap renderer and MC update cadence
    heatmap       = HeatmapRenderer(renderer.ax)
    mc_rng        = np.random.default_rng()
    mc_every      = max(1, int(1.0 / (cfg.sim.dt * cfg.sim.render_fps)))  # every ~1 s
    mc_counter    = 0
    goal_artists  = []

    print("[main] Phase 5 running.  U toggles MC heatmap.  M = SLAM.  Q = quit.")

    while plt.fignum_exists(renderer.fig.number):
        t0 = time.perf_counter()
        renderer.fig.canvas.flush_events()

        if keys.quit:
            break

        if keys.toggle_features:
            show_features        = not show_features
            keys.toggle_features = False
            print(f"[main] Feature overlay {'ON' if show_features else 'OFF'}")

        if keys.toggle_slam:
            show_slam        = not show_slam
            keys.toggle_slam = False
            print(f"[main] SLAM map {'ON' if show_slam else 'OFF'}")

        if keys.toggle_heatmap:
            show_heatmap        = not show_heatmap
            keys.toggle_heatmap = False
            if not show_heatmap:
                heatmap.clear()
            print(f"[main] MC heatmap {'ON' if show_heatmap else 'OFF'}")

        if keys.reset:
            robot.reset(cfg.robot.start_x, cfg.robot.start_y,
                        cfg.robot.start_theta)
            slam_state = SLAMState(init_pose=robot.pose,
                                   init_pose_cov=np.zeros((3, 3)))
            heatmap.clear()
            goal_artists = draw_goal(renderer.ax, None, goal_artists)
            mc_counter   = 0
            keys.reset   = False
            print("[main] Robot + SLAM state reset.")

        if keys.preset is not None:
            renderer.close()
            world, robot, sensor, extractor, renderer, slam_state = \
                build_sim(cfg, keys.preset)
            renderer.init()
            renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
            renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)
            plt.show(block=False)
            keys.preset   = None
            keys._held.clear()
            slam_artists  = []
            heatmap       = HeatmapRenderer(renderer.ax)
            goal_artists  = []
            mc_counter    = 0
            continue

        # -- physics + sensing -----------------------------------------------
        v, omega = keys.compute_command(cfg.robot.max_v, cfg.robot.max_omega)
        robot.step(v, omega, dt)

        # -- EKF predict ------------------------------------------------------
        predict(slam_state, v, omega, dt,
                cfg.ekf.Q_v, cfg.ekf.Q_w)

        # -- sense + extract --------------------------------------------------
        last_scan              = sensor.scan(robot, world)
        last_corners, last_lines = extractor.extract(robot.pose, world, last_scan)
        all_obs = list(last_corners) + list(last_lines)

        # -- data association + update ----------------------------------------
        associations = associate_observations(
            slam_state, all_obs, R_corner, R_line, cfg.ekf.gate_chi2)

        for obs, feat_idx in associations:
            R = R_corner if obs.feature_kind == 'corner' else R_line
            if feat_idx == -1:
                # new feature — initialise
                if obs.feature_kind == 'corner':
                    init_corner(slam_state, obs.z, R,
                                cfg.ekf.init_cov_corner)
                else:
                    init_line(slam_state, obs.z, R,
                              cfg.ekf.init_cov_line,
                              seg_p0=obs.seg_p0, seg_p1=obs.seg_p1)
            else:
                # update existing
                update_single(slam_state, feat_idx, obs.z, R)

        # -- MC uncertainty map (every ~1 s when heatmap visible) -----------
        mc_counter += 1
        if show_heatmap and slam_state.n_features > 0 and mc_counter >= mc_every:
            mc_counter = 0
            umap = build_uncertainty_map(
                state          = slam_state,
                world          = world,
                robot_pos      = robot.pos,
                n_samples      = cfg.montecarlo.n_samples_local,
                robot_radius   = cfg.robot.radius,
                virtual_cov    = cfg.montecarlo.virtual_cov,
                uncertainty_lo = cfg.montecarlo.uncertainty_lo,
                uncertainty_hi = cfg.montecarlo.uncertainty_hi,
                rng            = mc_rng,
            )
            heatmap.update(umap)

            # Pick navigation target: max P(p)/distance among uncertain pts
            if len(umap.uncertain_points) > 0:
                goal = _select_goal(umap, robot.pos)
                goal_artists = draw_goal(renderer.ax, goal, goal_artists)

        # -- render at target FPS --------------------------------------------
        now = time.perf_counter()
        if now - last_frame >= frame_dur:
            # remove old SLAM artists
            for art in slam_artists:
                try:
                    art.remove()
                except Exception:
                    pass
            slam_artists = []

            renderer.update(
                robot,
                laser_scan  = last_scan,
                corner_obs  = last_corners if show_features else [],
                line_obs    = last_lines   if show_features else [],
            )

            # draw SLAM map on top
            if show_slam and slam_state.n_features > 0:
                slam_artists = _draw_slam(renderer.ax, slam_state)

            plt.pause(0.001)
            last_frame = now

        elapsed = time.perf_counter() - t0
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    renderer.close()
    print("[main] Simulation ended.")


# ---------------------------------------------------------------- SLAM draw

def _draw_slam(ax, state: SLAMState) -> list:
    """
    Draw SLAM-estimated features and their 2-sigma covariance ellipses.

    Visual guide
    ------------
    Corners  (red  ✕ + ellipse):
        The ✕ marks the EKF's best estimate of a corner's world position.
        The ellipse shows 2-sigma positional uncertainty — it shrinks as the
        robot re-observes the corner from different angles.

    Lines  (teal diamond ◆ + dashed outline):
        The ◆ marks the midpoint of the EKF's estimated wall segment.
        A dashed teal line is drawn along the segment from its stored
        endpoints.  The uncertainty ellipse is drawn at the segment midpoint
        and represents uncertainty in (rho, alpha) mapped to XY.
        Lines are drawn with a dashed style so they don't obscure the
        ground-truth white walls underneath.

    Returns list of matplotlib artists so they can be removed next frame.
    """
    from viz.covariance import cov_ellipse

    CORNER_COLOR = '#f87171'   # red
    LINE_COLOR   = '#34d399'   # teal

    artists = []
    for feat in state.features:
        mean = state.feature_mean(feat.idx)
        cov  = state.feature_cov(feat.idx)

        if feat.kind == 'corner':
            # ── corner: X marker at estimated position + uncertainty ellipse ──
            pt, = ax.plot(mean[0], mean[1], 'x',
                          color=CORNER_COLOR, markersize=11,
                          markeredgewidth=2.5, zorder=12)
            artists.append(pt)

            ell = cov_ellipse(mean, cov, ax,
                              edgecolor=CORNER_COLOR, linewidth=1.2,
                              alpha=0.45, zorder=11)
            if ell is not None:
                artists.append(ell)

        else:  # line
            # ── line: dashed segment + diamond midpoint + uncertainty ellipse ──
            rho, alpha = mean

            if feat.seg_p0 is not None and feat.seg_p1 is not None:
                # Draw as dashed line along stored segment endpoints
                ln, = ax.plot([feat.seg_p0[0], feat.seg_p1[0]],
                              [feat.seg_p0[1], feat.seg_p1[1]],
                              '--', color=LINE_COLOR, linewidth=1.8,
                              alpha=0.75, zorder=7)   # zorder < walls (3) avoidance
                artists.append(ln)
                mid = 0.5 * (feat.seg_p0 + feat.seg_p1)
            else:
                # Fallback: no segment stored, use foot-of-perpendicular
                mid = np.array([rho * np.cos(alpha), rho * np.sin(alpha)])

            # Diamond marker at segment midpoint
            pt, = ax.plot(mid[0], mid[1], 'D',
                          color=LINE_COLOR, markersize=6,
                          markeredgewidth=0, alpha=0.9, zorder=12)
            artists.append(pt)

            # Uncertainty ellipse centred at midpoint (not foot-of-perp).
            # We project the (rho, alpha) covariance into XY at the midpoint
            # via a 2x2 Jacobian: d[x,y]/d[rho,alpha] at mid.
            ca, sa  = np.cos(alpha), np.sin(alpha)
            J       = np.array([[ca, -rho * sa],
                                 [sa,  rho * ca]])
            cov_xy  = J @ cov @ J.T
            # clamp to avoid degenerate ellipses from near-zero rho
            cov_xy += np.eye(2) * 0.01
            ell = cov_ellipse(mid, cov_xy, ax,
                              edgecolor=LINE_COLOR, linewidth=0.9,
                              linestyle='--', alpha=0.35, zorder=10)
            if ell is not None:
                artists.append(ell)

    return artists


# ---------------------------------------------------------------- goal selector

def _select_goal(umap, robot_pos: np.ndarray) -> np.ndarray:
    """
    Choose the navigation target from uncertain MC points (Eq. 11).

    Returns the point maximising P(p) / distance(robot, p).
    """
    pts    = umap.uncertain_points
    scores = umap.uncertain_scores
    dists  = np.linalg.norm(pts - robot_pos, axis=1)
    dists  = np.maximum(dists, 0.1)   # avoid div-by-zero
    metric = scores / dists
    return pts[np.argmax(metric)]


# ---------------------------------------------------------------------- CLI

def parse_args():
    p = argparse.ArgumentParser(description='Active SLAM 2-D Simulator')
    p.add_argument('--world',  choices=['lab', 'corridor', 'open'],
                   default=None)
    p.add_argument('--config', default='config.yaml')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = Config.load(args.config)
    run(cfg, preset=args.world)