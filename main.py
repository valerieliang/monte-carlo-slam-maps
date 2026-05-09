"""
main.py
-------
Simulation entry point — Phases 1-6.

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
  P         : print path report / save path.csv
  (manual drive disabled in --auto mode)

Run
---
  python main.py                     # manual drive
  python main.py --auto              # autonomous exploration
  python main.py --auto --world corridor
"""

from __future__ import annotations

# Backend: Qt5 > Qt6 > TkAgg — chosen before any other plt import
import matplotlib, sys, os

def _pick_backend():
    """
    Try backends in order of performance.  matplotlib.use() alone doesn't
    validate that the Qt bindings are actually installed — we must attempt
    to import the backend module directly to confirm it works.
    """
    import importlib
    candidates = [
        ('Qt5Agg',  'matplotlib.backends.backend_qt5agg'),
        ('Qt6Agg',  'matplotlib.backends.backend_qtagg'),
        ('TkAgg',   'matplotlib.backends.backend_tkagg'),
        ('Agg',     'matplotlib.backends.backend_agg'),   # headless fallback
    ]
    for name, module in candidates:
        try:
            importlib.import_module(module)
            matplotlib.use(name)
            return name
        except Exception:
            continue
    return 'Agg'

_chosen_backend = _pick_backend()

import argparse, time
import numpy as np
import matplotlib.pyplot as plt

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
from viz.heatmap       import HeatmapRenderer
from viz.logger        import PathLogger
from montecarlo.uncertainty_map import build_uncertainty_map
from navigation.selector        import GoalSelector
from navigation.controller      import Controller, Mode


# ---------------------------------------------------------------------------
# Key state
# ---------------------------------------------------------------------------

class KeyState:
    def __init__(self):
        self._held: set[str]    = set()
        self.quit               = False
        self.reset              = False
        self.preset: str | None = None
        self.toggle_features    = False
        self.toggle_slam        = False
        self.toggle_heatmap     = False
        self.clicked_goal: np.ndarray | None = None
        self.dump_path: bool = False

    def on_press(self, event) -> None:
        k = (event.key or '').lower()
        self._held.add(k)
        if   k == 'q':           self.quit            = True
        elif k == 'r':           self.reset           = True
        elif k in ('1','2','3'): self.preset = {'1':'lab','2':'corridor','3':'open'}[k]
        elif k == 'f':           self.toggle_features = True
        elif k == 'm':           self.toggle_slam     = True
        elif k == 'u':           self.toggle_heatmap  = True
        elif k == 'h':           print(__doc__)
        elif k == 'p':           self.dump_path = True

    def on_click(self, event) -> None:
        """Left-click on the axes sets a manual goal position."""
        if event.button == 1 and event.inaxes is not None:
            self.clicked_goal = np.array([event.xdata, event.ydata])

    def on_release(self, event) -> None:
        self._held.discard((event.key or '').lower())

    def compute_command(self, max_v, max_omega):
        v = omega = 0.0
        if 'w' in self._held or 'up'    in self._held: v     += max_v
        if 's' in self._held or 'down'  in self._held: v     -= max_v
        if 'a' in self._held or 'left'  in self._held: omega += max_omega
        if 'd' in self._held or 'right' in self._held: omega -= max_omega
        return v, omega


# ---------------------------------------------------------------------------
# Sim factory
# ---------------------------------------------------------------------------

def build_sim(cfg: Config, preset: str | None = None, auto: bool = False):
    p          = preset or cfg.world.preset
    world      = World.from_preset(p)
    robot      = Robot(cfg.robot.start_x, cfg.robot.start_y,
                       cfg.robot.start_theta, cfg.robot.radius)
    sensor     = Sensor.from_cfg(cfg.sensor)
    extractor  = FeatureExtractor.from_cfg(cfg)
    phase      = "6 - Autonomous" if auto else "5"
    renderer   = Renderer(world,
                          figsize=tuple(cfg.renderer.figsize),
                          dpi=cfg.renderer.dpi,
                          title=f'Active SLAM - Phase {phase}  [{p}]')
    slam_state = SLAMState(init_pose=robot.pose,
                           init_pose_cov=np.zeros((3, 3)))
    selector   = GoalSelector(
        local_area_size = cfg.montecarlo.local_area_size,
        min_dist        = 1.0,
        dedup_dist      = cfg.navigation.duplicate_thresh,
    )
    controller = Controller(
        k_v            = cfg.navigation.controller_k_v,
        k_w            = cfg.navigation.controller_k_w,
        max_v          = cfg.robot.max_v,
        max_omega      = cfg.robot.max_omega,
        goal_tolerance = cfg.navigation.goal_tolerance,
    )
    return world, robot, sensor, extractor, renderer, slam_state, selector, controller


def _build_R(cfg):
    R = np.diag([cfg.ekf.R_range**2, cfg.ekf.R_bearing**2])
    return R, R


# ---------------------------------------------------------------------------
# SLAM overlay
# ---------------------------------------------------------------------------

def _draw_slam(ax, state: SLAMState) -> list:
    """Draw EKF features + 2-sigma ellipses.  Returns artist list."""
    from viz.covariance import cov_ellipse
    CORNER = '#f87171'
    LINE   = '#34d399'
    arts   = []

    for feat in state.features:
        mean = state.feature_mean(feat.idx)
        cov  = state.feature_cov(feat.idx)

        if feat.kind == 'corner':
            pt, = ax.plot(mean[0], mean[1], 'x',
                          color=CORNER, markersize=11,
                          markeredgewidth=2.5, zorder=12)
            arts.append(pt)
            ell = cov_ellipse(mean, cov, ax,
                              edgecolor=CORNER, linewidth=1.2,
                              alpha=0.45, zorder=11)
            if ell is not None:
                arts.append(ell)
        else:
            rho, alpha = mean
            if feat.seg_p0 is not None and feat.seg_p1 is not None:
                ln, = ax.plot([feat.seg_p0[0], feat.seg_p1[0]],
                              [feat.seg_p0[1], feat.seg_p1[1]],
                              '--', color=LINE, linewidth=1.8,
                              alpha=0.75, zorder=7)
                arts.append(ln)
                mid = 0.5 * (feat.seg_p0 + feat.seg_p1)
            else:
                mid = np.array([rho * np.cos(alpha), rho * np.sin(alpha)])

            pt, = ax.plot(mid[0], mid[1], 'D',
                          color=LINE, markersize=6,
                          markeredgewidth=0, alpha=0.9, zorder=12)
            arts.append(pt)

            ca, sa = np.cos(alpha), np.sin(alpha)
            J      = np.array([[ca, -rho*sa], [sa, rho*ca]])
            cov_xy = J @ cov @ J.T + np.eye(2) * 0.01
            ell = cov_ellipse(mid, cov_xy, ax,
                              edgecolor=LINE, linewidth=0.9,
                              linestyle='--', alpha=0.35, zorder=10)
            if ell is not None:
                arts.append(ell)

    return arts


def _clear_artists(artists: list) -> list:
    for a in artists:
        try:
            a.remove()
        except Exception:
            pass
    return []


def _draw_goal_ring(ax, goal: np.ndarray, artists: list) -> list:
    artists = _clear_artists(artists)
    if goal is None:
        return []
    ring, = ax.plot(goal[0], goal[1], 'o',
                    color='#f59e0b', markersize=18,
                    markerfacecolor='none', markeredgewidth=2.5, zorder=15)
    dot,  = ax.plot(goal[0], goal[1], 'o',
                    color='#f59e0b', markersize=5, zorder=16)
    return [ring, dot]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: Config, preset: str | None = None, auto: bool = False) -> None:

    world, robot, sensor, extractor, renderer, slam_state, selector, controller = \
        build_sim(cfg, preset, auto=auto)
    renderer.init()

    keys          = KeyState()
    show_features = True
    show_slam     = True
    show_heatmap  = auto

    renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
    renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)
    renderer.fig.canvas.mpl_connect('button_press_event', keys.on_click)

    plt.ion()
    plt.show(block=False)

    dt         = cfg.sim.dt
    frame_dur  = 1.0 / cfg.sim.render_fps
    last_frame = time.perf_counter()

    R_corner, R_line = _build_R(cfg)

    last_scan              = sensor.scan(robot, world)
    last_corners, last_lines = extractor.extract(robot.pose, world, last_scan)

    # SLAM overlay — only redrawn when map actually changes
    slam_artists    = []
    slam_feat_count = 0
    slam_obs_total  = 0
    slam_force_draw = False

    # Heatmap
    heatmap            = HeatmapRenderer(renderer.ax)
    mc_rng             = np.random.default_rng()
    # Run MC once per simulated second: round(1s / dt) steps.
    # The old formula divided by render_fps too, giving mc_every=1
    # when dt=0.05 and fps=20, which ran MC every single step.
    mc_every           = max(1, round(1.0 / cfg.sim.dt))
    mc_counter         = 0
    last_umap          = None
    heatmap_force_draw = False

    # Path logger
    path_log         = PathLogger(world)
    path_log_artists = False   # True once overlay drawn

    # Phase 6 state
    start_pos    = robot.pos.copy()
    returning    = False
    nav_goal_art = []

    renderer.update_legend(show_heatmap=show_heatmap,
                           show_features=show_features,
                           show_slam=show_slam)
    if auto:
        print("[main] Autonomous mode.  U=heatmap  M=SLAM  Q=quit.")
    else:
        print("[main] Manual drive.  WASD=move  U=heatmap  M=SLAM  Q=quit.")

    while plt.fignum_exists(renderer.fig.number):
        t0 = time.perf_counter()
        renderer.fig.canvas.flush_events()

        if keys.quit:
            break

        # -- Toggles ----------------------------------------------------------

        if keys.toggle_features:
            show_features        = not show_features
            keys.toggle_features = False
            renderer.update_legend(show_heatmap=show_heatmap,
                                   show_features=show_features,
                                   show_slam=show_slam)
            print(f"[main] Features {'ON' if show_features else 'OFF'}")

        if keys.toggle_slam:
            show_slam        = not show_slam
            keys.toggle_slam = False
            if not show_slam:
                slam_artists    = _clear_artists(slam_artists)
                slam_feat_count = 0
                slam_obs_total  = 0
            else:
                slam_force_draw = True
            renderer.update_legend(show_heatmap=show_heatmap,
                                   show_features=show_features,
                                   show_slam=show_slam)
            print(f"[main] SLAM map {'ON' if show_slam else 'OFF'}")

        if keys.toggle_heatmap:
            show_heatmap        = not show_heatmap
            keys.toggle_heatmap = False
            if not show_heatmap:
                heatmap.clear()
            elif last_umap is not None:
                heatmap_force_draw = True
            renderer.update_legend(show_heatmap=show_heatmap,
                                   show_features=show_features,
                                   show_slam=show_slam)
            print(f"[main] Heatmap {'ON' if show_heatmap else 'OFF'}")

        # -- Reset ------------------------------------------------------------

        if keys.reset:
            robot.reset(cfg.robot.start_x, cfg.robot.start_y,
                        cfg.robot.start_theta)
            slam_state      = SLAMState(init_pose=robot.pose,
                                        init_pose_cov=np.zeros((3, 3)))
            slam_artists    = _clear_artists(slam_artists)
            slam_feat_count = 0
            slam_obs_total  = 0
            heatmap.clear()
            selector.reset()
            controller.set_goal(None)
            nav_goal_art = _clear_artists(nav_goal_art)
            returning    = False
            last_umap    = None
            mc_counter   = 0
            keys.reset   = False
            path_log.reset()
            path_log.clear_overlay()
            print("[main] Reset.")

        # -- Preset switch ----------------------------------------------------

        if keys.preset is not None:
            renderer.close()
            world, robot, sensor, extractor, renderer, slam_state, selector, controller = \
                build_sim(cfg, keys.preset, auto=auto)
            renderer.init()
            renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
            renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)
            renderer.fig.canvas.mpl_connect('button_press_event', keys.on_click)
            plt.show(block=False)
            keys.preset        = None
            keys._held.clear()
            slam_artists       = []
            slam_feat_count    = 0
            slam_obs_total     = 0
            slam_force_draw    = False
            heatmap            = HeatmapRenderer(renderer.ax)
            heatmap_force_draw = False
            last_umap          = None
            nav_goal_art       = []
            returning          = False
            mc_counter         = 0
            path_log           = PathLogger(world)
            path_log.clear_overlay()
            show_features      = True
            show_slam          = True
            show_heatmap       = auto
            renderer.update_legend(show_heatmap=auto,
                                   show_features=True,
                                   show_slam=True)
            continue

        # -- Path dump
        if keys.dump_path:
            keys.dump_path = False
            path_log.report()
            path_log.save('path.csv')
            path_log.clear_overlay()
            path_log.draw(renderer.ax)
            renderer.fig.canvas.draw_idle()

        # -- Click-to-goal ----------------------------------------------------
        if keys.clicked_goal is not None:
            goal = keys.clicked_goal
            keys.clicked_goal = None
            controller.set_goal(goal)
            # Interrupt any autonomous return-to-start
            returning    = False
            nav_goal_art = _draw_goal_ring(renderer.ax, goal, nav_goal_art)
            print(f"[click] Goal set: ({goal[0]:.1f}, {goal[1]:.1f})")

        # -- Physics ----------------------------------------------------------

        if auto:
            v, omega = controller.step(slam_state.pose, world, dt)
        else:
            v, omega = keys.compute_command(cfg.robot.max_v, cfg.robot.max_omega)

        robot.step(v, omega, dt)
        path_log.record(robot.pos, robot.theta)

        # -- EKF predict ------------------------------------------------------

        predict(slam_state, v, omega, dt, cfg.ekf.Q_v, cfg.ekf.Q_w)

        # -- Sense + extract --------------------------------------------------

        last_scan              = sensor.scan(robot, world)
        last_corners, last_lines = extractor.extract(robot.pose, world, last_scan)
        all_obs                  = list(last_corners) + list(last_lines)

        # -- EKF update -------------------------------------------------------

        associations = associate_observations(
            slam_state, all_obs, R_corner, R_line, cfg.ekf.gate_chi2)

        for obs, feat_idx in associations:
            R = R_corner if obs.feature_kind == 'corner' else R_line
            if feat_idx == -1:
                if obs.feature_kind == 'corner':
                    init_corner(slam_state, obs.z, R, cfg.ekf.init_cov_corner)
                else:
                    init_line(slam_state, obs.z, R, cfg.ekf.init_cov_line,
                              seg_p0=obs.seg_p0, seg_p1=obs.seg_p1)
            else:
                update_single(slam_state, feat_idx, obs.z, R)

        # -- MC uncertainty map  (every ~1 s) ---------------------------------

        mc_counter += 1
        run_mc = (slam_state.n_features > 0
                  and mc_counter >= mc_every
                  and (auto or show_heatmap))
        if run_mc:
            mc_counter = 0
            last_umap  = build_uncertainty_map(
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
            if show_heatmap:
                heatmap.update(last_umap)
                heatmap_force_draw = False

        # -- Autonomous navigation --------------------------------------------

        if auto and last_umap is not None:
            if returning:
                controller.set_goal(start_pos)
                if controller.goal_reached(robot.pos):
                    controller.set_goal(None)
                    returning    = False
                    nav_goal_art = _clear_artists(nav_goal_art)
                    print("[auto] Returned to start.  Exploration complete.")
            elif controller.goal_reached(robot.pos):
                selector.notify_goal_reached(robot.pos)
                nav_goal_art = _clear_artists(nav_goal_art)
                controller.set_goal(None)

            if not returning and controller.goal is None:
                result = selector.select(last_umap, robot.pos)
                if result.source == 'complete':
                    print("[auto] Map complete - returning to start.")
                    returning = True
                    controller.set_goal(start_pos)
                else:
                    controller.set_goal(result.goal)
                    nav_goal_art = _draw_goal_ring(
                        renderer.ax, result.goal, nav_goal_art)
                    print(f"[auto] Goal ({result.source}): "
                          f"({result.goal[0]:.1f}, {result.goal[1]:.1f})  "
                          f"n_uncertain={result.n_uncertain}")

        # -- Render at target FPS ---------------------------------------------

        now = time.perf_counter()
        if now - last_frame >= frame_dur:

            # SLAM overlay: only recreate artists when map has changed.
            # Avoids creating/destroying dozens of matplotlib objects every frame.
            if show_slam and slam_state.n_features > 0:
                feat_count  = slam_state.n_features
                obs_total   = sum(f.obs_count for f in slam_state.features)
                map_changed = (feat_count != slam_feat_count
                               or obs_total != slam_obs_total
                               or slam_force_draw)
                if map_changed:
                    slam_artists    = _clear_artists(slam_artists)
                    slam_artists    = _draw_slam(renderer.ax, slam_state)
                    slam_feat_count = feat_count
                    slam_obs_total  = obs_total
                    slam_force_draw = False

            # Heatmap: re-render if toggle just turned on with existing data
            if show_heatmap and heatmap_force_draw and last_umap is not None:
                heatmap.update(last_umap)
                heatmap_force_draw = False

            renderer.update(
                robot,
                laser_scan = last_scan,
                corner_obs = last_corners if show_features else [],
                line_obs   = last_lines   if show_features else [],
            )

            plt.pause(0.001)
            last_frame = now

        elapsed = time.perf_counter() - t0
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    path_log.report()
    renderer.close()
    print("[main] Simulation ended.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Active SLAM 2-D Simulator')
    p.add_argument('--world',  choices=['lab', 'corridor', 'open'], default=None)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--auto',   action='store_true',
                   help='Enable autonomous exploration (Phase 6)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = Config.load(args.config)
    run(cfg, preset=args.world, auto=args.auto)