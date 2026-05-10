"""
slam_analytics.py
-----------------
Headless evaluation harness for the EKF-SLAM simulator.

Runs a full autonomous simulation, records ground-truth vs estimated
quantities at every step, and produces a multi-panel report with:

  1.  Pose estimation error   (x, y, θ over time)
  2.  NEES consistency test   (Normalised Estimation Error Squared)
  3.  Feature map error       (per-feature, corner and line separately)
  4.  Filter covariance trace (shrinkage over time)
  5.  Data association quality (match rate, new-feature rate)
  6.  Navigation efficiency   (path length, goal success, wall crossings)
  7.  Exploration coverage    (fraction of world mapped over time)
  8.  Innovation whiteness    (chi-squared test per update)
  9.  Summary statistics table

Usage
-----
    # Default: lab world, 300 MC samples, saves report to slam_report.png
    python slam_analytics.py

    # Custom world and output path
    python slam_analytics.py --world corridor --out results/corridor_run.png

    # Longer run, more MC samples for smoother heatmap statistics
    python slam_analytics.py --world lab --samples 600 --seed 42

    # Print only the summary table (no plot)
    python slam_analytics.py --summary-only

Notes
-----
- Runs headless (Agg backend) — no window is opened.
- Ground-truth robot pose comes from env.Robot (the true physics).
  The EKF estimate comes from slam.state.SLAMState.pose.
- Ground-truth feature positions come from env.World (corners and
  segments) and are matched to EKF features by closest distance
  after the run completes.
- The run terminates when the selector returns 'complete' or after
  MAX_STEPS steps (whichever comes first).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup — works when run from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config          import Config
from env.world       import World
from env.robot       import Robot
from env.sensor      import Sensor
from slam.features   import FeatureExtractor
from slam.state      import SLAMState
from slam.predict    import predict
from slam.update     import update_single, init_corner, init_line
from slam.data_assoc import associate_observations
from montecarlo.uncertainty_map import build_uncertainty_map
from navigation.selector        import GoalSelector
from navigation.slam_world      import SLAMWorld
from navigation.controller      import Controller, Mode

warnings.filterwarnings('ignore', category=RuntimeWarning)

MAX_STEPS = 8000   # hard cap so the script always terminates


# ============================================================================
# Data recorder
# ============================================================================

class SimRecorder:
    """
    Accumulates all metrics during the simulation loop.
    All per-step lists are appended to in the main loop and converted
    to numpy arrays at the end via finalise().
    """

    def __init__(self):
        # -- Time axis
        self.times: list[float] = []

        # -- Ground-truth pose vs EKF estimate
        self.gt_x:   list[float] = []
        self.gt_y:   list[float] = []
        self.gt_th:  list[float] = []
        self.ekf_x:  list[float] = []
        self.ekf_y:  list[float] = []
        self.ekf_th: list[float] = []

        # -- EKF vehicle covariance trace (Pxx + Pyy + Pthth)
        self.pose_cov_trace: list[float] = []

        # -- NEES (Normalised Estimation Error Squared) for vehicle pose
        self.nees: list[float] = []

        # -- Innovation statistics (per EKF update call)
        self.innovations:   list[np.ndarray] = []    # raw ν vectors
        self.innov_covs:    list[np.ndarray] = []    # S matrices
        self.innov_chi2:    list[float] = []          # νᵀS⁻¹ν per update

        # -- Data association
        self.n_obs_total:   list[int] = []   # observations per step
        self.n_matched:     list[int] = []   # matched to existing features
        self.n_new:         list[int] = []   # initialised as new features
        self.n_features:    list[int] = []   # map size over time

        # -- Navigation
        self.goal_positions:  list[np.ndarray] = []
        self.goal_reached:    list[bool]        = []
        self.wall_crossings:  int = 0
        self.path_length:     float = 0.0
        self._prev_pos: np.ndarray | None = None

        # -- Finalised arrays (set by finalise())
        self.t:           np.ndarray | None = None
        self.pose_err_xy: np.ndarray | None = None   # Euclidean XY error (m)
        self.pose_err_th: np.ndarray | None = None   # heading error (rad)

    # ---------------------------------------------------------------- record

    def record_pose(self, t: float,
                    gt_pose: np.ndarray,
                    ekf_pose: np.ndarray,
                    P: np.ndarray) -> None:
        self.times.append(t)
        self.gt_x.append(float(gt_pose[0]))
        self.gt_y.append(float(gt_pose[1]))
        self.gt_th.append(float(gt_pose[2]))
        self.ekf_x.append(float(ekf_pose[0]))
        self.ekf_y.append(float(ekf_pose[1]))
        self.ekf_th.append(float(ekf_pose[2]))
        self.pose_cov_trace.append(float(P[0,0] + P[1,1] + P[2,2]))

        # NEES: eᵀ Pvv⁻¹ e  where e = gt - ekf (wrapped for angle)
        Pvv = P[:3, :3].copy()
        e   = gt_pose - ekf_pose
        e[2] = _wrap(e[2])
        try:
            nees_val = float(e @ np.linalg.inv(Pvv + 1e-9 * np.eye(3)) @ e)
        except np.linalg.LinAlgError:
            nees_val = np.nan
        self.nees.append(nees_val)

    def record_association(self, n_obs: int, n_matched: int,
                            n_new: int, n_features: int) -> None:
        self.n_obs_total.append(n_obs)
        self.n_matched.append(n_matched)
        self.n_new.append(n_new)
        self.n_features.append(n_features)

    def record_innovation(self, nu: np.ndarray, S: np.ndarray) -> None:
        self.innovations.append(nu.copy())
        self.innov_covs.append(S.copy())
        try:
            chi2 = float(nu @ np.linalg.inv(S) @ nu)
        except np.linalg.LinAlgError:
            chi2 = np.nan
        self.innov_chi2.append(chi2)

    def record_navigation(self, pos: np.ndarray, world,
                          crossed_wall: bool) -> None:
        if crossed_wall:
            self.wall_crossings += 1
        if self._prev_pos is not None:
            self.path_length += float(np.linalg.norm(pos - self._prev_pos))
        self._prev_pos = pos.copy()

    # ---------------------------------------------------------------- finalise

    def finalise(self) -> None:
        self.t = np.array(self.times)

        gt_x  = np.array(self.gt_x);  gt_y  = np.array(self.gt_y)
        gt_th = np.array(self.gt_th)
        ex_x  = np.array(self.ekf_x); ex_y  = np.array(self.ekf_y)
        ex_th = np.array(self.ekf_th)

        self.pose_err_xy = np.sqrt((gt_x - ex_x)**2 + (gt_y - ex_y)**2)
        self.pose_err_th = np.abs(_wrap_arr(gt_th - ex_th))

        self.pose_cov_trace = np.array(self.pose_cov_trace)
        self.nees           = np.array(self.nees)
        self.n_features     = np.array(self.n_features)
        self.innov_chi2     = np.array(self.innov_chi2)


# ============================================================================
# Ground-truth feature registry
# ============================================================================

class GTFeatureRegistry:
    """
    Builds the ground-truth feature list from env.World and tracks which
    EKF features were matched to which ground-truth features after the run.
    """

    def __init__(self, world: World):
        self.gt_corners: list[np.ndarray] = [c.pos.copy() for c in world.corners]
        self.gt_lines:   list[tuple]      = [s.as_polar_line() for s in world.segments]

    def match_ekf_to_gt(self, state: SLAMState) -> dict:
        """
        Greedy nearest-neighbour matching of EKF features to GT features.

        Returns
        -------
        {
          'corner_errors':   list of Euclidean XY errors (m) per matched corner
          'line_rho_errors': list of |Δρ| (m) per matched line
          'line_alpha_errors': list of |Δα| (rad) per matched line
          'n_gt_corners':    total GT corners
          'n_gt_lines':      total GT lines
          'n_matched_corners': how many GT corners had an EKF match
          'n_matched_lines':   how many GT lines had an EKF match
        }
        """
        ekf_corners = [(f, state.feature_mean(f.idx))
                       for f in state.features if f.kind == 'corner']
        ekf_lines   = [(f, state.feature_mean(f.idx))
                       for f in state.features if f.kind == 'line']

        corner_errors = []
        used_ekf_c = set()
        for gt_pos in self.gt_corners:
            best_err = np.inf
            best_idx = -1
            for i, (feat, mean) in enumerate(ekf_corners):
                if i in used_ekf_c:
                    continue
                err = float(np.linalg.norm(mean - gt_pos))
                if err < best_err:
                    best_err = err
                    best_idx = i
            if best_idx >= 0 and best_err < 3.0:  # 3 m match radius
                corner_errors.append(best_err)
                used_ekf_c.add(best_idx)

        line_rho_errors   = []
        line_alpha_errors = []
        used_ekf_l = set()
        for gt_rho, gt_alpha in self.gt_lines:
            best_err = np.inf
            best_idx = -1
            for i, (feat, mean) in enumerate(ekf_lines):
                if i in used_ekf_l:
                    continue
                d_rho   = abs(mean[0] - gt_rho)
                d_alpha = abs(_wrap(mean[1] - gt_alpha))
                err     = d_rho + d_alpha   # combined distance for matching
                if err < best_err:
                    best_err = err
                    best_idx = i
            if best_idx >= 0 and best_err < 2.0:
                rho_e, alpha_e = ekf_lines[best_idx][1]
                line_rho_errors.append(abs(rho_e - gt_rho))
                line_alpha_errors.append(abs(_wrap(alpha_e - gt_alpha)))
                used_ekf_l.add(best_idx)

        return {
            'corner_errors':      corner_errors,
            'line_rho_errors':    line_rho_errors,
            'line_alpha_errors':  line_alpha_errors,
            'n_gt_corners':       len(self.gt_corners),
            'n_gt_lines':         len(self.gt_lines),
            'n_matched_corners':  len(corner_errors),
            'n_matched_lines':    len(line_rho_errors),
        }


# ============================================================================
# Wall crossing detector (lightweight, inline)
# ============================================================================

def _check_crossing(world, p0: np.ndarray, p1: np.ndarray) -> bool:
    diff = p1 - p0
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return False
    angle  = float(np.arctan2(diff[1], diff[0]))
    result = world.ray_intersect(p0, angle, max_range=dist)
    if result is None:
        return False
    hit_dist, _ = result
    return hit_dist < dist * 0.85


# ============================================================================
# Helpers
# ============================================================================

def _wrap(angle):
    return float((np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi)

def _wrap_arr(arr):
    return (np.asarray(arr) + np.pi) % (2 * np.pi) - np.pi


class _CombinedWorld:
    def __init__(self, gt, slam):
        self._gt   = gt
        self._slam = slam
    def ray_intersect(self, origin, angle, max_range=30.0):
        r1 = self._gt.ray_intersect(origin, angle, max_range)
        r2 = self._slam.ray_intersect(origin, angle, max_range)
        if r1 is None: return r2
        if r2 is None: return r1
        return r1 if r1[0] <= r2[0] else r2


# ============================================================================
# Simulation runner
# ============================================================================

def run_simulation(cfg: Config,
                   preset: str,
                   n_samples: int,
                   seed: int
                   ) -> tuple[SimRecorder, GTFeatureRegistry, SLAMState, Config]:

    rng    = np.random.default_rng(seed)
    world  = World.from_preset(preset)
    robot  = Robot(cfg.robot.start_x, cfg.robot.start_y,
                   cfg.robot.start_theta, cfg.robot.radius)
    sensor = Sensor.from_cfg(cfg.sensor, rng=rng)
    extractor = FeatureExtractor.from_cfg(cfg, rng=rng)

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

    R_corner = np.diag([cfg.ekf.R_range**2,   cfg.ekf.R_bearing**2])
    R_line   = np.diag([cfg.ekf.R_range**2,   cfg.ekf.R_bearing**2])

    gt_registry = GTFeatureRegistry(world)
    recorder    = SimRecorder()
    mc_rng      = np.random.default_rng(seed + 1)
    mc_every    = max(1, round(1.0 / cfg.sim.dt))
    mc_counter  = 0
    mc_cycles   = 0        # total MC runs completed
    last_umap   = None
    umap_fresh  = False
    returning   = False
    start_pos   = robot.pos.copy()
    # Guard: don't allow 'complete' until the robot has driven at least
    # MIN_EXPLORE_DIST metres AND completed at least MIN_MC_CYCLES MC runs.
    # This prevents a false-complete when all features happen to be visible
    # from the start position and the first MC scores everything as low-uncertainty.
    MIN_EXPLORE_DIST = 3.0   # metres
    MIN_MC_CYCLES    = 5     # MC updates before completion is allowed
    prev_pos: np.ndarray | None = None
    t           = 0.0
    dt          = cfg.sim.dt
    step        = 0

    print(f"\n[analytics] Running '{preset}' world | seed={seed} | "
          f"max_steps={MAX_STEPS}")

    for step in range(MAX_STEPS):
        t = step * dt

        # -- Controller -------------------------------------------------------
        _slam_world = SLAMWorld.from_state(slam_state)
        _cw         = _CombinedWorld(world, _slam_world)
        v, omega    = controller.step(slam_state.pose, _cw, dt,
                                      slam_world=_slam_world)

        # -- Physics ----------------------------------------------------------
        prev_pos_gt = robot.pos.copy()
        robot.step(v, omega, dt)

        # Wall crossing detection
        crossed = _check_crossing(world, prev_pos_gt, robot.pos)
        recorder.record_navigation(robot.pos, world, crossed)

        # -- EKF predict ------------------------------------------------------
        predict(slam_state, v, omega, dt, cfg.ekf.Q_v, cfg.ekf.Q_w)

        # -- Sense + extract --------------------------------------------------
        scan = sensor.scan(robot, world)
        corners, lines = extractor.extract(robot.pose, world, scan)
        all_obs = list(corners) + list(lines)

        # -- EKF update + record innovations ----------------------------------
        associations = associate_observations(
            slam_state, all_obs, R_corner, R_line, cfg.ekf.gate_chi2)

        n_matched = 0
        n_new     = 0
        for obs, feat_idx in associations:
            R = R_corner if obs.feature_kind == 'corner' else R_line
            if feat_idx == -1:
                n_new += 1
                if obs.feature_kind == 'corner':
                    init_corner(slam_state, obs.z, R, cfg.ekf.init_cov_corner)
                else:
                    init_line(slam_state, obs.z, R, cfg.ekf.init_cov_line,
                              seg_p0=obs.seg_p0, seg_p1=obs.seg_p1)
            else:
                n_matched += 1
                # Compute innovation BEFORE update for recording
                feat = slam_state.features[feat_idx]
                from slam.update import (
                    _h_corner, _h_corner_jacobian,
                    _h_line,   _h_line_jacobian,
                )
                if feat.kind == 'corner':
                    z_hat = _h_corner(slam_state, feat_idx)
                    H     = _h_corner_jacobian(slam_state, feat_idx)
                else:
                    z_hat = _h_line(slam_state, feat_idx)
                    H     = _h_line_jacobian(slam_state, feat_idx)
                nu = obs.z - z_hat
                nu[1] = _wrap(nu[1])
                S  = H @ slam_state.P @ H.T + R
                recorder.record_innovation(nu, S)
                update_single(slam_state, feat_idx, obs.z, R)

        recorder.record_association(len(all_obs), n_matched, n_new,
                                    slam_state.n_features)

        # -- Record pose metrics ----------------------------------------------
        recorder.record_pose(t, robot.pose, slam_state.pose, slam_state.P)

        # -- MC map -----------------------------------------------------------
        mc_counter += 1
        if slam_state.n_features > 0 and mc_counter >= mc_every:
            mc_counter = 0
            last_umap  = build_uncertainty_map(
                state          = slam_state,
                world          = world,
                robot_pos      = robot.pos,
                n_samples      = n_samples,
                robot_radius   = cfg.robot.radius,
                virtual_cov    = cfg.montecarlo.virtual_cov,
                uncertainty_lo = cfg.montecarlo.uncertainty_lo,
                uncertainty_hi = cfg.montecarlo.uncertainty_hi,
                rng            = mc_rng,
            )
            umap_fresh = True
            mc_cycles += 1

        # -- Autonomous navigation -------------------------------------------
        if last_umap is not None:
            if returning:
                controller.set_goal(start_pos)
                if controller.goal_reached(robot.pos):
                    print(f"[analytics] Returned to start at t={t:.1f}s "
                          f"step={step}. Done.")
                    break
            elif controller.goal_reached(robot.pos):
                selector.notify_goal_reached(robot.pos)
                controller.set_goal(None)
                umap_fresh = False

            if not returning and controller.goal is None and umap_fresh:
                umap_fresh = False
                # Don't declare complete too early: robot must have moved
                # and the uncertainty map must have been refreshed enough
                # times to have actually explored, not just seen from start.
                explored_enough = (recorder.path_length >= MIN_EXPLORE_DIST
                                   and mc_cycles >= MIN_MC_CYCLES)
                result = selector.select(last_umap, robot.pos)
                if result.source == 'complete' and explored_enough:
                    print(f"[analytics] Map declared complete at t={t:.1f}s "
                          f"step={step}  path={recorder.path_length:.1f}m. "
                          f"Returning to start.")
                    returning = True
                    controller.set_goal(start_pos, world=_slam_world)
                elif result.source == 'complete' and not explored_enough:
                    # Too early — nudge the robot by issuing a random
                    # frontier goal toward the room centre so it starts moving.
                    bounds = world.bounds
                    cx = (bounds[0] + bounds[1]) / 2
                    cy = (bounds[2] + bounds[3]) / 2
                    nudge = np.array([cx, cy]) + np.array([
                        np.random.uniform(-2, 2),
                        np.random.uniform(-2, 2)])
                    controller.set_goal(nudge, world=_slam_world)
                    print(f"[analytics] Early complete suppressed "
                          f"(path={recorder.path_length:.1f}m < {MIN_EXPLORE_DIST}m "
                          f"or mc_cycles={mc_cycles} < {MIN_MC_CYCLES}). "
                          f"Nudging to ({nudge[0]:.1f},{nudge[1]:.1f}).")
                else:
                    controller.set_goal(result.goal, world=_slam_world)
                    recorder.goal_positions.append(result.goal.copy())

        # -- Progress print every 10 s ----------------------------------------
        if step % int(10.0 / dt) == 0:
            err_xy = np.sqrt(
                (robot.x - slam_state.pose[0])**2 +
                (robot.y - slam_state.pose[1])**2
            )
            print(f"  t={t:6.1f}s  feats={slam_state.n_features:3d}  "
                  f"pose_err={err_xy:.3f}m  "
                  f"crossings={recorder.wall_crossings}")

    else:
        print(f"[analytics] Reached MAX_STEPS={MAX_STEPS}.")

    recorder.finalise()
    return recorder, gt_registry, slam_state, cfg


# ============================================================================
# Report generation
# ============================================================================

STYLE = {
    'bg':       '#0f1117',
    'panel':    '#161b27',
    'text':     '#e2e8f0',
    'grid':     '#1e2130',
    'accent1':  '#818cf8',   # indigo
    'accent2':  '#34d399',   # teal
    'accent3':  '#f59e0b',   # amber
    'accent4':  '#f87171',   # red
    'accent5':  '#a78bfa',   # violet
    'accent6':  '#38bdf8',   # sky
}

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(STYLE['panel'])
    ax.tick_params(colors=STYLE['text'], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE['grid'])
    ax.xaxis.label.set_color(STYLE['text'])
    ax.yaxis.label.set_color(STYLE['text'])
    ax.title.set_color(STYLE['text'])
    ax.grid(True, color=STYLE['grid'], linewidth=0.5, alpha=0.7)
    if title:   ax.set_title(title, fontsize=8, pad=4)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=7)


def build_report(rec: SimRecorder,
                 gt_reg: GTFeatureRegistry,
                 slam_state: SLAMState,
                 cfg: Config,
                 preset: str,
                 out_path: str) -> dict:

    t = rec.t
    match_results = gt_reg.match_ekf_to_gt(slam_state)

    # ── Pre-compute statistics ─────────────────────────────────────────────

    # 1. Pose error
    rmse_xy = float(np.sqrt(np.mean(rec.pose_err_xy**2)))
    rmse_th = float(np.sqrt(np.mean(rec.pose_err_th**2)))
    max_xy  = float(rec.pose_err_xy.max())
    final_xy = float(rec.pose_err_xy[-1])

    # 2. NEES — consistency check
    #    95% chi2(3) bounds: [0.352, 7.815] for a single run
    #    A well-calibrated filter should have NEES ≈ 3 (dof) on average.
    nees_valid  = rec.nees[np.isfinite(rec.nees)]
    nees_mean   = float(np.mean(nees_valid)) if len(nees_valid) else np.nan
    nees_lo, nees_hi = stats.chi2.ppf([0.025, 0.975], df=3)
    frac_consistent = float(np.mean((nees_valid >= nees_lo) &
                                    (nees_valid <= nees_hi))) if len(nees_valid) else np.nan

    # 3. Innovation chi-squared
    innov_valid = np.array(rec.innov_chi2)[np.isfinite(rec.innov_chi2)]
    innov_mean  = float(np.mean(innov_valid)) if len(innov_valid) else np.nan
    # 95% chi2(2) bounds: [0.051, 5.991]
    innov_lo, innov_hi = stats.chi2.ppf([0.025, 0.975], df=2)
    frac_innov_ok = float(np.mean((innov_valid >= innov_lo) &
                                   (innov_valid <= innov_hi))) if len(innov_valid) else np.nan

    # 4. Association stats
    n_obs_arr     = np.array(rec.n_obs_total)
    n_matched_arr = np.array(rec.n_matched)
    n_new_arr     = np.array(rec.n_new)
    total_obs     = int(n_obs_arr.sum())
    total_matched = int(n_matched_arr.sum())
    total_new     = int(n_new_arr.sum())
    match_rate    = total_matched / max(total_obs, 1)

    # 5. Feature map quality
    ce  = match_results['corner_errors']
    lre = match_results['line_rho_errors']
    lae = match_results['line_alpha_errors']
    corner_rmse = float(np.sqrt(np.mean(np.array(ce)**2)))  if ce  else np.nan
    line_rho_rmse   = float(np.sqrt(np.mean(np.array(lre)**2))) if lre else np.nan
    line_alpha_rmse = float(np.sqrt(np.mean(np.array(lae)**2))) if lae else np.nan

    recall_corners = (match_results['n_matched_corners'] /
                      max(match_results['n_gt_corners'], 1))
    recall_lines   = (match_results['n_matched_lines'] /
                      max(match_results['n_gt_lines'], 1))

    # 6. Navigation
    total_time   = float(t[-1]) if len(t) else 0.0
    path_length  = rec.path_length
    n_goals      = len(rec.goal_positions)
    crossings    = rec.wall_crossings

    # ── Build figure ───────────────────────────────────────────────────────

    fig = plt.figure(figsize=(18, 14), facecolor=STYLE['bg'])
    fig.suptitle(
        f"EKF-SLAM Analytics Report — world: '{preset}'  |  "
        f"duration: {total_time:.0f}s  |  path: {path_length:.1f}m  |  "
        f"features: {slam_state.n_features}",
        color=STYLE['text'], fontsize=11, y=0.98
    )

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.06)

    # ── Panel 1: XY pose error over time ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(t, 0, rec.pose_err_xy,
                     color=STYLE['accent1'], alpha=0.3, linewidth=0)
    ax1.plot(t, rec.pose_err_xy, color=STYLE['accent1'], linewidth=0.9,
             label='XY error')
    # Smooth trend line
    if len(t) > 50:
        win = max(20, len(t)//40)
        smooth = np.convolve(rec.pose_err_xy,
                             np.ones(win)/win, mode='valid')
        t_sm = t[win//2: win//2 + len(smooth)]
        ax1.plot(t_sm, smooth, color=STYLE['accent3'], linewidth=1.5,
                 linestyle='--', label='smoothed trend')
    ax1.axhline(rmse_xy, color=STYLE['accent4'], linestyle=':',
                linewidth=1.2, label=f'RMSE={rmse_xy:.3f}m')
    ax1.legend(fontsize=6, labelcolor=STYLE['text'],
               facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    _style_ax(ax1, 'Pose Error — XY (m)', 'Time (s)', 'Error (m)')

    # ── Panel 2: Heading error over time ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.fill_between(t, 0, np.degrees(rec.pose_err_th),
                     color=STYLE['accent2'], alpha=0.3, linewidth=0)
    ax2.plot(t, np.degrees(rec.pose_err_th),
             color=STYLE['accent2'], linewidth=0.9)
    ax2.axhline(np.degrees(rmse_th), color=STYLE['accent4'], linestyle=':',
                linewidth=1.2,
                label=f'RMSE={np.degrees(rmse_th):.2f}°')
    ax2.legend(fontsize=6, labelcolor=STYLE['text'],
               facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    _style_ax(ax2, 'Pose Error — Heading (°)', 'Time (s)', 'Error (°)')

    # ── Panel 3: NEES over time ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(t, rec.nees, color=STYLE['accent5'], linewidth=0.6, alpha=0.7)
    ax3.axhline(3.0, color=STYLE['accent3'], linestyle='--', linewidth=1.2,
                label='Expected (dof=3)')
    ax3.axhline(nees_hi, color=STYLE['accent4'], linestyle=':',
                linewidth=1.0, label=f'95% upper={nees_hi:.1f}')
    ax3.axhline(nees_lo, color=STYLE['accent2'], linestyle=':',
                linewidth=1.0, label=f'95% lower={nees_lo:.2f}')
    ax3.set_ylim(-0.5, min(float(np.nanpercentile(rec.nees, 98)) * 1.5, 50))
    ax3.legend(fontsize=6, labelcolor=STYLE['text'],
               facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    _style_ax(ax3,
              f'NEES (Normalised Estimation Error Squared)  '
              f'mean={nees_mean:.2f}  '
              f'{frac_consistent*100:.0f}% within 95% bounds',
              'Time (s)', 'NEES')

    # ── Panel 4: Covariance trace over time ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.semilogy(t, rec.pose_cov_trace,
                 color=STYLE['accent6'], linewidth=0.9)
    _style_ax(ax4, 'Vehicle Pose Covariance Trace  tr(P_vv)',
              'Time (s)', 'tr(P_vv)  [log scale]')

    # ── Panel 5: Innovation chi-squared histogram ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if len(innov_valid) > 0:
        hist_max = min(float(np.percentile(innov_valid, 98)), 20.0)
        bins = np.linspace(0, hist_max, 40)
        ax5.hist(innov_valid, bins=bins, density=True,
                 color=STYLE['accent1'], alpha=0.75, edgecolor='none')
        xs = np.linspace(0, hist_max, 200)
        ax5.plot(xs, stats.chi2.pdf(xs, df=2),
                 color=STYLE['accent3'], linewidth=1.5,
                 label='χ²(2) expected')
        ax5.axvline(innov_lo, color=STYLE['accent2'], linestyle=':',
                    linewidth=0.9)
        ax5.axvline(innov_hi, color=STYLE['accent4'], linestyle=':',
                    linewidth=0.9, label='95% bounds')
        ax5.legend(fontsize=6, labelcolor=STYLE['text'],
                   facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    _style_ax(ax5,
              f'Innovation χ²  mean={innov_mean:.2f}\n'
              f'{frac_innov_ok*100:.0f}% within χ²(2) 95%',
              'νᵀS⁻¹ν', 'Density')

    # ── Panel 6: Feature count over time ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    feat_t = t[:len(rec.n_features)]
    ax6.step(feat_t, rec.n_features,
             color=STYLE['accent2'], linewidth=1.2, where='post')
    ax6.fill_between(feat_t, 0, rec.n_features,
                     color=STYLE['accent2'], alpha=0.15, step='post')
    _style_ax(ax6, f'Map Size Over Time\nfinal={slam_state.n_features} features',
              'Time (s)', '# features')

    # ── Panel 7: Association rates over time ────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    obs_t = t[:len(rec.n_obs_total)]
    # Rolling match rate (window = 5 s)
    window = max(1, int(5.0 / cfg.sim.dt))
    n_obs_arr2     = np.array(rec.n_obs_total,  dtype=float)
    n_matched_arr2 = np.array(rec.n_matched,    dtype=float)
    # Avoid division by zero
    safe_obs = np.where(n_obs_arr2 > 0, n_obs_arr2, np.nan)
    match_frac = n_matched_arr2 / safe_obs
    # Rolling mean ignoring NaN
    def rolling_mean_nan(arr, w):
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            seg = arr[max(0, i-w+1):i+1]
            valid = seg[np.isfinite(seg)]
            if len(valid):
                out[i] = np.mean(valid)
        return out
    roll_match = rolling_mean_nan(match_frac, window)
    ax7.plot(obs_t, roll_match * 100,
             color=STYLE['accent3'], linewidth=1.0,
             label='Match rate %')
    ax7.set_ylim(-5, 105)
    ax7.axhline(match_rate * 100, color=STYLE['accent4'],
                linestyle=':', linewidth=1.0,
                label=f'Overall {match_rate*100:.0f}%')
    ax7.legend(fontsize=6, labelcolor=STYLE['text'],
               facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    _style_ax(ax7, f'Data Association Match Rate\n(5s rolling window)',
              'Time (s)', 'Match rate (%)')

    # ── Panel 8: Map accuracy — corner and line errors ───────────────────────
    ax8 = fig.add_subplot(gs[2, 3])
    categories = []
    values     = []
    colours    = []
    if ce:
        categories.append(f'Corner XY\n(n={len(ce)})')
        values.append(corner_rmse * 100)   # cm
        colours.append(STYLE['accent1'])
    if lre:
        categories.append(f'Line ρ\n(n={len(lre)})')
        values.append(line_rho_rmse * 100)
        colours.append(STYLE['accent2'])
    if lae:
        categories.append(f'Line α\n(n={len(lae)})')
        values.append(np.degrees(line_alpha_rmse))
        colours.append(STYLE['accent3'])

    if categories:
        bars = ax8.bar(categories, values, color=colours,
                       edgecolor='none', alpha=0.85)
        for bar, val in zip(bars, values):
            ax8.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.01 * max(values),
                     f'{val:.2f}', ha='center', va='bottom',
                     color=STYLE['text'], fontsize=7)
    _style_ax(ax8, 'Feature Map RMSE\n(cm for XY/ρ, deg for α)',
              '', 'RMSE (cm or deg)')

    # ── Panel 9: Recall bar chart ────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[3, 0])
    labels  = ['Corner recall', 'Line recall']
    recalls = [recall_corners * 100, recall_lines * 100]
    cols    = [STYLE['accent1'], STYLE['accent2']]
    brs = ax9.bar(labels, recalls, color=cols, edgecolor='none', alpha=0.85)
    for b, v in zip(brs, recalls):
        ax9.text(b.get_x() + b.get_width()/2,
                 v + 1.5, f'{v:.0f}%',
                 ha='center', va='bottom',
                 color=STYLE['text'], fontsize=8)
    ax9.set_ylim(0, 115)
    ax9.axhline(100, color=STYLE['grid'], linestyle='--', linewidth=0.8)
    _style_ax(ax9, 'Feature Recall\n(GT features matched by EKF)',
              '', 'Recall (%)')

    # ── Panel 10: NEES distribution ──────────────────────────────────────────
    ax10 = fig.add_subplot(gs[3, 1])
    if len(nees_valid) > 0:
        nees_cap = min(float(np.percentile(nees_valid, 98)) * 1.5, 30.0)
        bins10 = np.linspace(0, nees_cap, 40)
        ax10.hist(nees_valid, bins=bins10, density=True,
                  color=STYLE['accent5'], alpha=0.75, edgecolor='none')
        xs10 = np.linspace(0, nees_cap, 200)
        ax10.plot(xs10, stats.chi2.pdf(xs10, df=3),
                  color=STYLE['accent3'], linewidth=1.5,
                  label='χ²(3) expected')
        ax10.axvline(nees_hi, color=STYLE['accent4'], linestyle=':',
                     linewidth=0.9)
        ax10.legend(fontsize=6, labelcolor=STYLE['text'],
                    facecolor=STYLE['panel'], edgecolor=STYLE['grid'])
    _style_ax(ax10, 'NEES Distribution vs χ²(3)', 'NEES', 'Density')

    # ── Panel 11: Summary statistics table ───────────────────────────────────
    ax11 = fig.add_subplot(gs[3, 2:])
    ax11.axis('off')
    ax11.set_facecolor(STYLE['panel'])

    summary_data = [
        ('World preset',                   preset),
        ('Simulation duration (s)',         f'{total_time:.1f}'),
        ('Total path length (m)',           f'{path_length:.2f}'),
        ('Navigation goals issued',         f'{n_goals}'),
        ('Wall crossings',                  f'{crossings}  {"✓ OK" if crossings==0 else "✗ FAIL"}'),
        ('─' * 30,                          '─' * 16),
        ('POSE ERROR',                      ''),
        ('  XY RMSE (m)',                   f'{rmse_xy:.4f}'),
        ('  XY max error (m)',              f'{max_xy:.4f}'),
        ('  XY final error (m)',            f'{final_xy:.4f}'),
        ('  Heading RMSE (deg)',            f'{np.degrees(rmse_th):.3f}'),
        ('─' * 30,                          '─' * 16),
        ('FILTER CONSISTENCY',              ''),
        ('  NEES mean (expect ≈3)',         f'{nees_mean:.2f}  {"↑ overconfident" if nees_mean>5 else ("↓ underconfident" if nees_mean<1.5 else "✓ consistent")}'),
        ('  % steps within χ²(3) 95%',     f'{frac_consistent*100:.0f}%'),
        ('  Innovation mean χ² (expect≈2)', f'{innov_mean:.2f}  {"↑" if innov_mean>4 else ("↓" if innov_mean<0.5 else "✓")}'),
        ('  % innovations within χ²(2) 95%', f'{frac_innov_ok*100:.0f}%'),
        ('─' * 30,                          '─' * 16),
        ('MAP QUALITY',                     ''),
        ('  GT corners',                    f'{match_results["n_gt_corners"]}'),
        ('  GT lines',                      f'{match_results["n_gt_lines"]}'),
        ('  Corner recall',                 f'{recall_corners*100:.0f}%  ({match_results["n_matched_corners"]} matched)'),
        ('  Line recall',                   f'{recall_lines*100:.0f}%  ({match_results["n_matched_lines"]} matched)'),
        ('  EKF features (total)',          f'{slam_state.n_features}  (incl. duplicates)'),
        ('  Corner RMSE (cm)',              f'{corner_rmse*100:.2f}' if not np.isnan(corner_rmse) else 'N/A'),
        ('  Line ρ RMSE (cm)',              f'{line_rho_rmse*100:.2f}' if not np.isnan(line_rho_rmse) else 'N/A'),
        ('  Line α RMSE (deg)',             f'{np.degrees(line_alpha_rmse):.3f}' if not np.isnan(line_alpha_rmse) else 'N/A'),
        ('─' * 30,                          '─' * 16),
        ('DATA ASSOCIATION',                ''),
        ('  Total observations',            f'{total_obs}'),
        ('  Matched (existing features)',   f'{total_matched}  ({match_rate*100:.0f}%)'),
        ('  New (feature initialisations)', f'{total_new}'),
    ]

    col_x = [0.02, 0.55]
    row_y = 0.97
    dy    = 0.95 / len(summary_data)
    for label, value in summary_data:
        if label.startswith('─'):
            ax11.plot([0.01, 0.99], [row_y, row_y],
                      color=STYLE['grid'], linewidth=0.5,
                      transform=ax11.transAxes, clip_on=False)
        else:
            bold = value == '' or label.startswith('  ') is False
            weight = 'bold' if (value == '' and not label.startswith('  ')) else 'normal'
            ax11.text(col_x[0], row_y, label,
                      transform=ax11.transAxes,
                      color=STYLE['text'], fontsize=7.5,
                      fontweight=weight, va='top',
                      fontfamily='monospace')
            if value:
                colour = STYLE['accent3'] if ('✓' in value or '✗' in value
                                               or '↑' in value or '↓' in value) \
                                          else STYLE['accent2']
                ax11.text(col_x[1], row_y, value,
                          transform=ax11.transAxes,
                          color=colour, fontsize=7.5,
                          va='top', fontfamily='monospace')
        row_y -= dy

    ax11.set_title('Summary Statistics', color=STYLE['text'],
                   fontsize=9, pad=6, loc='left')

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=STYLE['bg'])
    plt.close(fig)
    print(f"\n[analytics] Report saved → {out_path}")

    # Return summary dict for programmatic use
    return {
        'rmse_xy':            rmse_xy,
        'rmse_th_deg':        float(np.degrees(rmse_th)),
        'max_xy':             max_xy,
        'final_xy':           final_xy,
        'nees_mean':          nees_mean,
        'frac_consistent':    frac_consistent,
        'innov_mean_chi2':    innov_mean,
        'frac_innov_ok':      frac_innov_ok,
        'corner_rmse_cm':     corner_rmse * 100 if not np.isnan(corner_rmse) else np.nan,
        'line_rho_rmse_cm':   line_rho_rmse * 100 if not np.isnan(line_rho_rmse) else np.nan,
        'line_alpha_rmse_deg': float(np.degrees(line_alpha_rmse)) if not np.isnan(line_alpha_rmse) else np.nan,
        'recall_corners':     recall_corners,
        'recall_lines':       recall_lines,
        'n_ekf_features':     slam_state.n_features,
        'match_rate':         match_rate,
        'path_length':        path_length,
        'wall_crossings':     crossings,
        'total_time':         total_time,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EKF-SLAM analytics — headless evaluation run')
    parser.add_argument('--world',   choices=['lab', 'corridor', 'open'],
                        default='lab')
    parser.add_argument('--config',  default='config.yaml')
    parser.add_argument('--out',     default='slam_report.png',
                        help='Output image path')
    parser.add_argument('--samples', type=int, default=300,
                        help='MC samples per uncertainty update')
    parser.add_argument('--seed',    type=int, default=0)
    parser.add_argument('--summary-only', action='store_true',
                        help='Print summary table to stdout only, no plot saved')
    args = parser.parse_args()

    cfg = Config.load(args.config)

    t0 = time.perf_counter()
    rec, gt_reg, slam_state, cfg = run_simulation(
        cfg, args.world, args.samples, args.seed)
    elapsed = time.perf_counter() - t0
    print(f"[analytics] Simulation completed in {elapsed:.1f}s wall-clock time.")

    if args.summary_only:
        # Quick text table only
        match_results = gt_reg.match_ekf_to_gt(slam_state)
        print(f"\n{'─'*60}")
        print(f"  World: {args.world}  |  seed: {args.seed}  |  "
              f"features: {slam_state.n_features}")
        print(f"{'─'*60}")
        print(f"  Pose RMSE XY:   {np.sqrt(np.mean(rec.pose_err_xy**2)):.4f} m")
        print(f"  Pose RMSE θ:    {np.degrees(np.sqrt(np.mean(rec.pose_err_th**2))):.3f} °")
        print(f"  NEES mean:      {np.nanmean(rec.nees):.2f}  (expect ≈3)")
        print(f"  Corner recall:  "
              f"{match_results['n_matched_corners']}/{match_results['n_gt_corners']}")
        print(f"  Line recall:    "
              f"{match_results['n_matched_lines']}/{match_results['n_gt_lines']}")
        print(f"  Wall crossings: {rec.wall_crossings}")
        print(f"  Path length:    {rec.path_length:.2f} m")
        print(f"{'─'*60}")
        return

    summary = build_report(rec, gt_reg, slam_state, cfg,
                            args.world, args.out)

    print(f"\n{'─'*60}")
    print("  KEY RESULTS")
    print(f"{'─'*60}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<30s}: {v:.4f}")
        else:
            print(f"  {k:<30s}: {v}")
    print(f"{'─'*60}\n")


if __name__ == '__main__':
    main()