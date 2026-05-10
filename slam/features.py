"""
slam/features.py
----------------
Feature measurement models, analytic Jacobians, and the synthetic
observation generator used throughout Phases 3 and 4.

Recap of the two feature types (paper Eqs. 4 & 5)
--------------------------------------------------

Corner  (Cartesian state: [cx, cy])
    z_corner = [r,    beta ]
    r    = sqrt((xv-cx)^2 + (yv-cy)^2)
    beta = atan2(cy-yv, cx-xv) - theta_v        [robot frame bearing]

Line    (polar state: [rho, alpha])
    z_line = [rho_obs, alpha_obs]
    rho_obs   = rho - xv*cos(alpha) - yv*sin(alpha)
    alpha_obs = alpha - theta_v                  [robot frame bearing]

Noise model (diagonal R matrix)
    corners : diag(sigma_r^2,  sigma_beta^2)
    lines   : diag(sigma_rho^2, sigma_alpha^2)

Jacobian convention
-------------------
H  = dz/dx_full   where x_full = [x_v, y_v, theta_v, ..., feat_params, ...]
     shape (2, 3 + 2)  for both feature types when computing w.r.t.
     [vehicle pose | this feature's 2 params] — the EKF update.py will
     scatter these into the full-state Jacobian.

All angles wrapped to (-pi, pi].
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Observation containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CornerObs:
    """
    A single noisy corner observation produced by the feature extractor.

    Attributes
    ----------
    z            : (2,)  [range, bearing]  in robot frame
    R            : (2,2) measurement noise covariance
    kind         : 'convex' | 'concave' — geometry type of the detected corner.
                   This is what Phase 3 tests assert on c.kind.
    gt_pos       : (2,) ground-truth world position (sim only, not used by EKF)
    feature_kind : property, always returns 'corner' — used by data_assoc /
                   main.py for EKF dispatch (obs.feature_kind == 'corner').
    """
    z:      np.ndarray           # (2,)  [r, beta]
    R:      np.ndarray           # (2,2)
    kind:   str = 'convex'       # geometry: 'convex' | 'concave'
    gt_pos: Optional[np.ndarray] = None

    @property
    def feature_kind(self) -> str:
        """EKF dispatch identifier — always 'corner'."""
        return 'corner'

    @property
    def range(self) -> float:
        return float(self.z[0])

    @property
    def bearing(self) -> float:
        return float(self.z[1])


@dataclass
class LineObs:
    """
    A single noisy line observation produced by the feature extractor.

    Attributes
    ----------
    z               : (2,)  [rho_obs, alpha_obs]  in robot frame
    R               : (2,2) measurement noise covariance
    kind            : 'line'  — identifies this as a line observation for
                      data_assoc.py and main.py (obs.kind == 'line')
    seg_p0 / seg_p1 : (2,) world-frame endpoints of the source segment —
                      passed through to state.py for line rendering
    cluster_midpoint: (2,) world-frame centre of the observed wall cluster.
                      Used by the renderer to place the '+' marker.
    gt_rho          : ground-truth rho   (sim only)
    gt_alpha        : ground-truth alpha (sim only)
    """
    z:                np.ndarray           # (2,)  [rho_obs, alpha_obs]
    R:                np.ndarray           # (2,2)
    kind:             str = 'line'         # always 'line' — used by data_assoc & main
    seg_p0:           Optional[np.ndarray] = None
    seg_p1:           Optional[np.ndarray] = None
    cluster_midpoint: Optional[np.ndarray] = None
    gt_rho:           Optional[float]      = None
    gt_alpha:         Optional[float]      = None

    @property
    def feature_kind(self) -> str:
        """EKF dispatch identifier — always 'line'."""
        return 'line'


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Jacobians:
    """
    Analytic Jacobians of a measurement w.r.t. the relevant state variables.

    H_v   : (2, 3)  dz / d[xv, yv, theta_v]
    H_f   : (2, 2)  dz / d[feature params]

    The full H row for the EKF update is assembled by update.py as:
        H_full[0:2, v_idx]  = H_v
        H_full[0:2, f_idx]  = H_f
    """
    H_v: np.ndarray   # (2, 3)
    H_f: np.ndarray   # (2, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Pure measurement-model functions
# ─────────────────────────────────────────────────────────────────────────────

def h_corner(robot_pose: np.ndarray,
             corner_pos: np.ndarray) -> np.ndarray:
    """
    Noiseless corner measurement (Eq. 4).

    Parameters
    ----------
    robot_pose : (3,) [xv, yv, theta_v]
    corner_pos : (2,) [cx, cy]

    Returns
    -------
    z : (2,) [r, beta]
    """
    xv, yv, tv = robot_pose
    cx, cy     = corner_pos
    dx, dy     = cx - xv, cy - yv
    r          = np.sqrt(dx**2 + dy**2)
    beta       = _wrap(np.arctan2(dy, dx) - tv)
    return np.array([r, beta])


def h_line(robot_pose: np.ndarray,
           rho: float,
           alpha: float) -> np.ndarray:
    """
    Noiseless line measurement (Eq. 5).

    Parameters
    ----------
    robot_pose : (3,) [xv, yv, theta_v]
    rho        : perpendicular distance from world origin (m)
    alpha      : angle of line's outward normal (rad)

    Returns
    -------
    z : (2,) [rho_obs, alpha_obs]
    """
    xv, yv, tv = robot_pose
    rho_obs    = rho - xv * np.cos(alpha) - yv * np.sin(alpha)
    alpha_obs  = _wrap(alpha - tv)
    return np.array([rho_obs, alpha_obs])


# ─────────────────────────────────────────────────────────────────────────────
# Analytic Jacobians
# ─────────────────────────────────────────────────────────────────────────────

def feature_jacobians(robot_pose:     np.ndarray,
                      feature_type:   str,
                      feature_params: np.ndarray) -> Jacobians:
    """
    Compute analytic Jacobians of the measurement model.

    Parameters
    ----------
    robot_pose     : (3,) [xv, yv, theta_v]
    feature_type   : 'corner' | 'line'   (also accepts old kwarg name 'kind')
    feature_params : (2,)
        corner -> [cx, cy]
        line   -> [rho, alpha]

    Returns
    -------
    Jacobians with H_v (2,3) and H_f (2,2).
    """
    if feature_type == 'corner':
        return _jacobian_corner(robot_pose, feature_params)
    elif feature_type == 'line':
        return _jacobian_line(robot_pose, feature_params)
    else:
        raise ValueError(f"Unknown feature type '{feature_type}'")


def _jacobian_corner(robot_pose: np.ndarray,
                     corner_pos: np.ndarray) -> Jacobians:
    """
    Analytic Jacobian for corner measurement.

    z = [r, beta]
    r    = sqrt(dx^2 + dy^2),       dx = cx-xv,  dy = cy-yv
    beta = atan2(dy, dx) - theta_v

    dz/d[xv, yv, tv]:
        dr/dxv    = -dx/r,  dr/dyv    = -dy/r,  dr/dtv    =  0
        dbeta/dxv =  dy/r², dbeta/dyv = -dx/r², dbeta/dtv = -1

    dz/d[cx, cy]:
        dr/dcx    =  dx/r,  dr/dcy    =  dy/r
        dbeta/dcx = -dy/r², dbeta/dcy =  dx/r²
    """
    xv, yv, tv = robot_pose
    cx, cy     = corner_pos
    dx, dy     = cx - xv, cy - yv
    r2         = dx**2 + dy**2
    r          = max(np.sqrt(r2), 1e-6)

    H_v = np.array([
        [-dx/r,    -dy/r,    0.0],
        [ dy/r2,  -dx/r2,  -1.0],
    ])

    H_f = np.array([
        [ dx/r,    dy/r  ],
        [-dy/r2,   dx/r2 ],
    ])

    return Jacobians(H_v=H_v, H_f=H_f)


def _jacobian_line(robot_pose: np.ndarray,
                   line_params: np.ndarray) -> Jacobians:
    """
    Analytic Jacobian for line measurement.

    z = [rho_obs, alpha_obs]
    rho_obs   = rho - xv*cos(α) - yv*sin(α)
    alpha_obs = alpha - tv

    dz/d[xv, yv, tv]:
        drho_obs/dxv   = -cos(α),  drho_obs/dyv   = -sin(α),  drho_obs/dtv   = 0
        dalpha_obs/dxv =  0,        dalpha_obs/dyv =  0,        dalpha_obs/dtv = -1

    dz/d[rho, alpha]:
        drho_obs/drho      =  1
        drho_obs/dalpha    =  xv*sin(α) - yv*cos(α)
        dalpha_obs/drho    =  0
        dalpha_obs/dalpha  =  1
    """
    xv, yv, tv  = robot_pose
    rho, alpha  = line_params
    ca, sa      = np.cos(alpha), np.sin(alpha)

    H_v = np.array([
        [-ca,  -sa,  0.0],
        [0.0,  0.0, -1.0],
    ])

    H_f = np.array([
        [1.0,  xv * sa - yv * ca],
        [0.0,  1.0               ],
    ])

    return Jacobians(H_v=H_v, H_f=H_f)


# ─────────────────────────────────────────────────────────────────────────────
# Numerical Jacobians (for testing only)
# ─────────────────────────────────────────────────────────────────────────────

def numerical_jacobian_corner(robot_pose:  np.ndarray,
                               corner_pos: np.ndarray,
                               eps:        float = 1e-6
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobians for corner model.  Used in tests to validate
    the analytic versions.

    Returns (H_v_num, H_f_num), both (2, 3) and (2, 2).
    """
    H_v = np.zeros((2, 3))
    for i in range(3):
        rp_p = robot_pose.copy(); rp_p[i] += eps
        rp_m = robot_pose.copy(); rp_m[i] -= eps
        H_v[:, i] = (h_corner(rp_p, corner_pos) - h_corner(rp_m, corner_pos)) / (2 * eps)

    H_f = np.zeros((2, 2))
    for i in range(2):
        cp_p = corner_pos.copy(); cp_p[i] += eps
        cp_m = corner_pos.copy(); cp_m[i] -= eps
        H_f[:, i] = (h_corner(robot_pose, cp_p) - h_corner(robot_pose, cp_m)) / (2 * eps)

    return H_v, H_f


def numerical_jacobian_line(robot_pose:  np.ndarray,
                             line_params: np.ndarray,
                             eps:         float = 1e-6
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobians for line model."""
    H_v = np.zeros((2, 3))
    for i in range(3):
        rp_p = robot_pose.copy(); rp_p[i] += eps
        rp_m = robot_pose.copy(); rp_m[i] -= eps
        dz   = h_line(rp_p, line_params[0], line_params[1]) \
             - h_line(rp_m, line_params[0], line_params[1])
        dz[1] = _wrap(dz[1])
        H_v[:, i] = dz / (2 * eps)

    H_f = np.zeros((2, 2))
    for i in range(2):
        lp_p = line_params.copy(); lp_p[i] += eps
        lp_m = line_params.copy(); lp_m[i] -= eps
        dz   = h_line(robot_pose, lp_p[0], lp_p[1]) \
             - h_line(robot_pose, lp_m[0], lp_m[1])
        dz[1] = _wrap(dz[1])
        H_f[:, i] = dz / (2 * eps)

    return H_v, H_f


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic observation generator
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Synthetic feature extractor for simulation.

    Instead of running a real line/corner detector on the laser point cloud
    (a research problem in its own right), this class uses ground-truth world
    geometry to generate noisy corner and line observations directly.

    Visibility criteria
    -------------------
    A feature is observable if:
      1. It is within sensor max_range of the robot.
      2. It is within the sensor FOV (bearing within ±fov_half).
      3. Line-of-sight is not occluded (ray-cast to the feature).

    For lines, additionally:
      4. The robot must be on the outward-normal side of the wall.
      5. When a scan is provided, at least SUPPORT_MIN_HITS laser returns
         must fall on the wall (prevents phantom wall observations).
      6. Along-segment gaps in laser returns produce separate observations
         (gap detection via angular-spacing analysis).

    Parameters
    ----------
    fov_rad       : total FOV in radians (half-angle = fov_rad/2)
    max_range     : maximum detection range (m)
    noise_range   : 1-sigma range / rho noise (m)
    noise_bearing : 1-sigma bearing / alpha noise (rad)
    noise_rho     : if provided, overrides noise_range for line rho noise
    noise_alpha   : if provided, overrides noise_bearing for line alpha noise
    rng           : numpy Generator
    """

    # minimum laser hits on a segment before accepting it as a line obs
    SUPPORT_MIN_HITS = 2
    # gap multiplier: gap > GAP_MULTIPLIER × expected_angular_spacing → split
    GAP_MULTIPLIER   = 3.0
    # minimum cluster size after gap-split to emit an observation
    GAP_MIN_CLUSTER  = 2

    def __init__(self,
                 fov_rad:       float,
                 max_range:     float,
                 noise_range:   float,
                 noise_bearing: float,
                 noise_rho:     float | None = None,
                 noise_alpha:   float | None = None,
                 rng: np.random.Generator | None = None):
        self.fov_half      = fov_rad / 2.0
        self.max_range     = float(max_range)
        self.noise_range   = float(noise_range)
        self.noise_bearing = float(noise_bearing)
        # line-specific noise (fall back to range/bearing noise if not given)
        self.noise_rho     = float(noise_rho)   if noise_rho   is not None else self.noise_range
        self.noise_alpha   = float(noise_alpha) if noise_alpha is not None else self.noise_bearing
        self._rng          = rng or np.random.default_rng()

    @classmethod
    def from_cfg(cls, cfg, rng=None) -> 'FeatureExtractor':
        return cls(
            fov_rad       = np.radians(cfg.sensor.fov_deg),
            max_range     = cfg.sensor.max_range,
            noise_range   = cfg.sensor.noise_range,
            noise_bearing = cfg.sensor.noise_bearing,
            rng           = rng,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self,
                robot_pose: np.ndarray,
                world,
                scan=None,
                ) -> Tuple[List[CornerObs], List[LineObs]]:
        """
        Generate all visible corner and line observations from the robot's
        current pose.

        Parameters
        ----------
        robot_pose : (3,) [xv, yv, theta_v]
        world      : env.World
        scan       : ScanResult | None — when provided, enables laser-support
                     gating and gap detection for lines

        Returns
        -------
        corners : list[CornerObs]
        lines   : list[LineObs]
        """
        corners = self._extract_corners(robot_pose, world)
        lines   = self._extract_lines(robot_pose, world, scan)
        return corners, lines

    # ── corners ───────────────────────────────────────────────────────────────

    def _extract_corners(self,
                         robot_pose: np.ndarray,
                         world) -> List[CornerObs]:
        xv, yv, tv = robot_pose
        origin     = np.array([xv, yv])
        R          = self._R_corner()
        obs        = []

        for corner in world.corners:
            cx, cy = corner.pos
            dx, dy = cx - xv, cy - yv
            dist   = np.sqrt(dx**2 + dy**2)

            if dist > self.max_range or dist < 1e-6:
                continue

            bearing = _wrap(np.arctan2(dy, dx) - tv)
            if abs(bearing) > self.fov_half:
                continue

            # occlusion check
            angle  = np.arctan2(dy, dx)
            result = world.ray_intersect(origin, angle, self.max_range)
            if result is None:
                continue
            hit_dist, _ = result
            if hit_dist < dist - 0.15:   # 15 cm tolerance for wall junctions
                continue

            r_noisy    = max(0.0, dist + self._rng.normal(0, self.noise_range))
            beta_noisy = _wrap(bearing + self._rng.normal(0, self.noise_bearing))

            obs.append(CornerObs(
                z      = np.array([r_noisy, beta_noisy]),
                R      = R.copy(),
                kind   = corner.kind,   # 'convex' | 'concave' — geometry type
                gt_pos = corner.pos.copy(),
            ))

        return obs

    # ── lines ────────────────────────────────────────────────────────────────

    def _extract_lines(self,
                       robot_pose: np.ndarray,
                       world,
                       scan=None) -> List[LineObs]:
        """
        Extract line observations with:
          - Facing-side check (only observe walls from the outward-normal side)
          - Collinear segment grouping (one (rho,alpha) → one or more obs)
          - Laser-support gating when a scan is provided
          - Gap detection: along-segment laser gaps → separate observations
        """
        xv, yv, tv = robot_pose
        origin      = np.array([xv, yv])
        R           = self._R_line()
        obs         = []
        hit_pts     = scan.valid_hits if scan is not None else None

        # Group segments that share the same infinite polar line
        groups: dict = {}
        for seg in world.segments:
            rho, alpha = seg.as_polar_line()
            key = (round(rho, 2), round(np.degrees(alpha), 1))
            groups.setdefault(key, []).append(seg)

        for _key, seg_list in groups.items():
            ref_seg    = seg_list[0]
            rho, alpha = ref_seg.as_polar_line()
            seg_normal = ref_seg.normal
            seg_dir    = ref_seg.direction

            # 1. Compute noiseless measurement; positive rho_obs means
            #    the robot is on the observable side of this wall.
            #    This replaces the normal-dot check which fails for walls
            #    whose segment direction convention places the normal on the
            #    wrong side (e.g. branch corridor interior walls).
            z_clean = h_line(robot_pose, rho, alpha)
            if z_clean[0] <= 0:
                continue

            # 3. Range gate: closest point on any segment in group
            any_close = any(
                _closest_point_on_segment(origin, seg.p0, seg.p1)[1] <= self.max_range
                for seg in seg_list)
            if not any_close:
                continue

            # 4. FOV gate: at least one close point is within bearing cone
            in_fov = False
            for seg in seg_list:
                closest, _ = _closest_point_on_segment(origin, seg.p0, seg.p1)
                brg = _wrap(np.arctan2(closest[1] - yv, closest[0] - xv) - tv)
                if abs(brg) <= self.fov_half:
                    in_fov = True
                    break
            if not in_fov:
                continue

            # 5. Laser-support gating + gap detection (only when scan available)
            if hit_pts is not None and len(hit_pts) > 0:
                tol_perp     = max(0.10, 3 * self.noise_range)
                endpoint_tol = max(0.15, 4 * self.noise_range)

                group_on = np.zeros(len(hit_pts), dtype=bool)
                for seg in seg_list:
                    rel    = hit_pts - seg.p0
                    t_proj = rel @ seg.direction
                    d_perp = np.abs(rel @ seg.normal)
                    d_p0   = np.linalg.norm(hit_pts - seg.p0, axis=1)
                    d_p1   = np.linalg.norm(hit_pts - seg.p1, axis=1)
                    group_on |= (
                        (t_proj  >  endpoint_tol) &
                        (t_proj  <  seg.length - endpoint_tol) &
                        (d_perp  <= tol_perp) &
                        (d_p0    >  endpoint_tol) &
                        (d_p1    >  endpoint_tol)
                    )

                if int(group_on.sum()) < self.SUPPORT_MIN_HITS:
                    continue

                support_hits = hit_pts[group_on]
                t_along      = (support_hits - ref_seg.p0) @ seg_dir
                sort_idx     = np.argsort(t_along)
                t_sorted     = t_along[sort_idx]
                pts_sorted   = support_hits[sort_idx]

                clusters = self._split_into_clusters(
                    t_sorted, pts_sorted,
                    seg_p0=ref_seg.p0, seg_dir=seg_dir, robot_pos=origin)

                for cluster_pts in clusters:
                    if len(cluster_pts) < self.GAP_MIN_CLUSTER:
                        continue
                    cluster_mean = cluster_pts.mean(axis=0)
                    full_mean    = support_hits.mean(axis=0)
                    delta_rho    = float(np.dot(cluster_mean - full_mean, seg_normal))
                    rho_obs_c    = z_clean[0] + delta_rho
                    if rho_obs_c < 0:
                        continue
                    rho_n   = rho_obs_c + self._rng.normal(0, self.noise_rho)
                    alpha_n = _wrap(z_clean[1] + self._rng.normal(0, self.noise_alpha))
                    # find which segment this cluster came from (best overlap)
                    src_seg = _best_segment_for_cluster(cluster_pts, seg_list)
                    obs.append(LineObs(
                        z                = np.array([rho_n, alpha_n]),
                        R                = R.copy(),
                        kind             = 'line',
                        seg_p0           = src_seg.p0.copy(),
                        seg_p1           = src_seg.p1.copy(),
                        cluster_midpoint = cluster_mean.copy(),
                        gt_rho           = rho,
                        gt_alpha         = alpha,
                    ))

            else:
                # No scan: one observation per group, use ref segment endpoints
                rho_n   = z_clean[0] + self._rng.normal(0, self.noise_rho)
                alpha_n = _wrap(z_clean[1] + self._rng.normal(0, self.noise_alpha))
                mid     = ref_seg.midpoint
                obs.append(LineObs(
                    z                = np.array([rho_n, alpha_n]),
                    R                = R.copy(),
                    kind             = 'line',
                    seg_p0           = ref_seg.p0.copy(),
                    seg_p1           = ref_seg.p1.copy(),
                    cluster_midpoint = mid.copy(),
                    gt_rho           = rho,
                    gt_alpha         = alpha,
                ))

        return obs

    # ── gap splitting ─────────────────────────────────────────────────────────

    def _split_into_clusters(self,
                             t_sorted:   np.ndarray,
                             pts_sorted: np.ndarray,
                             seg_p0:     np.ndarray,
                             seg_dir:    np.ndarray,
                             robot_pos:  np.ndarray
                             ) -> List[np.ndarray]:
        """
        Split sorted along-segment hit points into contiguous clusters by
        detecting gaps wider than GAP_MULTIPLIER × expected angular spacing.
        """
        d_theta    = np.radians(1.0)   # ~1 degree between adjacent rays
        seg_normal = np.array([-seg_dir[1], seg_dir[0]])

        if len(t_sorted) == 0:
            return []
        if len(t_sorted) == 1:
            return [pts_sorted]

        cluster_starts = [0]
        for i in range(len(t_sorted) - 1):
            gap       = t_sorted[i+1] - t_sorted[i]
            t_mid     = 0.5 * (t_sorted[i] + t_sorted[i+1])
            mid_world = seg_p0 + t_mid * seg_dir
            range_mid = np.linalg.norm(mid_world - robot_pos)
            to_mid    = mid_world - robot_pos
            cos_inc   = abs(np.dot(to_mid / (range_mid + 1e-9), seg_normal))
            expected  = range_mid * d_theta / max(cos_inc, 0.05)
            if gap > self.GAP_MULTIPLIER * expected:
                cluster_starts.append(i + 1)

        clusters = []
        for k, start in enumerate(cluster_starts):
            end = cluster_starts[k+1] if k+1 < len(cluster_starts) else len(pts_sorted)
            clusters.append(pts_sorted[start:end])
        return clusters

    # ── noise covariances ─────────────────────────────────────────────────────

    def _R_corner(self) -> np.ndarray:
        sr = max(self.noise_range,   1e-4)
        sb = max(self.noise_bearing, 1e-4)
        return np.diag([sr**2, sb**2])

    def _R_line(self) -> np.ndarray:
        sr = max(self.noise_rho,   1e-4)
        sa = max(self.noise_alpha, 1e-4)
        return np.diag([sr**2, sa**2])


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(angle) -> float:
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _closest_point_on_segment(p: np.ndarray,
                               a: np.ndarray,
                               b: np.ndarray
                               ) -> Tuple[np.ndarray, float]:
    """Return (closest_point, distance) from point p to segment [a, b]."""
    ab = b - a
    t  = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t  = np.clip(t, 0.0, 1.0)
    c  = a + t * ab
    return c, float(np.linalg.norm(p - c))


def _best_segment_for_cluster(cluster_pts: np.ndarray,
                               seg_list: list) -> object:
    """
    Return the segment in seg_list that has the most cluster points
    projecting onto it.  Used to pick seg_p0/seg_p1 for the LineObs.
    """
    best_seg   = seg_list[0]
    best_count = 0
    for seg in seg_list:
        t_proj = (cluster_pts - seg.p0) @ seg.direction
        count  = int(((t_proj >= 0) & (t_proj <= seg.length)).sum())
        if count > best_count:
            best_count = count
            best_seg   = seg
    return best_seg