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
from dataclasses import dataclass
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
    z       : (2,)  [range, bearing]  in robot frame
    R       : (2,2) measurement noise covariance
    kind    : 'convex' | 'concave'
    gt_pos  : (2,) ground-truth world position  (available in sim; not used
              by EKF — here for test assertions only)
    """
    z:      np.ndarray      # (2,)  [r, beta]
    R:      np.ndarray      # (2,2)
    kind:   str             = 'convex'
    gt_pos: Optional[np.ndarray] = None


@dataclass
class LineObs:
    """
    A single noisy line observation produced by the feature extractor.

    Attributes
    ----------
    z               : (2,)  [rho_obs, alpha_obs]  in robot frame
    R               : (2,2) measurement noise covariance
    gt_rho          : ground-truth rho   (sim only)
    gt_alpha        : ground-truth alpha (sim only)
    cluster_midpoint: (2,) world-frame centre of the observed wall cluster.
                      Used by the renderer to place the '+' marker at the
                      correct location when multiple sections of the same
                      infinite line are observed.  None = use foot-of-perp.
    """
    z:                np.ndarray    # (2,)  [rho_obs, alpha_obs]
    R:                np.ndarray    # (2,2)
    gt_rho:           Optional[float]          = None
    gt_alpha:         Optional[float]          = None
    cluster_midpoint: Optional[np.ndarray]     = None


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

def feature_jacobians(robot_pose:  np.ndarray,
                      feature_type: str,
                      feature_params: np.ndarray) -> Jacobians:
    """
    Compute analytic Jacobians of the measurement model.

    Parameters
    ----------
    robot_pose     : (3,) [xv, yv, theta_v]
    feature_type   : 'corner' | 'line'
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
        dr/dxv    = -dx/r
        dr/dyv    = -dy/r
        dr/dtv    =  0
        dbeta/dxv =  dy/r^2
        dbeta/dyv = -dx/r^2
        dbeta/dtv = -1

    dz/d[cx, cy]:
        dr/dcx    =  dx/r
        dr/dcy    =  dy/r
        dbeta/dcx = -dy/r^2
        dbeta/dcy =  dx/r^2
    """
    xv, yv, tv = robot_pose
    cx, cy     = corner_pos
    dx, dy     = cx - xv, cy - yv
    r2         = dx**2 + dy**2
    r          = np.sqrt(r2)

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
    rho_obs   = rho - xv*cos(alpha) - yv*sin(alpha)
    alpha_obs = alpha - tv

    dz/d[xv, yv, tv]:
        drho_obs/dxv   = -cos(alpha)
        drho_obs/dyv   = -sin(alpha)
        drho_obs/dtv   =  0
        dalpha_obs/dxv =  0
        dalpha_obs/dyv =  0
        dalpha_obs/dtv = -1

    dz/d[rho, alpha]:
        drho_obs/drho      =  1
        drho_obs/dalpha    =  xv*sin(alpha) - yv*cos(alpha)
        dalpha_obs/drho    =  0
        dalpha_obs/dalpha  =  1
    """
    xv, yv, tv  = robot_pose
    rho, alpha  = line_params

    ca, sa = np.cos(alpha), np.sin(alpha)

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
# Numerical Jacobian (for testing only)
# ─────────────────────────────────────────────────────────────────────────────

def numerical_jacobian_corner(robot_pose:  np.ndarray,
                               corner_pos: np.ndarray,
                               eps:        float = 1e-6
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobians for corner model.  Used in tests to validate
    the analytic versions.

    Returns (H_v_num, H_f_num), both (2, 2or3).
    """
    def f(rp, cp):
        return h_corner(rp, cp)

    H_v = np.zeros((2, 3))
    for i in range(3):
        rp_p = robot_pose.copy(); rp_p[i] += eps
        rp_m = robot_pose.copy(); rp_m[i] -= eps
        H_v[:, i] = (f(rp_p, corner_pos) - f(rp_m, corner_pos)) / (2 * eps)

    H_f = np.zeros((2, 2))
    for i in range(2):
        cp_p = corner_pos.copy(); cp_p[i] += eps
        cp_m = corner_pos.copy(); cp_m[i] -= eps
        H_f[:, i] = (f(robot_pose, cp_p) - f(robot_pose, cp_m)) / (2 * eps)

    return H_v, H_f


def numerical_jacobian_line(robot_pose:   np.ndarray,
                             line_params:  np.ndarray,
                             eps:          float = 1e-6
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobians for line model."""
    def f(rp, lp):
        return h_line(rp, lp[0], lp[1])

    H_v = np.zeros((2, 3))
    for i in range(3):
        rp_p = robot_pose.copy(); rp_p[i] += eps
        rp_m = robot_pose.copy(); rp_m[i] -= eps
        dz   = f(rp_p, line_params) - f(rp_m, line_params)
        # wrap alpha_obs component
        dz[1] = _wrap(dz[1])
        H_v[:, i] = dz / (2 * eps)

    H_f = np.zeros((2, 2))
    for i in range(2):
        lp_p = line_params.copy(); lp_p[i] += eps
        lp_m = line_params.copy(); lp_m[i] -= eps
        dz   = f(robot_pose, lp_p) - f(robot_pose, lp_m)
        dz[1] = _wrap(dz[1])
        H_f[:, i] = dz / (2 * eps)

    return H_v, H_f


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic observation generator  (the "cheat" extractor)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Synthetic feature extractor for simulation.

    Instead of running a real line/corner detector on the laser point cloud
    (a research problem in its own right), this class uses ground-truth world
    geometry to generate noisy corner and line observations directly.

    This is the "cheat" described in the Phase 3 procedure notes:
      - Check each world corner / wall segment for visibility from the robot
      - If visible: generate a noisy observation with the correct noise model
      - Return the list of observations for the EKF to consume

    Visibility criteria
    -------------------
    A feature is observable if:
      1. It is within sensor max_range of the robot.
      2. It is within the sensor FOV (bearing within ±fov/2).
      3. The line-of-sight from robot to feature is not occluded by any wall.
         (Tested via World.ray_intersect toward the feature.)

    Parameters
    ----------
    fov_rad       : half-angle of FOV on each side  (total = 2 * fov_rad)
    max_range     : maximum detection range (m)
    noise_range   : 1-sigma range noise (m)
    noise_bearing : 1-sigma bearing noise (rad)
    rng           : numpy Generator
    """

    def __init__(self,
                 fov_rad:       float,
                 max_range:     float,
                 noise_range:   float,
                 noise_bearing: float,
                 rng: np.random.Generator | None = None):
        self.fov_half      = fov_rad / 2.0        # ±fov_half from heading
        self.max_range     = float(max_range)
        self.noise_range   = float(noise_range)
        self.noise_bearing = float(noise_bearing)
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
        robot_pose : (3,) [xv, yv, theta_v]   — true pose (EKF uses estimated)
        world      : env.World
        scan       : ScanResult | None  — if provided, used for laser-support
                     validation of line features (rejects phantom walls)

        Returns
        -------
        corners : list of CornerObs
        lines   : list of LineObs
        """
        corners = self._extract_corners(robot_pose, world)
        lines   = self._extract_lines(robot_pose, world, scan)
        # No deduplication needed: _extract_lines groups collinear segments,
        # so each unique (rho,alpha) line is processed exactly once.
        # Multiple observations per line (different rho_obs) represent
        # distinct wall sections separated by a physical gap.
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

            # range gate
            if dist > self.max_range or dist < 1e-6:
                continue

            # bearing gate
            bearing = _wrap(np.arctan2(dy, dx) - tv)
            if abs(bearing) > self.fov_half:
                continue

            # occlusion check — cast ray toward corner;
            # hit must be at least as far as the corner itself
            angle  = np.arctan2(dy, dx)
            result = world.ray_intersect(origin, angle, self.max_range)
            if result is None:
                continue
            hit_dist, _ = result
            if hit_dist < dist - 0.15:   # 15 cm tolerance for wall junctions
                continue

            # generate noisy observation
            r_noisy    = dist    + self._rng.normal(0, self.noise_range)
            beta_noisy = bearing + self._rng.normal(0, self.noise_bearing)
            r_noisy    = max(0.0, r_noisy)
            beta_noisy = _wrap(beta_noisy)

            obs.append(CornerObs(
                z      = np.array([r_noisy, beta_noisy]),
                R      = R.copy(),
                kind   = corner.kind,
                gt_pos = corner.pos.copy(),
            ))

        return obs

    # ── lines ────────────────────────────────────────────────────────────────

    # minimum fraction of a segment's length that must be covered by laser
    # returns before accepting the segment as a valid line observation.
    SUPPORT_FRACTION = 0.25
    # minimum absolute number of supporting laser hits
    SUPPORT_MIN_HITS = 2
    # gap detection: a spacing between adjacent along-segment hits is a real
    # gap if it exceeds GAP_MULTIPLIER * expected_angular_spacing at that range.
    # Higher = fewer splits (more tolerant of angular thinning).
    GAP_MULTIPLIER   = 3.0
    # minimum cluster size to emit a sub-segment observation after splitting
    GAP_MIN_CLUSTER  = 2

    def _extract_lines(self,
                       robot_pose: np.ndarray,
                       world,
                       scan=None) -> List[LineObs]:
        """
        Extract line observations, grouping collinear segments so that gaps
        between adjacent collinear wall sections produce separate observations.
        """
        xv, yv, tv = robot_pose
        origin      = np.array([xv, yv])
        R           = self._R_line()
        obs         = []
        hit_pts     = scan.valid_hits if scan is not None else None

        # Group segments by shared polar line identity
        groups: dict = {}
        for seg in world.segments:
            rho, alpha = seg.as_polar_line()
            key = (round(rho, 2), round(np.degrees(alpha), 1))
            groups.setdefault(key, []).append(seg)

        for (rho_q, alpha_q_deg), seg_list in groups.items():
            ref_seg    = seg_list[0]
            rho, alpha = ref_seg.as_polar_line()
            seg_normal = ref_seg.normal
            seg_dir    = ref_seg.direction

            # facing-side check
            robot_side = np.dot(origin - ref_seg.p0, seg_normal)
            if robot_side <= 0:
                continue

            # noiseless measurement
            z_clean = h_line(robot_pose, rho, alpha)
            if z_clean[0] < 0:
                continue

            # range gate: any segment must be within max_range
            any_close = any(
                _closest_point_on_segment(origin, seg.p0, seg.p1)[1] <= self.max_range
                for seg in seg_list)
            if not any_close:
                continue

            # bearing gate: any closest point must be within FOV
            in_fov = False
            for seg in seg_list:
                closest, _ = _closest_point_on_segment(origin, seg.p0, seg.p1)
                bearing = _wrap(np.arctan2(closest[1]-yv, closest[0]-xv) - tv)
                if abs(bearing) <= self.fov_half:
                    in_fov = True
                    break
            if not in_fov:
                continue

            if hit_pts is not None and len(hit_pts) > 0:
                tol_perp     = max(0.10, 3 * self.noise_range)
                endpoint_tol = max(0.15, 4 * self.noise_range)

                # collect hits from ALL segments in group
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

                # gap split on merged hit cloud
                support_hits = hit_pts[group_on]
                common_p0    = ref_seg.p0
                t_along      = (support_hits - common_p0) @ seg_dir
                sort_idx     = np.argsort(t_along)
                t_sorted     = t_along[sort_idx]
                pts_sorted   = support_hits[sort_idx]

                clusters = self._split_into_clusters(
                    t_sorted, pts_sorted,
                    seg_p0=common_p0, seg_dir=seg_dir, robot_pos=origin)

                for cluster_pts in clusters:
                    if len(cluster_pts) < self.GAP_MIN_CLUSTER:
                        continue
                    full_mean    = support_hits.mean(axis=0)
                    cluster_mean = cluster_pts.mean(axis=0)
                    delta_rho    = float(np.dot(cluster_mean - full_mean, seg_normal))
                    rho_obs_c    = z_clean[0] + delta_rho
                    if rho_obs_c < 0:
                        continue
                    rho_n   = rho_obs_c + self._rng.normal(0, self.noise_range)
                    alpha_n = _wrap(z_clean[1] + self._rng.normal(0, self.noise_bearing))
                    obs.append(LineObs(
                        z=np.array([rho_n, alpha_n]), R=R.copy(),
                        gt_rho=rho, gt_alpha=alpha,
                        cluster_midpoint=cluster_mean.copy()))
            else:
                rho_n   = z_clean[0] + self._rng.normal(0, self.noise_range)
                alpha_n = _wrap(z_clean[1] + self._rng.normal(0, self.noise_bearing))
                obs.append(LineObs(
                    z=np.array([rho_n, alpha_n]), R=R.copy(),
                    gt_rho=rho, gt_alpha=alpha))

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
        Split along-segment hit points into contiguous clusters by detecting
        gaps that exceed the expected angular ray spacing at that range.

        A gap between adjacent hits is "real" (open space, not angular thinning)
        when it is wider than GAP_MULTIPLIER × expected_angular_spacing, where
        expected_angular_spacing = range * d_theta / cos(incidence_angle).

        Returns a list of (K_i, 2) arrays, one per cluster.
        """
        d_theta = np.radians(180.0 / 180.0)   # 1-deg ray spacing (181 rays, 180-deg FOV)
        seg_normal = np.array([-seg_dir[1], seg_dir[0]])

        if len(t_sorted) == 0:
            return []
        if len(t_sorted) == 1:
            return [pts_sorted]

        cluster_starts = [0]
        for i in range(len(t_sorted) - 1):
            gap = t_sorted[i+1] - t_sorted[i]
            # midpoint of candidate gap in world frame
            t_mid      = 0.5 * (t_sorted[i] + t_sorted[i+1])
            mid_world  = seg_p0 + t_mid * seg_dir
            range_mid  = np.linalg.norm(mid_world - robot_pos)
            to_mid     = mid_world - robot_pos
            cos_inc    = abs(np.dot(to_mid / (range_mid + 1e-9), seg_normal))
            expected   = range_mid * d_theta / max(cos_inc, 0.05)
            threshold  = self.GAP_MULTIPLIER * expected
            if gap > threshold:
                cluster_starts.append(i + 1)

        clusters = []
        for k, start in enumerate(cluster_starts):
            end = cluster_starts[k+1] if k+1 < len(cluster_starts) else len(pts_sorted)
            clusters.append(pts_sorted[start:end])
        return clusters

    # ── deduplication ────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_lines(lines: List[LineObs],
                           rho_tol:   float = 0.15,
                           alpha_tol: float = 0.05) -> List[LineObs]:
        """
        Deduplicate line observations with identical (rho, alpha) that arise
        from collinear segments being processed twice.  Observations from
        gap-split clusters on the same infinite line have the same gt_alpha
        but deliberately different rho_obs (z[0]) and are NOT deduplicated.
        """
        kept = []
        for obs in lines:
            duplicate = False
            for existing in kept:
                drho   = abs(obs.z[0]     - existing.z[0])
                dalpha = abs(_wrap(obs.z[1] - existing.z[1]))
                if drho < rho_tol and dalpha < alpha_tol:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(obs)
        return kept

    # ── noise covariances ─────────────────────────────────────────────────────

    def _R_corner(self) -> np.ndarray:
        return np.diag([self.noise_range**2, self.noise_bearing**2])

    def _R_line(self) -> np.ndarray:
        return np.diag([self.noise_range**2, self.noise_bearing**2])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(angle) -> float:
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _closest_point_on_segment(p: np.ndarray,
                               a: np.ndarray,
                               b: np.ndarray
                               ) -> Tuple[np.ndarray, float]:
    """
    Return (closest_point, distance) from point p to segment [a, b].
    """
    ab = b - a
    t  = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t  = np.clip(t, 0.0, 1.0)
    c  = a + t * ab
    return c, float(np.linalg.norm(p - c))