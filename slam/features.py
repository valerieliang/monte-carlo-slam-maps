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
    z        : (2,)  [rho_obs, alpha_obs]  in robot frame
    R        : (2,2) measurement noise covariance
    gt_rho   : ground-truth rho   (sim only)
    gt_alpha : ground-truth alpha (sim only)
    """
    z:        np.ndarray    # (2,)  [rho_obs, alpha_obs]
    R:        np.ndarray    # (2,2)
    gt_rho:   Optional[float] = None
    gt_alpha: Optional[float] = None


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

    def _extract_lines(self,
                       robot_pose: np.ndarray,
                       world,
                       scan=None) -> List[LineObs]:
        xv, yv, tv = robot_pose
        origin     = np.array([xv, yv])
        R          = self._R_line()
        obs        = []

        # Pre-compute valid hit points from scan for support check
        hit_pts = scan.valid_hits if scan is not None else None  # (M,2) or None

        for seg in world.segments:
            # closest point on segment to robot
            closest, dist = _closest_point_on_segment(origin, seg.p0, seg.p1)

            # range gate on closest point
            if dist > self.max_range:
                continue

            # bearing of closest point
            dx, dy  = closest - origin
            bearing = _wrap(np.arctan2(dy, dx) - tv)
            if abs(bearing) > self.fov_half:
                continue

            # ── facing-side check ────────────────────────────────────────────
            # The robot must be on the same side as the wall's outward normal.
            # If it's on the opposite side the wall is facing away — physically
            # impossible to observe with a laser scanner.
            seg_normal = seg.normal
            robot_side = np.dot(origin - seg.p0, seg_normal)
            if robot_side <= 0:
                continue

            # ── laser support check ───────────────────────────────────────────
            # Project scan hit points onto the segment and count those that:
            #   (a) fall within the segment region (between endpoint normals)
            #   (b) are within tol_perp of the segment's infinite line
            #   (c) are NOT within an endpoint exclusion zone — hits that
            #       land within endpoint_tol of p0 or p1 are junction grazes
            #       from an adjacent wall and must not count as support.
            if hit_pts is not None and len(hit_pts) > 0:
                seg_dir      = seg.direction
                rel          = hit_pts - seg.p0           # (M, 2)
                t_proj       = rel @ seg_dir              # (M,) along-seg coord
                d_perp       = np.abs(rel @ seg_normal)   # (M,) dist to line
                tol_perp     = max(0.10, 3 * self.noise_range)
                endpoint_tol = max(0.15, 4 * self.noise_range)
                # distance to each endpoint
                d_p0 = np.linalg.norm(hit_pts - seg.p0, axis=1)
                d_p1 = np.linalg.norm(hit_pts - seg.p1, axis=1)
                on_seg = (
                    (t_proj  >  endpoint_tol) &          # not a p0 graze
                    (t_proj  <  seg.length - endpoint_tol) &  # not a p1 graze
                    (d_perp  <= tol_perp) &
                    (d_p0    >  endpoint_tol) &
                    (d_p1    >  endpoint_tol)
                )
                n_support = int(on_seg.sum())
                if n_support < self.SUPPORT_MIN_HITS:
                    continue

            # polar representation of the segment's infinite line
            rho, alpha = seg.as_polar_line()

            # noiseless observation
            z_clean = h_line(robot_pose, rho, alpha)

            # add noise
            rho_n   = z_clean[0] + self._rng.normal(0, self.noise_range)
            alpha_n  = _wrap(z_clean[1] + self._rng.normal(0, self.noise_bearing))

            obs.append(LineObs(
                z        = np.array([rho_n, alpha_n]),
                R        = R.copy(),
                gt_rho   = rho,
                gt_alpha = alpha,
            ))

        return obs

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