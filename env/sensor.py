"""
env/sensor.py
-------------
Simulated 2-D laser range scanner (SICK-style, matching the paper).

The scanner casts `num_rays` evenly-spaced rays across a symmetric FOV
centred on the robot's heading.  Each ray finds its closest wall intersection
via World.ray_intersect(), then adds independent Gaussian noise to both
range and bearing.

Returned data
-------------
scan()  ->  ScanResult
    .ranges      : (N,)   noisy range readings  (m),   np.inf if no hit
    .bearings    : (N,)   noisy bearing angles  (rad), robot frame
    .hit_xy      : (N,2)  Cartesian hit points in WORLD frame (nan if no hit)
    .ray_angles  : (N,)   true ray angles in world frame (no noise)

All angles are in the robot body frame unless stated otherwise.
Bearing convention: 0 = straight ahead, positive = left (CCW).

Coordinate frames
-----------------
  World frame  : fixed global 2-D Cartesian
  Robot frame  : origin at robot position, x-axis along robot heading

Phase 3 (feature extraction) will consume ScanResult directly.
Phase 2 only needs hit_xy for the raw point-cloud visualisation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.world import World
    from env.robot import Robot


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class ScanResult:
    """
    All data produced by one laser scan.

    Attributes
    ----------
    ranges      : noisy range for each ray (m); np.inf = no return
    bearings    : noisy bearing for each ray (rad, robot frame)
    hit_xy      : world-frame Cartesian coordinates of each hit point;
                  row is [nan, nan] when the ray had no return
    ray_angles  : true (noiseless) world-frame angle of each ray -- used
                  for rendering the beam fan
    true_ranges : ground-truth ranges before noise (for test assertions)
    """
    ranges:      np.ndarray   # (N,)
    bearings:    np.ndarray   # (N,)
    hit_xy:      np.ndarray   # (N, 2)
    ray_angles:  np.ndarray   # (N,)   world frame, no noise
    true_ranges: np.ndarray   # (N,)   ground truth

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: True where the ray returned a finite range."""
        return np.isfinite(self.ranges)

    @property
    def valid_hits(self) -> np.ndarray:
        """World-frame hit points for rays that returned (M, 2)."""
        return self.hit_xy[self.valid_mask]

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())


# -----------------------------------------------------------------------------
# Sensor
# -----------------------------------------------------------------------------

class Sensor:
    """
    2-D laser range scanner.

    Parameters (from SensorCfg)
    ---------------------------
    fov_deg       : total field of view in degrees (e.g. 180)
    num_rays      : number of evenly-spaced rays across the FOV (e.g. 181)
    max_range     : maximum measurable range in metres (e.g. 8.0)
    noise_range   : 1-sigma Gaussian noise on range  (m)
    noise_bearing : 1-sigma Gaussian noise on bearing (rad)
    rng           : optional numpy random Generator for reproducibility
    """

    def __init__(self,
                 fov_deg:       float = 180.0,
                 num_rays:      int   = 181,
                 max_range:     float = 8.0,
                 noise_range:   float = 0.03,
                 noise_bearing: float = 0.01,
                 rng: np.random.Generator | None = None):

        self.fov_rad       = np.radians(fov_deg)
        self.num_rays      = int(num_rays)
        self.max_range     = float(max_range)
        self.noise_range   = float(noise_range)
        self.noise_bearing = float(noise_bearing)
        self._rng          = rng or np.random.default_rng()

        # ray offsets in robot frame -- fixed, computed once
        # symmetric about 0: from -fov/2 to +fov/2
        self._ray_offsets: np.ndarray = np.linspace(
            -self.fov_rad / 2,
             self.fov_rad / 2,
             self.num_rays
        )

    # -- factory -----------------------------------------------------------------

    @classmethod
    def from_cfg(cls, cfg, rng=None) -> 'Sensor':
        """Construct from a SensorCfg dataclass."""
        return cls(
            fov_deg       = cfg.fov_deg,
            num_rays      = cfg.num_rays,
            max_range     = cfg.max_range,
            noise_range   = cfg.noise_range,
            noise_bearing = cfg.noise_bearing,
            rng           = rng,
        )

    # -- main API ----------------------------------------------------------------

    def scan(self, robot, world) -> ScanResult:
        """
        Cast all rays from the robot's current pose and return a ScanResult.

        Parameters
        ----------
        robot : env.Robot   (provides .x, .y, .theta)
        world : env.World   (provides .ray_intersect())
        """
        rx, ry, rtheta = robot.x, robot.y, robot.theta
        origin = np.array([rx, ry])

        N = self.num_rays
        # world-frame angle of each ray
        ray_angles = rtheta + self._ray_offsets   # (N,)

        true_ranges = np.full(N, np.inf)
        hit_xy      = np.full((N, 2), np.nan)

        # -- cast each ray -------------------------------------------------------
        for i, angle in enumerate(ray_angles):
            result = world.ray_intersect(origin, angle, self.max_range)
            if result is not None:
                dist, pt          = result
                true_ranges[i]    = dist
                hit_xy[i]         = pt

        # -- add noise -----------------------------------------------------------
        rng = self._rng
        range_noise   = rng.normal(0.0, self.noise_range,   N)
        bearing_noise = rng.normal(0.0, self.noise_bearing, N)

        noisy_ranges   = true_ranges.copy()
        valid          = np.isfinite(true_ranges)
        noisy_ranges[valid] += range_noise[valid]
        # clip to physical limits
        noisy_ranges   = np.clip(noisy_ranges, 0.0, self.max_range)

        # bearings in robot frame with noise
        noisy_bearings = self._ray_offsets + bearing_noise

        # recompute noisy hit_xy from noisy ranges (robot-frame bearings
        # converted to world frame)
        noisy_world_angles = rtheta + noisy_bearings
        hit_xy_noisy       = np.full((N, 2), np.nan)
        hit_xy_noisy[valid, 0] = (rx
                                   + noisy_ranges[valid]
                                   * np.cos(noisy_world_angles[valid]))
        hit_xy_noisy[valid, 1] = (ry
                                   + noisy_ranges[valid]
                                   * np.sin(noisy_world_angles[valid]))

        return ScanResult(
            ranges      = noisy_ranges,
            bearings    = noisy_bearings,
            hit_xy      = hit_xy_noisy,
            ray_angles  = ray_angles,
            true_ranges = true_ranges,
        )

    # -- measurement model (used by EKF in Phase 4) ------------------------------

    def expected_corner(self,
                        robot_pose: np.ndarray,
                        corner_pos: np.ndarray
                        ) -> np.ndarray:
        """
        Noiseless expected observation of a corner feature (Eq. 4).

        Parameters
        ----------
        robot_pose : (3,) [x, y, theta]
        corner_pos : (2,) [cx, cy]

        Returns
        -------
        z : (2,) [range, bearing_in_robot_frame]
        """
        dx    = corner_pos[0] - robot_pose[0]
        dy    = corner_pos[1] - robot_pose[1]
        r     = np.sqrt(dx**2 + dy**2)
        beta  = np.arctan2(dy, dx) - robot_pose[2]
        beta  = self._wrap(beta)
        return np.array([r, beta])

    def expected_line(self,
                      robot_pose: np.ndarray,
                      rho: float,
                      alpha: float
                      ) -> np.ndarray:
        """
        Noiseless expected observation of a line feature (Eq. 5).

        Parameters
        ----------
        robot_pose : (3,) [x, y, theta]
        rho        : perpendicular distance from origin (m)
        alpha      : angle of line normal (rad)

        Returns
        -------
        z : (2,) [rho_obs, alpha_obs]  in robot frame
        """
        rx, ry, rtheta = robot_pose
        rho_obs   = rho - rx * np.cos(alpha) - ry * np.sin(alpha)
        alpha_obs = alpha - rtheta
        alpha_obs = self._wrap(alpha_obs)
        return np.array([rho_obs, alpha_obs])

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def __repr__(self) -> str:
        return (f"Sensor(fov={np.degrees(self.fov_rad):.0f}, "
                f"rays={self.num_rays}, "
                f"max_range={self.max_range}m)")