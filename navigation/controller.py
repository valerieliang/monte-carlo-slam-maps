"""
navigation/controller.py
------------------------
Unicycle controller with pure-pursuit, proximity steering, and wall-following
escape for when the robot gets stuck.

State machine:

  PURSUE — align to goal bearing then drive.
           A forward-facing ray fan detects nearby walls and applies:
             1. Speed reduction proportional to proximity
             2. Angular steering away from the closest wall
           When the wall is directly between robot and goal (robot can see the
           wall but not the goal) the controller switches to FOLLOW.

  FOLLOW — wall-following escape.  Arcs at constant speed in the direction
           with more clearance until follow_duration seconds have elapsed.

Stuck detection (PURSUE only):
  If the robot hasn't reduced distance to the goal by stuck_thresh metres in
  check_interval seconds, FOLLOW is triggered.
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple

ALIGN_THRESH = np.radians(20)   # heading error below which we start driving


class Mode(Enum):
    PURSUE = auto()
    FOLLOW = auto()
    DONE   = auto()


class Controller:
    """
    Unicycle controller: pursue → wall-follow escape → pursue.

    Parameters
    ----------
    k_v             : linear velocity gain (cruise speed when clear)
    k_w             : angular velocity gain
    max_v           : maximum linear velocity (m/s)
    max_omega       : maximum angular velocity (rad/s)
    goal_tolerance  : goal reached when dist < this (m)
    clearance_dist  : wall proximity detection range (m).
                      Robot slows and steers when walls are closer than this.
                      Should be at least 2 × max_v × dt to give enough warning.
    stuck_thresh    : min forward progress per check_interval to avoid stuck (m)
    check_interval  : progress check period (s)
    follow_duration : wall-follow escape duration (s)
    """

    def __init__(self,
                 k_v:             float = 0.4,
                 k_w:             float = 1.5,
                 max_v:           float = 0.5,
                 max_omega:       float = 1.0,
                 goal_tolerance:  float = 0.30,
                 clearance_dist:  float = 0.8,
                 stuck_thresh:    float = 0.10,
                 check_interval:  float = 1.5,
                 follow_duration: float = 2.0):
        self.k_v             = k_v
        self.k_w             = k_w
        self.max_v           = max_v
        self.max_omega       = max_omega
        self.goal_tolerance  = goal_tolerance
        self.clearance_dist  = clearance_dist
        self.stuck_thresh    = stuck_thresh
        self.check_interval  = check_interval
        self.follow_duration = follow_duration

        self._goal:           Optional[np.ndarray] = None
        self._mode:           Mode  = Mode.PURSUE
        self._time:           float = 0.0
        self._last_check:     float = 0.0
        self._dist_at_check:  float = np.inf
        self._stuck_timer:    float = 0.0
        self._follow_timer:   float = 0.0
        self._follow_sign:    float = 1.0

    @classmethod
    def from_cfg(cls, cfg) -> 'Controller':
        return cls(
            k_v            = cfg.navigation.controller_k_v,
            k_w            = cfg.navigation.controller_k_w,
            max_v          = cfg.robot.max_v,
            max_omega      = cfg.robot.max_omega,
            goal_tolerance = cfg.navigation.goal_tolerance,
        )

    # ---------------------------------------------------------------- public

    def set_goal(self, goal: Optional[np.ndarray]) -> None:
        self._goal          = goal.copy() if goal is not None else None
        self._mode          = Mode.PURSUE if goal is not None else Mode.DONE
        self._dist_at_check = np.inf
        self._stuck_timer   = 0.0
        self._follow_timer  = 0.0
        self._last_check    = self._time

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    @property
    def mode(self) -> Mode:
        return self._mode

    def goal_reached(self, robot_pos: np.ndarray) -> bool:
        if self._goal is None:
            return False
        return float(np.linalg.norm(robot_pos - self._goal)) < self.goal_tolerance

    def step(self,
             robot_pose: np.ndarray,
             world,
             dt: float) -> Tuple[float, float]:
        self._time += dt

        if self._goal is None or self._mode == Mode.DONE:
            return 0.0, 0.0

        pos = robot_pose[:2]
        th  = robot_pose[2]

        if self.goal_reached(pos):
            self._mode = Mode.DONE
            return 0.0, 0.0

        if self._mode == Mode.PURSUE:
            v, omega = self._pursue(pos, th, world)

            self._update_stuck(pos, dt)
            if self._stuck_timer >= self.check_interval:
                self._mode         = Mode.FOLLOW
                self._follow_timer = 0.0
                self._stuck_timer  = 0.0
                self._follow_sign  = self._choose_follow_side(pos, th, world)
        else:
            v, omega = self._follow(th)
            self._follow_timer += dt
            if self._follow_timer >= self.follow_duration:
                self._mode          = Mode.PURSUE
                self._dist_at_check = np.inf
                self._stuck_timer   = 0.0
                self._last_check    = self._time

        v     = float(np.clip(v,     -self.max_v,     self.max_v))
        omega = float(np.clip(omega, -self.max_omega,  self.max_omega))
        return v, omega

    # ---------------------------------------------------------------- private

    def _pursue(self, pos: np.ndarray, th: float,
                world=None) -> Tuple[float, float]:
        """
        Align-then-drive with proximity steering.

        Proximity steering works in two steps:
          1. Scan: cast 9 rays in a half-sphere (±80° in 20° steps).
             For each ray, compute a repulsion vector pointing away from
             the wall, scaled by (1 - hit/clearance)^2.
          2. Blend: the summed repulsion vector is converted to an angular
             correction.  Speed is reduced when any wall is very close,
             preventing the robot from closing faster than it can react.
        """
        dx      = self._goal[0] - pos[0]
        dy      = self._goal[1] - pos[1]
        bearing = np.arctan2(dy, dx)
        d_theta = _wrap(bearing - th)
        omega   = float(np.clip(self.k_w * d_theta, -self.max_omega, self.max_omega))

        if abs(d_theta) > ALIGN_THRESH:
            return 0.0, omega

        if world is None or self.clearance_dist <= 0:
            return self.k_v, omega

        # --- Proximity scan ---
        # Cast rays every 20° across ±80° (9 rays total, including straight ahead)
        offsets = np.radians(np.arange(-80, 81, 20))
        min_clearance = self.clearance_dist   # closest wall in any direction
        repulse_x = 0.0   # repulsion vector in world frame
        repulse_y = 0.0

        for off in offsets:
            ray_angle = _wrap(th + off)
            result    = world.ray_intersect(pos, ray_angle,
                                            max_range=self.clearance_dist)
            if result is None:
                continue
            hit_dist, _ = result
            # Quadratic falloff: strong repulsion when close
            strength = (1.0 - hit_dist / self.clearance_dist) ** 2
            min_clearance = min(min_clearance, hit_dist)
            # Repulsion vector: directly away from the wall
            repulse_x -= strength * np.cos(ray_angle)
            repulse_y -= strength * np.sin(ray_angle)

        # Convert repulsion vector to angular correction
        if abs(repulse_x) > 1e-6 or abs(repulse_y) > 1e-6:
            repulse_bearing = np.arctan2(repulse_y, repulse_x)
            repulse_dtheta  = _wrap(repulse_bearing - th)
            # Scale by how strong the repulsion is
            repulse_mag     = np.hypot(repulse_x, repulse_y)
            repulse_omega   = self.k_w * repulse_dtheta * min(repulse_mag, 1.0)
            omega = float(np.clip(omega + repulse_omega,
                                  -self.max_omega, self.max_omega))

        # Speed reduction: slow down proportionally when walls are close.
        # At min_clearance=0 the robot stops; at clearance_dist it goes full speed.
        speed_factor = np.clip(min_clearance / self.clearance_dist, 0.2, 1.0)
        v = self.k_v * speed_factor

        return v, omega

    def _update_stuck(self, pos: np.ndarray, dt: float) -> None:
        dist_to_goal = float(np.linalg.norm(pos - self._goal))
        self._stuck_timer += dt
        if self._time - self._last_check >= self.check_interval:
            progress = self._dist_at_check - dist_to_goal
            if progress >= self.stuck_thresh:
                self._stuck_timer = 0.0
            self._dist_at_check = dist_to_goal
            self._last_check    = self._time

    def _choose_follow_side(self, pos, th, world) -> float:
        dx      = self._goal[0] - pos[0]
        dy      = self._goal[1] - pos[1]
        bearing = np.arctan2(dy, dx)

        def clearance(angle):
            r = world.ray_intersect(pos, angle, max_range=5.0)
            return r[0] if r else 5.0

        left  = clearance(_wrap(bearing + np.pi / 4))
        right = clearance(_wrap(bearing - np.pi / 4))
        return 1.0 if left >= right else -1.0

    def _follow(self, th: float) -> Tuple[float, float]:
        return self.k_v * 0.5, self._follow_sign * self.max_omega * 0.7


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)