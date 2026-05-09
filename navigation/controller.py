"""
navigation/controller.py
------------------------
Unicycle controller that drives toward a goal with a wall-following
escape for when the robot gets stuck.

Architecture
------------
The controller runs as a state machine with two modes:

  PURSUE  — pure-pursuit toward the goal.
            Compute heading error Δθ = goal_bearing - robot_heading.
            ω = k_w * Δθ   (proportional heading control)
            v = k_v * (1 - |Δθ|/π)  (slow down on sharp turns)

  FOLLOW  — wall-following escape when stuck.
            Detected when the robot hasn't made progress toward the goal
            for `stuck_time` seconds.  The robot turns and moves parallel
            to the nearest obstacle until it can see the goal again.

The stuck detector measures displacement toward the goal every
`check_interval` seconds.  If the robot moved less than `stuck_thresh`
metres toward the goal in that window, it's stuck.

Parameters (all tunable via NavigationCfg)
------------------------------------------
  k_v             : linear  velocity gain
  k_w             : angular velocity gain
  goal_tolerance  : distance at which goal is "reached" (m)
  stuck_thresh    : minimum progress per check_interval to avoid stuck (m)
  check_interval  : seconds between progress checks
  stuck_time      : seconds of no progress before triggering wall-follow
  follow_duration : seconds to wall-follow before re-evaluating goal
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple


class Mode(Enum):
    PURSUE = auto()
    FOLLOW = auto()
    DONE   = auto()


class Controller:
    """
    Unicycle controller: pursue → wall-follow escape → pursue.

    Parameters
    ----------
    k_v            : linear velocity gain
    k_w            : angular velocity gain
    max_v          : maximum linear velocity (m/s)
    max_omega      : maximum angular velocity (rad/s)
    goal_tolerance : goal reached when dist < this (m)
    stuck_thresh   : min forward progress per check (m)
    check_interval : progress check period (s)
    follow_duration: wall-follow escape duration (s)
    """

    def __init__(self,
                 k_v:             float = 0.4,
                 k_w:             float = 1.2,
                 max_v:           float = 0.5,
                 max_omega:       float = 1.0,
                 goal_tolerance:  float = 0.30,
                 stuck_thresh:    float = 0.10,
                 check_interval:  float = 1.0,
                 follow_duration: float = 2.0):
        self.k_v             = k_v
        self.k_w             = k_w
        self.max_v           = max_v
        self.max_omega       = max_omega
        self.goal_tolerance  = goal_tolerance
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
        self._follow_sign:    float = 1.0   # +1 or -1 wall-follow direction

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
        """Set a new navigation goal.  Pass None to stop."""
        self._goal        = goal.copy() if goal is not None else None
        self._mode        = Mode.PURSUE
        self._dist_at_check = np.inf
        self._stuck_timer   = 0.0
        self._follow_timer  = 0.0

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
        """
        Compute (v, omega) for this timestep.

        Parameters
        ----------
        robot_pose : (3,) [x, y, theta]
        world      : env.World  (for wall-follow obstacle sensing)
        dt         : timestep (s)

        Returns
        -------
        (v, omega) clipped to [−max_v, max_v] and [−max_omega, max_omega]
        """
        self._time += dt

        if self._goal is None or self._mode == Mode.DONE:
            return 0.0, 0.0

        pos = robot_pose[:2]
        th  = robot_pose[2]

        if self.goal_reached(pos):
            self._mode = Mode.DONE
            return 0.0, 0.0

        if self._mode == Mode.PURSUE:
            v, omega = self._pursue(pos, th)
            self._update_stuck(pos, dt)
            if self._stuck_timer >= self.check_interval:
                self._mode        = Mode.FOLLOW
                self._follow_timer = 0.0
                self._follow_sign  = self._choose_follow_side(pos, th, world)
                self._stuck_timer  = 0.0
        else:  # FOLLOW
            v, omega = self._follow(th)
            self._follow_timer += dt
            if self._follow_timer >= self.follow_duration:
                self._mode          = Mode.PURSUE
                self._dist_at_check = np.inf
                self._stuck_timer   = 0.0

        v     = float(np.clip(v,     -self.max_v,     self.max_v))
        omega = float(np.clip(omega, -self.max_omega,  self.max_omega))
        return v, omega

    # ---------------------------------------------------------------- private

    def _pursue(self, pos: np.ndarray, th: float) -> Tuple[float, float]:
        """Pure-pursuit toward goal."""
        dx      = self._goal[0] - pos[0]
        dy      = self._goal[1] - pos[1]
        bearing = np.arctan2(dy, dx)
        d_theta = _wrap(bearing - th)

        omega = self.k_w * d_theta
        # Slow down proportionally when heading error is large
        v     = self.k_v * max(0.0, 1.0 - abs(d_theta) / np.pi)

        return v, omega

    def _update_stuck(self, pos: np.ndarray, dt: float) -> None:
        """Accumulate stuck timer; reset if making progress."""
        dist_to_goal = float(np.linalg.norm(pos - self._goal))

        self._stuck_timer += dt

        if self._time - self._last_check >= self.check_interval:
            progress = self._dist_at_check - dist_to_goal
            if progress >= self.stuck_thresh:
                # Making progress — reset stuck timer
                self._stuck_timer = 0.0
            self._dist_at_check = dist_to_goal
            self._last_check    = self._time

    def _choose_follow_side(self,
                             pos:   np.ndarray,
                             th:    float,
                             world) -> float:
        """
        Decide whether to wall-follow left (+1) or right (-1).

        Cast two rays at ±45° from the goal bearing — whichever side
        has more clearance is the follow side.
        """
        dx      = self._goal[0] - pos[0]
        dy      = self._goal[1] - pos[1]
        bearing = np.arctan2(dy, dx)

        def clearance(angle: float) -> float:
            result = world.ray_intersect(pos, angle, max_range=5.0)
            return result[0] if result else 5.0

        left_clear  = clearance(_wrap(bearing + np.pi / 4))
        right_clear = clearance(_wrap(bearing - np.pi / 4))
        return 1.0 if left_clear >= right_clear else -1.0

    def _follow(self, th: float) -> Tuple[float, float]:
        """
        Wall-following: drive forward slowly while turning away from the wall.
        The robot arcs around the obstacle by applying a constant turn
        in the direction of more clearance.
        """
        v     = self.k_v * 0.5
        omega = self._follow_sign * self.max_omega * 0.6
        return v, omega


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)