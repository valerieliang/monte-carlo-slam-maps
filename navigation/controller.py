"""
navigation/controller.py
------------------------
Unicycle controller with waypoint routing, proximity steering, and
wall-following escape.

When set_goal() is called, the direct path is checked for wall collisions.
If blocked, intermediate waypoints are automatically inserted by routing
around wall segment endpoints (visibility graph, up to 2 hops).  The robot
drives through each waypoint in sequence before reaching the final goal.

State machine per waypoint:
  PURSUE  — align to bearing then drive with proximity repulsion
  FOLLOW  — wall-following escape when stuck

Proximity steering:
  9 rays at ±80° cast against the combined world (ground-truth + SLAM).
  Each hit applies a repulsion vector scaled by proximity.
  Speed is reduced near walls so the robot can't outrun its own sensors.
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Optional, List, Tuple

ALIGN_THRESH = np.radians(20)


class Mode(Enum):
    PURSUE = auto()
    FOLLOW = auto()
    DONE   = auto()


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _path_clear(world, p0: np.ndarray, p1: np.ndarray,
                margin: float = 0.90) -> bool:
    """True if the direct path p0→p1 is not blocked by any wall."""
    diff = p1 - p0
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return True
    angle  = np.arctan2(diff[1], diff[0])
    result = world.ray_intersect(p0, angle, max_range=dist * margin)
    return result is None


def _route(world, robot_pos: np.ndarray, goal: np.ndarray,
           clearance: float = 0.6) -> List[np.ndarray]:
    """
    Return a list of intermediate waypoints [wp1, wp2, ...] such that
    robot_pos → wp1 → ... → goal is free of wall intersections.

    Uses wall segment endpoints offset by `clearance` metres toward the
    path midpoint.  Clearance must be larger than clearance_dist so the
    waypoint lands in open space, not right against the wall.
    Tries 1-hop then 2-hop routing.
    Returns [] if the direct path is already clear.
    """
    if _path_clear(world, robot_pos, goal):
        return []

    mid = 0.5 * (robot_pos + goal)
    candidates: List[np.ndarray] = []
    seen: set = set()

    for seg in _iter_segments(world):
        for ep in seg:
            key = (round(ep[0], 2), round(ep[1], 2))
            if key in seen:
                continue
            seen.add(key)
            # Try offsets at clearance, 1.5×clearance, and 2× toward midpath
            # so we get candidates that land clearly in open space
            d = mid - ep
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            unit = d / n
            for scale in [1.0, 1.5, 2.0]:
                candidates.append(ep + scale * clearance * unit)

    # 1-hop
    best1: Optional[np.ndarray] = None
    best1_cost = np.inf
    for wp in candidates:
        if (_path_clear(world, robot_pos, wp) and
                _path_clear(world, wp, goal)):
            cost = (np.linalg.norm(wp - robot_pos) +
                    np.linalg.norm(goal - wp))
            if cost < best1_cost:
                best1_cost = cost
                best1 = wp

    if best1 is not None:
        return [best1]

    # 2-hop
    best2: Optional[List[np.ndarray]] = None
    best2_cost = np.inf
    for wp1 in candidates:
        if not _path_clear(world, robot_pos, wp1):
            continue
        for wp2 in candidates:
            if wp2 is wp1:
                continue
            if (_path_clear(world, wp1, wp2) and
                    _path_clear(world, wp2, goal)):
                cost = (np.linalg.norm(wp1 - robot_pos) +
                        np.linalg.norm(wp2 - wp1) +
                        np.linalg.norm(goal - wp2))
                if cost < best2_cost:
                    best2_cost = cost
                    best2 = [wp1, wp2]

    return best2 or []


def _iter_segments(world):
    """Yield (p0, p1) pairs from a World or SLAMWorld-like object."""
    if hasattr(world, 'segments'):
        for seg in world.segments:
            yield (seg.p0, seg.p1)
    elif hasattr(world, '_segments'):
        yield from world._segments
    # CombinedWorld — recurse into sub-worlds
    if hasattr(world, '_gt'):
        yield from _iter_segments(world._gt)
    if hasattr(world, '_slam'):
        yield from _iter_segments(world._slam)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class Controller:
    """
    Unicycle controller with automatic waypoint routing.

    Parameters
    ----------
    k_v             : linear velocity gain
    k_w             : angular velocity gain
    max_v           : max linear velocity (m/s)
    max_omega       : max angular velocity (rad/s)
    goal_tolerance  : reached when dist < this (m)
    clearance_dist  : proximity detection range (m)
    stuck_thresh    : min progress per check_interval (m)
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

        self._final_goal: Optional[np.ndarray] = None
        self._waypoints:  List[np.ndarray]     = []   # remaining waypoints
        self._mode:       Mode  = Mode.DONE
        self._time:       float = 0.0
        self._last_check: float = 0.0
        self._dist_at_check: float = np.inf
        self._stuck_timer:   float = 0.0
        self._follow_timer:  float = 0.0
        self._follow_sign:   float = 1.0

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

    def set_goal(self, goal: Optional[np.ndarray],
                 world=None) -> None:
        """
        Set a new navigation goal.  If `world` is supplied and the direct
        path is blocked, waypoints are inserted automatically.
        """
        if goal is None:
            self._final_goal = None
            self._waypoints  = []
            self._mode       = Mode.DONE
            return

        self._final_goal = goal.copy()

        if world is not None:
            robot_pos = self._current_pos if hasattr(self, '_current_pos') else goal
            wps = _route(world, robot_pos, goal)
        else:
            wps = []

        self._waypoints       = [wp.copy() for wp in wps]
        self._mode            = Mode.PURSUE
        self._dist_at_check   = np.inf
        self._stuck_timer     = 0.0
        self._follow_timer    = 0.0
        self._last_check      = self._time

    @property
    def goal(self) -> Optional[np.ndarray]:
        """The final goal (not the current intermediate waypoint)."""
        return self._final_goal

    @property
    def current_target(self) -> Optional[np.ndarray]:
        """The immediate driving target (next waypoint or final goal)."""
        if self._waypoints:
            return self._waypoints[0]
        return self._final_goal

    @property
    def mode(self) -> Mode:
        return self._mode

    def goal_reached(self, robot_pos: np.ndarray) -> bool:
        """True when the robot is within goal_tolerance of the FINAL goal."""
        if self._final_goal is None:
            return False
        return float(np.linalg.norm(robot_pos - self._final_goal)) < self.goal_tolerance

    def step(self,
             robot_pose: np.ndarray,
             world,
             dt: float,
             slam_world=None) -> Tuple[float, float]:
        """
        world      : combined world (gt + slam) used for proximity steering
        slam_world : SLAM-only world used for re-routing after FOLLOW escape.
                     If None, falls back to `world` for re-routing.
        """
        self._time += dt
        self._current_pos = robot_pose[:2].copy()

        target = self.current_target
        if target is None or self._mode == Mode.DONE:
            return 0.0, 0.0

        pos = robot_pose[:2]
        th  = robot_pose[2]

        # Check if final goal reached
        if self.goal_reached(pos):
            self._mode      = Mode.DONE
            self._waypoints = []
            return 0.0, 0.0

        # Check if current waypoint reached — advance to next
        if (self._waypoints and
                float(np.linalg.norm(pos - self._waypoints[0])) < self.goal_tolerance):
            self._waypoints.pop(0)
            self._dist_at_check = np.inf
            self._stuck_timer   = 0.0
            self._last_check    = self._time
            target = self.current_target
            if target is None:
                self._mode = Mode.DONE
                return 0.0, 0.0

        if self._mode == Mode.PURSUE:
            v, omega = self._pursue(pos, th, world, target)
            self._update_stuck(pos, target, dt)
            if self._stuck_timer >= self.check_interval:
                self._mode         = Mode.FOLLOW
                self._follow_timer = 0.0
                self._stuck_timer  = 0.0
                self._follow_sign  = self._choose_follow_side(pos, th, world, target)
        else:  # FOLLOW
            v, omega = self._follow(th)
            self._follow_timer += dt
            if self._follow_timer >= self.follow_duration:
                # Re-route from current position after escaping.
                # Use slam_world only so routing is based on what
                # the robot has actually mapped, not ground truth.
                route_world = slam_world if slam_world is not None else world
                if route_world is not None and self._final_goal is not None:
                    remaining = _route(route_world, pos, self._final_goal)
                    self._waypoints = [wp.copy() for wp in remaining]
                self._mode          = Mode.PURSUE
                self._dist_at_check = np.inf
                self._stuck_timer   = 0.0
                self._last_check    = self._time

        v     = float(np.clip(v,     -self.max_v,     self.max_v))
        omega = float(np.clip(omega, -self.max_omega,  self.max_omega))
        return v, omega

    # ---------------------------------------------------------------- private

    def _pursue(self, pos, th, world, target) -> Tuple[float, float]:
        dx      = target[0] - pos[0]
        dy      = target[1] - pos[1]
        bearing = np.arctan2(dy, dx)
        d_theta = _wrap(bearing - th)
        omega   = float(np.clip(self.k_w * d_theta, -self.max_omega, self.max_omega))

        if abs(d_theta) > ALIGN_THRESH:
            return 0.0, omega

        if world is None or self.clearance_dist <= 0:
            return self.k_v, omega

        offsets = np.radians(np.arange(-80, 81, 20))
        min_clearance = self.clearance_dist
        repulse_x = repulse_y = 0.0

        for off in offsets:
            ray_angle = _wrap(th + off)
            result    = world.ray_intersect(pos, ray_angle,
                                            max_range=self.clearance_dist)
            if result is None:
                continue
            hit_dist, _ = result
            strength      = (1.0 - hit_dist / self.clearance_dist) ** 2
            min_clearance = min(min_clearance, hit_dist)
            repulse_x    -= strength * np.cos(ray_angle)
            repulse_y    -= strength * np.sin(ray_angle)

        if abs(repulse_x) > 1e-6 or abs(repulse_y) > 1e-6:
            rep_bearing = np.arctan2(repulse_y, repulse_x)
            rep_dtheta  = _wrap(rep_bearing - th)
            rep_mag     = np.hypot(repulse_x, repulse_y)
            rep_omega   = self.k_w * rep_dtheta * min(rep_mag, 1.0)
            omega = float(np.clip(omega + rep_omega,
                                  -self.max_omega, self.max_omega))

        speed_factor = np.clip(min_clearance / self.clearance_dist, 0.2, 1.0)
        return self.k_v * speed_factor, omega

    def _update_stuck(self, pos, target, dt) -> None:
        dist_to_target = float(np.linalg.norm(pos - target))
        self._stuck_timer += dt
        if self._time - self._last_check >= self.check_interval:
            progress = self._dist_at_check - dist_to_target
            if progress >= self.stuck_thresh:
                self._stuck_timer = 0.0
            self._dist_at_check = dist_to_target
            self._last_check    = self._time

    def _choose_follow_side(self, pos, th, world, target) -> float:
        dx      = target[0] - pos[0]
        dy      = target[1] - pos[1]
        bearing = np.arctan2(dy, dx)

        def clearance(angle):
            r = world.ray_intersect(pos, angle, max_range=5.0)
            return r[0] if r else 5.0

        left  = clearance(_wrap(bearing + np.pi / 4))
        right = clearance(_wrap(bearing - np.pi / 4))
        return 1.0 if left >= right else -1.0

    def _follow(self, th) -> Tuple[float, float]:
        return self.k_v * 0.5, self._follow_sign * self.max_omega * 0.7


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)