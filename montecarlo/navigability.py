"""
montecarlo/navigability.py
--------------------------
Filter MC sample points to keep only those reachable from the robot's
current position (Section III-B of the paper).

Implementation: for each candidate point, cast a ray from the robot.
If the ray hits a wall before reaching the point, that point is inside
an obstacle or behind a wall and is discarded.

This is the "ray-cast navigability" approach — simpler than the full
metric path-generation algorithm used in the real robot but sufficient
for a 2-D simulation.
"""

from __future__ import annotations
import numpy as np
from typing import List


def navigable_mask(points:     np.ndarray,
                   robot_pos:  np.ndarray,
                   world,
                   clearance:  float = 0.20
                   ) -> np.ndarray:
    """
    Return a boolean mask: True where the point is reachable from robot_pos.

    A point is navigable if the direct ray from the robot to the point is
    not blocked by any wall segment (within a clearance tolerance).

    Parameters
    ----------
    points    : (N, 2) candidate MC points
    robot_pos : (2,)   robot position
    world     : env.World  (provides ray_intersect)
    clearance : tolerance in metres — a wall hit within `clearance` of the
                point itself is treated as the wall the point sits on,
                not as an obstacle between robot and point

    Returns
    -------
    mask : (N,) bool array
    """
    mask      = np.zeros(len(points), dtype=bool)
    rx, ry    = robot_pos

    for i, pt in enumerate(points):
        dx   = pt[0] - rx
        dy   = pt[1] - ry
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1e-3:
            mask[i] = True
            continue

        angle  = np.arctan2(dy, dx)
        result = world.ray_intersect(
            np.array([rx, ry]), angle, dist + clearance)

        if result is None:
            # Ray reached max_range without hitting anything — navigable
            mask[i] = True
        else:
            hit_dist, _ = result
            # Navigable if wall is at or beyond the point (within clearance)
            mask[i] = hit_dist >= dist - clearance

    return mask


def filter_navigable(points:    np.ndarray,
                     robot_pos: np.ndarray,
                     world,
                     clearance: float = 0.20
                     ) -> np.ndarray:
    """
    Return only the navigable subset of points.

    Parameters
    ----------
    points    : (N, 2)
    robot_pos : (2,)
    world     : env.World

    Returns
    -------
    navigable_points : (M, 2)  where M <= N
    """
    mask = navigable_mask(points, robot_pos, world, clearance)
    return points[mask]