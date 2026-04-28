"""
env/world.py
------------
Ground-truth environment: walls as line Segments and corners as Corner points.
The robot never has direct access to this data - it only receives noisy sensor
observations derived from it (added in Phase 2).

Coordinate system: standard 2-D Cartesian, metres, X right, Y up.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# -----------------------------------------------------------------------------
# Primitive types
# -----------------------------------------------------------------------------

@dataclass
class Segment:
    """A finite wall segment defined by two endpoints (metres)."""
    p0: np.ndarray          # shape (2,)  start point
    p1: np.ndarray          # shape (2,)  end  point

    def __post_init__(self):
        self.p0 = np.asarray(self.p0, dtype=float)
        self.p1 = np.asarray(self.p1, dtype=float)

    # -- derived geometry ------------------------------------------------------

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p0 + self.p1)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from p0 to p1."""
        d = self.p1 - self.p0
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else np.array([1.0, 0.0])

    @property
    def normal(self) -> np.ndarray:
        """Left-hand normal to the segment direction."""
        dx, dy = self.direction
        return np.array([-dy, dx])

    def as_polar_line(self) -> Tuple[float, float]:
        """
        Return (rho, alpha) – polar representation of the infinite line
        containing this segment, as used by the EKF line-feature model (Eq. 5).

          rho   = signed perpendicular distance from origin to the line
          alpha = angle of the line's normal from the x-axis, in (-pi, pi]
        """
        dx = self.p1[0] - self.p0[0]
        dy = self.p1[1] - self.p0[1]
        # normal direction
        alpha = np.arctan2(-dx, dy)           # angle of outward normal
        rho   = self.p0[0]*np.cos(alpha) + self.p0[1]*np.sin(alpha)
        if rho < 0:                           # enforce rho >= 0 convention
            rho   = -rho
            alpha = alpha + np.pi if alpha <= 0 else alpha - np.pi
        return float(rho), float(alpha)

    def point_in_region(self, p: np.ndarray) -> bool:
        """
        True if point p falls within the 'region' of this segment as defined
        in the paper (Section III-D, Fig. 3): the infinite strip bounded by
        the two normals passing through p0 and p1.
        """
        p  = np.asarray(p, dtype=float)
        d  = self.direction
        t  = np.dot(p - self.p0, d)
        return 0.0 <= t <= self.length


@dataclass
class Corner:
    """
    A detected corner in the environment.

    kind    : 'convex' or 'concave' (from the robot's perspective looking in)
    pos     : (x, y) position in metres
    """
    pos:  np.ndarray
    kind: str = 'convex'          # 'convex' | 'concave'

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        if self.kind not in ('convex', 'concave'):
            raise ValueError(f"Corner kind must be 'convex' or 'concave', got '{self.kind}'")


# -----------------------------------------------------------------------------
# World
# -----------------------------------------------------------------------------

class World:
    """
    Container for all ground-truth geometry.

    Provides helper methods used by the sensor model (Phase 2) and the
    renderer (Phase 1).

    Usage
    -----
    >>> w = World.from_preset('lab')
    >>> w.segments     # list of Segment
    >>> w.corners      # list of Corner
    """

    def __init__(self,
                 segments: List[Segment],
                 corners:  List[Corner],
                 name:     str = 'world'):
        self.segments: List[Segment] = segments
        self.corners:  List[Corner]  = corners
        self.name = name

    # -- factory presets -------------------------------------------------------

    @classmethod
    def from_preset(cls, preset: str = 'lab') -> 'World':
        """
        Return a named preset environment.

        Presets
        -------
        'lab'     – rectangular room with one interior partial wall, mimicking
                    the Instituto de Automatica layout used in the paper.
        'corridor' – long narrow corridor with one branch.
        'open'    – bare rectangular room, minimal features.
        """
        if preset == 'lab':
            return cls._build_lab()
        elif preset == 'corridor':
            return cls._build_corridor()
        elif preset == 'open':
            return cls._build_open()
        else:
            raise ValueError(f"Unknown preset '{preset}'. "
                             f"Choose from: 'lab', 'corridor', 'open'.")

    # -- preset builders -------------------------------------------------------

    @classmethod
    def _build_lab(cls) -> 'World':
        """
        A roughly 12 m × 8 m room with one interior partial wall creating
        an L-shaped recess, plus a small alcove — enough feature variety to
        exercise both corner and line feature types.

        Layout (metres, approximate)
        -----------------------------
         (0,8)------------------(12,8)
           |                        |
           |   alcove               |
           |   (3,5)--(6,5)         |
           |          |             |
           |          (6,3)--(9,3)  |  ← interior partial wall
           |                        |
         (0,0)------------------(12,0)
        """
        segs = [
            # outer boundary
            Segment([0, 0],  [12, 0]),   # south wall
            Segment([12, 0], [12, 8]),   # east  wall
            Segment([12, 8], [0, 8]),    # north wall
            Segment([0, 8],  [0, 0]),    # west  wall
            # interior partial wall (creates an L-shaped obstacle)
            Segment([6, 3],  [9, 3]),    # east arm
            Segment([6, 3],  [6, 5]),    # vertical connector
            Segment([3, 5],  [6, 5]),    # west arm of alcove
        ]

        # Corners: endpoints that form actual convex/concave corners
        # (junction between two wall segments meeting at an angle)
        cors = [
            # outer room -- convex from inside
            Corner([0, 0],   'convex'),
            Corner([12, 0],  'convex'),
            Corner([12, 8],  'convex'),
            Corner([0, 8],   'convex'),
            # interior wall -- concave junctions (robot looks into the recess)
            Corner([6, 3],   'concave'),
            Corner([9, 3],   'concave'),
            Corner([6, 5],   'concave'),
            Corner([3, 5],   'concave'),
        ]

        return cls(segs, cors, name='lab')

    @classmethod
    def _build_corridor(cls) -> 'World':
        """Long corridor (20 m × 3 m) with one perpendicular branch."""
        segs = [
            # main corridor
            Segment([0, 0],   [20, 0]),    # south wall - solid, no opening            
            Segment([20, 0],  [20, 3]),    # east wall
            Segment([20, 3],  [12, 3]),    # right part of north wall
            Segment([8, 3],   [0, 3]),     # left part of north wall (gap from 8-12)
            Segment([0, 3],   [0, 0]),     # west wall
            
            # branch corridor (goes UP from the gap)
            Segment([8, 3],   [8, 9]),     # branch left wall
            Segment([8, 9],   [12, 9]),    # branch top wall
            Segment([12, 9],  [12, 3]),    # branch right wall
        ]
        cors = [
            # outer room corners
            Corner([0,  0],  'convex'),
            Corner([20, 0],  'convex'),
            Corner([20, 3],  'convex'),
            Corner([0,  3],  'convex'),
            
            # opening corners on north wall
            Corner([12, 3],  'convex'),    # right side of opening
            Corner([8,  3],  'convex'),    # left side of opening
            
            # branch corners
            Corner([8,  3],  'concave'),   # where branch meets main corridor (left)
            Corner([12, 3],  'concave'),   # where branch meets main corridor (right)
            Corner([8,  9],  'convex'),
            Corner([12, 9],  'convex'),
        ]
        return cls(segs, cors, name='corridor')

    @classmethod
    def _build_open(cls) -> 'World':
        """Bare 10 m × 10 m rectangle."""
        segs = [
            Segment([0,  0],  [10, 0]),
            Segment([10, 0],  [10, 10]),
            Segment([10, 10], [0,  10]),
            Segment([0,  10], [0,  0]),
        ]
        cors = [
            Corner([0,  0],  'convex'),
            Corner([10, 0],  'convex'),
            Corner([10, 10], 'convex'),
            Corner([0,  10], 'convex'),
        ]
        return cls(segs, cors, name='open')

    # -- geometry helpers ------------------------------------------------------

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) across all segment endpoints."""
        pts = np.array([ep for s in self.segments for ep in (s.p0, s.p1)])
        return (pts[:, 0].min(), pts[:, 0].max(),
                pts[:, 1].min(), pts[:, 1].max())

    def ray_intersect(self,
                      origin: np.ndarray,
                      angle:  float,
                      max_range: float = 30.0
                      ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Cast a ray from *origin* at *angle* (radians, world frame) and find
        the closest intersection with any wall segment.

        Returns
        -------
        (distance, hit_point) if an intersection exists within max_range,
        else None.

        Used by env/sensor.py (Phase 2).  Implemented here so the World owns
        all geometry queries.
        """
        origin = np.asarray(origin, dtype=float)
        ray_d  = np.array([np.cos(angle), np.sin(angle)])

        best_t   = max_range
        best_pt  = None

        for seg in self.segments:
            # Solve: origin + t*ray_d = seg.p0 + s*(seg.p1 - seg.p0)
            # [ray_d | -(seg.p1-seg.p0)] [t; s] = seg.p0 - origin
            d  = seg.p1 - seg.p0
            b  = seg.p0 - origin
            # 2x2 system  A * [t, s]^T = b
            A  = np.column_stack([ray_d, -d])
            det = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
            if abs(det) < 1e-12:
                continue                   # parallel
            ts = np.linalg.solve(A, b)
            t, s = ts
            if t > 1e-6 and t < best_t and 0.0 <= s <= 1.0:
                best_t  = t
                best_pt = origin + t * ray_d

        if best_pt is not None:
            return float(best_t), best_pt
        return None

    def __repr__(self) -> str:
        return (f"World('{self.name}', "
                f"{len(self.segments)} segments, "
                f"{len(self.corners)} corners)")