"""
navigation/selector.py
----------------------
Goal selection implementing the two-phase strategy from Section IV of the paper.

Phase 1 (local): search within a LOCAL_AREA_SIZE × LOCAL_AREA_SIZE box
    centred on the robot's current pose.  Pick the uncertain point with
    the highest P(p) / distance(robot, p).

Phase 2 (global): if no local uncertain points exist, fall back to the
    global transitory array — uncertain points discovered in previous MC
    runs that weren't chosen at the time.

Completion: if neither local nor global uncertain points are available,
    the map is assumed complete and the robot should return to start.

All selected goals are subject to a minimum standoff distance from the
robot (to avoid the ring snapping onto the robot itself).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from montecarlo.uncertainty_map import UncertaintyMap


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GoalResult:
    """
    Output of one goal-selection call.

    goal       : (2,) world-frame goal position, or None if map is complete
    source     : 'local' | 'global' | 'complete'
    n_uncertain: number of uncertain points available when goal was chosen
    """
    goal:        Optional[np.ndarray]
    source:      str          # 'local' | 'global' | 'complete'
    n_uncertain: int = 0


# ---------------------------------------------------------------------------
# Transitory array (global uncertain points not yet visited)
# ---------------------------------------------------------------------------

class TransitoryArray:
    """
    Stores discarded local uncertain points for later global search.

    When multiple local uncertain points are found, only one is chosen —
    the rest are stored here.  Points that are too close together are
    deduplicated (they likely refer to the same unexplored region).
    """

    def __init__(self, dedup_dist: float = 0.5):
        self._points: list[np.ndarray] = []
        self._dedup_dist = dedup_dist

    def add(self, points: np.ndarray) -> None:
        """Add a batch of candidate points, deduplicating as we go."""
        for pt in points:
            if not self._is_duplicate(pt):
                self._points.append(pt.copy())

    def remove_near(self, pos: np.ndarray, radius: float = 1.0) -> None:
        """Remove all stored points within `radius` of `pos` (goal reached)."""
        self._points = [
            p for p in self._points
            if np.linalg.norm(p - pos) > radius
        ]

    def as_array(self) -> Optional[np.ndarray]:
        if not self._points:
            return None
        return np.array(self._points)

    def clear(self) -> None:
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)

    def _is_duplicate(self, pt: np.ndarray) -> bool:
        for existing in self._points:
            if np.linalg.norm(pt - existing) < self._dedup_dist:
                return True
        return False


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class GoalSelector:
    """
    Two-phase goal selector (local → global → complete).

    Parameters
    ----------
    local_area_size : side length of the local search square (m)
    min_dist        : minimum distance from robot to accept a goal (m)
    dedup_dist      : distance below which two global points are duplicates (m)
    """

    def __init__(self,
                 local_area_size: float = 5.0,
                 min_dist:        float = 1.0,
                 dedup_dist:      float = 0.5):
        self.local_area_size = local_area_size
        self.min_dist        = min_dist
        self._transitory     = TransitoryArray(dedup_dist)

    def select(self,
               umap:      'UncertaintyMap',
               robot_pos: np.ndarray) -> GoalResult:
        """
        Select the next navigation goal.

        Parameters
        ----------
        umap      : current UncertaintyMap (already has uncertain_mask set)
        robot_pos : (2,) current robot position

        Returns
        -------
        GoalResult
        """
        pts    = umap.uncertain_points
        scores = umap.uncertain_scores

        if len(pts) == 0:
            # No uncertain points at all — check transitory array
            return self._select_global(robot_pos)

        # ── Phase 1: local search ──────────────────────────────────────────
        half  = self.local_area_size / 2.0
        local = ((pts[:, 0] >= robot_pos[0] - half) &
                 (pts[:, 0] <= robot_pos[0] + half) &
                 (pts[:, 1] >= robot_pos[1] - half) &
                 (pts[:, 1] <= robot_pos[1] + half))

        local_pts    = pts[local]
        local_scores = scores[local]

        # Apply minimum standoff distance
        if len(local_pts) > 0:
            dists     = np.linalg.norm(local_pts - robot_pos, axis=1)
            far_mask  = dists >= self.min_dist
            local_pts    = local_pts[far_mask]
            local_scores = local_scores[far_mask]
            dists        = dists[far_mask]

        if len(local_pts) > 0:
            # Eq. 11: maximise P(p) / distance
            metric   = local_scores / np.maximum(dists, 0.1)
            best_idx = int(np.argmax(metric))
            goal     = local_pts[best_idx]

            # Store unchosen local points in the transitory array
            mask = np.ones(len(local_pts), dtype=bool)
            mask[best_idx] = False
            if mask.any():
                self._transitory.add(local_pts[mask])

            return GoalResult(goal=goal, source='local',
                              n_uncertain=len(pts))

        # ── Phase 2: global fallback ───────────────────────────────────────
        # Add ALL current uncertain points to transitory array and pick from it
        dists_all = np.linalg.norm(pts - robot_pos, axis=1)
        far_all   = dists_all >= self.min_dist
        if far_all.any():
            self._transitory.add(pts[far_all])

        return self._select_global(robot_pos)

    def notify_goal_reached(self, goal: np.ndarray) -> None:
        """Call when the robot reaches its current goal."""
        self._transitory.remove_near(goal, radius=1.0)

    def reset(self) -> None:
        self._transitory.clear()

    # ── private ──────────────────────────────────────────────────────────────

    def _select_global(self, robot_pos: np.ndarray) -> GoalResult:
        stored = self._transitory.as_array()
        if stored is None:
            return GoalResult(goal=None, source='complete', n_uncertain=0)

        dists = np.linalg.norm(stored - robot_pos, axis=1)
        far   = dists >= self.min_dist
        if not far.any():
            return GoalResult(goal=None, source='complete', n_uncertain=0)

        stored = stored[far]
        dists  = dists[far]
        # Pick closest global point (we don't have scores for global points)
        best   = int(np.argmin(dists))
        return GoalResult(goal=stored[best], source='global',
                          n_uncertain=len(stored))