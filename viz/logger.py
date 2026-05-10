"""
viz/logger.py
-------------
Records the robot's driven path and detects segments where the robot
passed through a wall (physically impossible — indicates a controller
or navigability bug).

Usage
-----
    log = PathLogger(world)
    # in the main loop, every physics step:
    log.record(robot.pos, robot.theta)
    # at the end or on demand:
    log.draw(ax)           # overlay path on the existing axes
    log.report()           # print wall-crossing summary to stdout
    log.save('path.csv')   # dump full trajectory to CSV

Wall-crossing detection
-----------------------
After each new pose is recorded, a ray is cast from the previous position
to the new one.  If any wall segment intersects that ray closer than the
step length, the segment is flagged as a wall-crossing and highlighted in
red on the overlay.

This catches two distinct failure modes:
  1. Controller drove through a wall   (navigation bug)
  2. EKF pose diverged through a wall  (filter inconsistency)
"""

from __future__ import annotations
import numpy as np
import csv
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class PathLogger:
    """
    Records robot trajectory and detects wall-crossing moves.

    Parameters
    ----------
    world       : env.World — used for wall-crossing checks
    min_step    : minimum distance (m) between recorded poses.
                  Filters out near-duplicate entries when the robot is
                  stationary (saves memory and keeps the CSV clean).
    """

    def __init__(self, world, min_step: float = 0.05):
        self._world    = world
        self._min_step = min_step

        # Full trajectory: list of (x, y, theta)
        self._poses: List[np.ndarray] = []

        # Indices into _poses where a wall crossing was detected
        # Each entry is (i, j) meaning the move from pose[i] to pose[j] crossed a wall
        self._crossings: List[Tuple[int, int]] = []

        # matplotlib artists so the overlay can be cleared and redrawn
        self._path_artist   = None
        self._cross_artists: list = []

    # ---------------------------------------------------------------- public

    def record(self, pos: np.ndarray, theta: float) -> bool:
        """
        Record a new robot pose.  Returns True if a wall crossing was detected
        on the move from the previous pose to this one.

        Parameters
        ----------
        pos   : (2,) world-frame position
        theta : heading (rad)
        """
        pose = np.array([pos[0], pos[1], theta])

        # Distance filter — skip if barely moved
        if self._poses:
            if np.linalg.norm(pos - self._poses[-1][:2]) < self._min_step:
                return False

        crossing = False
        if self._poses:
            crossing = self._check_crossing(self._poses[-1][:2], pos)
            if crossing:
                i = len(self._poses) - 1
                j = len(self._poses)      # index of pose about to be appended
                self._crossings.append((i, j))

        self._poses.append(pose)
        return crossing

    @property
    def n_crossings(self) -> int:
        return len(self._crossings)

    @property
    def n_poses(self) -> int:
        return len(self._poses)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._poses.clear()
        self._crossings.clear()

    # ---------------------------------------------------------------- overlay

    def draw(self, ax: 'plt.Axes',
             path_color:     str   = '#818cf8',
             crossing_color: str   = '#ef4444',
             path_alpha:     float = 0.6,
             path_lw:        float = 1.2,
             crossing_lw:    float = 3.0,
             zorder:         int   = 20) -> None:
        """
        Draw the logged path on *ax*.

        The full trajectory is drawn as a thin indigo line.
        Wall-crossing segments are overdrawn in thick red.

        Call clear_overlay() first if redrawing over an existing overlay.
        """
        if len(self._poses) < 2:
            return

        pts = np.array(self._poses)

        # Full path
        self._path_artist, = ax.plot(
            pts[:, 0], pts[:, 1],
            color=path_color, linewidth=path_lw,
            alpha=path_alpha, zorder=zorder,
            label='logged path')

        # Wall-crossing segments — thick red
        for i, j in self._crossings:
            if j < len(self._poses):
                seg_x = [self._poses[i][0], self._poses[j][0]]
                seg_y = [self._poses[i][1], self._poses[j][1]]
                art, = ax.plot(seg_x, seg_y,
                               color=crossing_color, linewidth=crossing_lw,
                               alpha=0.9, zorder=zorder + 1)
                self._cross_artists.append(art)

        # Label crossing count if any
        if self._crossings:
            xi, yi = self._poses[self._crossings[0][0]][:2]
            ax.annotate(
                f'  {len(self._crossings)} wall crossing(s)',
                xy=(xi, yi),
                color=crossing_color, fontsize=7,
                fontfamily='monospace', zorder=zorder + 2)

    def clear_overlay(self) -> None:
        """Remove all overlay artists from the axes."""
        if self._path_artist is not None:
            try:
                self._path_artist.remove()
            except Exception:
                pass
            self._path_artist = None

        for a in self._cross_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._cross_artists.clear()

    # ---------------------------------------------------------------- report

    def report(self) -> None:
        """Print a summary of the logged path to stdout."""
        n = len(self._poses)
        if n < 2:
            print("[PathLogger] No path recorded yet.")
            return

        pts      = np.array(self._poses)[:, :2]
        diffs    = np.diff(pts, axis=0)
        dists    = np.linalg.norm(diffs, axis=1)
        total_m  = float(dists.sum())

        print(f"[PathLogger] {n} poses  |  {total_m:.1f} m total distance  "
              f"|  {self.n_crossings} wall crossing(s)")

        if self._crossings:
            print("  Wall crossings at:")
            for i, j in self._crossings:
                p0 = self._poses[i][:2]
                p1 = self._poses[j][:2] if j < len(self._poses) else p0
                print(f"    ({p0[0]:.2f},{p0[1]:.2f}) → "
                      f"({p1[0]:.2f},{p1[1]:.2f})")

    # ---------------------------------------------------------------- CSV

    def save(self, path: str = 'path.csv') -> None:
        """
        Save the full trajectory to a CSV file.

        Columns: step, x, y, theta_deg, wall_crossing
        """
        # Build crossing index set for fast lookup
        cross_set = {j for _, j in self._crossings}

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'x', 'y', 'theta_deg', 'wall_crossing'])
            for k, pose in enumerate(self._poses):
                writer.writerow([
                    k,
                    round(float(pose[0]), 4),
                    round(float(pose[1]), 4),
                    round(float(np.degrees(pose[2])), 2),
                    int(k in cross_set),
                ])
        print(f"[PathLogger] Saved {len(self._poses)} poses to '{path}'")

    # ---------------------------------------------------------------- private

    def _check_crossing(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        """
        Return True if the straight line from p0 to p1 genuinely crosses
        a wall (not just grazes an endpoint or travels parallel to a segment).

        Strategy:
          1. Cast a ray from p0 toward p1.
          2. If a wall is hit at distance < 40% of the step, it is definitely
             a crossing.
          3. If hit is between 40% and 90%, confirm with a reverse ray —
             a real crossing blocks both directions.

        No minimum step size is imposed here; that is handled by the
        min_step filter in record() which accumulates positions until a
        meaningful displacement has occurred before checking.
        """
        diff  = p1 - p0
        dist  = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return False

        angle   = float(np.arctan2(diff[1], diff[0]))
        result  = self._world.ray_intersect(p0, angle, max_range=dist)

        if result is None:
            return False

        hit_dist, _ = result

        # Clear crossing: wall hit well before p1
        if hit_dist < dist * 0.40:
            return True

        # Ambiguous zone: could be an endpoint/corner graze.
        # Confirm with reverse ray — a real wall blocks both directions.
        if hit_dist < dist * 0.90:
            back_angle  = float(np.arctan2(-diff[1], -diff[0]))
            back_result = self._world.ray_intersect(p1, back_angle,
                                                     max_range=dist)
            if back_result is None:
                return False
            back_dist, _ = back_result
            return back_dist < dist * 0.95

        return False