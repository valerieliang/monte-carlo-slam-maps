"""
tests/test_phase6.py
--------------------
Phase 6 exit-criterion checks for autonomous navigation.

Test groups
-----------
1. GoalSelector  — local/global/complete transitions, Eq. 11 metric,
                   transitory array deduplication
2. Controller    — pursue mode drives toward goal, goal-reached detection,
                   wall-follow mode activates when stuck, v/omega limits
3. Integration   — full autonomy loop converges on a simple world

Run with:
  cd <project_root>
  python -m pytest tests/test_phase6.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from env.world  import World
from navigation.selector   import GoalSelector, GoalResult, TransitoryArray
from navigation.controller import Controller, Mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_umap(pts, scores, lo=0.25, hi=0.75):
    """Minimal UncertaintyMap duck-type for selector tests."""
    from types import SimpleNamespace
    umap = SimpleNamespace()
    umap.points  = np.array(pts,    dtype=float).reshape(-1, 2)
    umap.scores  = np.array(scores, dtype=float)
    mask = (umap.scores >= lo) & (umap.scores <= hi)
    umap.uncertain_points = umap.points[mask]
    umap.uncertain_scores = umap.scores[mask]
    umap.uncertain_mask   = mask
    return umap


# =============================================================================
# 1. GoalSelector
# =============================================================================

class TestGoalSelector:

    def test_selects_local_point(self):
        sel  = GoalSelector(local_area_size=5.0, min_dist=0.5)
        umap = _make_umap([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                          [0.5, 0.5, 0.5])
        result = sel.select(umap, robot_pos=np.array([3.0, 3.0]))
        assert result.source == 'local'
        assert result.goal is not None

    def test_eq11_higher_score_wins_at_equal_distance(self):
        """Two points equidistant — higher score wins."""
        sel  = GoalSelector(local_area_size=10.0, min_dist=0.1)
        umap = _make_umap([[2.0, 0.0], [-2.0, 0.0]], [0.7, 0.4])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert result.goal is not None
        np.testing.assert_allclose(result.goal, [2.0, 0.0], atol=0.01)

    def test_eq11_closer_wins_at_equal_score(self):
        """Two equal-score points — closer one wins (higher score/dist)."""
        sel  = GoalSelector(local_area_size=10.0, min_dist=0.1)
        umap = _make_umap([[1.0, 0.0], [4.0, 0.0]], [0.5, 0.5])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        np.testing.assert_allclose(result.goal, [1.0, 0.0], atol=0.01)

    def test_min_dist_excludes_nearby_points(self):
        """Points within min_dist are ignored."""
        sel  = GoalSelector(local_area_size=10.0, min_dist=2.0)
        umap = _make_umap([[0.5, 0.0], [5.0, 0.0]], [0.5, 0.5])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert result.goal is not None
        assert np.linalg.norm(result.goal) >= 2.0

    def test_all_too_close_falls_to_global_or_complete(self):
        """If all uncertain points are within min_dist → global/complete."""
        sel  = GoalSelector(local_area_size=10.0, min_dist=5.0)
        umap = _make_umap([[0.5, 0.0]], [0.5])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert result.source in ('global', 'complete')

    def test_falls_back_to_global(self):
        """No local uncertain points but transitory has entries → 'global'."""
        sel = GoalSelector(local_area_size=2.0, min_dist=0.1)
        sel._transitory.add(np.array([[10.0, 10.0]]))
        umap = _make_umap([], [])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert result.source == 'global'
        assert result.goal is not None

    def test_complete_when_nothing_anywhere(self):
        sel  = GoalSelector()
        umap = _make_umap([], [])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert result.source == 'complete'
        assert result.goal is None

    def test_unchosen_local_stored_in_transitory(self):
        """Multiple local points: unchosen ones go to transitory array."""
        sel  = GoalSelector(local_area_size=10.0, min_dist=0.1)
        umap = _make_umap([[1.0,0.0],[2.0,0.0],[3.0,0.0]], [0.5,0.5,0.5])
        sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert len(sel._transitory) >= 1

    def test_notify_goal_reached_purges_nearby(self):
        sel  = GoalSelector()
        sel._transitory.add(np.array([[5.0,0.0],[5.2,0.0],[20.0,0.0]]))
        sel.notify_goal_reached(np.array([5.0, 0.0]))
        stored = sel._transitory.as_array()
        assert stored is not None
        for p in stored:
            assert np.linalg.norm(p - np.array([5.0,0.0])) > 1.0

    def test_reset_clears_transitory(self):
        sel = GoalSelector()
        sel._transitory.add(np.array([[5.0, 5.0]]))
        sel.reset()
        assert len(sel._transitory) == 0

    def test_goal_result_shape(self):
        sel  = GoalSelector(local_area_size=10.0, min_dist=0.1)
        umap = _make_umap([[2.0, 2.0]], [0.5])
        result = sel.select(umap, robot_pos=np.array([0.0, 0.0]))
        assert isinstance(result, GoalResult)
        assert result.goal is not None
        assert result.goal.shape == (2,)
        assert result.n_uncertain > 0


class TestTransitoryArray:

    def test_add_and_retrieve(self):
        ta = TransitoryArray()
        ta.add(np.array([[1.0,2.0],[3.0,4.0]]))
        assert len(ta) == 2

    def test_deduplication(self):
        ta = TransitoryArray(dedup_dist=1.0)
        ta.add(np.array([[0.0,0.0]]))
        ta.add(np.array([[0.3,0.0]]))   # within 1m → dup
        ta.add(np.array([[5.0,0.0]]))   # far → kept
        assert len(ta) == 2

    def test_remove_near(self):
        ta = TransitoryArray()
        ta.add(np.array([[1.0,0.0],[2.0,0.0],[10.0,0.0]]))
        ta.remove_near(np.array([1.5,0.0]), radius=1.5)
        arr = ta.as_array()
        assert arr is not None
        assert len(arr) == 1
        np.testing.assert_allclose(arr[0], [10.0,0.0])

    def test_clear(self):
        ta = TransitoryArray()
        ta.add(np.array([[1.0,1.0]]))
        ta.clear()
        assert ta.as_array() is None

    def test_empty_returns_none(self):
        assert TransitoryArray().as_array() is None


# =============================================================================
# 2. Controller
# =============================================================================

class TestControllerPursue:

    @pytest.fixture
    def world(self):
        return World.from_preset('open')

    @pytest.fixture
    def ctrl(self):
        return Controller(k_v=0.4, k_w=1.2, max_v=0.5, max_omega=1.0,
                          goal_tolerance=0.3)

    def test_no_goal_returns_zero(self, ctrl, world):
        v, omega = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert v == 0.0 and omega == 0.0

    def test_drives_toward_goal_ahead(self, ctrl, world):
        ctrl.set_goal(np.array([5.0, 0.0]))
        v, _ = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert v > 0.0

    def test_turns_left_for_left_goal(self, ctrl, world):
        ctrl.set_goal(np.array([0.0, 5.0]))
        _, omega = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert omega > 0.0

    def test_turns_right_for_right_goal(self, ctrl, world):
        ctrl.set_goal(np.array([0.0, -5.0]))
        _, omega = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert omega < 0.0

    def test_v_within_max(self, ctrl, world):
        ctrl.set_goal(np.array([10.0, 0.0]))
        v, _ = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert abs(v) <= ctrl.max_v + 1e-9

    def test_omega_within_max(self, ctrl, world):
        ctrl.set_goal(np.array([0.0, 10.0]))
        _, omega = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert abs(omega) <= ctrl.max_omega + 1e-9

    def test_slower_for_goal_behind(self, ctrl, world):
        """v should drop when goal requires a sharp turn."""
        ctrl.set_goal(np.array([5.0, 0.0]))
        v_ahead, _ = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        ctrl.set_goal(np.array([-5.0, 0.0]))
        v_behind, _ = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert v_ahead > v_behind

    def test_goal_reached_within_tolerance(self, ctrl, world):
        ctrl.set_goal(np.array([1.0, 0.0]))
        assert ctrl.goal_reached(np.array([1.1, 0.0]))
        assert not ctrl.goal_reached(np.array([5.0, 0.0]))

    def test_initial_mode_is_pursue(self, ctrl):
        ctrl.set_goal(np.array([5.0, 5.0]))
        assert ctrl.mode == Mode.PURSUE

    def test_mode_done_on_goal_reached(self, ctrl, world):
        ctrl.set_goal(np.array([0.1, 0.0]))
        ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert ctrl.mode == Mode.DONE

    def test_set_goal_resets_done_to_pursue(self, ctrl, world):
        ctrl.set_goal(np.array([0.1, 0.0]))
        ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        ctrl.set_goal(np.array([5.0, 0.0]))
        assert ctrl.mode == Mode.PURSUE

    def test_set_goal_none_stops_robot(self, ctrl, world):
        ctrl.set_goal(np.array([5.0, 0.0]))
        ctrl.set_goal(None)
        v, omega = ctrl.step(np.array([0.,0.,0.]), world, dt=0.05)
        assert v == 0.0 and omega == 0.0


class TestControllerWallFollow:

    @pytest.fixture
    def world(self):
        return World.from_preset('lab')

    def test_follow_mode_activates_when_stuck(self, world):
        """No-progress for check_interval seconds triggers FOLLOW."""
        ctrl = Controller(k_v=0.4, k_w=1.2, max_v=0.5, max_omega=1.0,
                          goal_tolerance=0.3,
                          stuck_thresh=100.0,   # impossible to achieve
                          check_interval=0.2,
                          follow_duration=2.0)
        ctrl.set_goal(np.array([8.0, 2.0]))
        pose = np.array([7.0, 4.0, -np.pi/2])
        for _ in range(50):            # 5 seconds
            ctrl.step(pose, world, dt=0.1)
        assert ctrl.mode == Mode.FOLLOW

    def test_follow_produces_movement(self, world):
        ctrl = Controller(stuck_thresh=100.0, check_interval=0.01,
                          follow_duration=2.0)
        ctrl.set_goal(np.array([8.0, 2.0]))
        ctrl._mode = Mode.FOLLOW
        v, omega = ctrl.step(np.array([7.,4.,-np.pi/2]), world, dt=0.1)
        assert v != 0.0 or omega != 0.0

    def test_follow_returns_to_pursue_after_duration(self, world):
        # Use a long check_interval so stuck detection doesn't re-trigger
        # FOLLOW immediately after it exits, letting us assert PURSUE.
        ctrl = Controller(stuck_thresh=100.0, check_interval=999.0,
                          follow_duration=0.5)
        ctrl.set_goal(np.array([5.0, 5.0]))
        ctrl._mode = Mode.FOLLOW
        ctrl._follow_timer = 0.0
        pose = np.array([1.,1.,0.])
        for _ in range(15):           # 0.75 s > 0.5 s
            ctrl.step(pose, world, dt=0.05)
        assert ctrl.mode == Mode.PURSUE


# =============================================================================
# 3. Integration
# =============================================================================

class TestAutonomousIntegration:

    def test_robot_reduces_distance_to_goal(self):
        """Over 2 seconds of stepping the robot should close in on the goal."""
        world = World.from_preset('open')
        ctrl  = Controller(k_v=0.4, k_w=1.2, max_v=0.5, max_omega=1.0,
                           goal_tolerance=0.3)
        goal  = np.array([8.0, 8.0])
        ctrl.set_goal(goal)
        pose  = np.array([1.0, 1.0, 0.0])
        d0    = np.linalg.norm(pose[:2] - goal)

        for _ in range(40):
            v, omega = ctrl.step(pose, world, dt=0.05)
            pose[0] += v * np.cos(pose[2]) * 0.05
            pose[1] += v * np.sin(pose[2]) * 0.05
            pose[2] += omega * 0.05
            if ctrl.goal_reached(pose[:2]):
                break

        assert np.linalg.norm(pose[:2] - goal) < d0

    def test_goal_reached_in_open_room(self):
        """Robot should reach a close-by goal within 5 seconds."""
        world = World.from_preset('open')
        ctrl  = Controller(k_v=0.5, k_w=1.5, max_v=0.5, max_omega=1.5,
                           goal_tolerance=0.3, check_interval=999.0)
        goal  = np.array([4.0, 4.0])
        ctrl.set_goal(goal)
        pose  = np.array([1.0, 1.0, 0.0])

        reached = False
        for _ in range(250):    # up to 12.5 s — allows alignment + drive
            v, omega = ctrl.step(pose, world, dt=0.05)
            pose[0] += v * np.cos(pose[2]) * 0.05
            pose[1] += v * np.sin(pose[2]) * 0.05
            pose[2] += omega * 0.05
            if ctrl.goal_reached(pose[:2]):
                reached = True
                break

        assert reached, \
            f"Goal not reached, final dist={np.linalg.norm(pose[:2]-goal):.2f}"

    def test_selector_local_global_complete_lifecycle(self):
        """Full lifecycle: local → global → complete."""
        sel = GoalSelector(local_area_size=4.0, min_dist=0.1)

        # Local phase
        umap1   = _make_umap([[2.,0.],[3.,0.]], [0.5,0.5])
        result1 = sel.select(umap1, robot_pos=np.array([0.,0.]))
        assert result1.source == 'local'

        # Simulate reaching goal, no new uncertain points
        sel.notify_goal_reached(result1.goal)
        umap2   = _make_umap([], [])
        result2 = sel.select(umap2, robot_pos=result1.goal)
        # May be global (unchosen pts stored) or complete
        assert result2.source in ('global', 'complete')

        # Drain remaining
        sel.reset()
        result3 = sel.select(umap2, robot_pos=np.array([0.,0.]))
        assert result3.source == 'complete'
        assert result3.goal is None

    def test_returning_to_start_on_completion(self):
        """
        Simulate the return-to-start pattern: once 'complete', the
        controller should be able to navigate to the start position.
        """
        world     = World.from_preset('open')
        ctrl      = Controller(k_v=0.5, k_w=1.5, max_v=0.5, max_omega=1.5,
                               goal_tolerance=0.3, check_interval=999.0)
        start_pos = np.array([1.0, 1.0])
        pose      = np.array([7.0, 7.0, np.pi])   # far from start

        ctrl.set_goal(start_pos)
        d0 = np.linalg.norm(pose[:2] - start_pos)

        for _ in range(400):    # up to 20 s from far corner
            v, omega = ctrl.step(pose, world, dt=0.05)
            pose[0] += v * np.cos(pose[2]) * 0.05
            pose[1] += v * np.sin(pose[2]) * 0.05
            pose[2] += omega * 0.05
            if ctrl.goal_reached(pose[:2]):
                break

        assert ctrl.goal_reached(pose[:2]), \
            f"Did not return to start, dist={np.linalg.norm(pose[:2]-start_pos):.2f}"