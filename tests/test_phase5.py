"""
tests/test_phase5.py
--------------------
Phase 5 exit-criterion checks for the Monte Carlo uncertainty map pipeline.

Test groups
-----------
1. VirtualFeatures  — bounding box computation and virtual line placement
2. Sampler          — uniform distribution of MC points
3. Navigability     — ray-cast filter correctly accepts/rejects points
4. Scoring          — fresh features score high, well-observed score low
5. UncertaintyMap   — full pipeline integration

Run with:
  cd <project_root>
  python -m pytest tests/test_phase5.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from env.world  import World
from slam.state import SLAMState
from slam.features import h_corner, h_line
from slam.update import init_corner, init_line, update_single

from montecarlo.virtual_features import build_virtual_features
from montecarlo.sampler           import sample_points
from montecarlo.navigability      import navigable_mask, filter_navigable
from montecarlo.uncertainty_map   import (build_uncertainty_map,
                                          _score_uncertainty,
                                          UncertaintyMap)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

R = np.diag([0.03**2, 0.01**2])


def _state_with_features(pose=(6.0, 4.0, 0.0), world_preset='lab',
                          n_updates=0) -> tuple:
    """Return (SLAMState, World) with a few features seeded from pose."""
    world = World.from_preset(world_preset)
    state = SLAMState(np.array(pose, dtype=float), np.zeros((3, 3)))
    for c in world.corners[:4]:
        init_corner(state, h_corner(state.pose, c.pos), R, 0.5)
        for _ in range(n_updates):
            update_single(state, state.n_features - 1,
                          h_corner(state.pose, c.pos), R)
    for s in world.segments[:3]:
        rho, alpha = s.as_polar_line()
        init_line(state, h_line(state.pose, rho, alpha), R, 0.5,
                  seg_p0=s.p0, seg_p1=s.p1)
        for _ in range(n_updates):
            update_single(state, state.n_features - 1,
                          h_line(state.pose, rho, alpha), R)
    return state, world


# =============================================================================
# 1. VirtualFeatures
# =============================================================================

class TestVirtualFeatures:

    def test_returns_four_lines(self):
        state, _ = _state_with_features()
        vlines, bounds = build_virtual_features(state)
        assert len(vlines) == 4

    def test_bounds_are_valid_rectangle(self):
        """Bounds must form a valid non-degenerate rectangle."""
        state, _ = _state_with_features()
        _, (xmin, xmax, ymin, ymax) = build_virtual_features(state)
        assert xmin < xmax, f"xmin={xmin} >= xmax={xmax}"
        assert ymin < ymax, f"ymin={ymin} >= ymax={ymax}"
        assert (xmax - xmin) > 0.1
        assert (ymax - ymin) > 0.1

    def test_bounds_xmin_le_xmax(self):
        state, _ = _state_with_features()
        _, (xmin, xmax, ymin, ymax) = build_virtual_features(state)
        assert xmin < xmax
        assert ymin < ymax

    def test_bounds_expand_with_more_features(self):
        state1, world = _state_with_features()
        state2 = SLAMState(np.array([6.0, 4.0, 0.0]), np.zeros((3, 3)))
        for c in world.corners:
            init_corner(state2, h_corner(state2.pose, c.pos), R, 0.5)
        _, b1 = build_virtual_features(state1)
        _, b2 = build_virtual_features(state2)
        span1 = (b1[1] - b1[0]) * (b1[3] - b1[2])
        span2 = (b2[1] - b2[0]) * (b2[3] - b2[2])
        assert span2 >= span1 - 1e-9   # allow floating-point tolerance

    def test_empty_state_returns_small_box(self):
        state = SLAMState(np.array([3.0, 3.0, 0.0]), np.zeros((3, 3)))
        vlines, (xmin, xmax, ymin, ymax) = build_virtual_features(state)
        assert xmax > xmin
        assert ymax > ymin
        assert len(vlines) == 4

    def test_virtual_line_seg_endpoints_defined(self):
        """Each virtual line should have valid (non-None) segment endpoints."""
        state, _ = _state_with_features()
        vlines, _ = build_virtual_features(state)
        for vl in vlines:
            assert vl.seg_p0 is not None
            assert vl.seg_p1 is not None
            assert vl.seg_p0.shape == (2,)
            assert vl.seg_p1.shape == (2,)
            # Segment should have non-zero length
            length = np.linalg.norm(vl.seg_p1 - vl.seg_p0)
            assert length > 0.1, f"Virtual line segment has near-zero length {length}"


# =============================================================================
# 2. Sampler
# =============================================================================

class TestSampler:

    def test_returns_correct_count(self):
        bounds = (0.0, 10.0, 0.0, 8.0)
        pts = sample_points(bounds, 200, rng=np.random.default_rng(0))
        assert pts.shape == (200, 2)

    def test_all_points_inside_bounds(self):
        xmin, xmax, ymin, ymax = 1.0, 9.0, 2.0, 7.0
        pts = sample_points((xmin, xmax, ymin, ymax), 500,
                            rng=np.random.default_rng(1))
        assert np.all(pts[:, 0] >= xmin)
        assert np.all(pts[:, 0] <= xmax)
        assert np.all(pts[:, 1] >= ymin)
        assert np.all(pts[:, 1] <= ymax)

    def test_distribution_is_uniform(self):
        """Mean of N samples should be close to the box centre."""
        bounds = (0.0, 10.0, 0.0, 10.0)
        pts = sample_points(bounds, 5000, rng=np.random.default_rng(42))
        np.testing.assert_allclose(pts[:, 0].mean(), 5.0, atol=0.3)
        np.testing.assert_allclose(pts[:, 1].mean(), 5.0, atol=0.3)

    def test_different_seeds_give_different_points(self):
        bounds = (0.0, 10.0, 0.0, 10.0)
        p1 = sample_points(bounds, 50, rng=np.random.default_rng(1))
        p2 = sample_points(bounds, 50, rng=np.random.default_rng(2))
        assert not np.allclose(p1, p2)

    def test_same_seed_reproducible(self):
        bounds = (0.0, 10.0, 0.0, 10.0)
        p1 = sample_points(bounds, 50, rng=np.random.default_rng(7))
        p2 = sample_points(bounds, 50, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(p1, p2)


# =============================================================================
# 3. Navigability
# =============================================================================

class TestNavigability:

    @pytest.fixture
    def lab(self):
        return World.from_preset('lab')

    def test_robot_position_navigable(self, lab):
        robot_pos = np.array([6.0, 4.0])
        pts = np.array([[6.0, 4.0]])   # robot's own position
        mask = navigable_mask(pts, robot_pos, lab)
        assert mask[0]

    def test_point_behind_wall_not_navigable(self, lab):
        """A point clearly outside the room boundary is not reachable."""
        robot_pos = np.array([6.0, 4.0])
        # outside the south wall at y=0
        pts = np.array([[6.0, -2.0]])
        mask = navigable_mask(pts, robot_pos, lab)
        assert not mask[0]

    def test_point_across_interior_wall_blocked(self, lab):
        """
        Lab has an interior wall section.  A point on the far side of it
        should be unreachable via direct ray from the room centre.
        """
        # Interior wall: Segment([6,3],[9,3])
        # From (7, 4) casting south, hits the wall at y=3.
        # A point at (7, 1) is behind that wall.
        robot_pos = np.array([7.0, 4.0])
        pts       = np.array([[7.0, 1.0]])
        mask      = navigable_mask(pts, robot_pos, lab)
        assert not mask[0]

    def test_open_floor_point_navigable(self, lab):
        """A clear floor point should be reachable."""
        robot_pos = np.array([2.0, 4.0])
        pts       = np.array([[4.0, 4.0]])
        mask      = navigable_mask(pts, robot_pos, lab)
        assert mask[0]

    def test_filter_navigable_reduces_count(self, lab):
        robot_pos = np.array([6.0, 4.0])
        # Mix of inside and outside points
        pts = np.vstack([
            np.random.default_rng(0).uniform([0.5, 0.5], [11.5, 7.5], (50, 2)),
            np.array([[6.0, -1.0], [13.0, 4.0], [-1.0, 4.0]])  # outside
        ])
        navigable = filter_navigable(pts, robot_pos, lab)
        assert len(navigable) < len(pts)
        assert len(navigable) > 0

    def test_all_navigable_points_inside_bounds(self, lab):
        robot_pos = np.array([6.0, 4.0])
        pts = sample_points((0.5, 11.5, 0.5, 7.5), 200,
                            rng=np.random.default_rng(5))
        nav = filter_navigable(pts, robot_pos, lab)
        xmin, xmax, ymin, ymax = lab.bounds
        assert np.all(nav[:, 0] >= xmin - 0.5)
        assert np.all(nav[:, 0] <= xmax + 0.5)


# =============================================================================
# 4. Scoring
# =============================================================================

class TestScoring:

    def test_empty_state_all_zeros(self):
        state = SLAMState(np.zeros(3), np.zeros((3, 3)))
        pts   = np.array([[1.0, 1.0], [5.0, 5.0]])
        scores = _score_uncertainty(pts, state)
        np.testing.assert_array_equal(scores, 0.0)

    def test_fresh_feature_scores_high_nearby(self):
        """A brand-new feature (obs=1) should give high scores close to it."""
        state = SLAMState(np.array([2.0, 2.0, 0.0]), np.zeros((3, 3)))
        cpos  = np.array([5.0, 5.0])
        init_corner(state, h_corner(state.pose, cpos), R, 0.5)
        assert state.features[0].obs_count == 1

        pts    = np.array([[5.0, 5.0]])   # right at the feature
        scores = _score_uncertainty(pts, state)
        assert scores[0] > 0.5, f"Score at fresh feature = {scores[0]:.3f}, want > 0.5"

    def test_well_observed_feature_scores_lower(self):
        """After many updates the same feature should score lower nearby."""
        state = SLAMState(np.array([2.0, 2.0, 0.0]), np.zeros((3, 3)))
        cpos  = np.array([5.0, 5.0])
        init_corner(state, h_corner(state.pose, cpos), R, 0.5)

        score_fresh = _score_uncertainty(np.array([[5.0, 5.0]]), state)[0]

        # Update many times
        for _ in range(30):
            update_single(state, 0, h_corner(state.pose, cpos), R)

        score_mapped = _score_uncertainty(np.array([[5.0, 5.0]]), state)[0]
        assert score_mapped < score_fresh, \
            f"Mapped score {score_mapped:.3f} should be < fresh {score_fresh:.3f}"

    def test_scores_decay_with_distance(self):
        """Score should be highest near the feature and decay with distance."""
        state = SLAMState(np.array([0.0, 0.0, 0.0]), np.zeros((3, 3)))
        cpos  = np.array([5.0, 5.0])
        init_corner(state, h_corner(state.pose, cpos), R, 0.5)

        dists = [0.0, 1.0, 2.0, 4.0]
        pts   = np.array([[5.0 + d, 5.0] for d in dists])
        scores = _score_uncertainty(pts, state)
        # Scores should be non-increasing with distance
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i+1] - 1e-6, \
                f"Score increased at dist {dists[i+1]}: {scores[i]:.4f} -> {scores[i+1]:.4f}"

    def test_scores_in_0_1(self):
        state, _ = _state_with_features()
        pts    = sample_points((0.5, 11.5, 0.5, 7.5), 100,
                               rng=np.random.default_rng(9))
        scores = _score_uncertainty(pts, state)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_fresh_features_score_higher_than_mapped(self):
        """
        The same spatial location should score higher when the nearby feature
        is freshly initialised vs well-observed.
        """
        pose  = np.array([2.0, 2.0, 0.0])
        cpos  = np.array([4.0, 4.0])
        probe = np.array([[4.0, 4.0]])

        # Fresh
        state_fresh = SLAMState(pose, np.zeros((3, 3)))
        init_corner(state_fresh, h_corner(pose, cpos), R, 0.5)
        s_fresh = _score_uncertainty(probe, state_fresh)[0]

        # Mapped
        state_mapped = SLAMState(pose, np.zeros((3, 3)))
        init_corner(state_mapped, h_corner(pose, cpos), R, 0.5)
        for _ in range(50):
            update_single(state_mapped, 0, h_corner(pose, cpos), R)
        s_mapped = _score_uncertainty(probe, state_mapped)[0]

        assert s_fresh > s_mapped, \
            f"Fresh {s_fresh:.4f} should > mapped {s_mapped:.4f}"

    def test_line_feature_scores_along_segment(self):
        """
        Points along a freshly-seen wall segment should score higher than
        points far from it.
        """
        world = World.from_preset('lab')
        state = SLAMState(np.array([6.0, 4.0, 0.0]), np.zeros((3, 3)))
        seg   = world.segments[0]   # south wall
        rho, alpha = seg.as_polar_line()
        init_line(state, h_line(state.pose, rho, alpha), R, 0.5,
                  seg_p0=seg.p0, seg_p1=seg.p1)

        pts_near = np.array([[3.0, 0.5], [6.0, 0.5], [9.0, 0.5]])
        pts_far  = np.array([[3.0, 6.0], [6.0, 6.0], [9.0, 6.0]])

        s_near = _score_uncertainty(pts_near, state)
        s_far  = _score_uncertainty(pts_far,  state)

        assert s_near.mean() > s_far.mean(), \
            f"Near wall {s_near.mean():.4f} should > far {s_far.mean():.4f}"

    def test_corridor_walls_only_detected_from_correct_side(self):
        """
        In the corridor, all visible walls should be mapped from inside
        and score correctly — no wall that was never observed should score
        high just because of its polar form.
        """
        world = World.from_preset('corridor')
        state = SLAMState(np.array([4.0, 1.5, 0.0]), np.zeros((3, 3)))
        from slam.features import FeatureExtractor
        ex = FeatureExtractor(fov_rad=np.radians(180), max_range=20.0,
                              noise_range=0.0, noise_bearing=0.0,
                              rng=np.random.default_rng(0))
        corners, lines = ex.extract(state.pose, world)
        assert len(lines) > 0, "No lines observed — rho_obs check may be wrong"
        # All observed lines should have positive rho_obs
        for l in lines:
            assert l.z[0] > 0, \
                f"Line with negative rho_obs={l.z[0]:.3f} should be rejected"


# =============================================================================
# 5. UncertaintyMap pipeline
# =============================================================================

class TestUncertaintyMapPipeline:

    @pytest.fixture
    def lab(self):
        return World.from_preset('lab')

    def test_returns_uncertainty_map(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=100, rng=np.random.default_rng(0))
        assert isinstance(umap, UncertaintyMap)

    def test_points_and_scores_same_length(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=150, rng=np.random.default_rng(1))
        assert len(umap.points) == len(umap.scores)

    def test_navigable_points_fewer_than_sampled(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=200, rng=np.random.default_rng(2))
        assert len(umap.points) <= umap.n_total

    def test_scores_in_valid_range(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=150, rng=np.random.default_rng(3))
        assert np.all(umap.scores >= 0.0)
        assert np.all(umap.scores <= 1.0)

    def test_uncertain_band_is_subset_of_navigable(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=200, rng=np.random.default_rng(4))
        assert len(umap.uncertain_points) <= len(umap.points)

    def test_uncertain_points_scores_in_band(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=200, rng=np.random.default_rng(5))
        lo, hi = 0.25, 0.75
        umap.set_band(lo, hi)
        if len(umap.uncertain_scores) > 0:
            assert np.all(umap.uncertain_scores >= lo)
            assert np.all(umap.uncertain_scores <= hi)

    def test_fresh_map_has_uncertain_points(self, lab):
        """With freshly-initialised features, there should be uncertain points."""
        state, _ = _state_with_features(n_updates=0)
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=300, rng=np.random.default_rng(6))
        assert len(umap.uncertain_points) > 0, \
            "Expected uncertain points near freshly-seen features"

    def test_well_mapped_has_fewer_uncertain_points(self, lab):
        """After many updates the uncertain point count should drop."""
        state_fresh,  _ = _state_with_features(n_updates=0)
        state_mapped, _ = _state_with_features(n_updates=50)

        rng = np.random.default_rng(7)
        u_fresh  = build_uncertainty_map(state_fresh,  lab,
                                         robot_pos=np.array([6.0, 4.0]),
                                         n_samples=300, rng=rng)
        rng = np.random.default_rng(7)
        u_mapped = build_uncertainty_map(state_mapped, lab,
                                         robot_pos=np.array([6.0, 4.0]),
                                         n_samples=300, rng=rng)
        assert len(u_fresh.uncertain_points) >= len(u_mapped.uncertain_points), \
            (f"Fresh map should have >= uncertain points than mapped map: "
             f"{len(u_fresh.uncertain_points)} vs {len(u_mapped.uncertain_points)}")

    def test_empty_state_returns_zeros(self, lab):
        state = SLAMState(np.array([6.0, 4.0, 0.0]), np.zeros((3, 3)))
        umap  = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                      n_samples=100, rng=np.random.default_rng(8))
        # No features → all scores should be 0
        assert np.all(umap.scores == 0.0)

    def test_all_navigable_points_in_world_bounds(self, lab):
        state, _ = _state_with_features()
        umap = build_uncertainty_map(state, lab, robot_pos=np.array([6.0, 4.0]),
                                     n_samples=200, rng=np.random.default_rng(9))
        xmin, xmax, ymin, ymax = lab.bounds
        pad = 1.0
        if len(umap.points) > 0:
            assert np.all(umap.points[:, 0] >= xmin - pad)
            assert np.all(umap.points[:, 0] <= xmax + pad)
            assert np.all(umap.points[:, 1] >= ymin - pad)
            assert np.all(umap.points[:, 1] <= ymax + pad)

    def test_fresh_features_score_higher_than_mapped_in_full_pipeline(self, lab):
        """
        Integration test: the mean score across navigable points should be
        higher when features are fresh than when they are well-observed,
        using the full build_uncertainty_map pipeline.
        """
        state_fresh,  _ = _state_with_features(n_updates=0)
        state_mapped, _ = _state_with_features(n_updates=50)

        rng = np.random.default_rng(10)
        u_fresh  = build_uncertainty_map(state_fresh, lab,
                                         robot_pos=np.array([6.0, 4.0]),
                                         n_samples=200, rng=rng)
        rng = np.random.default_rng(10)
        u_mapped = build_uncertainty_map(state_mapped, lab,
                                         robot_pos=np.array([6.0, 4.0]),
                                         n_samples=200, rng=rng)
        if len(u_fresh.scores) > 0 and len(u_mapped.scores) > 0:
            assert u_fresh.scores.mean() > u_mapped.scores.mean(), \
                (f"Fresh mean {u_fresh.scores.mean():.4f} should > "
                 f"mapped mean {u_mapped.scores.mean():.4f}")