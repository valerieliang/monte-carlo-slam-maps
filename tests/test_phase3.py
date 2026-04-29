"""
tests/test_phase3.py

Phase 3 exit-criterion checks.

1. Measurement model functions  (h_corner, h_line)
2. Analytic Jacobian correctness -- finite-difference vs analytic
3. FeatureExtractor visibility and output structure
4. Noise statistics on extracted observations

Run with:
  cd slam_sim
  python -m pytest tests/test_phase3.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from slam.features import (
    h_corner, h_line,
    feature_jacobians,
    numerical_jacobian_corner,
    numerical_jacobian_line,
    FeatureExtractor,
    _wrap,
)
from env.world import World
from env.robot import Robot


# -----------------------------------------------------------------------------
# Measurement model: h_corner
# -----------------------------------------------------------------------------

class TestHCorner:

    def test_range_pythagorean(self):
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([3.0, 4.0])
        z    = h_corner(pose, cpos)
        assert pytest.approx(z[0], abs=1e-9) == 5.0

    def test_bearing_straight_ahead(self):
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([5.0, 0.0])
        z    = h_corner(pose, cpos)
        assert pytest.approx(z[1], abs=1e-9) == 0.0

    def test_bearing_90_left(self):
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([0.0, 3.0])
        z    = h_corner(pose, cpos)
        assert pytest.approx(z[1], abs=1e-9) == np.pi / 2

    def test_bearing_90_right(self):
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([0.0, -3.0])
        z    = h_corner(pose, cpos)
        assert pytest.approx(z[1], abs=1e-9) == -np.pi / 2

    def test_bearing_behind(self):
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([-2.0, 0.0])
        z    = h_corner(pose, cpos)
        assert pytest.approx(abs(z[1]), abs=1e-9) == np.pi

    def test_heading_compensated(self):
        """Rotating the robot by phi should shift bearing by -phi."""
        pose0 = np.array([0.0, 0.0, 0.0])
        phi   = np.pi / 4
        pose1 = np.array([0.0, 0.0, phi])
        cpos  = np.array([3.0, 3.0])
        z0    = h_corner(pose0, cpos)
        z1    = h_corner(pose1, cpos)
        assert pytest.approx(z0[0], abs=1e-9) == z1[0]   # range unchanged
        assert pytest.approx(_wrap(z1[1] - z0[1]), abs=1e-6) == -phi

    def test_range_positive(self):
        for _ in range(20):
            pose = np.random.uniform(-5, 5, 3)
            cpos = np.random.uniform(-5, 5, 2)
            z    = h_corner(pose, cpos)
            assert z[0] >= 0

    def test_bearing_wrapped(self):
        for _ in range(50):
            pose = np.random.uniform(-5, 5, 3)
            cpos = np.random.uniform(-5, 5, 2)
            z    = h_corner(pose, cpos)
            assert -np.pi <= z[1] <= np.pi


# -----------------------------------------------------------------------------
# Measurement model: h_line
# -----------------------------------------------------------------------------

class TestHLine:

    def test_rho_obs_horizontal_wall(self):
        """
        Horizontal wall at y=5: rho=5, alpha=pi/2.
        Robot at origin: rho_obs = 5 - 0*cos(pi/2) - 0*sin(pi/2) = 5.
        """
        pose = np.array([0.0, 0.0, 0.0])
        z    = h_line(pose, rho=5.0, alpha=np.pi/2)
        assert pytest.approx(z[0], abs=1e-9) == 5.0

    def test_rho_obs_robot_offset(self):
        """Robot 2m toward wall: rho_obs decreases by 2."""
        pose = np.array([0.0, 2.0, 0.0])
        z    = h_line(pose, rho=5.0, alpha=np.pi/2)
        assert pytest.approx(z[0], abs=1e-9) == 3.0

    def test_alpha_obs_compensated_for_heading(self):
        """alpha_obs = alpha - theta_v."""
        pose = np.array([0.0, 0.0, np.pi/4])
        rho, alpha = 3.0, np.pi/2
        z    = h_line(pose, rho, alpha)
        expected_alpha_obs = _wrap(alpha - pose[2])
        assert pytest.approx(z[1], abs=1e-9) == expected_alpha_obs

    def test_vertical_wall(self):
        """Vertical wall at x=6: rho=6, alpha=0. Robot at (0,0,0)."""
        pose = np.array([0.0, 0.0, 0.0])
        z    = h_line(pose, rho=6.0, alpha=0.0)
        assert pytest.approx(z[0], abs=1e-9) == 6.0
        assert pytest.approx(z[1], abs=1e-9) == 0.0

    def test_alpha_obs_wrapped(self):
        for _ in range(30):
            pose      = np.random.uniform(-3, 3, 3)
            rho, alpha = abs(np.random.uniform(1, 5)), np.random.uniform(-np.pi, np.pi)
            z         = h_line(pose, rho, alpha)
            assert -np.pi <= z[1] <= np.pi


# -----------------------------------------------------------------------------
# Jacobian validation -- analytic vs finite difference
# -----------------------------------------------------------------------------

class TestJacobians:

    POSES = [
        np.array([0.0,  0.0,  0.0]),
        np.array([2.0,  3.0,  np.pi/4]),
        np.array([-1.0, 1.0, -np.pi/3]),
        np.array([4.0, -2.0,  np.pi/2]),
        np.array([0.5,  0.5,  np.pi]),
    ]

    CORNERS = [
        np.array([5.0, 0.0]),
        np.array([3.0, 4.0]),
        np.array([-1.0, 3.0]),
        np.array([0.0, -2.0]),
    ]

    LINES = [
        (5.0, np.pi/2),
        (3.0, 0.0),
        (4.0, np.pi/4),
        (2.0, -np.pi/3),
    ]

    @pytest.mark.parametrize('pose', POSES)
    @pytest.mark.parametrize('cpos', CORNERS)
    def test_corner_H_v(self, pose, cpos):
        J_ana           = feature_jacobians(pose, 'corner', cpos)
        H_v_num, _      = numerical_jacobian_corner(pose, cpos)
        np.testing.assert_allclose(J_ana.H_v, H_v_num, atol=1e-5,
            err_msg=f"Corner H_v mismatch at pose={pose}, cpos={cpos}")

    @pytest.mark.parametrize('pose', POSES)
    @pytest.mark.parametrize('cpos', CORNERS)
    def test_corner_H_f(self, pose, cpos):
        J_ana         = feature_jacobians(pose, 'corner', cpos)
        _, H_f_num    = numerical_jacobian_corner(pose, cpos)
        np.testing.assert_allclose(J_ana.H_f, H_f_num, atol=1e-5,
            err_msg=f"Corner H_f mismatch at pose={pose}, cpos={cpos}")

    @pytest.mark.parametrize('pose', POSES)
    @pytest.mark.parametrize('rho,alpha', LINES)
    def test_line_H_v(self, pose, rho, alpha):
        params          = np.array([rho, alpha])
        J_ana           = feature_jacobians(pose, 'line', params)
        H_v_num, _      = numerical_jacobian_line(pose, params)
        np.testing.assert_allclose(J_ana.H_v, H_v_num, atol=1e-5,
            err_msg=f"Line H_v mismatch at pose={pose}, line=({rho},{alpha})")

    @pytest.mark.parametrize('pose', POSES)
    @pytest.mark.parametrize('rho,alpha', LINES)
    def test_line_H_f(self, pose, rho, alpha):
        params        = np.array([rho, alpha])
        J_ana         = feature_jacobians(pose, 'line', params)
        _, H_f_num    = numerical_jacobian_line(pose, params)
        np.testing.assert_allclose(J_ana.H_f, H_f_num, atol=1e-5,
            err_msg=f"Line H_f mismatch at pose={pose}, line=({rho},{alpha})")

    def test_jacobian_shapes(self):
        pose   = np.array([1.0, 2.0, 0.5])
        cpos   = np.array([4.0, 3.0])
        params = np.array([3.0, np.pi/4])

        Jc = feature_jacobians(pose, 'corner', cpos)
        assert Jc.H_v.shape == (2, 3)
        assert Jc.H_f.shape == (2, 2)

        Jl = feature_jacobians(pose, 'line', params)
        assert Jl.H_v.shape == (2, 3)
        assert Jl.H_f.shape == (2, 2)

    def test_unknown_feature_type_raises(self):
        with pytest.raises(ValueError):
            feature_jacobians(np.zeros(3), 'blob', np.zeros(2))


# -----------------------------------------------------------------------------
# FeatureExtractor
# -----------------------------------------------------------------------------

@pytest.fixture
def lab():
    return World.from_preset('lab')

@pytest.fixture
def extractor():
    return FeatureExtractor(
        fov_rad       = np.radians(180),
        max_range     = 10.0,
        noise_range   = 0.03,
        noise_bearing = 0.01,
        rng           = np.random.default_rng(42),
    )

@pytest.fixture
def noiseless_extractor():
    return FeatureExtractor(
        fov_rad       = np.radians(180),
        max_range     = 10.0,
        noise_range   = 0.0,
        noise_bearing = 0.0,
        rng           = np.random.default_rng(0),
    )


class TestFeatureExtractor:

    def test_returns_lists(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, lines = extractor.extract(pose, lab)
        assert isinstance(corners, list)
        assert isinstance(lines,   list)

    def test_obs_shapes(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, lines = extractor.extract(pose, lab)
        for c in corners:
            assert c.z.shape == (2,)
            assert c.R.shape == (2, 2)
        for l in lines:
            assert l.z.shape == (2,)
            assert l.R.shape == (2, 2)

    def test_some_corners_detected(self, extractor, lab):
        """From centre of lab the extractor should see at least 2 corners."""
        pose = np.array([6.0, 4.0, 0.0])
        corners, _ = extractor.extract(pose, lab)
        assert len(corners) >= 2, \
            f"Expected >= 2 corners, got {len(corners)}"

    def test_some_lines_detected(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        _, lines = extractor.extract(pose, lab)
        assert len(lines) >= 2, \
            f"Expected >= 2 lines, got {len(lines)}"

    def test_corner_ranges_positive(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, _ = extractor.extract(pose, lab)
        for c in corners:
            assert c.z[0] > 0, "Corner range must be positive"

    def test_corner_ranges_within_max(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, _ = extractor.extract(pose, lab)
        for c in corners:
            assert c.z[0] <= extractor.max_range + 0.1

    def test_corner_bearings_within_fov(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, _ = extractor.extract(pose, lab)
        for c in corners:
            assert abs(c.z[1]) <= extractor.fov_half + 1e-6, \
                f"Corner bearing {c.z[1]:.3f} outside FOV"

    def test_R_matrices_positive_definite(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, lines = extractor.extract(pose, lab)
        for c in corners:
            eigvals = np.linalg.eigvalsh(c.R)
            assert np.all(eigvals > 0), "Corner R not positive definite"
        for l in lines:
            eigvals = np.linalg.eigvalsh(l.R)
            assert np.all(eigvals > 0), "Line R not positive definite"

    def test_noiseless_corner_range_matches_model(self, noiseless_extractor, lab):
        """
        Noiseless extracted range must match h_corner() to within
        floating-point precision.
        """
        pose    = np.array([2.0, 2.0, 0.0])
        corners, _ = noiseless_extractor.extract(pose, lab)
        for obs in corners:
            if obs.gt_pos is None:
                continue
            z_model = h_corner(pose, obs.gt_pos)
            assert pytest.approx(obs.z[0], abs=1e-6) == z_model[0], \
                f"Range mismatch: extracted={obs.z[0]:.4f}, model={z_model[0]:.4f}"

    def test_noiseless_corner_bearing_matches_model(self, noiseless_extractor, lab):
        pose    = np.array([2.0, 2.0, 0.0])
        corners, _ = noiseless_extractor.extract(pose, lab)
        for obs in corners:
            if obs.gt_pos is None:
                continue
            z_model = h_corner(pose, obs.gt_pos)
            diff    = abs(_wrap(obs.z[1] - z_model[1]))
            assert diff < 1e-6, \
                f"Bearing mismatch: extracted={obs.z[1]:.4f}, model={z_model[1]:.4f}"

    def test_noiseless_line_matches_model(self, noiseless_extractor, lab):
        pose    = np.array([2.0, 2.0, 0.0])
        _, lines = noiseless_extractor.extract(pose, lab)
        for obs in lines:
            if obs.gt_rho is None:
                continue
            z_model = h_line(pose, obs.gt_rho, obs.gt_alpha)
            assert pytest.approx(obs.z[0], abs=1e-5) == z_model[0], \
                f"rho_obs mismatch: {obs.z[0]:.4f} vs {z_model[0]:.4f}"
            diff = abs(_wrap(obs.z[1] - z_model[1]))
            assert diff < 1e-5, \
                f"alpha_obs mismatch: {obs.z[1]:.4f} vs {z_model[1]:.4f}"

    def test_no_features_behind_wall(self, noiseless_extractor, lab):
        """
        With a narrow FOV facing east, corners on the west wall should
        not be extracted (they are behind the robot and outside FOV).
        """
        narrow = FeatureExtractor(
            fov_rad       = np.radians(60),
            max_range     = 15.0,
            noise_range   = 0.0,
            noise_bearing = 0.0,
            rng           = np.random.default_rng(0),
        )
        # facing east from centre -- west wall corners at x=0 are behind
        pose    = np.array([6.0, 4.0, 0.0])
        corners, _ = narrow.extract(pose, lab)
        for obs in corners:
            # all detected corners must be in FOV
            assert abs(obs.z[1]) <= narrow.fov_half + 1e-6

    def test_no_detections_beyond_max_range(self, lab):
        """With max_range=0.1 almost nothing should be detectable."""
        tiny = FeatureExtractor(
            fov_rad   = np.radians(180),
            max_range = 0.1,
            noise_range=0.0, noise_bearing=0.0,
            rng=np.random.default_rng(0))
        pose           = np.array([6.0, 4.0, 0.0])
        corners, lines = tiny.extract(pose, lab)
        assert len(corners) == 0
        assert len(lines)   == 0

    def test_corner_kind_preserved(self, extractor, lab):
        pose = np.array([6.0, 4.0, 0.0])
        corners, _ = extractor.extract(pose, lab)
        for c in corners:
            assert c.kind in ('convex', 'concave')


# -----------------------------------------------------------------------------
# Noise statistics on extracted observations
# -----------------------------------------------------------------------------

class TestExtractionNoise:

    def test_corner_range_noise_std(self, lab):
        """Over many extractions, range noise std should match sigma."""
        sigma = 0.05
        ex    = FeatureExtractor(fov_rad=np.radians(180), max_range=10.0,
                                 noise_range=sigma, noise_bearing=0.0,
                                 rng=np.random.default_rng(7))
        pose  = np.array([6.0, 4.0, 0.0])

        errors = []
        for _ in range(500):
            corners, _ = ex.extract(pose, lab)
            for obs in corners:
                if obs.gt_pos is None: continue
                z_true  = h_corner(pose, obs.gt_pos)
                errors.append(obs.z[0] - z_true[0])

        assert len(errors) > 50, "Too few observations to measure noise"
        measured = np.std(errors)
        assert pytest.approx(measured, rel=0.20) == sigma, \
            f"Range noise std {measured:.4f} != {sigma}"

    def test_corner_bearing_noise_std(self, lab):
        sigma = 0.02
        ex    = FeatureExtractor(fov_rad=np.radians(180), max_range=10.0,
                                 noise_range=0.0, noise_bearing=sigma,
                                 rng=np.random.default_rng(13))
        pose  = np.array([6.0, 4.0, 0.0])

        errors = []
        for _ in range(500):
            corners, _ = ex.extract(pose, lab)
            for obs in corners:
                if obs.gt_pos is None: continue
                z_true  = h_corner(pose, obs.gt_pos)
                errors.append(_wrap(obs.z[1] - z_true[1]))

        assert len(errors) > 50
        measured = np.std(errors)
        assert pytest.approx(measured, rel=0.20) == sigma, \
            f"Bearing noise std {measured:.4f} != {sigma}"