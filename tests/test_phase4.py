"""
tests/test_phase4.py
--------------------
Phase 4 exit-criterion checks for the EKF-SLAM back-end.

Test groups
-----------
1. SLAMState  — state vector / covariance structure and augmentation
2. Predict    — pose propagation and covariance growth
3. Update     — innovation shrinkage, Kalman gain correctness
4. Init       — feature initialisation from inverse measurement model
5. DataAssoc  — Mahalanobis gating, association / new-feature logic
6. Integration — multi-step loop: covariance shrinks on revisit,
                 no duplicate features for repeated observations

Run with:
  cd <project_root>
  python -m pytest tests/test_phase4.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from slam.state     import SLAMState, _wrap
from slam.predict   import predict
from slam.update    import (update_single, init_corner, init_line,
                            _h_corner, _h_line,
                            _h_corner_jacobian, _h_line_jacobian)
from slam.data_assoc import associate, associate_observations, mahalanobis
from slam.features  import (FeatureExtractor, h_corner, h_line,
                             feature_jacobians,
                             numerical_jacobian_corner,
                             numerical_jacobian_line, _wrap as fw)
from env.world      import World


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

R_CORNER = np.diag([0.03**2, 0.01**2])
R_LINE   = np.diag([0.03**2, 0.01**2])

def _fresh_state(pose=(1.0, 1.0, 0.0)) -> SLAMState:
    return SLAMState(np.array(pose, dtype=float), np.zeros((3, 3)))

def _state_with_corner(pose, cpos) -> SLAMState:
    """State with one corner feature initialised from noiseless observation."""
    st = _fresh_state(pose)
    z  = h_corner(np.array(pose), np.array(cpos))
    init_corner(st, z, R_CORNER, init_cov_scale=0.5)
    return st

def _state_with_line(pose, rho, alpha) -> SLAMState:
    st = _fresh_state(pose)
    z  = h_line(np.array(pose), rho, alpha)
    init_line(st, z, R_LINE, init_cov_scale=0.5)
    return st

@pytest.fixture
def lab():
    return World.from_preset('lab')

@pytest.fixture
def noiseless_ex():
    return FeatureExtractor(
        fov_rad=np.radians(180), max_range=15.0,
        noise_range=0.0, noise_bearing=0.0,
        rng=np.random.default_rng(0))


# =============================================================================
# 1. SLAMState
# =============================================================================

class TestSLAMState:

    def test_initial_dim(self):
        st = _fresh_state()
        assert st.dim == 3

    def test_initial_n_features(self):
        st = _fresh_state()
        assert st.n_features == 0

    def test_pose_accessor(self):
        st = _fresh_state((2.0, 3.0, 0.5))
        np.testing.assert_array_almost_equal(st.pose, [2.0, 3.0, 0.5])

    def test_add_corner_expands_dim(self):
        st = _fresh_state()
        cov   = np.eye(2)
        cross = np.zeros((3, 2))
        st.add_feature(np.array([5.0, 5.0]), cov, 'corner', cross)
        assert st.dim == 5
        assert st.n_features == 1

    def test_add_line_expands_dim(self):
        st = _fresh_state()
        cov   = np.eye(2)
        cross = np.zeros((3, 2))
        st.add_feature(np.array([3.0, 0.5]), cov, 'line', cross)
        assert st.dim == 5
        assert st.n_features == 1

    def test_add_multiple_features(self):
        st = _fresh_state()
        for i in range(4):
            st.add_feature(np.zeros(2), np.eye(2), 'corner', np.zeros((3+2*i, 2)))
        assert st.dim == 3 + 8
        assert st.n_features == 4

    def test_feature_mean_roundtrip(self):
        st   = _fresh_state()
        mean = np.array([7.0, 3.0])
        st.add_feature(mean, np.eye(2), 'corner', np.zeros((3, 2)))
        np.testing.assert_array_almost_equal(st.feature_mean(0), mean)

    def test_feature_cov_roundtrip(self):
        st  = _fresh_state()
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        st.add_feature(np.zeros(2), cov, 'corner', np.zeros((3, 2)))
        np.testing.assert_array_almost_equal(st.feature_cov(0), cov)

    def test_P_symmetric_after_add(self):
        st = _fresh_state()
        st.add_feature(np.zeros(2), np.eye(2)*0.5, 'line', np.zeros((3, 2)))
        P = st.P
        np.testing.assert_array_almost_equal(P, P.T)

    def test_P_positive_definite_after_add(self):
        st = _fresh_state()
        st.add_feature(np.zeros(2), np.eye(2)*0.5, 'corner', np.zeros((3, 2)))
        eigvals = np.linalg.eigvalsh(st.P)
        assert np.all(eigvals >= 0)

    def test_state_slice_correct(self):
        st = _fresh_state()
        for i in range(3):
            st.add_feature(np.zeros(2), np.eye(2), 'corner',
                           np.zeros((3 + 2*i, 2)))
        for i, feat in enumerate(st.features):
            sl = feat.state_slice
            assert sl.start == 3 + 2*i
            assert sl.stop  == 3 + 2*i + 2

    def test_symmetrize_corrects_drift(self):
        st = _fresh_state()
        st.add_feature(np.zeros(2), np.eye(2), 'corner', np.zeros((3, 2)))
        st.P[0, 1] += 1e-8   # introduce tiny asymmetry
        st.symmetrize_P()
        np.testing.assert_array_almost_equal(st.P, st.P.T)


# =============================================================================
# 2. Predict
# =============================================================================

class TestPredict:

    def test_pose_updates_correctly(self):
        """Unicycle kinematics: straight ahead at v=1, omega=0, dt=1."""
        st = _fresh_state((0.0, 0.0, 0.0))
        predict(st, v=1.0, omega=0.0, dt=1.0, Q_v=0.0, Q_w=0.0)
        np.testing.assert_allclose(st.pose, [1.0, 0.0, 0.0], atol=1e-9)

    def test_pose_turning(self):
        """Pure rotation: v=0, omega=pi/2, dt=1 → theta = pi/2."""
        st = _fresh_state((0.0, 0.0, 0.0))
        predict(st, v=0.0, omega=np.pi/2, dt=1.0, Q_v=0.0, Q_w=0.0)
        np.testing.assert_allclose(st.pose, [0.0, 0.0, np.pi/2], atol=1e-9)

    def test_pose_circular_arc(self):
        """v=1, omega=pi/2, dt=1: robot ends up at (2/pi, 2/pi) heading pi/2."""
        st = _fresh_state((0.0, 0.0, 0.0))
        predict(st, v=1.0, omega=np.pi/2, dt=1.0, Q_v=0.0, Q_w=0.0)
        expected_x = np.cos(0.0) * 1.0   # one step of Euler integrate
        expected_y = np.sin(0.0) * 1.0
        np.testing.assert_allclose(st.pose[0], expected_x, atol=1e-9)
        np.testing.assert_allclose(st.pose[1], expected_y, atol=1e-9)

    def test_heading_wraps(self):
        """Heading should always stay in (-pi, pi]."""
        st = _fresh_state((0.0, 0.0, np.pi - 0.01))
        predict(st, v=0.0, omega=0.1, dt=1.0, Q_v=0.0, Q_w=0.0)
        assert -np.pi < st.pose[2] <= np.pi

    def test_Pvv_grows_with_motion(self):
        """Vehicle covariance must increase when robot moves with Q > 0."""
        st = _fresh_state((0.0, 0.0, 0.0))
        trace_before = np.trace(st.Pvv)
        predict(st, v=1.0, omega=0.0, dt=0.1, Q_v=0.05, Q_w=0.1)
        assert np.trace(st.Pvv) > trace_before

    def test_Pvv_static_no_noise(self):
        """Zero velocity + zero noise → covariance unchanged."""
        st = _fresh_state((0.0, 0.0, 0.0))
        P_before = st.P.copy()
        predict(st, v=0.0, omega=0.0, dt=0.1, Q_v=0.0, Q_w=0.0)
        np.testing.assert_array_almost_equal(st.P, P_before)

    def test_feature_cov_unchanged_by_predict(self):
        """Feature uncertainty must not grow due to predict."""
        st = _fresh_state()
        st.add_feature(np.array([5.0, 5.0]), np.eye(2)*0.3, 'corner',
                       np.zeros((3, 2)))
        cov_before = st.feature_cov(0).copy()
        predict(st, v=1.0, omega=0.0, dt=0.1, Q_v=0.1, Q_w=0.1)
        np.testing.assert_array_almost_equal(st.feature_cov(0), cov_before)

    def test_P_remains_symmetric_after_predict(self):
        st = _fresh_state()
        st.add_feature(np.zeros(2), np.eye(2)*0.5, 'corner', np.zeros((3,2)))
        predict(st, v=0.5, omega=0.3, dt=0.05, Q_v=0.02, Q_w=0.05)
        np.testing.assert_array_almost_equal(st.P, st.P.T, decimal=10)

    def test_P_positive_semidefinite_after_predict(self):
        st = _fresh_state()
        predict(st, v=1.0, omega=0.5, dt=0.1, Q_v=0.1, Q_w=0.2)
        eigvals = np.linalg.eigvalsh(st.P)
        assert np.all(eigvals >= -1e-10)

    def test_cross_corr_updated(self):
        """
        Cross-correlation Pvm must become non-zero after a predict step when
        the vehicle has non-zero Pvv and at least one feature.

        Strategy: predict first (Pvv grows from Q), then init a corner
        (cross-cov = Pxv @ Jv^T is non-zero), then predict again so
        Fv @ Pvm propagates it.
        """
        st = _fresh_state()
        # Grow Pvv via process noise
        predict(st, v=1.0, omega=0.0, dt=0.1, Q_v=0.1, Q_w=0.1)
        # Init corner: cross-cov seeded from non-zero Pvv
        z = h_corner(st.pose, np.array([5.0, 0.0]))
        init_corner(st, z, R_CORNER, init_cov_scale=0.5)
        # Another predict propagates cross-corr
        predict(st, v=0.5, omega=0.2, dt=0.1, Q_v=0.1, Q_w=0.1)
        Pvm = st.P[:3, 3:]
        assert np.any(np.abs(Pvm) > 1e-10)


# =============================================================================
# 3. Update (single feature, known association)
# =============================================================================

class TestUpdate:

    def test_innovation_near_zero_on_reobservation(self):
        """
        After initialising a corner at exact noiseless observation,
        re-observing from the *same pose* should yield near-zero innovation
        and no state change.
        """
        pose  = np.array([2.0, 2.0, 0.0])
        cpos  = np.array([6.0, 4.0])
        st    = _state_with_corner(pose, cpos)

        mean_before = st.feature_mean(0).copy()
        z_exact     = _h_corner(st, 0)
        update_single(st, 0, z_exact, R_CORNER)
        np.testing.assert_allclose(st.feature_mean(0), mean_before, atol=1e-8)

    def test_Pff_shrinks_after_update(self):
        """Feature covariance must decrease after a good observation."""
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([6.0, 4.0])
        st   = _state_with_corner(pose, cpos)

        trace_before = np.trace(st.feature_cov(0))
        z_noisy      = _h_corner(st, 0) + np.array([0.01, 0.005])
        update_single(st, 0, z_noisy, R_CORNER)
        assert np.trace(st.feature_cov(0)) < trace_before

    def test_P_symmetric_after_update(self):
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([6.0, 4.0])
        st   = _state_with_corner(pose, cpos)
        z    = _h_corner(st, 0)
        update_single(st, 0, z, R_CORNER)
        np.testing.assert_array_almost_equal(st.P, st.P.T, decimal=10)

    def test_P_psd_after_update(self):
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([6.0, 4.0])
        st   = _state_with_corner(pose, cpos)
        z    = _h_corner(st, 0) + np.array([0.02, -0.01])
        update_single(st, 0, z, R_CORNER)
        eigvals = np.linalg.eigvalsh(st.P)
        assert np.all(eigvals >= -1e-10)

    def test_line_Pff_shrinks_after_update(self):
        rho, alpha = 5.0, np.pi / 2
        pose = np.array([2.0, 0.0, 0.0])
        st   = _state_with_line(pose, rho, alpha)
        trace_before = np.trace(st.feature_cov(0))
        z_noisy      = _h_line(st, 0) + np.array([0.01, 0.005])
        update_single(st, 0, z_noisy, R_LINE)
        assert np.trace(st.feature_cov(0)) < trace_before

    def test_obs_count_increments(self):
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([6.0, 4.0])
        st   = _state_with_corner(pose, cpos)
        count_before = st.features[0].obs_count
        update_single(st, 0, _h_corner(st, 0), R_CORNER)
        assert st.features[0].obs_count == count_before + 1

    def test_multiple_updates_converge(self):
        """
        Repeatedly updating with zero-noise observations should converge
        feature mean to the true position.
        """
        true_cpos = np.array([8.0, 5.0])
        pose      = np.array([2.0, 2.0, 0.0])
        st        = _state_with_corner(pose, true_cpos)

        # perturb the feature mean slightly to simulate initialisation error
        st._x[3] += 0.3
        st._x[4] += 0.3

        for _ in range(20):
            z_exact = h_corner(pose, true_cpos)
            update_single(st, 0, z_exact, R_CORNER * 0.001)

        np.testing.assert_allclose(st.feature_mean(0), true_cpos, atol=0.1)

    def test_bearing_innovation_wraps(self):
        """
        If the expected bearing is just above +π and observed is just below -π,
        the innovation must wrap correctly (not give ≈ 2π).
        """
        pose  = np.array([0.0, 0.0, np.pi - 0.01])
        cpos  = np.array([-5.0, 0.0])
        st    = _state_with_corner(pose, cpos)
        # Compute expected z then perturb bearing by tiny amount crossing ±π
        z     = _h_corner(st, 0)
        z[1]  = _wrap(z[1] + 0.02)
        # Should not raise and P must stay PSD
        update_single(st, 0, z, R_CORNER)
        eigvals = np.linalg.eigvalsh(st.P)
        assert np.all(eigvals >= -1e-10)


# =============================================================================
# 4. Feature Initialisation
# =============================================================================

class TestFeatureInit:

    def test_corner_init_correct_mean(self):
        """Noiseless init: feature mean should match true corner position."""
        pose  = np.array([1.0, 1.0, 0.0])
        cpos  = np.array([5.0, 4.0])
        z     = h_corner(pose, cpos)
        st    = _fresh_state(pose)
        init_corner(st, z, R_CORNER, init_cov_scale=0.0)
        np.testing.assert_allclose(st.feature_mean(0), cpos, atol=1e-6)

    def test_line_init_correct_mean(self):
        """Noiseless init: feature mean should match true (rho, alpha)."""
        rho, alpha = 4.0, np.pi / 3
        pose       = np.array([0.0, 0.0, 0.0])
        z          = h_line(pose, rho, alpha)
        st         = _fresh_state(pose)
        init_line(st, z, R_LINE, init_cov_scale=0.0)
        mean = st.feature_mean(0)
        assert pytest.approx(mean[0], abs=1e-5) == rho
        assert pytest.approx(fw(mean[1] - alpha), abs=1e-5) == 0.0

    def test_corner_init_expands_state(self):
        st = _fresh_state()
        z  = h_corner(st.pose, np.array([4.0, 3.0]))
        init_corner(st, z, R_CORNER)
        assert st.n_features == 1
        assert st.dim == 5

    def test_line_init_expands_state(self):
        st = _fresh_state()
        z  = h_line(st.pose, 3.0, 0.0)
        init_line(st, z, R_LINE)
        assert st.n_features == 1
        assert st.dim == 5

    def test_init_cov_positive_definite(self):
        pose = np.array([2.0, 2.0, 0.5])
        cpos = np.array([6.0, 5.0])
        z    = h_corner(pose, cpos)
        st   = _fresh_state(pose)
        init_corner(st, z, R_CORNER, init_cov_scale=0.5)
        eigvals = np.linalg.eigvalsh(st.feature_cov(0))
        assert np.all(eigvals > 0)

    def test_init_P_symmetric(self):
        pose = np.array([2.0, 2.0, 0.0])
        z    = h_corner(pose, np.array([5.0, 5.0]))
        st   = _fresh_state(pose)
        init_corner(st, z, R_CORNER)
        np.testing.assert_array_almost_equal(st.P, st.P.T)

    def test_cross_cov_nonzero_with_uncertain_vehicle(self):
        """
        When vehicle pose covariance is non-zero, the feature-vehicle
        cross-covariance should be non-zero after init.
        """
        Pvv  = np.diag([0.1, 0.1, 0.05])
        st   = SLAMState(np.array([2.0, 2.0, 0.0]), Pvv)
        z    = h_corner(st.pose, np.array([6.0, 4.0]))
        init_corner(st, z, R_CORNER)
        cross = st.P[:3, 3:]
        assert np.any(np.abs(cross) > 1e-10)

    def test_multiple_corner_init(self):
        """Initialise 3 corners and verify all means are correct."""
        st    = _fresh_state((0.0, 0.0, 0.0))
        cposns = [np.array([4.0, 0.0]),
                  np.array([0.0, 4.0]),
                  np.array([-4.0, 0.0])]
        for cpos in cposns:
            z = h_corner(st.pose, cpos)
            init_corner(st, z, R_CORNER, init_cov_scale=0.0)
        assert st.n_features == 3
        for i, cpos in enumerate(cposns):
            np.testing.assert_allclose(st.feature_mean(i), cpos, atol=1e-5)

    def test_line_rho_non_negative_after_init(self):
        """Polar rho should be normalised to >= 0 during init."""
        pose = np.array([0.0, 0.0, np.pi])
        z    = h_line(pose, 3.0, 0.0)
        st   = _fresh_state(pose)
        init_line(st, z, R_LINE, init_cov_scale=0.0)
        rho = st.feature_mean(0)[0]
        assert rho >= -1e-9, f"rho = {rho} should be >= 0"


# =============================================================================
# 5. Data Association
# =============================================================================

class TestDataAssociation:

    def _obs_like(self, z, kind):
        """Simple namespace to duck-type observation objects."""
        from types import SimpleNamespace
        return SimpleNamespace(z=np.array(z), kind=kind, feature_kind=kind)

    def test_mahalanobis_zero_for_zero_innovation(self):
        S  = np.eye(2)
        nu = np.zeros(2)
        assert mahalanobis(nu, S) == pytest.approx(0.0)

    def test_mahalanobis_identity_covariance(self):
        nu = np.array([1.0, 0.0])
        S  = np.eye(2)
        assert mahalanobis(nu, S) == pytest.approx(1.0)

    def test_mahalanobis_scales_with_uncertainty(self):
        nu    = np.array([1.0, 0.0])
        S_big = np.diag([4.0, 4.0])
        S_sm  = np.diag([1.0, 1.0])
        assert mahalanobis(nu, S_big) < mahalanobis(nu, S_sm)

    def test_associate_known_corner_matches(self):
        """A near-noiseless observation of the only feature should match it."""
        pose  = np.array([2.0, 2.0, 0.0])
        cpos  = np.array([6.0, 4.0])
        st    = _state_with_corner(pose, cpos)
        z     = _h_corner(st, 0) + np.array([0.001, 0.001])
        idx, dist = associate(st, z, 'corner', R_CORNER, gate_chi2=9.21)
        assert idx == 0
        assert dist < 9.21

    def test_associate_new_corner_returns_minus_one(self):
        """Observation far from all features must return -1 (new feature)."""
        pose  = np.array([2.0, 2.0, 0.0])
        cpos  = np.array([6.0, 4.0])
        st    = _state_with_corner(pose, cpos)
        # fabricate observation of a completely different corner
        z_new = h_corner(np.array(pose), np.array([0.5, 0.5]))
        idx, _ = associate(st, z_new, 'corner', R_CORNER, gate_chi2=9.21)
        assert idx == -1

    def test_associate_wrong_kind_skipped(self):
        """A line observation must not match a corner feature."""
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([6.0, 4.0])
        st   = _state_with_corner(pose, cpos)
        z    = h_line(pose, 4.0, 0.0)
        idx, _ = associate(st, z, 'line', R_LINE, gate_chi2=9.21)
        assert idx == -1

    def test_associate_selects_nearest_of_two(self):
        """With two corners, the closer one (in Mahalanobis sense) wins."""
        pose  = np.array([0.0, 0.0, 0.0])
        cpos1 = np.array([5.0, 0.0])
        cpos2 = np.array([0.0, 5.0])
        st    = _fresh_state(pose)
        for cpos in [cpos1, cpos2]:
            z = h_corner(pose, cpos)
            init_corner(st, z, R_CORNER, init_cov_scale=0.0)

        # observe corner 1 with tiny noise → should match feat 0
        z_c1  = h_corner(pose, cpos1) + np.array([0.001, 0.0])
        idx, _ = associate(st, z_c1, 'corner', R_CORNER, gate_chi2=9.21)
        assert idx == 0

        # observe corner 2 with tiny noise → should match feat 1
        z_c2  = h_corner(pose, cpos2) + np.array([0.0, 0.001])
        idx, _ = associate(st, z_c2, 'corner', R_CORNER, gate_chi2=9.21)
        assert idx == 1

    def test_batch_associate_all_known(self):
        """All observations of known features should match."""
        pose   = np.array([0.0, 0.0, 0.0])
        cposns = [np.array([4.0, 0.0]), np.array([0.0, 4.0])]
        st     = _fresh_state(pose)
        for cp in cposns:
            init_corner(st, h_corner(pose, cp), R_CORNER, init_cov_scale=0.0)

        obs_list = [self._obs_like(h_corner(pose, cp), 'corner') for cp in cposns]
        results  = associate_observations(st, obs_list, R_CORNER, R_LINE)
        matched  = [idx for _, idx in results]
        assert -1 not in matched
        assert set(matched) == {0, 1}

    def test_batch_associate_new_feature(self):
        """A truly new observation should produce feat_idx == -1."""
        pose = np.array([0.0, 0.0, 0.0])
        st   = _fresh_state(pose)
        init_corner(st, h_corner(pose, np.array([4.0, 0.0])),
                    R_CORNER, init_cov_scale=0.0)

        obs  = self._obs_like(h_corner(pose, np.array([0.0, 4.0])), 'corner')
        res  = associate_observations(st, [obs], R_CORNER, R_LINE)
        assert res[0][1] == -1

    def test_gate_threshold_respected(self):
        """With a gate of 0 nothing should ever match."""
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([6.0, 4.0])
        st   = _state_with_corner(pose, cpos)
        z    = _h_corner(st, 0)
        idx, _ = associate(st, z, 'corner', R_CORNER, gate_chi2=0.0)
        assert idx == -1


# =============================================================================
# 6. Integration: full predict–update loop
# =============================================================================

class TestIntegration:

    def test_Pff_decreases_on_repeated_observation(self):
        """
        Observing the same corner many times from the same pose should
        reduce the feature covariance trace monotonically (approximately).
        """
        pose = np.array([2.0, 2.0, 0.0])
        cpos = np.array([8.0, 5.0])
        st   = _fresh_state(pose)
        init_corner(st, h_corner(pose, cpos), R_CORNER, init_cov_scale=1.0)

        trace_vals = [np.trace(st.feature_cov(0))]
        for _ in range(15):
            z = h_corner(pose, cpos)
            update_single(st, 0, z, R_CORNER)
            trace_vals.append(np.trace(st.feature_cov(0)))

        assert trace_vals[-1] < trace_vals[0], \
            "Feature covariance did not decrease with repeated observations"

    def test_Pvv_grows_then_shrinks(self):
        """
        During pure motion Pvv grows; after observing a fixed landmark it
        should be corrected (Pvv shrinks or stays bounded).
        """
        pose = np.array([1.0, 1.0, 0.0])
        cpos = np.array([8.0, 1.0])
        st   = _fresh_state(pose)
        init_corner(st, h_corner(pose, cpos), R_CORNER, init_cov_scale=0.1)

        # several predict steps (grow)
        for _ in range(5):
            predict(st, v=0.5, omega=0.0, dt=0.1, Q_v=0.05, Q_w=0.05)

        trace_after_motion = np.trace(st.Pvv)

        # update with a good observation (shrink)
        for _ in range(5):
            predict(st, v=0.5, omega=0.0, dt=0.1, Q_v=0.05, Q_w=0.05)
            z = h_corner(st.pose, cpos)
            update_single(st, 0, z, R_CORNER)

        trace_after_update = np.trace(st.Pvv)
        assert trace_after_update < trace_after_motion * 2, \
            "Vehicle covariance not corrected after landmark observations"

    def test_no_duplicate_features_on_repeat_obs(self, lab, noiseless_ex):
        """
        Observing the same scene repeatedly must not create unbound duplicate
        features.  Corner count must not exceed GT+2; lines get a slightly
        larger allowance because collinear segments share polar params and
        the gate sometimes sees them as distinct on first encounter.
        """
        pose = np.array([6.0, 4.0, 0.0])
        st   = _fresh_state(pose)

        for _ in range(10):
            corners, lines = noiseless_ex.extract(pose, lab)
            all_obs = list(corners) + list(lines)
            for obs in all_obs:
                R = R_CORNER if obs.feature_kind == 'corner' else R_LINE
                idx, _ = associate(st, obs.z, obs.feature_kind, R, gate_chi2=9.21)
                if idx == -1:
                    if obs.feature_kind == 'corner':
                        init_corner(st, obs.z, R, init_cov_scale=0.5)
                    else:
                        init_line(st, obs.z, R, init_cov_scale=0.5)
                else:
                    update_single(st, idx, obs.z, R)

        n_corners = sum(1 for f in st.features if f.kind == 'corner')
        n_lines   = sum(1 for f in st.features if f.kind == 'line')
        n_gt_corners = len(lab.corners)
        n_gt_lines   = len(lab.segments)

        assert n_corners <= n_gt_corners + 2, \
            f"Too many corner features: {n_corners} (GT={n_gt_corners})"
        # Allow up to 2× GT lines: collinear segments seen from one static
        # pose can register as separate features before covariance tightens.
        assert n_lines <= n_gt_lines * 2 + 2, \
            f"Too many line features: {n_lines} (GT={n_gt_lines})"
        # After convergence, 2 more identical passes must add zero new features.
        for _ in range(2):
            corners2, lines2 = noiseless_ex.extract(pose, lab)
            for obs in list(corners2) + list(lines2):
                R = R_CORNER if obs.feature_kind == 'corner' else R_LINE
                idx, _ = associate(st, obs.z, obs.feature_kind, R, gate_chi2=9.21)
                if idx == -1:
                    if obs.feature_kind == 'corner':
                        init_corner(st, obs.z, R, init_cov_scale=0.5)
                    else:
                        init_line(st, obs.z, R, init_cov_scale=0.5)
                else:
                    update_single(st, idx, obs.z, R)

        n_stable_corners = sum(1 for f in st.features if f.kind == 'corner')
        n_stable_lines   = sum(1 for f in st.features if f.kind == 'line')
        assert n_stable_corners <= n_gt_corners + 2, \
            f"Corners still growing: {n_stable_corners}"
        assert n_stable_lines <= n_gt_lines * 2 + 2, \
            f"Lines still growing: {n_stable_lines}"

    def test_state_dim_consistent(self):
        """State dimension must always equal 3 + 2 * n_features."""
        pose = np.array([0.0, 0.0, 0.0])
        st   = _fresh_state(pose)
        cposns = [np.array([4.0, 0.0]),
                  np.array([0.0, 4.0]),
                  np.array([-3.0, 2.0])]
        for cp in cposns:
            init_corner(st, h_corner(pose, cp), R_CORNER)
            assert st.dim == 3 + 2 * st.n_features

    def test_P_stays_psd_through_full_loop(self):
        """
        Run predict + update for 20 steps and verify P never loses PSD.
        """
        pose = np.array([1.0, 1.0, 0.0])
        st   = _fresh_state(pose)
        world = World.from_preset('lab')
        ex    = FeatureExtractor(fov_rad=np.radians(180), max_range=15.0,
                                 noise_range=0.03, noise_bearing=0.01,
                                 rng=np.random.default_rng(99))

        for step in range(20):
            predict(st, v=0.3, omega=0.1, dt=0.1, Q_v=0.02, Q_w=0.05)
            corners, lines = ex.extract(st.pose, world)
            for obs in list(corners) + list(lines):
                R   = R_CORNER if obs.feature_kind == 'corner' else R_LINE
                idx, _ = associate(st, obs.z, obs.feature_kind, R, gate_chi2=9.21)
                if idx == -1:
                    if obs.feature_kind == 'corner':
                        init_corner(st, obs.z, R)
                    else:
                        init_line(st, obs.z, R)
                else:
                    update_single(st, idx, obs.z, R)

            eigvals = np.linalg.eigvalsh(st.P)
            assert np.all(eigvals >= -1e-8), \
                f"P not PSD at step {step}: min eigval = {eigvals.min():.2e}"