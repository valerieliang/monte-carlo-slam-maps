"""
tests/test_phase2.py
--------------------
Phase 2 exit-criterion checks: sensor raycasting, noise statistics,
ScanResult structure, and expected-measurement model functions.

Run with:
  cd slam_sim
  python -m pytest tests/test_phase2.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from env.world  import World, Segment
from env.robot  import Robot
from env.sensor import Sensor, ScanResult


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def lab():
    return World.from_preset('lab')

@pytest.fixture
def sensor():
    # deterministic RNG for reproducibility
    return Sensor(fov_deg=180, num_rays=181, max_range=10.0,
                  noise_range=0.03, noise_bearing=0.01,
                  rng=np.random.default_rng(42))

@pytest.fixture
def noiseless():
    return Sensor(fov_deg=180, num_rays=181, max_range=10.0,
                  noise_range=0.0, noise_bearing=0.0,
                  rng=np.random.default_rng(0))

@pytest.fixture
def robot_centre():
    """Robot at centre of lab room facing east."""
    return Robot(6.0, 4.0, 0.0)


# ---------------------------------------------------------------- ScanResult

class TestScanResult:

    def test_shape(self, sensor, lab, robot_centre):
        scan = sensor.scan(robot_centre, lab)
        assert scan.ranges.shape    == (181,)
        assert scan.bearings.shape  == (181,)
        assert scan.hit_xy.shape    == (181, 2)
        assert scan.ray_angles.shape == (181,)

    def test_valid_mask_type(self, sensor, lab, robot_centre):
        scan = sensor.scan(robot_centre, lab)
        assert scan.valid_mask.dtype == bool

    def test_valid_hits_shape(self, sensor, lab, robot_centre):
        scan = sensor.scan(robot_centre, lab)
        m = scan.n_valid
        assert scan.valid_hits.shape == (m, 2)

    def test_no_negative_ranges(self, sensor, lab, robot_centre):
        scan = sensor.scan(robot_centre, lab)
        valid_r = scan.ranges[scan.valid_mask]
        assert np.all(valid_r >= 0)

    def test_ranges_within_max(self, sensor, lab, robot_centre):
        scan = sensor.scan(robot_centre, lab)
        valid_r = scan.ranges[scan.valid_mask]
        assert np.all(valid_r <= sensor.max_range + 1e-6)

    def test_inf_where_no_hit(self, sensor, lab, robot_centre):
        """Any ray with no hit must have inf range."""
        scan = sensor.scan(robot_centre, lab)
        no_hit = ~scan.valid_mask
        assert np.all(np.isinf(scan.ranges[no_hit]))

    def test_nan_hit_xy_where_no_hit(self, sensor, lab, robot_centre):
        scan = sensor.scan(robot_centre, lab)
        no_hit = ~scan.valid_mask
        assert np.all(np.isnan(scan.hit_xy[no_hit]))


# -------------------------------------------------------- geometry / coverage

class TestScanGeometry:

    def test_all_rays_hit_in_lab(self, noiseless, lab):
        """From the centre of lab, all 181 rays must hit something."""
        robot = Robot(6.0, 4.0, 0.0)
        scan  = noiseless.scan(robot, lab)
        assert scan.n_valid == 181, \
            f"Only {scan.n_valid}/181 rays hit (expected all)"

    def test_facing_east_range_to_wall(self, noiseless, lab):
        """
        Robot at (6,4) facing east (theta=0).
        Centre ray (bearing=0) should hit the east wall at x=12 -> range=6.
        """
        robot = Robot(6.0, 4.0, 0.0)
        scan  = noiseless.scan(robot, lab)
        centre_idx = 90          # middle ray of 181
        assert pytest.approx(scan.ranges[centre_idx], abs=0.05) == 6.0

    def test_facing_west_range_to_wall(self, noiseless, lab):
        """Robot at (6,4) facing west -> centre ray hits x=0 wall, range=6."""
        robot = Robot(6.0, 4.0, np.pi)
        scan  = noiseless.scan(robot, lab)
        centre_idx = 90
        assert pytest.approx(scan.ranges[centre_idx], abs=0.05) == 6.0

    def test_hit_points_near_walls(self, noiseless, lab):
        """All noiseless hit points should lie on wall segments (within 1 cm)."""
        robot = Robot(6.0, 4.0, 0.0)
        scan  = noiseless.scan(robot, lab)

        def point_on_any_segment(pt, segs, tol=0.01):
            for seg in segs:
                d   = seg.p1 - seg.p0
                t   = np.dot(pt - seg.p0, d) / (np.dot(d, d) + 1e-12)
                t   = np.clip(t, 0, 1)
                proj = seg.p0 + t * d
                if np.linalg.norm(pt - proj) < tol:
                    return True
            return False

        hits = scan.valid_hits
        for pt in hits:
            assert point_on_any_segment(pt, lab.segments), \
                f"Hit point {pt} is not on any wall segment"

    def test_ray_angles_span_fov(self, noiseless, lab):
        """Ray angles should span exactly the configured FOV."""
        robot = Robot(6.0, 4.0, 0.0)
        scan  = noiseless.scan(robot, lab)
        span  = scan.ray_angles[-1] - scan.ray_angles[0]
        assert pytest.approx(span, abs=1e-6) == np.radians(180.0)

    def test_hit_xy_consistent_with_range(self, noiseless, lab):
        """
        For each valid hit, the Euclidean distance from the robot to the
        hit point must equal the reported range (noiseless case).
        """
        robot = Robot(6.0, 4.0, 0.0)
        scan  = noiseless.scan(robot, lab)
        origin = np.array([robot.x, robot.y])
        mask   = scan.valid_mask
        dists  = np.linalg.norm(scan.hit_xy[mask] - origin, axis=1)
        np.testing.assert_allclose(dists, scan.true_ranges[mask], atol=1e-6)

    def test_range_symmetric_fov(self, noiseless):
        """
        In an open square room, a robot at the centre facing north should
        see equal ranges on left and right (symmetric FOV).
        """
        world = World.from_preset('open')   # 10x10 square
        robot = Robot(5.0, 5.0, np.pi / 2) # facing north
        scan  = noiseless.scan(robot, world)
        # left-most and right-most rays should be symmetric
        assert pytest.approx(scan.true_ranges[0],
                              abs=0.05) == scan.true_ranges[-1]


# ------------------------------------------------------------------- noise

class TestNoise:

    def test_noise_is_present(self, sensor, lab, robot_centre):
        """With noise enabled, ranges must differ from true ranges."""
        scan = sensor.scan(robot_centre, lab)
        mask = scan.valid_mask
        diff = np.abs(scan.ranges[mask] - scan.true_ranges[mask])
        assert diff.max() > 1e-6, "No noise detected — noise_range may be 0"

    def test_range_noise_magnitude(self, lab):
        """
        Over many scans, range noise std should be close to configured sigma.
        Use a fixed robot position for repeatability.
        """
        sigma = 0.05
        s     = Sensor(fov_deg=180, num_rays=181, max_range=10.0,
                       noise_range=sigma, noise_bearing=0.0,
                       rng=np.random.default_rng(7))
        robot = Robot(6.0, 4.0, 0.0)
        world = World.from_preset('lab')

        errors = []
        for _ in range(200):
            scan   = s.scan(robot, world)
            mask   = scan.valid_mask
            errors.extend((scan.ranges[mask] - scan.true_ranges[mask]).tolist())

        measured_std = np.std(errors)
        assert pytest.approx(measured_std, rel=0.15) == sigma, \
            f"Range noise std {measured_std:.4f} != configured {sigma}"

    def test_bearing_noise_magnitude(self, lab):
        """Bearing noise std should match configured sigma."""
        sigma = 0.02
        s     = Sensor(fov_deg=180, num_rays=181, max_range=10.0,
                       noise_range=0.0, noise_bearing=sigma,
                       rng=np.random.default_rng(13))
        robot    = Robot(6.0, 4.0, 0.0)
        world    = World.from_preset('lab')
        true_brg = s._ray_offsets.copy()

        errors = []
        for _ in range(200):
            scan   = s.scan(robot, world)
            errors.extend((scan.bearings - true_brg).tolist())

        measured_std = np.std(errors)
        assert pytest.approx(measured_std, rel=0.15) == sigma, \
            f"Bearing noise std {measured_std:.4f} != configured {sigma}"

    def test_different_seeds_differ(self, lab, robot_centre):
        s1 = Sensor(noise_range=0.05, rng=np.random.default_rng(1))
        s2 = Sensor(noise_range=0.05, rng=np.random.default_rng(2))
        r1 = s1.scan(robot_centre, lab).ranges
        r2 = s2.scan(robot_centre, lab).ranges
        assert not np.allclose(r1, r2)

    def test_same_seed_reproduces(self, lab, robot_centre):
        s1 = Sensor(noise_range=0.05, rng=np.random.default_rng(99))
        s2 = Sensor(noise_range=0.05, rng=np.random.default_rng(99))
        r1 = s1.scan(robot_centre, lab).ranges
        r2 = s2.scan(robot_centre, lab).ranges
        np.testing.assert_array_equal(r1, r2)


# ------------------------------------------- expected measurement models

class TestMeasurementModels:

    def test_expected_corner_range(self):
        """Corner at (4,0) from robot at (0,0,0): range = 4."""
        s    = Sensor()
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([4.0, 0.0])
        z    = s.expected_corner(pose, cpos)
        assert pytest.approx(z[0], abs=1e-9) == 4.0

    def test_expected_corner_bearing_zero(self):
        """Corner directly ahead: bearing should be 0."""
        s    = Sensor()
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([3.0, 0.0])
        z    = s.expected_corner(pose, cpos)
        assert pytest.approx(z[1], abs=1e-9) == 0.0

    def test_expected_corner_bearing_left(self):
        """Corner at 90 deg to the left: bearing = +pi/2."""
        s    = Sensor()
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([0.0, 3.0])
        z    = s.expected_corner(pose, cpos)
        assert pytest.approx(z[1], abs=1e-6) == np.pi / 2

    def test_expected_corner_behind(self):
        """Corner directly behind: bearing = +/-pi."""
        s    = Sensor()
        pose = np.array([0.0, 0.0, 0.0])
        cpos = np.array([-2.0, 0.0])
        z    = s.expected_corner(pose, cpos)
        assert pytest.approx(abs(z[1]), abs=1e-6) == np.pi

    def test_expected_line_ahead(self):
        """
        Horizontal wall at y=5 (rho=5, alpha=pi/2) seen from (0,0,0).
        Expected rho_obs = 5 - 0*cos(pi/2) - 0*sin(pi/2) = 5.
        Expected alpha_obs = pi/2 - 0 = pi/2.
        """
        s     = Sensor()
        pose  = np.array([0.0, 0.0, 0.0])
        rho, alpha = 5.0, np.pi / 2
        z     = s.expected_line(pose, rho, alpha)
        assert pytest.approx(z[0], abs=1e-9) == 5.0
        assert pytest.approx(z[1], abs=1e-9) == np.pi / 2

    def test_expected_line_robot_offset(self):
        """
        Wall at y=5 (rho=5, alpha=pi/2), robot at (0, 2, 0).
        rho_obs = 5 - 0*cos(pi/2) - 2*sin(pi/2) = 5 - 2 = 3.
        """
        s     = Sensor()
        pose  = np.array([0.0, 2.0, 0.0])
        rho, alpha = 5.0, np.pi / 2
        z     = s.expected_line(pose, rho, alpha)
        assert pytest.approx(z[0], abs=1e-9) == 3.0

    def test_bearing_wrapping(self):
        """expected_corner bearing must stay in (-pi, pi]."""
        s = Sensor()
        for angle in np.linspace(-np.pi + 1e-9, np.pi, 37):
            pose = np.array([0.0, 0.0, angle])
            cpos = np.array([1.0, 0.0])
            z    = s.expected_corner(pose, cpos)
            assert -np.pi <= z[1] <= np.pi, \
                f"Bearing {z[1]} out of range for robot angle {angle}"


# ---------------------------------------- integration: scan -> model round-trip

class TestScanModelConsistency:

    def test_corner_range_consistent_with_scan(self):
        """
        Place a corner at a known position.  The noiseless scan range for
        the ray pointing at the corner should match expected_corner's range.
        """
        # minimal world: just one wall + one corner at (5, 0)
        world  = World.from_preset('open')
        robot  = Robot(0.0, 5.0, 0.0)           # at (0,5) facing east
        sensor = Sensor(fov_deg=10, num_rays=11,
                        max_range=15.0,
                        noise_range=0.0, noise_bearing=0.0,
                        rng=np.random.default_rng(0))

        scan = sensor.scan(robot, world)
        # centre ray points east; east wall at x=10 -> range = 10
        centre_idx = 5
        model_z    = sensor.expected_corner(robot.pose,
                                             np.array([10.0, 5.0]))
        assert pytest.approx(scan.true_ranges[centre_idx],
                             abs=0.05) == model_z[0]