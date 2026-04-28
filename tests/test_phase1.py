"""
tests/test_phase1.py
--------------------
Phase 1 exit-criterion checks.

Run with:
  cd slam_sim
  python -m pytest tests/test_phase1.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from env.world  import World, Segment, Corner
from env.robot  import Robot
from config     import Config


# -----------------------------------------------------------------------------
# Segment
# -----------------------------------------------------------------------------

class TestSegment:

    def test_midpoint(self):
        s = Segment([0, 0], [4, 0])
        np.testing.assert_allclose(s.midpoint, [2, 0])

    def test_length(self):
        s = Segment([0, 0], [3, 4])
        assert pytest.approx(s.length, rel=1e-6) == 5.0

    def test_direction_unit(self):
        s = Segment([0, 0], [3, 4])
        d = s.direction
        assert pytest.approx(np.linalg.norm(d), rel=1e-6) == 1.0

    def test_normal_perpendicular(self):
        s = Segment([0, 0], [1, 0])
        assert pytest.approx(np.dot(s.direction, s.normal), abs=1e-9) == 0.0

    def test_polar_line_rho_nonneg(self):
        for p0, p1 in [([0,0],[12,0]), ([0,8],[12,8]), ([0,0],[0,8])]:
            s = Segment(p0, p1)
            rho, alpha = s.as_polar_line()
            assert rho >= 0, f"rho={rho} for segment {p0}->{p1}"

    def test_point_in_region_inside(self):
        s = Segment([0, 0], [4, 0])
        assert s.point_in_region([2, 3])   # above midpoint of horizontal seg

    def test_point_in_region_outside(self):
        s = Segment([0, 0], [4, 0])
        assert not s.point_in_region([5, 0])  # past endpoint


# -----------------------------------------------------------------------------
# World
# -----------------------------------------------------------------------------

class TestWorld:

    @pytest.mark.parametrize('preset', ['lab', 'corridor', 'open'])
    def test_preset_loads(self, preset):
        w = World.from_preset(preset)
        assert len(w.segments) > 0
        assert len(w.corners)  > 0

    def test_bounds_lab(self):
        w = World.from_preset('lab')
        xmin, xmax, ymin, ymax = w.bounds
        assert xmin <= 0 and xmax >= 12
        assert ymin <= 0 and ymax >= 8

    def test_ray_hits_south_wall(self):
        """
        Ray cast south from (6, 4) hits the interior wall at y=3 first
        (the east arm Segment([6,3],[9,3]) passes through x=6..9, y=3).
        From x=2 (west of interior wall) the south wall at y=0 is first.
        """
        w = World.from_preset('lab')
        # from west side of room, no interior wall in the way
        result = w.ray_intersect(np.array([2.0, 4.0]), angle=-np.pi/2)
        assert result is not None
        dist, pt = result
        assert pytest.approx(pt[1], abs=1e-3) == 0.0

    def test_ray_hits_interior_wall_before_south(self):
        """From (7, 4) casting south, interior wall at y=3 is hit first."""
        w = World.from_preset('lab')
        result = w.ray_intersect(np.array([7.0, 4.0]), angle=-np.pi/2)
        assert result is not None
        dist, pt = result
        assert pytest.approx(pt[1], abs=1e-3) == 3.0

    def test_ray_hits_east_wall(self):
        """Ray cast east from interior should hit x=12 wall."""
        w = World.from_preset('lab')
        result = w.ray_intersect(np.array([6.0, 4.0]), angle=0.0)
        assert result is not None
        dist, pt = result
        assert pytest.approx(pt[0], abs=1e-3) == 12.0

    def test_ray_max_range(self):
        """Ray with max_range shorter than wall should return None."""
        w = World.from_preset('lab')
        result = w.ray_intersect(np.array([6.0, 4.0]), angle=0.0,
                                 max_range=1.0)
        assert result is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            World.from_preset('nonexistent')

    def test_corner_kinds(self):
        w = World.from_preset('lab')
        kinds = {c.kind for c in w.corners}
        assert kinds <= {'convex', 'concave'}


# -----------------------------------------------------------------------------
# Robot kinematics
# -----------------------------------------------------------------------------

class TestRobot:

    def test_initial_pose(self):
        r = Robot(2.0, 3.0, np.pi/4)
        np.testing.assert_allclose(r.pose, [2.0, 3.0, np.pi/4])

    def test_straight_drive(self):
        """Driving straight for 1 s at 1 m/s along x-axis."""
        r = Robot(0.0, 0.0, 0.0)
        for _ in range(20):          # 20 steps × dt=0.05 s = 1.0 s
            r.step(v=1.0, omega=0.0, dt=0.05)
        np.testing.assert_allclose(r.x, 1.0, atol=1e-6)
        np.testing.assert_allclose(r.y, 0.0, atol=1e-6)

    def test_spin_in_place(self):
        """omega=pi rad/s for 1 s -> 180° turn, no translation."""
        r = Robot(0.0, 0.0, 0.0)
        for _ in range(20):
            r.step(v=0.0, omega=np.pi, dt=0.05)
        np.testing.assert_allclose(r.x, 0.0, atol=1e-6)
        np.testing.assert_allclose(r.y, 0.0, atol=1e-6)
        np.testing.assert_allclose(abs(r.theta), np.pi, atol=1e-6)

    def test_angle_wrapping(self):
        """Theta should always stay in (-pi, pi]."""
        r = Robot(0.0, 0.0, 0.0)
        for _ in range(100):
            r.step(v=0.0, omega=1.0, dt=0.1)
        assert -np.pi < r.theta <= np.pi

    def test_trail_grows(self):
        r = Robot(0.0, 0.0, 0.0)
        assert len(r.trail) == 1
        for _ in range(10):
            r.step(1.0, 0.0, 0.05)
        assert len(r.trail) == 11

    def test_reset(self):
        r = Robot(1.0, 2.0, 0.5)
        r.step(1.0, 0.5, 0.1)
        r.reset(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r.pose, [0.0, 0.0, 0.0])
        assert len(r.trail) == 1

    def test_pos_property(self):
        r = Robot(3.0, 4.0, 0.0)
        np.testing.assert_allclose(r.pos, [3.0, 4.0])

    def test_semicircle(self):
        """
        Drive in a circle: v=r*omega, after pi/omega seconds the robot
        should be diametrically opposite its start.
        """
        radius_circ = 1.0
        omega       = 1.0          # rad/s
        v           = radius_circ * omega   # 1 m/s

        r   = Robot(0.0, 0.0, 0.0)
        T   = np.pi / omega        # half-period = pi s
        dt  = 0.001
        steps = int(T / dt)
        for _ in range(steps):
            r.step(v, omega, dt)

        # after half a circle the robot should be at (0, 2*radius_circ)
        np.testing.assert_allclose(r.x,     0.0,            atol=0.01)
        np.testing.assert_allclose(r.y,     2*radius_circ,  atol=0.01)
        np.testing.assert_allclose(r.theta, np.pi,          atol=0.01)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        cfg = Config()
        assert cfg.robot.radius == 0.25
        assert cfg.sim.dt       == 0.05
        assert cfg.sensor.num_rays == 181

    def test_load_yaml(self, tmp_path):
        yaml_content = """
robot:
  start_x: 5.0
  max_v: 1.5
sim:
  dt: 0.1
"""
        p = tmp_path / 'test.yaml'
        p.write_text(yaml_content)
        cfg = Config.load(str(p))
        assert cfg.robot.start_x == 5.0
        assert cfg.robot.max_v   == 1.5
        assert cfg.sim.dt        == 0.1
        # unspecified values keep their defaults
        assert cfg.robot.radius  == 0.25

    def test_missing_file_uses_defaults(self):
        cfg = Config.load('nonexistent_file.yaml')
        assert cfg.robot.radius == 0.25


# -----------------------------------------------------------------------------
# Integration: ray geometry consistent with world geometry
# -----------------------------------------------------------------------------

class TestRayWorldConsistency:

    def test_all_rays_hit_something(self):
        """From the centre of 'lab', all 181 rays should hit a wall."""
        w = World.from_preset('lab')
        origin = np.array([6.0, 4.0])
        fov    = np.linspace(-np.pi/2, np.pi/2, 181)
        for angle in fov:
            result = w.ray_intersect(origin, angle, max_range=20.0)
            assert result is not None, \
                f"Ray at {np.degrees(angle):.1f}° found no intersection"

    def test_ray_distance_positive(self):
        w = World.from_preset('lab')
        origin = np.array([6.0, 4.0])
        result = w.ray_intersect(origin, 0.0)
        assert result[0] > 0