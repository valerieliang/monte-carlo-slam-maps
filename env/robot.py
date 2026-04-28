"""
env/robot.py
------------
Robot state and unicycle kinematics.

The robot is a non-holonomic unicycle (like the Pioneer 3AT in the paper).
State: (x, y, theta) in world frame.

Phase 1 only exposes the kinematic model and manual-drive interface.
The EKF will consume the same model in Phase 4 via slam/predict.py.
"""

from __future__ import annotations
import numpy as np


class Robot:
    """
    Unicycle mobile robot with pose (x, y, theta).

    Parameters
    ----------
    x, y        : initial position (metres)
    theta       : initial heading (radians, world frame)
    radius      : physical robot radius used for collision / virtual-feature
                  offset (metres) -- 0.25 m matching the paper
    """

    def __init__(self,
                 x: float = 1.0,
                 y: float = 1.0,
                 theta: float = 0.0,
                 radius: float = 0.25):
        self.x      = float(x)
        self.y      = float(y)
        self.theta  = float(theta)
        self.radius = float(radius)

        # history of (x, y) positions for drawing the trail
        self._trail: list[tuple[float, float]] = [(self.x, self.y)]

    # -- pose access -----------------------------------------------------------

    @property
    def pos(self) -> np.ndarray:
        """Current (x, y) position as a numpy array."""
        return np.array([self.x, self.y])

    @property
    def pose(self) -> np.ndarray:
        """Current (x, y, theta) as a numpy array."""
        return np.array([self.x, self.y, self.theta])

    @property
    def trail(self) -> list[tuple[float, float]]:
        return self._trail

    # -- kinematics ------------------------------------------------------------

    def step(self, v: float, omega: float, dt: float) -> None:
        """
        Apply unicycle kinematics for one timestep (Eq. 3 in the paper).

            x_dot     = v * cos(theta)
            y_dot     = v * sin(theta)
            theta_dot = omega

        Parameters
        ----------
        v     : linear  velocity (m/s)
        omega : angular velocity (rad/s)
        dt    : timestep (s)
        """
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta += omega * dt
        self.theta  = self._wrap(self.theta)

        self._trail.append((self.x, self.y))
        if len(self._trail) > 2000:
            self._trail = self._trail[-2000:]

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self, x: float, y: float, theta: float) -> None:
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self._trail = [(self.x, self.y)]

    def __repr__(self) -> str:
        return (f"Robot(x={self.x:.2f}, y={self.y:.2f}, "
                f"theta={np.degrees(self.theta):.1f}\u00b0)")