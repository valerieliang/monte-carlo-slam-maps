"""
config.py
---------
Loads config.yaml and exposes all parameters as a typed, dot-accessible
object.  All modules import from here rather than reading YAML directly.

Usage
-----
>>> from slam_sim.config import Config
>>> cfg = Config.load('config.yaml')
>>> cfg.robot.max_v
0.5
>>> cfg.sensor.num_rays
181
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# -----------------------------------------------------------------------------
# Section dataclasses
# -----------------------------------------------------------------------------

@dataclass
class WorldCfg:
    preset: str = 'lab'

@dataclass
class RobotCfg:
    start_x:     float = 1.0
    start_y:     float = 1.0
    start_theta: float = 0.0
    radius:      float = 0.25
    max_v:       float = 0.5
    max_omega:   float = 1.0

@dataclass
class SimCfg:
    dt:          float = 0.05
    render_fps:  int   = 20

@dataclass
class SensorCfg:
    fov_deg:       float = 180.0
    num_rays:      int   = 181
    max_range:     float = 8.0
    noise_range:   float = 0.03
    noise_bearing: float = 0.01

@dataclass
class EKFCfg:
    Q_v:              float = 0.02
    Q_w:              float = 0.05
    R_range:          float = 0.03
    R_bearing:        float = 0.01
    gate_chi2:        float = 9.21
    init_cov_corner:  float = 0.5
    init_cov_line:    float = 0.5

@dataclass
class MonteCarloCfg:
    n_samples_local:  int   = 300
    n_samples_global: int   = 800
    local_area_size:  float = 5.0
    uncertainty_lo:   float = 0.40
    uncertainty_hi:   float = 0.60
    virtual_cov:      float = 0.32

@dataclass
class NavigationCfg:
    goal_tolerance:   float = 0.30
    controller_k_v:   float = 0.4
    controller_k_w:   float = 1.2
    duplicate_thresh: float = 0.50

@dataclass
class RendererCfg:
    figsize: Tuple[int, int] = (10, 7)
    dpi:     int             = 110


# -----------------------------------------------------------------------------
# Top-level config
# -----------------------------------------------------------------------------

@dataclass
class Config:
    world:       WorldCfg      = field(default_factory=WorldCfg)
    robot:       RobotCfg      = field(default_factory=RobotCfg)
    sim:         SimCfg        = field(default_factory=SimCfg)
    sensor:      SensorCfg     = field(default_factory=SensorCfg)
    ekf:         EKFCfg        = field(default_factory=EKFCfg)
    montecarlo:  MonteCarloCfg = field(default_factory=MonteCarloCfg)
    navigation:  NavigationCfg = field(default_factory=NavigationCfg)
    renderer:    RendererCfg   = field(default_factory=RendererCfg)

    @classmethod
    def load(cls, path: str | Path = 'config.yaml') -> 'Config':
        """Load a YAML config file and return a populated Config object."""
        p = Path(path)
        if not p.exists():
            print(f"[Config] '{path}' not found — using all defaults.")
            return cls()

        with open(p) as f:
            raw = yaml.safe_load(f) or {}

        def _get(section_cls, key):
            data = raw.get(key, {})
            # filter to only known fields
            valid = {f: v for f, v in data.items()
                     if f in section_cls.__dataclass_fields__}
            return section_cls(**valid)

        return cls(
            world      = _get(WorldCfg,      'world'),
            robot      = _get(RobotCfg,      'robot'),
            sim        = _get(SimCfg,        'sim'),
            sensor     = _get(SensorCfg,     'sensor'),
            ekf        = _get(EKFCfg,        'ekf'),
            montecarlo = _get(MonteCarloCfg, 'montecarlo'),
            navigation = _get(NavigationCfg, 'navigation'),
            renderer   = _get(RendererCfg,   'renderer'),
        )

    def __repr__(self) -> str:
        return f"Config(world={self.world.preset}, dt={self.sim.dt}s)"