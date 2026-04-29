"""
main.py
-------
Simulation entry point.

Phase 1: static world + manual keyboard drive.
Phase 2: live laser scan overlay.
Phase 3: synthetic feature extraction overlay  (ACTIVE).

Controls
--------
  W / up    : forward        A / left  : turn left
  S / down  : backward       D / right : turn right
  Q         : quit           R         : reset robot
  1/2/3     : switch preset  (lab / corridor / open)
  F         : toggle feature overlay on/off
  H         : print help

Run
---
  python main.py
  python main.py --world corridor
"""

from __future__ import annotations
import argparse, sys, os, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config            import Config
from env.world         import World
from env.robot         import Robot
from env.sensor        import Sensor
from slam.features     import FeatureExtractor
from viz.renderer      import Renderer


# ----------------------------------------------------------------- key state

class KeyState:
    def __init__(self):
        self._held: set[str] = set()
        self.quit            = False
        self.reset           = False
        self.preset: str | None = None
        self.toggle_features = False

    def on_press(self, event) -> None:
        k = (event.key or '').lower()
        self._held.add(k)
        if k == 'q':
            self.quit = True
        elif k == 'r':
            self.reset = True
        elif k in ('1', '2', '3'):
            self.preset = {'1': 'lab', '2': 'corridor', '3': 'open'}[k]
        elif k == 'f':
            self.toggle_features = True
        elif k == 'h':
            print(__doc__)

    def on_release(self, event) -> None:
        self._held.discard((event.key or '').lower())

    def compute_command(self, max_v, max_omega):
        v = omega = 0.0
        if 'w' in self._held or 'up'    in self._held: v     += max_v
        if 's' in self._held or 'down'  in self._held: v     -= max_v
        if 'a' in self._held or 'left'  in self._held: omega += max_omega
        if 'd' in self._held or 'right' in self._held: omega -= max_omega
        return v, omega


# ---------------------------------------------------------------- sim factory

def build_sim(cfg: Config, preset: str | None = None):
    p         = preset or cfg.world.preset
    world     = World.from_preset(p)
    robot     = Robot(cfg.robot.start_x, cfg.robot.start_y,
                      cfg.robot.start_theta, cfg.robot.radius)
    sensor    = Sensor.from_cfg(cfg.sensor)
    extractor = FeatureExtractor.from_cfg(cfg)
    renderer  = Renderer(world,
                         figsize=tuple(cfg.renderer.figsize),
                         dpi=cfg.renderer.dpi,
                         title=f'Active SLAM — Phase 3  [{p}]')
    return world, robot, sensor, extractor, renderer


# ----------------------------------------------------------------- main loop

def run(cfg: Config, preset: str | None = None) -> None:
    world, robot, sensor, extractor, renderer = build_sim(cfg, preset)
    renderer.init()

    keys            = KeyState()
    show_features   = True

    renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
    renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)

    plt.ion()
    plt.show(block=False)

    dt         = cfg.sim.dt
    frame_dur  = 1.0 / cfg.sim.render_fps
    last_frame = time.perf_counter()

    last_scan    = sensor.scan(robot, world)
    last_corners, last_lines = extractor.extract(robot.pose, world, last_scan)

    print("[main] Phase 3 running.  F toggles feature overlay.  Q to quit.")

    while plt.fignum_exists(renderer.fig.number):
        t0 = time.perf_counter()
        renderer.fig.canvas.flush_events()

        if keys.quit:
            break

        if keys.toggle_features:
            show_features       = not show_features
            keys.toggle_features = False
            print(f"[main] Feature overlay {'ON' if show_features else 'OFF'}")

        if keys.reset:
            robot.reset(cfg.robot.start_x, cfg.robot.start_y,
                        cfg.robot.start_theta)
            keys.reset = False
            print("[main] Robot reset.")

        if keys.preset is not None:
            renderer.close()
            world, robot, sensor, extractor, renderer = build_sim(cfg, keys.preset)
            renderer.init()
            renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
            renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)
            plt.show(block=False)
            keys.preset = None
            keys._held.clear()
            continue

        # physics + sensing
        v, omega = keys.compute_command(cfg.robot.max_v, cfg.robot.max_omega)
        robot.step(v, omega, dt)
        last_scan             = sensor.scan(robot, world)
        last_corners, last_lines = extractor.extract(robot.pose, world, last_scan)

        # render at target FPS
        now = time.perf_counter()
        if now - last_frame >= frame_dur:
            renderer.update(
                robot,
                laser_scan  = last_scan,
                corner_obs  = last_corners if show_features else [],
                line_obs    = last_lines   if show_features else [],
            )
            plt.pause(0.001)
            last_frame = now

        elapsed = time.perf_counter() - t0
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    renderer.close()
    print("[main] Simulation ended.")


# ---------------------------------------------------------------------- CLI

def parse_args():
    p = argparse.ArgumentParser(description='Active SLAM 2-D Simulator')
    p.add_argument('--world',  choices=['lab', 'corridor', 'open'], default=None)
    p.add_argument('--config', default='config.yaml')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = Config.load(args.config)
    run(cfg, preset=args.world)