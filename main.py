"""
main.py
-------
Simulation entry point -- Phase 1: static world + manual keyboard drive.

Controls
--------
  W / ↑   : drive forward
  S / ↓   : drive backward
  A / ←   : turn left  (counter-clockwise)
  D / →   : turn right (clockwise)
  Q        : quit
  R        : reset robot to start pose
  1/2/3    : switch world preset  (lab / corridor / open)
  H        : print this help

The window must have focus for key-presses to register.

Run
---
  python main.py
  python main.py --world corridor
  python main.py --world open
"""

from __future__ import annotations
import argparse
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# -- local imports -------------------------------------------------------------
# Allow running from the repo root without installing as a package
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config      import Config
from env.world   import World
from env.robot   import Robot
from viz.renderer import Renderer


# -----------------------------------------------------------------------------
# Keyboard state
# -----------------------------------------------------------------------------

class KeyState:
    """Tracks which keys are currently held down."""

    def __init__(self):
        self._held: set[str] = set()
        self.quit   = False
        self.reset  = False
        self.preset: str | None = None

    def on_press(self, event) -> None:
        k = event.key
        if k is None:
            return
        k = k.lower()
        self._held.add(k)

        if k == 'q':
            self.quit = True
        elif k == 'r':
            self.reset = True
        elif k == '1':
            self.preset = 'lab'
        elif k == '2':
            self.preset = 'corridor'
        elif k == '3':
            self.preset = 'open'
        elif k == 'h':
            _print_help()

    def on_release(self, event) -> None:
        k = (event.key or '').lower()
        self._held.discard(k)

    def compute_command(self, max_v: float, max_omega: float
                        ) -> tuple[float, float]:
        """
        Return (v, omega) from currently-held keys.

        Forward  = W or up
        Backward = S or down
        Left     = A or left
        Right    = D or right
        """
        v = omega = 0.0
        if 'w' in self._held or 'up' in self._held:
            v += max_v
        if 's' in self._held or 'down' in self._held:
            v -= max_v
        if 'a' in self._held or 'left' in self._held:
            omega += max_omega
        if 'd' in self._held or 'right' in self._held:
            omega -= max_omega
        return v, omega


def _print_help():
    print(__doc__)


# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------

def build_sim(cfg: Config, preset: str | None = None):
    """Instantiate world, robot, renderer from config."""
    p = preset or cfg.world.preset
    world    = World.from_preset(p)
    robot    = Robot(cfg.robot.start_x, cfg.robot.start_y,
                     cfg.robot.start_theta, cfg.robot.radius)
    renderer = Renderer(world,
                        figsize=tuple(cfg.renderer.figsize),
                        dpi=cfg.renderer.dpi,
                        title=f'Active SLAM -- Phase 1  [{p}]')
    return world, robot, renderer


def run(cfg: Config, preset: str | None = None) -> None:
    world, robot, renderer = build_sim(cfg, preset)
    renderer.init()

    keys = KeyState()
    renderer.fig.canvas.mpl_connect('key_press_event',   keys.on_press)
    renderer.fig.canvas.mpl_connect('key_release_event', keys.on_release)

    plt.ion()
    plt.show(block=False)

    dt         = cfg.sim.dt
    frame_dur  = 1.0 / cfg.sim.render_fps
    last_frame = time.perf_counter()

    print("[main] Phase 1 running.  Press H in terminal for help, "
          "Q in window to quit.")

    while plt.fignum_exists(renderer.fig.number):
        t0 = time.perf_counter()

        # -- keyboard events --------------------------------------------------
        renderer.fig.canvas.flush_events()

        if keys.quit:
            break

        if keys.reset:
            robot.reset(cfg.robot.start_x, cfg.robot.start_y,
                        cfg.robot.start_theta)
            keys.reset = False
            print("[main] Robot reset.")

        if keys.preset is not None:
            renderer.close()
            world, robot, renderer = build_sim(cfg, keys.preset)
            renderer.init()
            renderer.fig.canvas.mpl_connect('key_press_event',
                                            keys.on_press)
            renderer.fig.canvas.mpl_connect('key_release_event',
                                            keys.on_release)
            plt.show(block=False)
            keys.preset = None
            keys._held.clear()
            continue

        # -- physics step -----------------------------------------------------
        v, omega = keys.compute_command(cfg.robot.max_v, cfg.robot.max_omega)
        robot.step(v, omega, dt)

        # -- render -----------------------------------------------------------
        now = time.perf_counter()
        if now - last_frame >= frame_dur:
            renderer.update(robot)
            plt.pause(0.001)
            last_frame = now

        # -- pace the loop ----------------------------------------------------
        elapsed = time.perf_counter() - t0
        sleep   = dt - elapsed
        if sleep > 0:
            time.sleep(sleep)

    renderer.close()
    print("[main] Simulation ended.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Active SLAM 2-D Simulator')
    p.add_argument('--world', choices=['lab', 'corridor', 'open'],
                   default=None,
                   help='World preset (overrides config.yaml)')
    p.add_argument('--config', default='config.yaml',
                   help='Path to config YAML file')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = Config.load(args.config)
    run(cfg, preset=args.world)