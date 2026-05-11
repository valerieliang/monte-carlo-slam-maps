# Monte Carlo Active SLAM Simulator — Code Overview

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Mathematical and Theoretical Foundations](#2-mathematical-and-theoretical-foundations)
   - 2.1 Robot Kinematics
   - 2.2 Sensor Model
   - 2.3 Feature Extraction
   - 2.4 EKF-SLAM: State & Covariance
   - 2.5 EKF-SLAM: Predict Step
   - 2.6 EKF-SLAM: Update Step
   - 2.7 Data Association
   - 2.8 Monte Carlo Uncertainty Mapping
   - 2.9 Goal Selection & Navigation
3. [File-by-File Reference](#3-file-by-file-reference)
4. [The Main Pipeline](#4-the-main-pipeline)
5. [Configuration Guide](#5-configuration-guide)
6. [Running Experiments & What to Look For](#6-running-experiments--what-to-look-for)

---

## 1. Project Summary

This simulator implements **Active EKF-SLAM with Monte Carlo uncertainty-driven exploration** in a 2-D environment. The robot:

1. Drives through a walled environment (manual or autonomous).
2. Casts a laser scanner and extracts geometric features (corners and wall lines).
3. Maintains an **Extended Kalman Filter (EKF)** map of those features, tracking both their estimated positions and their uncertainty.
4. Periodically samples random candidate **Monte Carlo points** in the environment, scores them by proximity to uncertain/newly-discovered features, and selects the most informative unvisited location as the next navigation goal.
5. A unicycle controller with waypoint routing drives the robot to that goal, building the map as it goes.

The implementation follows the architecture described in the paper it is based on (Instituto de Automática, Argentina), particularly Equations 1–11.

---

## 2. Mathematical and Theoretical Foundations

### 2.1 Robot Kinematics — `env/robot.py`

The robot is modelled as a **unicycle** (non-holonomic), meaning it can only move in the direction it is facing and rotate in place. The continuous-time kinematic equations are:

```
ẋ     = v · cos(θ)
ẏ     = v · sin(θ)
θ̇     = ω
```

Integrated over one timestep `dt` (Euler method, matching paper Eq. 3):

```
x(t+dt) = x(t) + v · cos(θ) · dt
y(t+dt) = y(t) + v · sin(θ) · dt
θ(t+dt) = θ(t) + ω · dt          (wrapped to (-π, π])
```

The state is the robot **pose** `(x, y, θ)`. `v` is linear velocity (m/s) and `ω` is angular velocity (rad/s).

---

### 2.2 Sensor Model — `env/sensor.py`

The sensor is a simulated **2-D SICK laser scanner**. It casts `N` rays uniformly spaced across a symmetric field of view (default 180°). For each ray:

1. A ray-segment intersection test finds the closest wall hit (in `World.ray_intersect()`).
2. Independent **Gaussian noise** is added to both the range and bearing readings:

```
r_noisy    = r_true    + ε_r ,     ε_r  ~ N(0, σ_r²)
β_noisy    = β_true    + ε_β ,     ε_β  ~ N(0, σ_β²)
```

Default values: `σ_r = 0.03 m`, `σ_β = 0.01 rad`.

The sensor also provides **noiseless expected measurement functions** used by the EKF:

**Corner observation model (Eq. 4):**
```
r    = sqrt((x_c - x_v)² + (y_c - y_v)²)
β    = atan2(y_c - y_v, x_c - x_v) - θ_v
```

**Line observation model (Eq. 5):**
```
ρ_obs   = ρ - x_v · cos(α) - y_v · sin(α)
α_obs   = α - θ_v
```
where `(ρ, α)` is the global polar-form representation of the wall line (perpendicular distance from origin and normal angle).

---

### 2.3 Feature Extraction — `slam/features.py`

Raw laser scan points are processed to extract two geometric feature types:

**Corner features** — Points where two wall segments meet at an angle, detected by looking for abrupt depth discontinuities in adjacent scan rays. Each corner is expressed as a Cartesian `(x_c, y_c)` position.

**Line features** — Straight wall segments extracted by fitting lines to clusters of scan points (e.g., iterative end-point fit or split-and-merge). Each line is stored in **polar form** `(ρ, α)` where `ρ` is the perpendicular distance from the world origin to the line, and `α` is the angle of the line's outward normal. This representation is preferred over slope-intercept form because it handles vertical lines and is well-suited to the EKF's Gaussian noise model.

---

### 2.4 EKF-SLAM: State and Covariance — `slam/state.py`

The EKF-SLAM maintains a joint probability distribution over the robot pose and all map features simultaneously. The **full state vector** is (Eq. 1):

```
x̂ = [x_v, y_v, θ_v | x_c1, y_c1, x_c2, y_c2, ..., ρ_l1, α_l1, ...]
      └─── vehicle pose ───┘ └───── corner features ─────┘ └── line features ──┘
```

The **full covariance matrix** partitions as (Eq. 2):

```
P = | P_vv   P_vm |
    | P_mv   P_mm |
```

- `P_vv` (3×3): vehicle pose uncertainty
- `P_mm` (2k×2k): map feature uncertainty (k features, each 2D)
- `P_vm` / `P_mv` (3×2k): cross-correlations between vehicle and map — this is the critical component that makes SLAM different from independent localisation and mapping; the robot's pose uncertainty and the feature position uncertainties are **correlated**, not independent

Every feature occupies exactly 2 elements in the state vector regardless of type (both corner `(x, y)` and line `(ρ, α)` are 2D), making the state layout uniform.

The state dimension grows as new features are discovered. A map with `n` features has a state vector of length `3 + 2n` and a covariance matrix of size `(3+2n) × (3+2n)`.

---

### 2.5 EKF-SLAM: Predict Step — `slam/predict.py`

At each timestep, before processing any sensor data, the filter **predicts** how the state evolves under the motion command `(v, ω)`. Only the vehicle pose changes during prediction; features remain stationary.

**Mean propagation:**
```
x̂_v(t+dt) = f(x̂_v(t), v, ω, dt)   [unicycle kinematics, Section 2.1]
```

**Covariance propagation** uses the **Jacobian** of the motion model linearised at the current estimate:

```
F_v = ∂f/∂x_v = | 1  0  -v·sin(θ)·dt   |
                 | 0  1   v·cos(θ)·dt  |
                 | 0  0   1            |
```

**Process noise** is injected via the velocity input Jacobian `G` (3×2):

```
G = | cos(θ)·dt   0  |
    | sin(θ)·dt   0  |
    | 0           dt |
```

Process noise in velocity space: `Q = diag(Q_v, Q_ω)`

Additive process noise in state space: `Q_state = G · Q · Gᵀ`

**Full covariance prediction** exploits the block structure (avoids building the full n×n Jacobian):

```
P_vv_new = F_v · P_vv · F_vᵀ + Q_state
P_vm_new = F_v · P_vm          (cross-correlations rotated by vehicle motion)
P_mm_new = P_mm                (features unaffected by prediction)
```

This is the correct EKF prediction for SLAM: the vehicle-map cross-covariance `P_vm` gets updated because the robot's new uncertain pose propagates uncertainty into the relationship between the robot and all mapped features.

---

### 2.6 EKF-SLAM: Update Step — `slam/update.py`

When a feature observation `z` is associated to an existing map feature, the EKF performs a **Kalman update** using the linearised measurement model.

For a matched feature `i`:

**Step 1 — Expected measurement and Jacobian:**
```
z̃_i = h(x̂, feature_i)      [Eq. 4 or 5, evaluated at current estimate]
H_i  = ∂h/∂x̂               [Jacobian of measurement model, full state]
```

The Jacobian `H_i` is a `(2 × state_dim)` matrix. It is sparse — only the 3 vehicle pose columns and 2 feature columns are non-zero.

**Step 2 — Innovation:**
```
ν = z - z̃_i
```
Angular components of the innovation are wrapped to `(-π, π]` to handle the circular nature of angles.

**Step 3 — Innovation covariance:**
```
S = H · P · Hᵀ + R
```
where `R` is the 2×2 measurement noise covariance: `diag(σ_r², σ_β²)` for corners, `diag(σ_ρ², σ_α²)` for lines.

**Step 4 — Kalman gain:**
```
K = P · Hᵀ · S⁻¹
```

**Step 5 — State update:**
```
x̂ ← x̂ + K · ν
```

**Step 6 — Covariance update (Joseph form for numerical stability):**
```
P ← (I - K·H) · P · (I - K·H)ᵀ + K · R · Kᵀ
```

The Joseph form is numerically more stable than the standard `P ← (I - K·H)·P` because it guarantees symmetry and positive semi-definiteness even with floating-point rounding. After every update, `P` is explicitly symmetrised: `P ← 0.5(P + Pᵀ)`.

**Feature initialisation:** When a new feature is observed (no match found in data association), it is **augmented** into the state by inverting the measurement model:

For a new corner from observation `z = [r, β]`:
```
x_c = x_v + r · cos(θ_v + β)
y_c = y_v + r · sin(θ_v + β)
```

The initial covariance of the new feature is computed via error propagation through the inverse model Jacobians `J_v` (∂g/∂x_v) and `J_z` (∂g/∂z):
```
P_new_feature = J_v · P_vv · J_vᵀ + J_z · R · J_zᵀ + init_cov_scale · I
```

The cross-covariance between the new feature and the existing state is:
```
P_{existing, new} = P_{existing, v} · J_vᵀ
```

This correctly captures that the new feature's position is uncertain in proportion to the robot's own pose uncertainty at the moment of first observation.

---

### 2.7 Data Association — `slam/data_assoc.py`

Before applying an update, each observation must be matched to an existing map feature (or flagged as new). The algorithm is **nearest-neighbour gating** using the **Mahalanobis distance**:

```
d²_i = νᵢᵀ · S_i⁻¹ · νᵢ
```

This distance is **scale-normalised** — it measures the innovation in units of standard deviations of the innovation covariance `S_i`. Unlike Euclidean distance, it accounts for the varying uncertainty of each feature.

A match is accepted if `d² < gate_chi2`. The threshold `gate_chi2 = 9.21` is the 99th percentile of the chi-squared distribution with 2 degrees of freedom, meaning a correct association is accepted with 99% probability. If no feature falls within the gate, the observation is treated as a new feature.

---

### 2.8 Monte Carlo Uncertainty Mapping — `montecarlo/`

After the EKF map is built, the simulation must decide **where the robot should go next**. This is the "active" part of Active SLAM. The approach is a Monte Carlo method in three stages:

**Stage 1 — Sampling (Eqs. 6–7, `sampler.py`):**

Uniformly sample `N` candidate points inside the current map bounding box:
```
m_i = x_min + (x_max - x_min) · μ_i
n_i = y_min + (y_max - y_min) · μ_i     where μ_i ~ U(0, 1)
```

**Stage 2 — Navigability filter (`navigability.py`):**

For each candidate point, cast a ray from the robot. If the ray hits a wall before reaching the point, the point is unreachable (inside a wall or occluded) and is discarded. Only navigable points proceed.

**Stage 3 — Uncertainty scoring (`uncertainty_map.py`):**

Each navigable point is scored by how close it is to **recently discovered or uncertain features**. The scoring model:

```
score(p) = max_k [ w_k · exp(-dist(p, anchor_k)² / (2σ²)) ]
```

where:
- `w_k = 1 / (1 + obs_count_k)` — freshly observed features (obs=1) score high (`w=0.5`), well-mapped features (obs=20) fade toward zero (`w≈0.05`)
- The spatial kernel uses a **fixed width `σ`** (default 2.5 m), not the EKF covariance, so even well-localised walls still generate a spatial influence zone for navigation
- Taking the `max` rather than a `sum` prevents dense feature clusters from double-counting

Scores are normalised to `[0, 1]`. The "uncertain band" `[uncertainty_lo, uncertainty_hi]` (default `[0.40, 0.60]`) contains the points most worthy of exploration.

**Virtual boundary lines (`virtual_features.py`):** Four virtual line features are placed at the edges of the current map bounding box to score frontier points near unexplored regions. They use a fixed covariance `diag(0.32, 0.32)` as specified in the paper.

---

### 2.9 Goal Selection and Navigation — `navigation/`

**Goal selection (`selector.py`):** Two-phase strategy (Section IV of paper):

- **Phase 1 — Local search:** within a `local_area_size × local_area_size` box around the robot, pick the uncertain point maximising `P(p) / dist(robot, p)` (Eq. 11 — information gain per unit travel cost).
- **Phase 2 — Global fallback:** if no local uncertain points exist, pick the closest stored point from the **transitory array** (points from previous MC runs that weren't chosen at the time).
- **Completion:** if neither phase yields a candidate, the map is declared complete and the robot returns to its start position.

**Controller (`controller.py`):** A unicycle proportional controller with automatic waypoint routing. If the direct path to the goal is blocked by a wall, a visibility-graph style search inserts intermediate waypoints around wall segment endpoints. A 9-ray proximity fan applies repulsive steering forces to avoid collisions, and a wall-following escape manoeuvre is triggered if the robot gets stuck for too long.

**SLAM world (`slam_world.py`):** A lightweight collision-query object built from the EKF map's current line and corner features. The controller queries both the ground-truth world and this SLAM-derived world simultaneously (`_CombinedWorld` in `main.py`), so the robot respects walls it has explicitly mapped even if the ground-truth geometry would otherwise permit passage.

---

## 3. File-by-File Reference

### Environment — `env/`

| File | Purpose |
|------|---------|
| `env/robot.py` | Robot state `(x, y, θ)`, unicycle kinematics (`step()`), pose history trail. The single source of truth for ground-truth robot position. |
| `env/sensor.py`| 2-D laser scanner simulation. Casts rays via `World.ray_intersect()`, adds Gaussian noise to range and bearing. Also provides noiseless `expected_corner()` and `expected_line()` functions used by the EKF update. |
| `env/world.py` | Ground-truth environment geometry. Holds `Segment` (wall) and `Corner` objects. Provides `ray_intersect()` for both the sensor and the controller. Contains three preset environments: `lab`, `corridor`, `open`. The robot never has direct access to this data — it only receives sensor observations derived from it. |

### SLAM — `slam/`

| File | Purpose |
|------|---------|
| `slam/state.py` | `SLAMState` class: owns the full EKF state vector `x̂` and covariance matrix `P`. Handles state augmentation when new features are added (`add_feature()`). Also holds `Feature` metadata objects (kind, segment endpoints, observation count). |
| `slam/predict.py` | EKF prediction step. Propagates pose mean via unicycle kinematics and covariance via linearised Jacobian. Only touches the vehicle block and vehicle-map cross-correlations; feature-feature block is unchanged. |
| `slam/update.py` | EKF update step (one feature at a time). Computes Jacobians `H`, innovation `ν`, innovation covariance `S`, Kalman gain `K`, and applies the Joseph-form covariance update. Also contains `init_corner()` and `init_line()` for augmenting the state with newly observed features. |
| `slam/data_assoc.py` | Mahalanobis-distance nearest-neighbour gating. Associates each incoming observation to the closest existing feature within the chi-squared gate, or flags it as new. |
| `slam/features.py` | Feature extraction from raw `ScanResult`. Detects corners (depth discontinuities) and wall lines (point clustering / line fitting) in the laser scan. Produces `CornerObs` and `LineObs` objects with pre-computed `z` measurement vectors. |

### Monte Carlo — `montecarlo/`

| File | Purpose |
|------|---------|
| `montecarlo/sampler.py` | Draws uniform random `(x, y)` candidate points within the map bounding box (Eqs. 6–7). |
| `montecarlo/navigability.py` | Ray-cast navigability filter. Discards candidate points that are unreachable from the robot's current position. |
| `montecarlo/probability.py` | Sum-of-Gaussians scoring (Eqs. 8–10). Per-point probability under the current feature map, with both scalar and vectorised implementations. |
| `montecarlo/uncertainty_map.py` | Master pipeline. Calls `build_virtual_features()`, `sample_points()`, `navigable_mask()`, and `_score_uncertainty()`. Returns an `UncertaintyMap` object with points, scores, and the uncertain band mask. |
| `montecarlo/virtual_features.py` | Constructs four virtual boundary line features at the edges of the current map extent. These ensure frontier points score near 0.5 even before real features have been mapped there. |

### Navigation — `navigation/`

| File | Purpose |
|------|---------|
| `navigation/selector.py` | Two-phase goal selector (local → global → complete). Maintains a `TransitoryArray` of unvisited uncertain points for global fallback. |
| `navigation/controller.py` | Unicycle P-controller with visibility-graph waypoint routing and proximity-repulsion steering. State machine: `PURSUE → FOLLOW → PURSUE` to escape stuck situations. |
| `navigation/slam_world.py` | Builds a `ray_intersect()`-compatible object from the current EKF map's line and corner features, so the controller can avoid SLAM-mapped walls. |

### Visualisation — `viz/`

| File | Purpose |
|------|---------|
| `viz/renderer.py` | Core matplotlib renderer. Draws the world walls, robot body, laser scan fan, extracted feature observations, robot trail, and legend. Updates at the target FPS. |
| `viz/covariance.py` | Draws 2-D covariance ellipses (2σ contours) for EKF features using eigendecomposition. Used to visualise feature uncertainty on the SLAM map overlay. |
| `viz/heatmap.py` | Scatter-plot heatmap of MC uncertainty scores. Uses a custom dark-blue→amber→dark-red colormap where amber (score ≈ 0.5) is the navigation target colour. Highlights the uncertain band with larger markers. |
| `viz/logger.py` | Records the full robot trajectory and detects wall crossings by casting rays between consecutive poses. Can generate a `path.csv` and print a summary report. |

### Top-Level

| File | Purpose |
|------|---------|
| `main.py` | Simulation entry point. Wires all components together into the main loop. Handles keyboard/mouse input, preset switching, toggle overlays, CLI arguments. |
| `config.py` | Typed dataclass tree loaded from `config.yaml`. All modules import from here. |
| `config.yaml` | All tunable parameters in one place. See Section 5. |
| `requirements.txt` | `numpy`, `matplotlib`, `scipy`, `pyyaml`, `pytest`. |

---

## 4. The Main Pipeline

The `run()` function in `main.py` implements a fixed-rate control loop at `dt = 0.05 s` (20 Hz). Every iteration:

```
┌─────────────────────────────────────────────────────────────┐
│  1. CONTROLLER STEP                                         │
│     Auto:   controller.step(slam_pose, combined_world, dt)  │
│     Manual: keyboard WASD → (v, ω)                          │
│                                                             │
│  2. PHYSICS                                                 │
│     robot.step(v, ω, dt)     ← ground-truth motion          │
│     path_log.record(...)     ← trajectory logging           │
│                                                             │
│  3. EKF PREDICT                                             │
│     predict(slam_state, v, ω, dt, Q_v, Q_w)                 │
│     Updates x̂_v and P using linearised unicycle Jacobian    │
│                                                             │
│  4. SENSE + EXTRACT                                         │
│     scan   = sensor.scan(robot, world)                      │
│     corners, lines = extractor.extract(robot.pose, scan)    │
│                                                             │
│  5. DATA ASSOCIATION + EKF UPDATE                           │
│     For each observation:                                   │
│       if matched → update_single(state, feat_idx, z, R)     │
│       if new     → init_corner / init_line                  │
│                                                             │
│  6. MC UNCERTAINTY MAP  (every ~1 simulated second)         │
│     umap = build_uncertainty_map(slam_state, world, ...)    │
│                                                             │
│  7. GOAL SELECTION (auto mode only)                         │
│     result = selector.select(umap, robot.pos)               │
│     controller.set_goal(result.goal, world=slam_world)      │
│                                                             │
│  8. RENDER  (at target FPS, decoupled from control rate)    │
│     renderer.update(robot, scan, corners, lines)            │
│     heatmap.update(umap)    (if visible)                    │
│     _draw_slam(ax, slam_state)  (if visible)                │
└─────────────────────────────────────────────────────────────┘
```

**Key design choices:**

- **Predict-then-sense:** The EKF prediction runs before the sensor scan each cycle, which is the correct causal ordering — the robot moves, then observes.
- **SLAM-world separation:** The controller uses `_CombinedWorld` (ground-truth + SLAM) for proximity steering, but uses `SLAMWorld` alone for re-routing after a wall-follow escape. This simulates the real scenario where the robot only knows its own map, not the true environment.
- **MC rate:** The uncertainty map runs once per simulated second (`mc_every = round(1.0 / dt) = 20 steps`), not every frame. This is intentional — the map changes slowly and MC sampling is relatively expensive.
- **SLAM overlay efficiency:** The SLAM matplotlib artists are only recreated when the map actually changes (new feature added or observation count increases), avoiding per-frame object churn.

---

## 5. Configuration Guide

All parameters live in `config.yaml`. Edit and rerun — no code changes needed.

### World Preset

```yaml
world:
  preset: lab       # 'lab' | 'corridor' | 'open'
```

| Preset | Description | Good for |
|--------|-------------|----------|
| `lab` | 12×8 m room with interior L-shaped partial wall and alcove | Default; tests both corner and line feature extraction |
| `corridor` | 20×3 m corridor with one perpendicular branch | Tests navigation through narrow spaces and branch detection |
| `open` | Bare 10×10 m rectangle | Minimal features; tests pure corner detection at room corners |

Switch at runtime with keys **1**, **2**, **3** without restarting.

### Robot

```yaml
robot:
  start_x: 1.0       # Starting position
  start_y: 1.0
  start_theta: 0.0   # Heading in radians (0 = facing right)
  max_v: 1.0         # Max linear speed (m/s) — increase for faster exploration
  max_omega: 1.0     # Max turn rate (rad/s)
```

Moving `start_x/y` to different room corners tests whether SLAM can re-localise after starting with no map. Higher `max_v` reduces exploration time but may cause the controller to overshoot goals or miss narrow doorways.

### Sensor

```yaml
sensor:
  fov_deg: 180.0       # Field of view — reduce to test limited-FOV effects
  num_rays: 181        # Ray count — reduce for faster simulation
  max_range: 20.0      # Laser range — reduce to force more exploration steps
  noise_range: 0.03    # Range noise std dev (m)
  noise_bearing: 0.01  # Bearing noise std dev (rad)
```

Increasing `noise_range` or `noise_bearing` makes the EKF work harder and covariance ellipses will be larger. Setting `max_range` lower than the room diagonal forces the robot to get closer to walls before features are observed.

### EKF

```yaml
ekf:
  Q_v: 0.02     # Motion noise: translational — increase if robot slips
  Q_w: 0.05     # Motion noise: rotational — increase if robot spins unexpectedly
  R_range: 0.03
  R_bearing: 0.01
  gate_chi2: 9.21   # Association gate — reduce to 5.99 (95%) to be stricter
  init_cov_corner: 0.5   # Initial feature uncertainty — higher = more cautious
  init_cov_line: 0.5
```

**Q vs R balance is the most important EKF tuning knob:**
- `Q >> R`: the filter trusts sensors more than motion. Features converge quickly but the pose may jump.
- `Q << R`: the filter trusts motion more. Pose is smooth but new features may not be incorporated quickly enough.
- `gate_chi2`: lowering this rejects more observations as "new features" rather than matching existing ones, which can cause map fragmentation (same wall appearing twice). Raising it risks false matches.

### Monte Carlo

```yaml
montecarlo:
  n_samples_local: 150    # Points sampled for uncertainty scoring
  n_samples_global: 800   # Points for global fallback search
  local_area_size: 5.0    # Half-width of local search square (m)
  uncertainty_lo: 0.40    # Lower bound of "uncertain" band
  uncertainty_hi: 0.60    # Upper bound of "uncertain" band
```

Increasing `n_samples_local` gives a smoother heatmap but runs slower. The `uncertainty_lo/hi` band defines what counts as "worth exploring" — widening it (e.g., `0.30–0.70`) sends the robot to more places; narrowing it (e.g., `0.45–0.55`) makes it pickier.

### Navigation

```yaml
navigation:
  goal_tolerance: 0.30    # Distance to declare goal reached (m)
  controller_k_v: 0.4     # Linear speed proportional gain
  controller_k_w: 1.2     # Turning rate proportional gain
```

If the robot oscillates around goals, reduce `k_w`. If it's slow to reach goals, increase `k_v` (but don't exceed `max_v` in effect).

---

## 6. Running Experiments & What to Look For

### How to Run

```bash
# Manual drive (keyboard)
python main.py

# Autonomous exploration, lab environment
python main.py --auto

# Autonomous, different world
python main.py --auto --world corridor
python main.py --auto --world open

# Custom config
python main.py --auto --config my_experiment.yaml
```

**Keyboard controls (all modes):**

| Key | Action |
|-----|--------|
| W/A/S/D or arrow keys | Manual drive (forward/turn left/back/turn right) |
| F | Toggle raw feature observation overlay |
| M | Toggle SLAM map overlay (EKF features + covariance ellipses) |
| U | Toggle MC uncertainty heatmap |
| 1 / 2 / 3 | Switch to lab / corridor / open preset |
| R | Reset robot and wipe map |
| P | Print path report to console, save `path.csv` |
| H | Print help |
| Q | Quit |
| Left-click | Set manual goal (auto mode: interrupts autonomous navigation) |

---

### What You Should See in Each Mode

#### Manual Drive (no `--auto`)

**Laser scan fan:** Yellow/cream rays fanning out from the robot. Each ray should terminate at a wall or reach max range. The fan should smoothly follow the robot's heading as you turn.

**Extracted features (F key, green/pink):**
- Green dots: corner observations — should appear at wall junctions, doorway edges, and interior wall endpoints
- Pink ticks: line observations — should appear along flat wall stretches

These are **raw per-frame observations** and will flicker slightly due to sensor noise. That is normal.

**SLAM map (M key, red/teal):**
- Red `×` markers with ellipses: EKF corner estimates. The ellipses should be large on first observation, then shrink as the corner is revisited.
- Teal dashed lines with `◆` markers and ellipses: EKF line feature estimates.

Watch for: ellipses **shrinking over time** as features are re-observed. If they stay large or grow, the filter may be diverging.

**MC heatmap (U key):**
- Dark blue = free space (well-mapped, low uncertainty)
- Amber = uncertain frontier (good place to explore)
- Dark red = near a feature (occupied/boundary)

In manual mode the heatmap is static unless you press U to refresh. Try driving away from a wall and watch amber points appear in the direction you just came from.

---

#### Autonomous Mode (`--auto`)

**Normal, healthy exploration** looks like:

1. **Early phase:** robot turns in place or makes short drives to observe nearby walls. The SLAM map builds quickly with a few large-ellipse features. The heatmap is mostly amber (little is mapped yet).

2. **Mid phase:** robot makes purposeful drives to amber heatmap regions. Each time it arrives at a goal (yellow ring), new features are added along the newly-observed wall. Covariance ellipses on older features shrink. The heatmap becomes more blue in already-visited areas.

3. **Late phase:** the robot makes increasingly long excursions to reach the last amber regions (frontiers). The console prints `[auto] Goal (global):` when it is using stored transitory points rather than fresh local ones — this is expected and correct.

4. **Completion:** `[auto] Map complete - returning to start.` The robot drives back to its starting position and the autonomous loop ends. The SLAM map should cover the full room with small covariance ellipses on most features.

---

### Red Flags to Watch For

**Wall crossings (path logger):**  
Press **P** at any time to check. Zero wall crossings is the target. Occasional crossings near narrow doorways or tight corners are a controller/clearance bug. Frequent crossings indicate a navigation failure. `path.csv` records the full trajectory; wall-crossing steps are flagged with `wall_crossing=1`.

**Covariance ellipses not shrinking:**  
If a feature's ellipse stays large or grows over many observations, data association is probably failing (the feature keeps being re-initialised as "new" rather than matched). Try increasing `gate_chi2` or checking that the initial covariance scale is not so large it overlaps with unrelated features.

**Duplicate features (two ellipses on the same wall):**  
This indicates a data association false negative — the same physical feature was initialised twice. Lower `gate_chi2` slightly or reduce `init_cov_corner`/`init_cov_line`. In the `corridor` preset this is most likely near the branch junction.

**Robot spinning in place or oscillating near a goal:**  
The controller's angular gain `k_w` is too high or the `goal_tolerance` is too tight. Increase `goal_tolerance` from 0.30 to 0.50, or reduce `k_w`.

**Robot not exploring a region (stuck in one area):**  
The MC sampler may not be placing enough points in that region, or the navigability filter is over-pruning. Try increasing `n_samples_local` or checking that the region is actually reachable (not behind a wall). In the `lab` preset, the alcove interior can be a navigability edge case.

**Heatmap all blue immediately:**  
This means uncertainty scores are near zero everywhere, which happens when all features already have high `obs_count`. This is correct if the robot has genuinely mapped the space, but if it happens too early, the `uncertainty_lo` threshold may be too high. Lower it from 0.40 to 0.25.

**Heatmap all amber:**  
The scoring sigma (`spatial_sigma = 2.5 m` in `uncertainty_map.py`) may be too large for a small room, or too few features have been mapped yet. In the `open` preset with a small number of features, this is expected early in exploration.

**Console prints `[auto] Map complete` very early:**  
The `selector` ran out of uncertain points and transitory points too soon. Reduce `uncertainty_lo` to capture more candidate points, or increase `n_samples_local` so more uncertain points are found per MC run.

---

### Quantitative Checks After a Full Exploration Run

After pressing **P** (or at the end of the run), the console shows:

```
[PathLogger] 847 poses  |  42.3 m total distance  |  0 wall crossing(s)
```

A healthy run on `lab` with `--auto` typically covers 30–60 m with 0 wall crossings. If distance is unusually high (>100 m), the robot may be circling without converging. If it is very low (<15 m), the map was declared complete prematurely.

The number of EKF features at the end of a `lab` exploration should be in the range of **12–20 features** (7 wall segments → 7 line features + ~8 corners in the preset geometry, plus some duplicates from noisy observations). More than 30 features usually indicates excessive re-initialisation; fewer than 8 suggests the sensor range or extraction is too conservative.