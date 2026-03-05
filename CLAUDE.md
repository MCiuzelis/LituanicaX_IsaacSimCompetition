# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab reinforcement learning project training a **MuSHR nano v2** Ackermann/RC-car robot to navigate a cone track using PPO. The environment uses a front camera + OpenCV cone mask for observations and runs inside NVIDIA Isaac Sim via the Isaac Lab framework.

**Registered gym task name: `Mushr`** (defined in `tasks/mushr_maze/__init__.py`)

## Commands

### Training
```bash
# Can be run from any directory — log path is hardcoded as absolute
python /home/matasciuzelis/Documents/lituanicaXsim/scripts/rsl_rl/train.py \
    --task Mushr

# Override common parameters
python /home/matasciuzelis/Documents/lituanicaXsim/scripts/rsl_rl/train.py \
    --task Mushr \
    --num_envs 500 \
    --max_iterations 3000 \
    --seed 42 \
    --run_name my_experiment
```

### Inference / Playback
```bash
python /home/matasciuzelis/Documents/lituanicaXsim/scripts/rsl_rl/play.py \
    --task Mushr \
    --num_envs 1 \
    --checkpoint /home/matasciuzelis/Documents/lituanicaXsim/logs/rsl_rl/mushr_maze/<run_timestamp>/model_600.pt
```

### Camera Visualisation (no policy needed)
```bash
python /home/matasciuzelis/Documents/lituanicaXsim/scripts/rsl_rl/visualize.py --task Mushr
# Optional: change upscale factor for cone-mask windows (default 5 → 480×270)
python .../visualize.py --task Mushr --mask_scale 8
```
Spawns 1 env, robot stationary at start. Opens three cv2 windows:
- **Raw Camera** — full 1152×648 RGB from Isaac Sim
- **Cone Overlay** — 96×54 upscaled; detected cones highlighted cyan, near-field line red
- **Cone Mask** — 96×54 upscaled binary mask (what the policy network receives)
Press `Q` to quit.

### TensorBoard
```bash
tensorboard --logdir /home/matasciuzelis/Documents/lituanicaXsim/logs/rsl_rl/mushr_maze/
```

### Package Installation
```bash
pip install -e /home/matasciuzelis/Documents/lituanicaXsim/source/lituanicaXsim
```

## Architecture

```
scripts/rsl_rl/
  train.py          — Entry point: launches Isaac Sim, creates env, runs PPO via RSL-RL
  play.py           — Loads a checkpoint, runs inference, exports to JIT/ONNX
  visualize.py      — No-policy camera viewer: raw RGB + OpenCV cone mask (cv2 windows)
  cli_args.py       — Shared argparse helpers and cfg override utilities

source/lituanicaXsim/lituanicaXsim/
  tasks/mushr_maze/
    mushr_maze_env.py       — Environment and config (DirectRLEnv subclass)
    agents/
      rsl_rl_ppo_cfg.py     — PPO runner/network/algorithm hyperparameters

assets/
  mushr_nano_v2.usd — MuSHR nano v2 robot (RC car)
  CONES.usd         — Cone track (policy LiDAR target)
  WALLS.usd         — Wall boundary (termination only, invisible to policy)
  archive/          — Old/unused assets (TrackExport.usd, walls_export.usd, WheeledLab-main/, etc.)
```

### Environment (`mushr_maze_env.py`)

The environment is a `DirectRLEnv` (low-level Isaac Lab API, no manager hierarchy). Everything is driven by constants at the top of the file; all must be kept consistent:

| Constant | Value | Used in |
|---|---|---|
| `MAX_LIN_VEL` | 2.0 m/s | action scaling, reward normalization |
| `MAX_STEER` | 0.488 rad | action scaling (≈28°, from WheeledLab) |
| `MAX_ANG_VEL` | 4.0 rad/s | observation normalization for yaw |
| `WHEEL_RADIUS` | 0.05 m | Ackermann IK in `_pre_physics_step` |
| `BASE_LENGTH` | 0.325 m | wheelbase (front–rear axle distance) |
| `BASE_WIDTH` | 0.2 m | track width (left–right wheel distance) |
| `CAMERA_WIDTH` | 1152 px | Pi Cam 3 Wide half-res (native 2304) |
| `CAMERA_HEIGHT` | 648 px | Pi Cam 3 Wide half-res (native 1296) |
| `CAMERA_HFOV_DEG` | 94° | Pi Camera Module 3 Wide actual HFOV |
| `CONE_COLLISION_MASK_RATIO` | 0.30 | full-frame cone coverage termination |
| `WALL_CONTACT_FORCE_THRESH` | 15.0 N | horizontal force for wall termination |
| `action_lpf_alpha_throttle` | 0.5 | EMA alpha for throttle (τ≈47 ms at 30 Hz) |
| `action_lpf_alpha_steer` | 0.25 | EMA alpha for steering (τ≈108 ms at 30 Hz) |

Robot USD scale is **1.0** (1:1 real-world scale).

**Step pipeline per policy tick (4 physics sub-steps at 120 Hz → ~30 Hz policy):**
1. `_pre_physics_step` — EMA low-pass filter on actions → Ackermann IK → rear wheel ω + front steer tan position
2. `_apply_action` — writes velocity targets (rear) + position targets (front) to PhysX
3. Physics simulation (4×)
4. `_get_observations` — flattened 96×54 cone mask + body-frame lin/ang vel
5. `_get_rewards` — alive + forward + clearance + smooth + collision + wall + completion + backward + nospin
6. `_get_dones` — terminate if cone mask coverage > `CONE_COLLISION_MASK_RATIO` OR wall contact force > `WALL_CONTACT_FORCE_THRESH`
7. `_reset_idx` — restore default pose + env origin + yaw jitter + clear LPF state

### Ackermann Action Model (RWD)

The MuSHR nano v2 is rear-wheel drive with Ackermann steering:

| Policy output | Scaling | Physical command |
|---|---|---|
| `action[0]` ∈ [-1,1] | × MAX_LIN_VEL | forward speed m/s |
| `action[1]` ∈ [-1,1] | × MAX_STEER | steering angle rad |

Conversion in `_pre_physics_step`:
- `rear_wheel_ω = lin_vel / WHEEL_RADIUS` (rad/s, same for both rear wheels)
- `steer_joint_pos = tan(steer_angle)` (tan convention matching mushr_nano_v2.usd joint definition)

Front throttle joints are passive (stiffness=damping=0), suspension joints are spring-like passive (stiffness=1e8).

### Reward Terms

| Term | Formula | Purpose |
|---|---|---|
| `r_alive` | `alive_weight / max_episode_length` per step | Survival incentive |
| `r_forward` | `forward_weight * (v_fwd / MAX_LIN_VEL)` | Encourage forward speed |
| `r_smooth` | `-smooth_weight * |ang_vel_norm|` | Penalise erratic steering |
| `r_backward` | `-backward_weight * clamp(-v_fwd/MAX_LIN_VEL, 0, 1)` | Penalise sustained reversing |
| `r_slip` | `-slip_weight * |rear_wheel_vel − v_fwd| / MAX_LIN_VEL` | Penalise rear-wheel traction loss |
| `r_collision` | `-collision_penalty` on any termination | Base death penalty (cone OR wall OR flip) |
| `r_wall` | `-wall_penalty` on wall termination only | Additional catastrophic wall penalty |
| `r_completion` | `+completion_weight * (1 - elapsed/max_steps)` on lap | Speed bonus for lap completion |
| `r_nospin` | `-nospin_weight * |ang_vel_norm| * (1 - clamp(v_fwd, 0, 1))` | Penalise spinning in place |

**Disabled (weight=0):** `r_clearance` (no proximity penalty).

**Reward weights (defaults in `MushrMazeEnvCfg`):**

| Parameter | Value |
|---|---|
| `alive_weight` | 1.0 |
| `forward_weight` | 2.5 |
| `backward_weight` | 2.0 |
| `smooth_weight` | 0.025 |
| `slip_weight` | 1.0 |
| `collision_penalty` | 100.0 |
| `wall_penalty` | 100.0 |
| `completion_weight` | 8000.0 |
| `nospin_weight` | 0.2 |

**`episode_length_s = 900.0`** — Keep this value. At 30 Hz this gives `max_episode_length = 27 000` steps, making `alive_weight/max_episode_length ≈ 0.000037` per step (meaningful). Using large values like 12 000 s makes alive reward negligible.

### Wall System (Invisible Boundary)

Walls are loaded from `assets/WALLS.usd` and are **invisible to the policy** — the policy camera only sees cones. Walls cause immediate termination + large penalty.

**Termination detection:**
- **Cones:** camera-based — terminate if full-frame cone mask coverage > `CONE_COLLISION_MASK_RATIO` (0.30)
- **Walls:** contact-force-based — terminate if horizontal force on any robot link > `WALL_CONTACT_FORCE_THRESH` (15 N)
- **Flip/rollover:** orientation-based — terminate if robot tilt > 72° from vertical (`up_z < 0.3`)

Five `ContactSensor` instances monitor: `base_link`, all four wheel links.

### Multi-Robot Setup

Same as before: track USD is loaded once at `/World/Track`, walls at `/World/Walls`, all at world origin. `env_spacing=0.0` keeps every env origin at (0,0,0). Isaac Lab's per-environment collision groups prevent robot–robot interaction.

### PPO Configuration (`rsl_rl_ppo_cfg.py`)

Actor-critic MLP: `[256, 128, 64]` with ELU activation. Rollout buffer: 64 steps × N envs. Checkpoints saved every 25 iterations. Class: `MushrMazePPORunnerCfg`.

### Log Location

Logs are always written to **`/home/matasciuzelis/Documents/lituanicaXsim/logs/rsl_rl/mushr_maze/`** (absolute path hardcoded in `train.py` line 147, driven by `experiment_name = "mushr_maze"`).

## Asset Paths

All USD paths are absolute and machine-specific — update if moving to a different machine:

- **Robot:** `/home/matasciuzelis/Documents/lituanicaXsim/assets/mushr_nano_v2.usd`
- **Track (cones):** `/home/matasciuzelis/Documents/lituanicaXsim/assets/CONES.usd`
  - Scale: `(0.0075, 0.0125, 0.0075)` — same CAD export convention as previous TrackExport.usd
  - Orientation: `(0.70711, 0.70711, 0.0, 0.0)` — +90° around X to cancel Y-up baked rotation
  - Translation: `(0, 5, 0)` — global offset
  - **TUNE:** verify scale/orientation on first run; adjust if cones appear at wrong size/angle
- **Walls:** `/home/matasciuzelis/Documents/lituanicaXsim/assets/WALLS.usd`
  - Same scale, orientation, and translation as track
  - **TUNE:** same as cones
- **Robot spawn:** `(2.989, 5.9613, 0.05)` — **TUNE:** adjust if car spawns outside the track

## Key Design Decisions & Pitfalls

### Why no physics collision on cones OR walls?

Same as before — any `CollisionAPI` on these meshes causes robots to jiggle in place:
1. Flat base mesh at Z≈0 in `CONES.usd` fights `GroundPlaneCfg`
2. Closed-loop wall mesh convex hull spans entire track interior

**Fix:** No `CollisionAPI` on any cone mesh. Walls have `CollisionAPI` so that `ContactSensor` can detect hits; cones are purely visual + detected by camera mask coverage.

### Camera mounted at base_link

`camera_cfg.prim_path` ends at `…/base_link/front_camera`. Offset `(0.06, 0.0, 0.04)` m places the camera ≈6 cm forward and 4 cm up from the link origin at 1:1 scale. If the view looks wrong, inspect USD link offsets and adjust `OffsetCfg.pos`.

### Ackermann steering — tan joint convention

`mushr_nano_v2.usd` uses `tan(steering_angle)` as the steering joint position (not the angle directly). Do NOT change to raw angle without also verifying the joint definition in the USD.

### Why `env_spacing=0.0`?

Track and walls are single global prims at world origin. All robots must be inside them. Non-zero spacing would offset robots outside the track.

### Wall penalty rationale

- Cone hit total: **−100**
- Wall hit total: **−200** (collision_penalty + wall_penalty)

This makes wall contact 2× more expensive than cone contact, training the agent to stay within the boundary.

### Log path is absolute

`train.py` line 147 uses an absolute path. Logs always appear in `/home/matasciuzelis/Documents/lituanicaXsim/logs/rsl_rl/mushr_maze/`.
