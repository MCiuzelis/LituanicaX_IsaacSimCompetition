# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab reinforcement learning project training a TurtleBot3 Burger differential-drive robot to navigate a cone track using PPO. The environment uses a 2D LiDAR sensor for observations and runs inside NVIDIA Isaac Sim via the Isaac Lab framework.

## Commands

### Training
```bash
# Can be run from any directory — log path is now hardcoded as absolute
python /home/matasciuzelis/Documents/turtlebot_maze_rl/scripts/rsl_rl/train.py \
    --task Isaac-TurtleBot-Maze-Direct-v0

# Override common parameters
python /home/matasciuzelis/Documents/turtlebot_maze_rl/scripts/rsl_rl/train.py \
    --task Isaac-TurtleBot-Maze-Direct-v0 \
    --num_envs 500 \
    --max_iterations 3000 \
    --seed 42 \
    --run_name my_experiment
```

### Inference / Playback
```bash
python /home/matasciuzelis/Documents/turtlebot_maze_rl/scripts/rsl_rl/play.py \
    --task Isaac-TurtleBot-Maze-Direct-v0 \
    --num_envs 1 \
    --checkpoint /home/matasciuzelis/Documents/turtlebot_maze_rl/logs/rsl_rl/turtlebot_maze/<run_timestamp>/model_600.pt
```

### TensorBoard
```bash
tensorboard --logdir /home/matasciuzelis/Documents/turtlebot_maze_rl/logs/rsl_rl/turtlebot_maze/
```

### Package Installation
```bash
pip install -e /home/matasciuzelis/Documents/turtlebot_maze_rl/source/turtlebot_maze_rl
```

## Architecture

```
scripts/rsl_rl/
  train.py          — Entry point: launches Isaac Sim, creates env, runs PPO via RSL-RL
  play.py           — Loads a checkpoint, runs inference, exports to JIT/ONNX
  cli_args.py       — Shared argparse helpers and cfg override utilities

source/turtlebot_maze_rl/turtlebot_maze_rl/
  tasks/turtlebot_maze/
    turtlebot_maze_env.py   — Environment and config (DirectRLEnv subclass)
    agents/
      rsl_rl_ppo_cfg.py     — PPO runner/network/algorithm hyperparameters
```

### Environment (`turtlebot_maze_env.py`)

The environment is a `DirectRLEnv` (low-level Isaac Lab API, no manager hierarchy). Everything is driven by constants at the top of the file; all must be kept consistent:

| Constant | Value | Used in |
|---|---|---|
| `LIDAR_NUM_BEAMS` | 36 | `horizontal_res=360/LIDAR_NUM_BEAMS`, `observation_space=LIDAR_NUM_BEAMS+2` |
| `LIDAR_MAX_RANGE` | 2.5 m | `RayCasterCfg.max_distance`, observation normalization |
| `COLLISION_DIST` | 0.13 m | cone proximity termination in `_get_dones` |
| `WALL_CONTACT_DIST` | 0.10 m | wall proximity termination in `_get_dones` |
| `MAX_LIN_VEL` | 0.4 m/s | action scaling, reward normalization |
| `MAX_ANG_VEL` | 2.0 rad/s | action scaling, reward normalization |
| `WHEEL_RADIUS` | 0.033 m | differential-drive IK in `_pre_physics_step` |
| `HALF_SEPARATION` | 0.080 m | differential-drive IK in `_pre_physics_step` |

**Step pipeline per policy tick (4 physics sub-steps at 120 Hz → ~30 Hz policy):**
1. `_pre_physics_step` — normalised [lin_vel, ang_vel] → wheel velocities (rad/s) via diff-drive IK
2. `_apply_action` — writes joint velocity targets to PhysX
3. Physics simulation (4×)
4. `_get_observations` — 36 LiDAR beam distances (normalised) + body-frame lin/ang vel
5. `_get_rewards` — alive + forward + clearance + smooth + collision + wall + completion + backward + nospin
6. `_get_dones` — terminate if min cone-LiDAR < `COLLISION_DIST` OR min wall-LiDAR < `WALL_CONTACT_DIST`
7. `_reset_idx` — restore default pose + env origin + ±0.2 rad yaw jitter

### Reward Terms

| Term | Formula | Purpose |
|---|---|---|
| `r_alive` | `alive_weight / max_episode_length` per step | Survival incentive |
| `r_forward` | `forward_weight * (v_fwd / MAX_LIN_VEL)` | Encourage forward speed |
| `r_clearance` | `-clearance_weight * exp(-min_ray / 0.08)` | Smooth cone-clearance penalty |
| `r_smooth` | `-smooth_weight * |ang_vel_norm|` | Penalise erratic steering |
| `r_collision` | `-collision_penalty` on any termination | Base death penalty (cone OR wall) |
| `r_wall` | `-wall_penalty` on wall termination only | Additional catastrophic wall penalty |
| `r_completion` | `+completion_weight * (1 - elapsed/max_steps)` on lap | Speed bonus for lap completion |
| `r_backward` | `-backward_weight * clamp(-v_fwd/MAX_LIN_VEL, 0)` | Penalise reversing |
| `r_nospin` | `-nospin_weight * |ang_vel_norm| * (1 - clamp(v_fwd, 0, 1))` | Penalise spinning in place |

**Reward weights (defaults in `TurtleBotMazeEnvCfg`):**

| Parameter | Value | Notes |
|---|---|---|
| `alive_weight` | 1.0 | Normalised over `max_episode_length` |
| `forward_weight` | 2.5 | |
| `clearance_weight` | 0.5 | |
| `smooth_weight` | 0.02 | |
| `collision_penalty` | 5.0 | Fires on cone AND wall termination |
| `wall_penalty` | 200.0 | Fires ONLY on wall termination (total wall cost = 205.0 vs cone cost = 5.0) |
| `completion_weight` | 20.0 | |
| `backward_weight` | 3.0 | |
| `nospin_weight` | 0.5 | |

**`episode_length_s = 900.0`** — Keep this value. At 30 Hz this gives `max_episode_length = 27 000` steps, making `alive_weight/max_episode_length ≈ 0.000037` per step (meaningful). Using large values like 12 000 s makes alive reward negligible (~0.000007/step).

### Wall System (Invisible Boundary)

Walls are loaded from `walls_export.usd` and are **invisible to the policy** — the policy LiDAR only sees cones. Walls cause immediate termination + large penalty (`wall_penalty = 200`).

**Two RayCasters are used:**
- `self.lidar` → targets `/World/TrackMergedMesh` (cones) — included in observations
- `self.wall_lidar` → targets `/World/WallsMergedMesh` (walls) — used only for termination, never observed

**`self._wall_terminated`** is a bool tensor set in `_get_dones` and read in `_get_rewards` to apply `wall_penalty` only on wall hits.

### Multi-Robot Setup

The track USD is loaded once as a **global** prim at `/World/Track` (not inside any env prim). All cone geometry is merged into a single `/World/TrackMergedMesh` prim (no CollisionAPI). Similarly, walls are merged into `/World/WallsMergedMesh`. A physics ground plane is added at `/World/GroundPlane` via `GroundPlaneCfg`. `env_spacing=0.0` keeps every env origin at world (0,0,0) so all robots spawn inside the track. Isaac Lab's per-environment collision groups prevent robot–robot interaction.

**No physics collision is applied to any track or wall mesh.** See "Key Design Decisions" below.

If you increase `num_envs`, no other code changes are needed. If you change `LIDAR_NUM_BEAMS`, the `horizontal_res`, `observation_space`, and the docstring beam count at the top of the file must all be updated accordingly.

### RayCaster Merged Mesh

Because `RayCaster` only supports one mesh prim, `_setup_scene` iterates all `UsdGeom.Mesh` prims under a root prim, transforms their vertices to world space via `UsdGeom.XformCache`, combines them into a single `Vt.Vec3fArray` / face index array, and writes the result to a new `UsdGeom.Mesh` prim. This is done for both the track (`/World/TrackMergedMesh`) and walls (`/World/WallsMergedMesh`). These prims have no `CollisionAPI` — the RayCaster reads USD geometry directly.

### PPO Configuration (`rsl_rl_ppo_cfg.py`)

Actor-critic MLP: `[256, 128, 64]` with ELU activation. Rollout buffer: 64 steps × N envs. Checkpoints saved every 25 iterations.

### Log Location

Logs are always written to **`/home/matasciuzelis/Documents/turtlebot_maze_rl/logs/rsl_rl/turtlebot_maze/`** (absolute path hardcoded in `train.py` line 147). This was changed from a relative path to avoid logs landing in whatever directory training was launched from.

### Training Flow

`train.py` uses Hydra (via `@hydra_task_config`) to load `TurtleBotMazeEnvCfg` and `TurtleBotMazePPORunnerCfg`. CLI args override Hydra defaults after parsing. The gym environment is wrapped with `RslRlVecEnvWrapper` before being passed to `OnPolicyRunner.learn()`.

The task is registered as `Isaac-TurtleBot-Maze-Direct-v0` in `tasks/turtlebot_maze/__init__.py`.

## Asset Paths

All USD paths are absolute and machine-specific — update them if moving to a different machine:

- Robot: `/home/matasciuzelis/Documents/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd`
- Track (cones): `/home/matasciuzelis/Documents/turtlebot_maze_rl/TrackExport.usd`
  - Scale: `(0.0075, 0.0125, 0.0075)` — X/Z = 0.75×0.01 cm→m, Y = 1.25×0.01
  - Orientation: `(0.70711, 0.70711, 0.0, 0.0)` — +90° around X to cancel Y-up baked rotation
  - Placed at world origin `(0, 0, 0)`
  - No `CollisionAPI` on any mesh (see Key Design Decisions)
  - RayCaster mesh: `/World/TrackMergedMesh`
- Walls: `/home/matasciuzelis/Documents/turtlebot_maze_rl/walls_export.usd`
  - **Same** scale and orientation as the track
  - Loaded at `/World/Walls`; **no** `CollisionAPI`
  - Merged into `/World/WallsMergedMesh` for the wall LiDAR
  - Wall meshes are **not** in `TrackMergedMesh` — invisible to the policy
- Robot spawn: `(2.989, 0.9613, 0.05)` — 5 cm above ground so wheels settle cleanly

## Key Design Decisions & Pitfalls

### Why no physics collision on cones OR walls?

**Any form of `CollisionAPI` on these meshes causes robots to jiggle in place and not drive.** Two root causes:

1. **Flat mesh at Z≈0** — `TrackExport.usd` contains a flat base/floor element. Applying `CollisionAPI` creates a second physics surface at Z≈0 competing with `GroundPlaneCfg`.
2. **Closed-loop mesh** — `walls_export.usd` contains a closed-loop track boundary (`/World/Walls/BézierCurve_001`, 10 720 verts, Z=[0.0004, 0.40 m]). `convexHull` on a closed loop = a hull that spans the **entire track interior** down to Z≈0 = second ground plane. A Z-extent filter does NOT help because the mesh is tall (0–0.40 m) and passes the filter.

**Fix:** No `CollisionAPI` on any track or wall mesh. Use dedicated `RayCaster` sensors for proximity-based termination. `activate_contact_sensors=True` on the robot is not needed and must be left off.

### Why two RayCasters?

`RayCaster` raises `NotImplementedError` if given more than one mesh prim path (the track has 4 meshes, walls have 1). The merged mesh approach solves this for each sensor independently. Having two separate sensors lets the policy see cones but not walls.

### Why `mesh_prim_paths` is set in `_setup_scene` not in the config class?

The actual merged prim paths only exist after the USDs are loaded and merged. Config classes hold placeholder paths (`["/World/Track"]`, `["/World/Walls"]`); `_setup_scene` overwrites them with the real paths before constructing the `RayCaster` objects.

### Why `env_spacing=0.0`?

The track and walls are single global prims at world origin. All robots must be inside them. With non-zero spacing, env origins would offset robots outside the track and the RayCaster BVH would return max-distance readings. Isaac Lab's collision groups prevent physical robot–robot interaction.

### Wall penalty rationale

At `wall_penalty = 200.0` and `collision_penalty = 5.0`:
- Cone hit total penalty: **−5.0**
- Wall hit total penalty: **−205.0** (collision_penalty + wall_penalty)
- Typical accumulated forward reward over ~1300 steps ≈ **~3 250**
- Wall death costs ~6% of total episode value — ~40× more than a cone hit

This makes wall contact catastrophic relative to cone contact, training the agent to navigate purely by cone geometry without ever touching the boundary.

### Why `episode_length_s = 900.0` (not larger)?

`alive_weight` is normalised: per-step alive reward = `alive_weight / max_episode_length`. With `episode_length_s = 12 000` → `max_episode_length = 360 000` → per-step alive reward ≈ 0.000003, effectively zero. Restoring to 900 s → 27 000 steps makes the survival incentive meaningful again (~0.000037/step).

### Log path is absolute

`train.py` line 147 was changed from `os.path.join("logs", ...)` (relative to CWD) to an absolute path pointing to the project directory. Previously, logs were silently written to whatever directory training was launched from (e.g., `~/logs/` when launched from home). Always check `/home/matasciuzelis/Documents/turtlebot_maze_rl/logs/` for checkpoints and TensorBoard events.
