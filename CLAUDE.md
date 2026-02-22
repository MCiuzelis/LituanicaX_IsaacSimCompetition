# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab reinforcement learning project training a TurtleBot3 Burger differential-drive robot to navigate a maze using PPO. The environment uses a 2D LiDAR sensor for observations and runs inside NVIDIA Isaac Sim via the Isaac Lab framework.

## Commands

### Training
```bash
# Run from the Isaac Lab installation directory
cd /path/to/IsaacLab
python /home/matasciuzelis/Documents/turtlebot_maze_rl/scripts/rsl_rl/train.py \
    --task Isaac-TurtleBot-Maze-Direct-v0

# Override common parameters
python scripts/rsl_rl/train.py \
    --task Isaac-TurtleBot-Maze-Direct-v0 \
    --num_envs 10 \
    --max_iterations 3000 \
    --seed 42 \
    --run_name my_experiment
```

### Inference / Playback
```bash
python /home/matasciuzelis/Documents/turtlebot_maze_rl/scripts/rsl_rl/play.py \
    --task Isaac-TurtleBot-Maze-Direct-v0 \
    --num_envs 1

# Load a specific checkpoint
python scripts/rsl_rl/play.py \
    --task Isaac-TurtleBot-Maze-Direct-v0 \
    --checkpoint /path/to/model_XXXX.pt
```

### Package Installation
```bash
# Install the extension so Isaac Lab can discover it
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
| `LIDAR_MAX_RANGE` | 3.5 m | `RayCasterCfg.max_distance`, observation normalization |
| `COLLISION_DIST` | 0.12 m | termination condition in `_get_dones` |
| `MAX_LIN_VEL` | 0.22 m/s | action scaling, reward normalization |
| `MAX_ANG_VEL` | 2.84 rad/s | action scaling, reward normalization |
| `WHEEL_RADIUS` | 0.033 m | differential-drive IK in `_pre_physics_step` |
| `HALF_SEPARATION` | 0.080 m | differential-drive IK in `_pre_physics_step` |

**Step pipeline per policy tick (4 physics sub-steps at 120 Hz → ~30 Hz policy):**
1. `_pre_physics_step` — normalised [lin_vel, ang_vel] → wheel velocities (rad/s) via diff-drive IK
2. `_apply_action` — writes joint velocity targets to PhysX
3. Physics simulation (4×)
4. `_get_observations` — 36 LiDAR beam distances (normalised) + body-frame lin/ang vel
5. `_get_rewards` — alive + forward progress + wall clearance + smoothness + collision
6. `_get_dones` — terminate if min LiDAR distance < `COLLISION_DIST`
7. `_reset_idx` — restore default pose + env origin + ±0.2 rad yaw jitter

### Multi-Robot Setup

The track USD is loaded once as a **global** prim at `/World/Track` (not inside any env prim). Isaac Lab's `RayCaster` bakes this mesh into a single Warp BVH shared by all environment sensors. To ensure all robots see the track walls, `env_spacing=0.0` keeps every env origin at world (0,0,0), so all 10 robots spawn at the same position inside the track. Isaac Lab's per-environment collision groups prevent robots from physically interacting.

If you increase `num_envs` beyond 10, no other code changes are needed. If you change `LIDAR_NUM_BEAMS`, the `horizontal_res`, `observation_space`, and the docstring beam count at the top of the file must all be updated accordingly.

### PPO Configuration (`rsl_rl_ppo_cfg.py`)

Actor-critic MLP: `[256, 128, 64]` with ELU activation. Rollout buffer: 64 steps × N envs. With 10 envs that is 640 samples per update, split into 4 mini-batches × 5 epochs. Checkpoints saved every 200 iterations under `logs/rsl_rl/turtlebot_maze/`.

### Training Flow

`train.py` uses Hydra (via `@hydra_task_config`) to load `TurtleBotMazeEnvCfg` and `TurtleBotMazePPORunnerCfg`. CLI args override Hydra defaults after parsing. The gym environment is wrapped with `RslRlVecEnvWrapper` before being passed to `OnPolicyRunner.learn()`.

The task is registered as `Isaac-TurtleBot-Maze-Direct-v0` in `tasks/turtlebot_maze/__init__.py`.

## Asset Paths

Both USD paths are absolute and machine-specific — update them if moving to a different machine:

- Robot: `/home/matasciuzelis/Documents/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd`
- Track: `/home/matasciuzelis/Documents/turtlebot_maze_rl/Track.usd`
  - Scale factor: `0.012` (1.2 user × 0.01 cm→m conversion from STEP import)
  - Mesh collision path: `/World/Track/Track/Shell1/Mesh`
