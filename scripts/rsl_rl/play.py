# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import subprocess
import sys

# Re-exec with conda's libstdc++ in LD_PRELOAD so the dynamic linker picks up
# CXXABI_1.3.15 before any C extensions load.  See train.py for rationale.
if "ISAACLAB_LIBSTDCPP_FIXED" not in os.environ:
    _conda_prefix = os.environ.get("CONDA_PREFIX")
    if _conda_prefix:
        _libstdcpp = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(_libstdcpp):
            os.environ["ISAACLAB_LIBSTDCPP_FIXED"] = "1"
            os.environ["LD_PRELOAD"] = _libstdcpp
            os.execv(sys.executable, [sys.executable] + sys.argv)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--fpv",
    action="store_true",
    default=False,
    help="First-person view: force 1 env and open live Raw Camera + Cone Mask windows.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video or for FPV mode
if args_cli.video or args_cli.fpv:
    args_cli.enable_cameras = True
if args_cli.fpv:
    args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

# ── FPV display subprocess ────────────────────────────────────────────────────
# Identical approach to visualize.py: a separate Python process with the
# GUI-capable opencv-python loaded from the conda site-packages polls two PNG
# files written atomically by the main process and shows them via cv2.imshow.
_CONDA_SP = "/home/matasciuzelis/miniconda3/envs/isaac/lib/python3.11/site-packages"
_FPV_DISPLAY_SCRIPT = rf"""
import sys, os
sys.path = [
    "{_CONDA_SP}",
    *[p for p in sys.path if "pip_prebundle" not in p and "omni" not in p],
]
import cv2, time, signal
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
OUT_DIR = "/tmp/mushr_debug"
FILES = [
    (os.path.join(OUT_DIR, "raw.png"),  "Raw Camera"),
    (os.path.join(OUT_DIR, "mask.png"), "Cone Mask"),
]
while not os.path.exists(os.path.join(OUT_DIR, "raw.png")):
    time.sleep(0.05)
while True:
    for fpath, name in FILES:
        img = cv2.imread(fpath)
        if img is not None:
            cv2.imshow(name, img)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
"""

import cv2
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import lituanicaXsim  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments (absolute path, matches train.py)
    log_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Hide wall geometry in play mode — walls are only used during training to
    # penalise boundary contact; they serve no purpose during inference.
    import omni.usd
    from pxr import UsdGeom
    _stage = omni.usd.get_context().get_stage()
    for _wall_path in ("/World/Walls", "/World/WallsMergedMesh"):
        _prim = _stage.GetPrimAtPath(_wall_path)
        if _prim.IsValid():
            UsdGeom.Imageable(_prim).MakeInvisible()

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ── FPV setup ─────────────────────────────────────────────────────────────
    _fpv_proc = None
    _fpv_out_dir = "/tmp/mushr_debug"
    if args_cli.fpv:
        os.makedirs(_fpv_out_dir, exist_ok=True)
        _mushr_env = env.unwrapped          # DirectRLEnv (MushrMazeEnv)
        _vision    = _mushr_env._vision_processor
        if os.environ.get("DISPLAY"):
            _display_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            _fpv_proc = subprocess.Popen(
                [sys.executable, "-c", _FPV_DISPLAY_SCRIPT], env=_display_env
            )
            print(f"[FPV] Display subprocess started (PID {_fpv_proc.pid})")
        else:
            print("[FPV] No $DISPLAY — writing frames to /tmp/mushr_debug/ (use feh --reload 0.25)")

    def _write_atomic(path: str, img: np.ndarray) -> None:
        base, ext = os.path.splitext(path)
        tmp = base + "_tmp" + ext
        cv2.imwrite(tmp, img)
        os.replace(tmp, path)

    def _fpv_step() -> None:
        """Grab camera frame, run cone-vision pipeline, write PNGs for display."""
        rgb = _mushr_env.camera.data.output["rgb"]          # [1, H, W, C]
        raw_np = rgb[0].cpu().numpy()
        if raw_np.dtype != np.uint8:
            raw_np = raw_np.astype(np.float32)
            if raw_np.max() <= 1.0:
                raw_np = raw_np * 255.0
            raw_np = np.clip(raw_np, 0, 255).astype(np.uint8)
        if raw_np.shape[-1] > 3:
            raw_np = raw_np[..., :3]

        raw_bgr = cv2.cvtColor(raw_np, cv2.COLOR_RGB2BGR)
        _write_atomic(os.path.join(_fpv_out_dir, "raw.png"), raw_bgr)

        _, _, _, debug_imgs = _vision.process_batch(rgb, debug_env_id=0)
        if debug_imgs is not None:
            _write_atomic(os.path.join(_fpv_out_dir, "mask.png"), debug_imgs["mask_bgr"])
    # ──────────────────────────────────────────────────────────────────────────

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.fpv:
            _fpv_step()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    if args_cli.fpv and _fpv_proc is not None and _fpv_proc.poll() is None:
        _fpv_proc.terminate()
        _fpv_proc.wait()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
