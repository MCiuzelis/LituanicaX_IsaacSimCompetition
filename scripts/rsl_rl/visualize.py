"""Visualize MuSHR camera streams from Isaac Sim without running any policy.

The robot is spawned at its start position.  WASD keys drive it manually
(requires pynput — install with: pip install pynput).

A separate display subprocess (spawned BEFORE Isaac Sim starts) owns the
cv2.imshow windows.  Isaac Sim runs its own Qt application internally; creating
a second QApplication in the same process silently blocks imshow.  Running the
display in a child process avoids the conflict entirely.

Two windows are shown:
  "Raw Camera"  — full 1152×648 RGB frame from Isaac Sim
  "Cone Mask"   — full 1152×648 binary mask (processed from camera feed)

Frames are exchanged via PNG files written to /tmp/mushr_debug/.
Press  Ctrl+C  in this terminal  OR  Q  in any display window to stop.

Usage:
    python /home/matasciuzelis/Documents/lituanicaXsim/scripts/rsl_rl/visualize.py --task Mushr
"""

import argparse
import os
import subprocess
import sys
import threading

# Re-exec with conda's libstdc++ in LD_PRELOAD — same rationale as train.py.
if "ISAACLAB_LIBSTDCPP_FIXED" not in os.environ:
    _conda_prefix = os.environ.get("CONDA_PREFIX")
    if _conda_prefix:
        _libstdcpp = os.path.join(_conda_prefix, "lib", "libstdc++.so.6")
        if os.path.exists(_libstdcpp):
            os.environ["ISAACLAB_LIBSTDCPP_FIXED"] = "1"
            os.environ["LD_PRELOAD"] = _libstdcpp
            os.execv(sys.executable, [sys.executable] + sys.argv)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize MuSHR camera streams (no policy).")
parser.add_argument("--task", type=str, default="Mushr", help="Gym task name.")
parser.add_argument(
    "--crop_top", type=float, default=None,
    help="Fraction of camera rows to discard from the top (sky). Defaults to ConeVisionCfg value.",
)
parser.add_argument(
    "--crop_bottom", type=float, default=None,
    help="Fraction of camera rows to discard from the bottom (near ground). Defaults to ConeVisionCfg value.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# ── display subprocess ─────────────────────────────────────────────────────────
# Spawned HERE — before AppLauncher initialises Isaac Sim's Qt application —
# so it gets its own clean QApplication and cv2.imshow works normally.
#
# The subprocess just polls /tmp/mushr_debug/ for PNG files written by the main
# process and calls imshow.  No shared memory or IPC needed.
_CONDA_SP = "/home/matasciuzelis/miniconda3/envs/isaac/lib/python3.11/site-packages"

_DISPLAY_SCRIPT = rf"""
import sys, os

# The conda env inherits PYTHONPATH from Isaac Lab which includes Isaac Sim's
# pip_prebundle (headless cv2).  Strip those entries and prepend the conda
# site-packages so the GUI-capable opencv-python is loaded first.
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

# Wait for the main process to write the first frame.
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

# ── keyboard state ─────────────────────────────────────────────────────────────
try:
    from pynput import keyboard as _pynput_kb
    _HAS_PYNPUT = True
except ImportError:
    _HAS_PYNPUT = False


class _KeyState:
    """Thread-safe WASD keyboard state for manual robot driving."""

    def __init__(self):
        self._down: set = set()
        self._lock = threading.Lock()

    def on_press(self, key):
        try:
            with self._lock:
                self._down.add(key.char.lower())
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            with self._lock:
                self._down.discard(key.char.lower())
        except AttributeError:
            pass

    def get_actions(self) -> tuple[float, float]:
        """Return (throttle, steer) in [-1, 1].

        When no key is pressed throttle is -1.0 (full deceleration) so the car
        decelerates to a stop and stays still.  W overrides to full acceleration.
        """
        with self._lock:
            if "w" in self._down:
                throttle = 1.0
            elif "s" in self._down:
                throttle = -1.0
            else:
                throttle = -1.0   # no key → brake to standstill
            steer = (1.0 if "a" in self._down else 0.0) - (1.0 if "d" in self._down else 0.0)
        return throttle, steer


display_proc = None
if os.environ.get("DISPLAY"):
    # Strip PYTHONPATH so Isaac Lab's prebundle paths don't leak into the
    # display subprocess before the sys.path fix inside the script runs.
    _display_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    display_proc = subprocess.Popen(
        [sys.executable, "-c", _DISPLAY_SCRIPT], env=_display_env
    )
    print(f"[visualize] Display subprocess started (PID {display_proc.pid})")
else:
    print("[visualize] No $DISPLAY — disk output only.")
    print("[visualize] Live view:  feh --reload 0.25 /tmp/mushr_debug/")

# ── launch Isaac Sim ───────────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
import numpy as np
import torch
import gymnasium as gym
import cv2  # headless is fine here — only imwrite/cvtColor needed in main process

import isaaclab_tasks  # noqa: F401
import lituanicaXsim   # noqa: F401

from lituanicaXsim.tasks.mushr_maze.mushr_maze_env import (
    MushrMazeEnvCfg,
    POLICY_IMAGE_WIDTH,
    POLICY_IMAGE_HEIGHT,
)
from lituanicaXsim.tasks.mushr_maze.cone_vision import ConeVisionCfg, ConeVisionProcessor


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert any Isaac camera frame (uint8 or float32) to uint8 RGB [H,W,3]."""
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype == np.uint8:
        return frame
    frame = frame.astype(np.float32)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)



# ── main ──────────────────────────────────────────────────────────────────────

def main():
    env_cfg = MushrMazeEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.show_camera_debug = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Override camera resolution for visualisation.
    # Training uses 640×360 to keep the TiledCamera tile small across many envs.
    # With only 1 env here the full 1024×576 fits in VRAM easily and gives a
    # much better raw-camera view for manual inspection.
    # The pinhole optical parameters (focal_length, horizontal_aperture) are
    # resolution-independent so the 120° FOV is preserved at any resolution.
    env_cfg.camera_cfg.width  = 1024
    env_cfg.camera_cfg.height = 576

    print("[visualize] Creating environment …")
    env = gym.make(args_cli.task, cfg=env_cfg)
    mushr_env = env.unwrapped

    device = mushr_env.device

    cam_w = mushr_env.cfg.camera_cfg.width
    cam_h = mushr_env.cfg.camera_cfg.height

    # Cone mask at EXACT policy resolution (what the policy network receives).
    # Camera stays at high-res for the raw view; detection still runs at 384×216
    # (max(256, min_detect_width=384)) — identical to the training pipeline.
    # crop_top / crop_bottom start from ConeVisionCfg defaults; CLI args override only
    # when explicitly provided (--crop_top X  or  --crop_bottom X).
    vision_cfg = ConeVisionCfg(
        output_width=POLICY_IMAGE_WIDTH,              # 256
        camera_height_width_ratio=cam_h / cam_w,     # e.g. 576/1024 = 0.5625 (16:9)
    )
    if args_cli.crop_top is not None:
        vision_cfg.crop_top_fraction = args_cli.crop_top
    if args_cli.crop_bottom is not None:
        vision_cfg.crop_bottom_fraction = args_cli.crop_bottom
    vision = ConeVisionProcessor(vision_cfg)

    out_dir = "/tmp/mushr_debug"
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[visualize] Camera: {cam_w}×{cam_h}  |  "
        f"Policy mask: {vision.output_width}×{vision.output_height}  |  "
        f"crop_top={vision_cfg.crop_top_fraction:.2f}  crop_bottom={vision_cfg.crop_bottom_fraction:.2f}"
    )
    print(f"[visualize] Frames → {out_dir}/   |   Press Ctrl+C to stop.")

    key_state = _KeyState()
    kb_listener = None
    if _HAS_PYNPUT:
        kb_listener = _pynput_kb.Listener(
            on_press=key_state.on_press, on_release=key_state.on_release
        )
        kb_listener.start()
        print("[visualize] WASD controls active: W=accelerate  S=brake  A=left  D=right  (no key=stop)\n")
    else:
        print("[visualize] pynput not found — robot stays stationary.")
        print("[visualize] Install with:  pip install pynput\n")

    env.reset()

    try:
        while simulation_app.is_running():
            throttle, steer = key_state.get_actions()
            actions = torch.tensor([[throttle, steer]], device=device)
            env.step(actions)

            # Grab raw camera frame
            rgb_tensor = mushr_env.camera.data.output["rgb"]  # [1, H, W, C]
            raw_np = _to_uint8_rgb(rgb_tensor[0].cpu().numpy())

            # Run cone-vision pipeline
            _, near_ratio, coverage, debug_imgs = vision.process_batch(
                rgb_tensor, debug_env_id=0
            )

            near  = float(near_ratio[0].item())
            cov   = float(coverage[0].item())
            label = f"near={near:.3f}  cov={cov:.3f}"

            # Write frames atomically (tmp → rename) so the display subprocess
            # never reads a partially-written PNG (avoids libpng read errors).
            def _write_atomic(path: str, img) -> None:
                base, ext = os.path.splitext(path)
                tmp = base + "_tmp" + ext   # e.g. raw_tmp.png — keeps codec hint
                cv2.imwrite(tmp, img)
                os.replace(tmp, path)

            raw_bgr = cv2.cvtColor(raw_np, cv2.COLOR_RGB2BGR)
            _write_atomic(os.path.join(out_dir, "raw.png"), raw_bgr)

            if debug_imgs is not None:
                mask_bgr = debug_imgs["mask_bgr"]
                # mask_bgr is POLICY_IMAGE_HEIGHT × POLICY_IMAGE_WIDTH (144×256).
                # Use small font so text fits without obscuring the mask.
                cv2.putText(mask_bgr, label, (2, POLICY_IMAGE_HEIGHT - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
                _write_atomic(os.path.join(out_dir, "mask.png"), mask_bgr)

    except KeyboardInterrupt:
        pass

    env.close()

    if kb_listener is not None:
        kb_listener.stop()

    if display_proc is not None and display_proc.poll() is None:
        display_proc.terminate()
        display_proc.wait()


if __name__ == "__main__":
    main()
    simulation_app.close()
