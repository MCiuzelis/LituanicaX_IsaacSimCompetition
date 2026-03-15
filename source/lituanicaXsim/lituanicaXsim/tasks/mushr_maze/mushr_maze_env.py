"""MuSHR nano v2 RC car — cone-track navigation environment (DirectRLEnv).

Observation space:
    [0:N]   flattened 2-channel OpenCV cone mask (left + right) from front camera
    [N]     normalized forward linear velocity  (v / vel_cap) — vel_cap grows with curriculum

Camera profile (Pi Camera Module 3 Wide — VRAM-optimised training resolution):
    - Sensor: Sony IMX708, 1/2.43" format (6.45 mm × 3.63 mm at 2304×1296 mode)
    - Horizontal FOV: 120° (Pi Camera Module 3 Wide product listing)
    - Listed resolution: 1536 × 864
    - Training resolution: 640 × 360 (16:9, ~2.4× downscale from listed)
      Rationale: TiledCamera tiles all env cameras into a single render texture.
      At 32 envs × 1024×576 the tile is 6144×3456 (21 MP) → OOM on 8 GB VRAM.
      At 640×360 the tile is 3840×2160 (8 MP) — fits comfortably.
      After 35% sky crop: 360×0.65=234 rows → downscales to 384×216 for detection.
      No quality loss for the policy; the mask stays at 256×144.
    - Sensor update rate: 28 FPS (synced with ~30 Hz policy)

Action space (2):
    [0]  throttle command in [-1, 1]  → remapped to [0, 1] inside the env:
         +1 → full acceleration at MAX_ACCEL m/s²
          0 → neutral (no velocity change)
         -1 → full deceleration at MAX_DECEL m/s²
         The env integrates a commanded velocity state clamped to [0, vel_cap].
         No reverse driving is possible; no wheel-lockup because rate is bounded.
    [1]  steering command in [-1, 1]  → scaled to [-MAX_STEER, +MAX_STEER] rad

Actuation (Ackermann / RWD):
    - Rear wheels receive a velocity target (rad/s) = commanded_vel / WHEEL_RADIUS
    - Front wheels receive a position target (rad) = tan(steering_angle)
      (tan-convention matches the joint representation in mushr_nano_v2.usd)
    - Front throttle joints are passive (stiffness=damping=0) — pure RWD
    - Suspension joints remain passive (not actuator-driven)

Reward:
    +distance_weight * v_fwd * dt                             (per step, rewards physical distance covered — sums to total odometry)
    +forward_weight * (v_fwd / INITIAL_VEL_CAP)               (fixed normalisation — stable scale as curriculum advances)
    +accel_reward_weight * max(accel_m_s2 / MAX_ACCEL, 0)     (per step, small bonus for forward acceleration)
    +height_reward_weight * clamp(z / ramp_top_z, 0, 1)       (per step, higher z → more reward)
    -slip_weight * |rear_wheel_vel − v_fwd| / INITIAL_VEL_CAP (traction loss penalty, same fixed normalisation)
    -wall_penalty       on wall termination only               (penalty for wall contact)
    +sector_gain_weight * improvement  when crossing a gate faster than global best  (sector timing reward)

Termination conditions:
    - Wall contact: any horizontal normal force on robot links above WALL_CONTACT_FORCE_THRESH
    - Robot tilt > 72° from vertical (up_z < 0.3) — catches flips and rollovers
    - Slow driving: speed < 40 % of vel_cap for every step; continuous per-step penalty
      and termination after 6 s of uninterrupted slow driving (timer resets the moment
      speed rises above the threshold).
    (Reverse driving is impossible — velocity is clamped [0, vel_cap] at all times.)

Cone system:
    CONES.usd is loaded as a visual-only prim — no CollisionAPI, no contact sensors.
    Cones are purely for the simulated camera; the robot drives through them freely.
    The policy learns to stay within the cone track because leaving it leads to walls.

Wall system:
    WALLS.usd has CollisionAPI enabled and is invisible to the policy camera.
    Any contact with a wall immediately terminates the episode.

TUNING NOTES (first run):
    - If the car spawns outside the track, adjust `pos` in robot_cfg.init_state
    - If track scale looks wrong, adjust `scale=(...)` in _setup_scene for CONES.usd/WALLS.usd
    - Run scripts/rsl_rl/visualize.py to inspect the live camera feed and cone mask
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import csv
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
try:
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
except Exception:  # pragma: no cover - compatibility across Isaac Lab versions
    from isaaclab.sensors import Camera as TiledCamera
    from isaaclab.sensors import CameraCfg as TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .cone_vision import ConeVisionCfg, ConeVisionProcessor


#NOTES: PER LETAI PRADEDA PRACIOJ PRADZIOJ, SUSTOJA, NUOLAT SWERVINA, NERA RACING LINES

# ---------------------------------------------------------------------------
# Asset USD paths  (relative to this file → project root → assets/)
# ---------------------------------------------------------------------------
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../assets")
)

MUSHR_USD         = os.path.join(_ASSETS_DIR, "mushr_nano_v2.usd")
TRACK_USD         = os.path.join(_ASSETS_DIR, "CONES.usd")
WALL_USD          = os.path.join(_ASSETS_DIR, "WALLS.usd")
# Simplified wall layout used during the initial training phase (before RAMP_ENABLE_LAPS laps).
WALL_INITIAL_USD  = os.path.join(_ASSETS_DIR, "WALLS_INITIAL.usd")
RAMP_USD          = os.path.join(_ASSETS_DIR, "RAMPS.usd")

# Number of total laps (across all envs) required before the ramp is enabled.
RAMP_ENABLE_LAPS = 50

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (Ackermann / AWD)
# Joint names / geometry verified against mushr_nano_v2.usd (see assets/mushr_nano_v2.usd)
# ---------------------------------------------------------------------------
WHEEL_RADIUS    = 0.05     # m
BASE_LENGTH     = 0.325    # m  (wheelbase front–rear axle distance)
BASE_WIDTH      = 0.2      # m  (track width between left/right wheels)

# ---------------------------------------------------------------------------
# Curriculum velocity schedule
# vel_cap starts at INITIAL_VEL_CAP and advances toward VEL_CAP_MAX as the
# rolling mean episode return exceeds VEL_CAP_REWARD_THRESHOLD.
# ---------------------------------------------------------------------------
INITIAL_VEL_CAP = 2.5    # m/s — starting velocity ceiling
VEL_CAP_MAX     = 2.5    # m/s — absolute maximum
VEL_CAP_STEP    = 0.05  # m/s — increment per curriculum advancement

# ---------------------------------------------------------------------------
# Acceleration limits — AWD, full vehicle weight on all drive wheels.
# Net ground friction (rubber wheel × basketball court, multiply combine):
#   μ_s = 0.8, μ_d = 0.7,  g = 9.81 m/s²
# AWD: all 4 wheels driven/braked → full normal force available (no weight-fraction penalty).
#   MAX_ACCEL: μ_s × g = 0.8 × 9.81 ≈ 7.85 → 7.8 m/s²
#   MAX_DECEL: μ_d × g = 0.7 × 9.81 ≈ 6.87 → 6.9 m/s²
# At 30 Hz (dt≈0.033 s): max Δv per step ≈ 0.26 m/s (accel) / 0.23 m/s (decel).
# ---------------------------------------------------------------------------
MAX_ACCEL = 7.8   # m/s² — max forward acceleration (AWD, full traction)
MAX_DECEL = 6.9   # m/s² — max deceleration (AWD, all wheels braking)

MAX_STEER    = 0.488   # rad  (~28°) max steering angle

# Normalization divisor for IMU angular velocity channels in the observation.
IMU_ANG_VEL_NORM = 10.0   # rad/s — generous ceiling for fast cornering / ramp transitions

# Camera capture resolution for RL training.
# 1152×648 is half the native Pi Camera Module 3 resolution (2304×1296).
# Same 16:9 aspect ratio and same optical model; uses 4× less VRAM than full res.
# The OpenCV pipeline downscales to 96×54 anyway for the policy input.
CAMERA_WIDTH = 640     # VRAM-optimised training resolution (see docstring above)
CAMERA_HEIGHT = 360    # 640×360 → 3840×2160 tiled texture at 32 envs (fits 8 GB)
CAMERA_FPS = 28.0           # synced with ~30 Hz policy
CAMERA_HFOV_DEG = 120.0    # Pi Camera Module 3 Wide product listing: 120° horizontal FOV

# Pi Camera Module 3 Wide pinhole parameters (IMX708 sensor, 16:9 aspect).
# Physical FOV is fixed at 120° regardless of capture resolution.
# Aperture and focal length below define the projection; vertical aperture is
# derived from horizontal by the 16:9 pixel aspect ratio (same at any resolution).
CAMERA_HORIZONTAL_APERTURE_MM = 6.45   # IMX708 sensor width (unchanged)
CAMERA_VERTICAL_APERTURE_MM = CAMERA_HORIZONTAL_APERTURE_MM * CAMERA_HEIGHT / CAMERA_WIDTH
CAMERA_FOCAL_LENGTH_MM = CAMERA_HORIZONTAL_APERTURE_MM / (
    2.0 * math.tan(math.radians(CAMERA_HFOV_DEG * 0.5))
)

# Policy image derived from OpenCV cone mask (resized from camera stream).
# 2-channel (left/right cones) at 256×144; obs_space = 2×256×144 + 2 = 73730.
# NOTE: first MLP layer is obs_space→256 = ~19 M weights; consider widening
#       the actor-critic hidden layers if training speed is a concern.
POLICY_IMAGE_WIDTH = 256
POLICY_IMAGE_HEIGHT = 144

# Wall contact force threshold (primary detection via ContactSensor in _apply_action).
# Sensors are filtered to /World/Walls only, so ground, ramp, and all other surfaces
# are excluded — false positives are not a concern regardless of threshold value.
# Low threshold is critical: a slow/glancing hit generates small horizontal force per
# sub-step (force ≈ penetration_depth × PhysX_stiffness); with a high threshold the
# robot "eases into" the wall for many sub-steps before detection triggers (~100 ms).
# 0.05 N catches any genuine first-touch while staying above PhysX numerical noise.
WALL_CONTACT_FORCE_THRESH = 0.0   # N

# Backup distance threshold: if the robot drifts more than this from the nearest


# Ramp contact detection uses a lower horizontal-force threshold so shallow ramp
# normals are still picked up while flat-ground contacts remain filtered out.

SLOW_SPEED_FRACTION = 0.3

# Robot links monitored for wall contact.
WALL_CONTACT_LINKS = (
    "base_link",
    "front_left_wheel_link",
    "front_right_wheel_link",
    "back_left_wheel_link",
    "back_right_wheel_link",
)

# ---------------------------------------------------------------------------
# Sector gate system — centerline exported from Blender/curve_points.csv
# ---------------------------------------------------------------------------
# N_SECTORS is NOT hardcoded — it is read dynamically from the CSV in __init__.

# Transform from CSV Blender-space XY → Isaac world XY.
#
# Blender and Isaac Sim use different coordinate conventions; the mapping is:
#   X_isaac = Y_blender * -0.3 + 16.0
#   Y_isaac = X_blender *  0.3
# (axes are swapped, Y is negated, and a scale of 0.3 + 16 m X-offset is applied)
SECTOR_CSV_TO_ISAAC_SCALE = 0.3
SECTOR_CSV_TO_ISAAC_X_OFFSET = 16.0


# ===========================================================================
# Environment Configuration
# ===========================================================================

@configclass
class MushrMazeEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # Core env parameters
    # ------------------------------------------------------------------
    decimation: int = 4                # policy at ~30 Hz (sim 120 Hz)
    # 1200 s × 30 Hz = 36 000 steps — long enough for many laps even at low speed.
    episode_length_s: float = 1200.0

    action_space:      int = 2
    # observation_space is patched in MushrMazeEnv.__init__ using ConeVisionProcessor.
    # The placeholder below keeps @configclass happy; the real value is set before
    # super().__init__() allocates observation buffers.
    observation_space: int = 3 * POLICY_IMAGE_WIDTH * POLICY_IMAGE_HEIGHT + 1  # 3 vision channels + 1 vel
    state_space:       int = 0

    # Crop fractions are defined once in ConeVisionCfg (cone_vision.py) and used
    # directly — no duplicate fields here.

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    # render_interval=decimation: render once per policy step (30 Hz).
    # No need to render on every physics sub-step — the camera only
    # needs one frame per policy observation.
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    # env_spacing=0.0 keeps all env origins at (0,0,0) so every robot spawns
    # inside the single global Track mesh.  Isaac Lab isolates each environment
    # via collision groups so robots do not physically interact with each other.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=12,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Robot articulation — MuSHR nano v2
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=MUSHR_USD,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10000.0,
                max_angular_velocity=100000.0,
                max_depenetration_velocity=1.0,   # low — prevents launching from penetration correction
                max_contact_impulse=5.0,           # non-zero so ContactSensor forces register; caps Δv to 2.5 m/s
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # TUNE: adjust pos after first run if car spawns outside the track
            pos=(15.15, 5, 0.02),
            joint_pos={
                "back_left_wheel_throttle":    0.0,
                "back_right_wheel_throttle":   0.0,
                "front_left_wheel_throttle":   0.0,
                "front_right_wheel_throttle":  0.0,
                "front_left_wheel_steer":      0.0,
                "front_right_wheel_steer":     0.0,
                "front_left_wheel_suspension": 0.0,
                "front_right_wheel_suspension":0.0,
                "back_left_wheel_suspension":  0.0,
                "back_right_wheel_suspension": 0.0,
            },
        ),
        actuators={
            # Front steering — position-controlled servo
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
                stiffness=100.0,
                damping=10.0,
                velocity_limit_sim=10.0,
                effort_limit_sim=3.2,
            ),
            # Rear drive wheels — velocity-controlled (AWD)
            # velocity_limit_sim / effort_limit from WheeledLab MUSHR_SUS_2WD_CFG reference.
            "rear_throttle": ImplicitActuatorCfg(
                joint_names_expr=["back_left_wheel_throttle", "back_right_wheel_throttle"],
                stiffness=0.0,
                damping=1000.0,
                velocity_limit_sim=450.0,
                effort_limit_sim=0.5,
            ),
            # Front drive wheels — velocity-controlled (AWD, mechanically coupled to rear)
            "front_throttle": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_throttle", "front_right_wheel_throttle"],
                stiffness=0.0,
                damping=1000.0,
                velocity_limit_sim=450.0,
                effort_limit_sim=0.5,
            ),
        },
    )

    # ------------------------------------------------------------------
    # Front-centered policy camera (Pi Cam 3 profile target)
    # ------------------------------------------------------------------
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/mushr_nano/base_link/front_camera",
        update_period=0.0,  # update every render call (once per policy step at render_interval=4)
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH_MM,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE_MM,
            vertical_aperture=CAMERA_VERTICAL_APERTURE_MM,
            clipping_range=(0.01, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1),
            # Same camera-frame convention used in WheeledLab MuSHR configs.
            rot=(-0.5, 0.5, -0.5, 0.5),
            convention="ros",
        ),
        debug_vis=False,
    )

    policy_image_width: int = POLICY_IMAGE_WIDTH
    policy_image_height: int = POLICY_IMAGE_HEIGHT

    # Live camera debug — use scripts/rsl_rl/visualize.py instead.
    show_camera_debug: bool = False
    camera_debug_env_id: int = 0
    camera_debug_interval: int = 2

    # ------------------------------------------------------------------
    # Reward weights
    # ------------------------------------------------------------------
    # Distance reward: rewards physical metres covered each step (v * dt), naturally
    # summing to total odometry over the episode.  Works correctly on a circular track
    # because it integrates incremental progress rather than straight-line displacement.
    distance_weight:    float = 3   # reward per metre of forward travel
    forward_weight:     float = 1.5

    # Extra one-time penalty specifically for wall contact (on top of collision_penalty).
    wall_penalty:       float = 100

    # Sector timing reward.
    # No reward is given until ALL sectors have been crossed at least once, building a
    # reference lap (best time per sector seen so far).  Once the reference is set,
    # any subsequent crossing that beats the reference time earns:
    #   reward = sector_gain_weight * (ref_time - sector_time) / ref_time
    sector_gain_weight: float = 10.0

    # Rear-wheel slip penalty: discourages wheel-locking hard braking.
    # slip_speed = |mean_rear_wheel_tangential_vel − body_forward_vel| / MAX_LIN_VEL
    slip_weight:        float = 4.0

    # Steering magnitude penalty using a fifth-root curve.
    # reward = steer_shape_weight * (|steer_angle| / MAX_STEER)^(1/5)
    # The fifth root is concave — it strongly penalises even small steering angles
    # while the penalty saturates gently at large angles.
    # Set negative to penalise; set to 0.0 to disable.
    steer_shape_weight: float = 4

    # One-time bonus awarded the moment an agent completes a full lap.
    lap_completion_reward: float = 500.0

    # Ramp reward: per-step bonus while any robot link contacts the ramp AND forward vel > 0.
    # Reward = ramp_reward_weight * clamp(v_fwd / INITIAL_VEL_CAP, 0, 1) per step on ramp.
    ramp_reward_weight: float = 3

    # Slow-driving penalty: applied every step the robot is below SLOW_SPEED_FRACTION × vel_cap.
    slow_penalty_weight: float = 2 #3.0

    # Fixed spawn → random spawn transition.
    # All agents spawn at fixed_spawn_pos with fixed_spawn_yaw until the rolling mean
    # episode return (over fixed_spawn_history_len episodes) exceeds fixed_spawn_reward_threshold,
    # after which random CSV-gate spawning is used for the rest of training.
    fixed_spawn_pos:              tuple = (15.239, 5)   # world XY (Z taken from USD default)
    fixed_spawn_yaw:              float = -math.pi / 2.0     # -90° = facing negative Y (track direction)
    fixed_spawn_reward_threshold: float = 10.0              # mean episode return to unlock random spawning
    fixed_spawn_history_len:      int   = 50                # rolling window size (episodes)

    # Low-pass filter (EMA) for steering — simulates servo bandwidth (~100 ms).
    # Throttle no longer uses LPF: velocity integration with MAX_ACCEL/MAX_DECEL
    # provides the same rate-limiting effect inherently.
    # alpha = 1 - exp(-dt / tau) where dt ≈ 1/30 s at 30 Hz policy.
    # alpha=0.25 → tau≈108 ms.
    action_lpf_alpha_steer: float = 0.4     # steering EMA coefficient


# ===========================================================================
# Sector gate geometry helper
# ===========================================================================

def _load_sector_gates(device: str):
    """Load Blender/curve_points.csv and return gate tensors for sector timing.

    Applies the same transform as the track USD in _setup_scene:
        world_xy = csv_xy * SECTOR_GATE_SCALE + SECTOR_GATE_OFFSET

    The number of sectors is read dynamically from the CSV (no hardcoded count).

    Returns:
        n_sectors  int            — number of sector gates (= rows in the CSV)
        gates      [n, 2]         — world XY of each gate point
        tangents   [n, 2]         — unit forward tangent at each gate (centred diff)
    """
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../Blender/curve_points.csv",
    )
    pts: list[tuple[float, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            row = [r.strip() for r in row if r.strip()]
            if len(row) >= 2:
                bx, by = float(row[0]), float(row[1])
                # X_isaac = Y_blender * -0.3 + 16, Y_isaac = X_blender * 0.3
                x = by * -SECTOR_CSV_TO_ISAAC_SCALE + SECTOR_CSV_TO_ISAAC_X_OFFSET
                y = bx *  SECTOR_CSV_TO_ISAAC_SCALE
                pts.append((x, y))

    n = len(pts)
    assert n >= 3, f"curve_points.csv must have at least 3 points, got {n}"
    print(f"[Sector] Loaded {n} sector gates from curve_points.csv")

    # Centred finite-difference tangent at each gate (wrap-around for closed loop)
    tangents: list[tuple[float, float]] = []
    for i in range(n):
        px, py = pts[(i - 1) % n]
        nx, ny = pts[(i + 1) % n]
        dx, dy = nx - px, ny - py
        mag = math.hypot(dx, dy)
        tangents.append((dx / mag, dy / mag))

    gates_t    = torch.tensor(pts,      dtype=torch.float32, device=device)  # [n, 2]
    tangents_t = torch.tensor(tangents, dtype=torch.float32, device=device)  # [n, 2]
    return n, gates_t, tangents_t


# ===========================================================================
# Environment Class
# ===========================================================================

class MushrMazeEnv(DirectRLEnv):
    cfg: MushrMazeEnvCfg

    def __init__(self, cfg: MushrMazeEnvCfg, render_mode: str | None = None, **kwargs):
        # Must exist before super().__init__(), because DirectRLEnv.__init__()
        # invokes _setup_scene(), where these sensors are created.
        self._wall_contact_sensors: dict[str, ContactSensor] = {}

        # Build a temporary ConeVisionProcessor (no GPU/sim needed) to get the exact
        # output dimensions (crop fractions live in ConeVisionCfg, single source of truth).
        # Patch cfg.observation_space BEFORE super().__init__() allocates buffers.
        _tmp_vision = ConeVisionProcessor(
            ConeVisionCfg(
                output_width=cfg.policy_image_width,
                camera_height_width_ratio=CAMERA_HEIGHT / CAMERA_WIDTH,
            )
        )
        cfg.policy_image_height = _tmp_vision.output_height
        # 3 vision channels (left cones, right cones, ramp) + forward vel
        cfg.observation_space   = _tmp_vision.obs_size + 1

        super().__init__(cfg, render_mode, **kwargs)

        # Throttle joint indices — all 4 wheels driven (AWD)
        self._rear_throttle_ids, _ = self.robot.find_joints([
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
        ])
        self._front_throttle_ids, _ = self.robot.find_joints([
            "front_left_wheel_throttle",
            "front_right_wheel_throttle",
        ])

        # Front steering joint indices (position-controlled)
        self._steer_ids, _ = self.robot.find_joints([
            "front_left_wheel_steer",
            "front_right_wheel_steer",
        ])

        # Buffers for Ackermann pre-computation
        self._rear_wheel_ang_vel = torch.zeros(self.num_envs, device=self.device)
        self._steer_tan          = torch.zeros(self.num_envs, device=self.device)

        # Commanded velocity state [N] — integrated from throttle actions each step,
        # clamped to [0, vel_cap].  Sent directly to rear-wheel velocity targets.
        self._current_vel = torch.zeros(self.num_envs, device=self.device)
        # Commanded acceleration state [N] — cached for reward shaping.
        self._accel_m_s2 = torch.zeros(self.num_envs, device=self.device)

        # Low-pass filtered steering [N] — persists across steps.
        self._filtered_steer = torch.zeros(self.num_envs, device=self.device)
        # High-pass steering penalty state: EMA of |Δsteer| per step.
        self._steer_integral  = torch.zeros(self.num_envs, device=self.device)
        self._prev_steer      = torch.zeros(self.num_envs, device=self.device)
        self._steer_hp = torch.zeros(self.num_envs, device=self.device)

        # Curriculum: current velocity ceiling.
        # Advances whenever any agent completes a full lap at the current vel_cap.
        self._vel_cap: float = INITIAL_VEL_CAP
        # Set to True (in _get_rewards) the moment any agent completes a full lap
        # at the current vel_cap.  Consumed (and reset) in _reset_idx to advance vel_cap.
        self._lap_at_current_vel: bool = False

        # ── Ramp curriculum state ─────────────────────────────────────────────────────
        # Ramp is enabled from the start (no lap threshold).
        self._total_laps_completed: int = 0
        self._ramp_enabled: bool = True

        # ── Fixed → random spawn transition ───────────────────────────────────────────
        # Always random from the start.
        self._random_spawn_unlocked: bool = True
        # Accumulates reward during each active episode [N] — reset to 0 at each reset.
        self._ep_return_buf = torch.zeros(self.num_envs, device=self.device)
        # Rolling window of completed episode returns used to compute the mean.
        self._ep_return_history: deque = deque(
            maxlen=self.cfg.fixed_spawn_history_len
        )

        # ── Sector gate geometry (precomputed constants, shared across all envs) ──────
        self._n_sectors, self._sector_gates, self._sector_tangents = _load_sector_gates(self.device)
        # Print first few gates so the user can verify alignment on first run.
        print("[Sector] First 3 world gate positions (world XY):")
        for i in range(min(3, self._n_sectors)):
            x, y = self._sector_gates[i].tolist()
            print(f"  gate[{i}]: ({x:.4f}, {y:.4f})")

        # ── Sector timing state ───────────────────────────────────────────────────────
        # Reference-lap system:
        #   Phase 1 (reference building): track best time per sector as envs explore.
        #              Once every sector has been crossed ≥1 time, take those best times
        #              as the reference and switch to Phase 2.
        #   Phase 2 (comparison): reward any crossing that beats the reference time.
        self._sector_best_times = torch.full(
            (self._n_sectors,), float("inf"),
            dtype=torch.float32, device=self.device,
        )
        # Reference times: set once when all sectors are first covered (inf = not set yet)
        self._sector_reference_times = torch.full(
            (self._n_sectors,), float("inf"),
            dtype=torch.float32, device=self.device,
        )
        # True once the reference lap has been established
        self._reference_established: bool = False
        # Per-env count of sector crossings in the current episode — resets each episode.
        # Also reset to 0 mid-episode when the agent completes a full lap, so each lap
        # is counted independently and the curriculum trigger fires at most once per lap.
        self._sectors_this_ep = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        # vel_cap that was active when each env last reset.  Curriculum only advances
        # when an agent that was spawned at the CURRENT vel_cap completes a full lap,
        # preventing carry-over agents from instantly chaining multiple increases.
        self._agent_spawn_vel_cap = torch.full(
            (self.num_envs,), INITIAL_VEL_CAP, dtype=torch.float32, device=self.device
        )

        # Per-env current sector index [N]
        self._current_sector = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        # Step number when each env entered its current sector [N]
        self._sector_entry_step = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        # Signed distance to the NEXT gate from previous step [N] — for crossing detection
        self._sector_d_prev = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # ── Quick-start tracking ──────────────────────────────────────────────────────
        # Spawn XY position for each env (set at reset, used for start-bonus distance check)
        self._spawn_pos = torch.zeros(self.num_envs, 2, device=self.device)
        # Whether the one-time start bonus has been given for each env this episode

        # Termination/contact flags — set in _get_dones, read in _get_rewards
        self._wall_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._on_ramp         = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Sticky wall-contact flag: set the moment any wall force is detected, cleared
        # only in _reset_idx.  This guarantees termination even when the bounce resolves
        # within the decimation sub-steps and the end-of-step sensor reads zero force.
        self._wall_ever_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Ramp completion state machine [N]:
        #   False → waiting for agent to reach ramp_top_z (while on_ramp)
        #   True  → agent has topped the ramp; waiting for descent below ramp_ground_z
        self._ramp_ascended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Slow-driving detection.
        # Counts consecutive steps where forward speed < SLOW_SPEED_FRACTION × vel_cap.
        # Resets to 0 the moment speed rises above the threshold.
        self._slow_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # OpenCV cone-vision pipeline — crop fractions come from ConeVisionCfg defaults.
        self._vision_processor = ConeVisionProcessor(
            ConeVisionCfg(
                output_width=self.cfg.policy_image_width,
                camera_height_width_ratio=CAMERA_HEIGHT / CAMERA_WIDTH,
            )
        )
        obs_dim = self._vision_processor.obs_size
        self._camera_obs = torch.zeros(self.num_envs, obs_dim, device=self.device)
        self._last_camera_update_step = -1
        self._camera_debug_prefix = "MuSHR Front Camera"

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # 1. Spawn robot articulation (one per environment via prim_path wildcard)
        self.robot = Articulation(self.cfg.robot_cfg)

    
        track_cfg = sim_utils.UsdFileCfg(
            usd_path=TRACK_USD,
            scale=(0.003, 0.003, 0.003),
        )
        track_cfg.func(
            "/World/Track",
            track_cfg,
            translation=(16, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # 2b. Load the full wall boundary — active from the start.
        wall_cfg = sim_utils.UsdFileCfg(
            usd_path=WALL_USD,
            scale=(0.003, 0.003, 0.01),
        )
        wall_cfg.func(
            "/World/Walls",
            wall_cfg,
            translation=(16, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Set convexDecomposition on wall prims.
        # Thick wall panels decompose into small outward-facing hulls — each panel's
        # hull has correct normals that push agents back into the track.
        # (Zero-thickness surfaces have inverted hulls spanning the track interior.)
        try:
            import omni.usd
            from pxr import UsdPhysics, Usd
            stage = omni.usd.get_context().get_stage()
            for wall_path in ("/World/Walls",):
                wall_root = stage.GetPrimAtPath(wall_path)
                if wall_root.IsValid():
                    for prim in Usd.PrimRange(wall_root):
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexDecomposition")
                        elif prim.HasAPI(UsdPhysics.CollisionAPI):
                            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                            mesh_api.GetApproximationAttr().Set("convexDecomposition")
        except Exception as e:
            print(f"[Wall] convexDecomposition setup failed: {e}")


        # 2c. Load ramps — rigid static colliders, visible to the policy camera.
        ramp_cfg = sim_utils.UsdFileCfg(
            usd_path=RAMP_USD,
            scale=(0.003, 0.003, 0.003),
        )
        ramp_cfg.func(
            "/World/Ramps",
            ramp_cfg,
            translation=(16, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # 3. Configure collision approximations.
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        stage = omni.usd.get_context().get_stage()

        # Dynamic wheel links cannot use triangle-mesh collision in PhysX.
        # Force wheel collision meshes to convexHull to avoid parse-time fallback.
        wheel_link_names = {
            "front_left_wheel_link",
            "front_right_wheel_link",
            "back_left_wheel_link",
            "back_right_wheel_link",
        }
        wheel_root = stage.GetPrimAtPath("/World/envs/env_0/Robot/mushr_nano")
        if wheel_root.IsValid():
            for prim in Usd.PrimRange(wheel_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                parent_name = prim.GetParent().GetName()
                if parent_name not in wheel_link_names:
                    continue
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.GetApproximationAttr().Set("convexHull")

        # Apply CollisionAPI, kinematic RigidBodyAPI, and invisible to both wall prims.
        # /World/Walls starts with collision DISABLED (active after RAMP_ENABLE_LAPS laps).
        # /World/Walls has collision ENABLED from the start.
        for wall_path, collision_enabled in (
            ("/World/Walls", True),
        ):
            wall_root = stage.GetPrimAtPath(wall_path)
            if not wall_root.IsValid():
                continue
            for prim in Usd.PrimRange(wall_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                wall_collision = UsdPhysics.CollisionAPI.Apply(prim)
                wall_collision.GetCollisionEnabledAttr().Set(collision_enabled)
                wall_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                wall_mesh_collision.GetApproximationAttr().Set("none")
            # Kinematic rigid body on root so ContactSensor filter_prim_paths_expr matches.
            wall_rb = UsdPhysics.RigidBodyAPI.Apply(wall_root)
            wall_rb.GetRigidBodyEnabledAttr().Set(True)
            wall_rb.GetKinematicEnabledAttr().Set(True)
            # Walls are always invisible to the policy camera.
            UsdGeom.Imageable(wall_root).MakeInvisible()

        # Enable triangle-mesh collision on ramp meshes — ramp is active from the start.
        ramp_root = stage.GetPrimAtPath("/World/Ramps")
        if ramp_root.IsValid():
            UsdGeom.Imageable(ramp_root).MakeVisible()
            for prim in Usd.PrimRange(ramp_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                ramp_collision = UsdPhysics.CollisionAPI.Apply(prim)
                ramp_collision.GetCollisionEnabledAttr().Set(True)
                ramp_mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                ramp_mesh_col.GetApproximationAttr().Set("none")  # triangleMesh preserves ramp shape


        # Remove unsupported suspension drive gains from the imported USD joints.
        # PhysX ignores these attrs for articulation joints and spams warnings otherwise.
        suspension_joint_names = (
            "front_left_wheel_suspension",
            "front_right_wheel_suspension",
            "back_left_wheel_suspension",
            "back_right_wheel_suspension",
        )
        for env_id in range(self.num_envs):
            base_link_path = f"/World/envs/env_{env_id}/Robot/mushr_nano/base_link"
            for joint_name in suspension_joint_names:
                joint_prim = stage.GetPrimAtPath(f"{base_link_path}/{joint_name}")
                if not joint_prim.IsValid():
                    continue
                for prop in list(joint_prim.GetProperties()):
                    prop_name = prop.GetName()
                    if "stiffness" in prop_name or "damping" in prop_name:
                        # Suspension gains come from referenced USD layers.
                        # Block authored attrs at this stronger layer so PhysX
                        # does not see unsupported articulation joint gains.
                        attr = joint_prim.GetAttribute(prop_name)
                        if attr.IsValid():
                            attr.Block()

        # 4. Physics ground plane — friction tuned to rubber on basketball court.
        # Target: static ≈ 0.8, dynamic ≈ 0.7.  Wheel collision shapes are assigned
        # a rubber material (static=1.0, dynamic=1.0) below; with "multiply" combine
        # the net contact coefficients are 0.8 × 1.0 = 0.8 / 0.7 × 1.0 = 0.7.
        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.05,
                friction_combine_mode="multiply",
                restitution_combine_mode="min",
            )
        )
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        # 5. Register sensors.
        self.camera = TiledCamera(self.cfg.camera_cfg)
        # Wall termination uses per-link contact-force sensing. We detect wall hits
        # by horizontal normal force magnitude (ground normals are vertical).
        for link_name in WALL_CONTACT_LINKS:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/mushr_nano/{link_name}",
                filter_prim_paths_expr=["/World/Walls"],
                history_length=self.cfg.decimation,
                update_period=0.0,
                track_pose=False,
                debug_vis=False,
            )
            self._wall_contact_sensors[link_name] = ContactSensor(sensor_cfg)

        # Create shared rubber physics material prim (referenced after cloning)
        rubber_mat_path = "/World/PhysicsMaterials/WheelRubber"
        rubber_mat_prim = stage.DefinePrim(rubber_mat_path, "Material")
        UsdPhysics.MaterialAPI.Apply(rubber_mat_prim)
        phys_rubber = UsdPhysics.MaterialAPI(rubber_mat_prim)
        phys_rubber.CreateStaticFrictionAttr().Set(1.0)   # rubber coeff; × ground 0.8 = 0.8 net
        phys_rubber.CreateDynamicFrictionAttr().Set(1.0)  # rubber coeff; × ground 0.7 = 0.7 net
        phys_rubber.CreateRestitutionAttr().Set(0.05)

        # 6. Clone environments (robot only; track and walls are global)
        self.scene.clone_environments(copy_from_source=False)

        # Bind rubber material to every wheel collision mesh across all environments.
        # Must run after clone_environments so all env_N prims exist.
        rubber_mat = UsdShade.Material(rubber_mat_prim)
        for env_id in range(self.num_envs):
            wheel_root_env = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/mushr_nano")
            if not wheel_root_env.IsValid():
                continue
            for prim in Usd.PrimRange(wheel_root_env):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                if prim.GetParent().GetName() not in wheel_link_names:
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    rubber_mat,
                    UsdShade.Tokens.strongerThanDescendants,
                    "physics",
                )

        # 7. Register assets / sensors with the scene manager
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["front_camera"] = self.camera
        for link_name, sensor in self._wall_contact_sensors.items():
            self.scene.sensors[f"wall_contact_{link_name}"] = sensor


        # 8. Compute ramp world-space XY bounding box for contact masking.
        # Used in _apply_action to suppress wall termination when the robot is
        # within the ramp footprint — the only reliable discriminant since ramp
        # edge impacts produce horizontal forces indistinguishable from wall hits.
        try:
            from pxr import UsdGeom
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            )
            ramp_prim = stage.GetPrimAtPath("/World/Ramps")
            if ramp_prim.IsValid():
                bb = bbox_cache.ComputeWorldBound(ramp_prim).GetRange()
                margin = 1.0   # metres — covers robot half-width + ramp edge uncertainty
                self._ramp_xy_min = torch.tensor(
                    [bb.GetMin()[0] - margin, bb.GetMin()[1] - margin],
                    dtype=torch.float32, device=self.device,
                )
                self._ramp_xy_max = torch.tensor(
                    [bb.GetMax()[0] + margin, bb.GetMax()[1] + margin],
                    dtype=torch.float32, device=self.device,
                )
                print(f"[Ramp] XY bounding box (with {margin} m margin): "
                      f"min={self._ramp_xy_min.tolist()}  max={self._ramp_xy_max.tolist()}")
            else:
                self._ramp_xy_min = None
                self._ramp_xy_max = None
        except Exception as e:
            print(f"[Ramp] Bounding-box computation failed: {e}")
            self._ramp_xy_min = None
            self._ramp_xy_max = None

        # 9. Dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Ramp enable helper
    # ------------------------------------------------------------------

    def _enable_ramp(self) -> None:
        """Make the ramp visible and re-enable its collision after the lap threshold."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            ramp_root = stage.GetPrimAtPath("/World/Ramps")
            if ramp_root.IsValid():
                UsdGeom.Imageable(ramp_root).MakeVisible()
                for prim in Usd.PrimRange(ramp_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    col_api = UsdPhysics.CollisionAPI(prim)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(True)
                print(f"[Ramp] ENABLED after {self._total_laps_completed} total laps!")
        except Exception as e:
            print(f"[Ramp] Enable failed: {e}")

    def _swap_walls(self) -> None:
        """Swap from WALLS_INITIAL to WALLS after the lap threshold is reached."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            # Disable collision on the initial walls.
            initial_root = stage.GetPrimAtPath("/World/WallsInitial")
            if initial_root.IsValid():
                for prim in Usd.PrimRange(initial_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    col_api = UsdPhysics.CollisionAPI(prim)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(False)
            # Enable collision on the full walls.
            full_root = stage.GetPrimAtPath("/World/Walls")
            if full_root.IsValid():
                for prim in Usd.PrimRange(full_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    col_api = UsdPhysics.CollisionAPI(prim)
                    if col_api:
                        col_api.GetCollisionEnabledAttr().Set(True)
            print(f"[Walls] Swapped to WALLS.usd after {self._total_laps_completed} total laps!")
        except Exception as e:
            print(f"[Walls] Swap failed: {e}")

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert actions → Ackermann wheel commands using velocity integration.

        Throttle (actions[:, 0]) ∈ [-1, 1] is remapped to [0, 1]:
          +1 → full acceleration at MAX_ACCEL m/s²
           0 → neutral (no change)
          -1 → full deceleration at MAX_DECEL m/s²
        The commanded velocity is integrated and clamped to [0, vel_cap], preventing
        wheel lockup by rate-limiting the velocity command to MAX_ACCEL/MAX_DECEL.
        No throttle LPF is needed — the integration itself acts as a rate limiter.

        Steering (actions[:, 1]) ∈ [-1, 1] is still low-pass filtered to simulate
        servo bandwidth, then converted to the tan joint-position convention.
        """
        dt = self.cfg.decimation / 120.0   # policy step duration (≈ 0.033 s at 30 Hz)

        # Throttle: remap [-1, 1] → [0, 1], compute acceleration rate in m/s²
        throttle_norm = (actions[:, 0].clamp(-1.0, 1.0) + 1.0) * 0.5   # [0, 1]
        accel_m_s2 = throttle_norm * (MAX_ACCEL + MAX_DECEL) - MAX_DECEL  # [-MAX_DECEL, +MAX_ACCEL]
        self._accel_m_s2 = accel_m_s2

        # Integrate commanded velocity and clamp to [0, vel_cap]
        self._current_vel = (self._current_vel + accel_m_s2 * dt).clamp(0.0, self._vel_cap)

        # Steering: EMA low-pass filter (simulates servo bandwidth ~100 ms)
        alpha_s = self.cfg.action_lpf_alpha_steer
        self._filtered_steer = (
            alpha_s * actions[:, 1].clamp(-1.0, 1.0)
            + (1.0 - alpha_s) * self._filtered_steer
        )
        steer_ang = self._filtered_steer * MAX_STEER   # rad

        self._rear_wheel_ang_vel = self._current_vel / WHEEL_RADIUS   # rad/s
        self._steer_tan          = torch.tan(steer_ang)               # tan convention

    def _apply_action(self) -> None:
        # ── Sub-step wall contact detection (120 Hz, covers sub-steps 0–2) ─────────────
        # Decimation loop order: _apply_action → write → sim.step → scene.update.
        # At sub-step K, _apply_action reads the sensor data written by scene.update()
        # after sub-step K-1.  So contacts from sub-step K are caught here at sub-step K+1.
        # This covers sub-steps 0–2.  Sub-step 3 contacts (no K+1 exists) are caught by
        # the direct sensor read inside _get_dones.
        # Sensors are filtered to /World/Walls only via filter_prim_paths_expr,
        # so force_matrix_w contains wall-only contact forces — ramp contacts are
        # never included and need no masking.
        # force_matrix_w shape: (num_envs, num_bodies_per_sensor, num_filter_bodies, 3)
        wall_hit_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for sensor in self._wall_contact_sensors.values():
            fm = sensor.data.force_matrix_w   # (N, B, M, 3) or None
            if fm is not None:
                h = torch.linalg.vector_norm(fm[..., :2], dim=-1)   # horizontal magnitude
                wall_hit_now |= h.reshape(self.num_envs, -1).gt(WALL_CONTACT_FORCE_THRESH).any(dim=1)

        # Update sticky flag — _reset_idx handles the actual reset.
        self._wall_ever_touched |= wall_hit_now

        # ── Action application ────────────────────────────────────────────────────────
        # Envs that have hit a wall get zero velocity targets so the robot stops
        # within the same 120 Hz sub-step it made contact.  This removes the up-to-33 ms
        # gap between impact and the 30 Hz _get_dones termination — the robot is already
        # stationary by the time the episode is formally ended.
        wall_mask = self._wall_ever_touched  # [N] bool

        # All 4 wheels — same velocity target (AWD, mechanically coupled)
        wheel_vels = self._rear_wheel_ang_vel.unsqueeze(-1).expand(-1, 2).clone()  # [N, 2]
        wheel_vels[wall_mask] = 0.0
        self.robot.set_joint_velocity_target(wheel_vels, joint_ids=self._rear_throttle_ids)
        self.robot.set_joint_velocity_target(wheel_vels, joint_ids=self._front_throttle_ids)

        # Front steering — position target
        steer_positions = self._steer_tan.unsqueeze(-1).expand(-1, 2).clone()  # [N, 2]
        steer_positions[wall_mask] = 0.0
        self.robot.set_joint_position_target(steer_positions, joint_ids=self._steer_ids)



    # ------------------------------------------------------------------
    # Camera cache helper (called once per step by _get_dones)
    # ------------------------------------------------------------------

    def _update_camera_cache(self) -> None:
        """Process camera frames and update cached observations.

        Isaac Lab calls _get_dones → _get_rewards → _get_observations in that
        order.  We process the camera once here (guarded by a step sentinel) so
        all three methods share the same per-step result without redundant work.
        """
        current_step = int(self.episode_length_buf.max().item())
        if self._last_camera_update_step == current_step:
            return
        self._last_camera_update_step = current_step

        rgb = self.camera.data.output["rgb"]  # [N, H, W, C]
        debug_env = self.cfg.camera_debug_env_id if self.cfg.show_camera_debug else None
        camera_obs, _, _, debug_imgs = self._vision_processor.process_batch(rgb, debug_env)

        self._camera_obs = camera_obs.to(self.device)

        if self.cfg.show_camera_debug and debug_imgs is not None:
            self._vision_processor.show_debug(debug_imgs, self._camera_debug_prefix)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_camera_cache()

        # Forward velocity — normalised by current vel_cap (curriculum-aware)
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0:1] / self._vel_cap   # [N, 1]

        obs = torch.cat([self._camera_obs, lin_vel_b], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        self._update_camera_cache()

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]   # forward (m/s)

        # Distance reward: reward proportional to physical metres covered this step.
        # dt is the real-time duration of one policy step (decimation / sim_hz).
        # Summed across all steps this equals total odometry — correct for circular tracks
        # because it integrates incremental progress, not straight-line displacement.
        dt = self.cfg.decimation * self.cfg.sim.dt   # ≈ 1/30 s
        r_distance = self.cfg.distance_weight * lin_vel_b * dt

        # Speed reward — normalised by INITIAL_VEL_CAP (fixed), not self._vel_cap.
        # Using a fixed denominator keeps the reward scale stable as the curriculum
        # advances vel_cap; otherwise r_forward would shrink each time vel_cap grows
        # (the same physical speed gives less reward), collapsing the mean episode
        # return and causing the curriculum threshold to never be met again.
        r_forward = self.cfg.forward_weight * (lin_vel_b / INITIAL_VEL_CAP)

        # Wall termination penalty.
        r_wall = -self.cfg.wall_penalty * self._wall_terminated.float()


        pos_xy = self.robot.data.root_pos_w[:, :2]                          # [N, 2]


        # ── Sector timing reward ──────────────────────────────────────────────────────
        # Two-phase system:
        #   Phase 1 — reference building: track the best time seen per sector as envs
        #             explore the track.  No reward until every sector ≥ 1 crossing.
        #   Phase 2 — comparison: reward = sector_gain_weight * (ref - time) / ref
        #             for any crossing that beats the reference time.
        dt_policy = self.cfg.decimation * self.cfg.sim.dt                # ≈ 0.0333 s

        next_gate = (self._current_sector + 1) % self._n_sectors          # [N]
        gate_pts  = self._sector_gates[next_gate]                          # [N, 2]
        gate_tan  = self._sector_tangents[next_gate]                       # [N, 2]

        # Signed distance: positive = ahead of gate, negative = behind gate
        d_current = ((pos_xy - gate_pts) * gate_tan).sum(dim=-1)           # [N]

        # Edge detection: negative→positive sign change = forward crossing.
        # Also require agent to be within 1.8 m of the gate point to prevent false
        # crossings where the tangent line intersects the track far from point n.
        near_gate = torch.linalg.vector_norm(pos_xy - gate_pts, dim=-1) < 1.8
        crossed   = (self._sector_d_prev <= 0.0) & (d_current > 0.0) & near_gate  # [N] bool

        # Sector transit time in seconds
        sector_time = (
            (self.episode_length_buf - self._sector_entry_step).float() * dt_policy
        )                                                                   # [N]

        # lap_agents is set inside the crossed.any() block; initialise here so r_lap
        # is always defined even on steps where no gate is crossed.
        lap_agents = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Always track per-env crossing count (used for both reference and curriculum)
        if crossed.any():
            crossed_sectors = next_gate[crossed]                            # [K]
            crossed_times   = sector_time[crossed]                         # [K]
            # Update global best times
            candidate = torch.full(
                (self._n_sectors,), float("inf"),
                dtype=torch.float32, device=self.device,
            )
            candidate.scatter_reduce_(
                0, crossed_sectors, crossed_times,
                reduce="amin", include_self=False,
            )
            torch.minimum(
                self._sector_best_times, candidate,
                out=self._sector_best_times,
            )
            self._sectors_this_ep[crossed] += 1
            # Detect which envs just completed a full lap this step
            lap_agents = self._sectors_this_ep >= self._n_sectors          # [N] bool
            if lap_agents.any():
                # Reset counters immediately so each lap fires exactly once
                self._sectors_this_ep[lap_agents] = 0

                # Count laps first so the print shows the correct running total.
                self._total_laps_completed += int(lap_agents.sum().item())

                # Print each lap completion with env id, episode time, and running total.
                for eid in lap_agents.nonzero(as_tuple=False).squeeze(-1).tolist():
                    ep_steps = int(self.episode_length_buf[eid].item())
                    lap_time_s = ep_steps * self.cfg.decimation * self.cfg.sim.dt
                    print(
                        f"[Lap] env={eid:>3d}  step={ep_steps:>6d}  "
                        f"time={lap_time_s:.1f}s  "
                        f"total_laps={self._total_laps_completed}"
                    )

                if not self._reference_established:
                    self._sector_reference_times = self._sector_best_times.clone()
                    self._reference_established = True
                    ref_mean = self._sector_reference_times.mean().item()
                    print(
                        f"[Sector] Reference lap established! "
                        f"{self._n_sectors} sectors, mean ref time = {ref_mean:.2f} s"
                    )

                # Curriculum: only advance if at least one lapping agent was spawned
                # at the current vel_cap (not a carry-over from a lower level).
                fresh_lap = lap_agents & (self._agent_spawn_vel_cap == self._vel_cap)
                if fresh_lap.any() and not self._lap_at_current_vel:
                    self._lap_at_current_vel = True

                # Ramp/wall curriculum: disabled — ramp is active from the start.
                # if not self._ramp_enabled and self._total_laps_completed >= RAMP_ENABLE_LAPS:
                #     self._ramp_enabled = True
                #     self._enable_ramp()
                #     # self._swap_walls()  # WALLS.usd is active from the start; no swap needed

        # ── Lap completion bonus ───────────────────────────────────────────────────────
        r_lap = torch.zeros(self.num_envs, device=self.device)
        r_lap[lap_agents] = self.cfg.lap_completion_reward

        if not self._reference_established:
            r_sector = torch.zeros(self.num_envs, device=self.device)
        else:
            # ── Phase 2: reward crossings that beat the reference ──────────────
            ref_time    = self._sector_reference_times[next_gate]           # [N]
            beats_ref   = crossed & (sector_time < ref_time)               # [N] bool
            improvement = (
                (ref_time - sector_time) / ref_time.clamp(min=1e-6)
            ).clamp(0.0, 1.0)
            r_sector = self.cfg.sector_gain_weight * beats_ref.float() * improvement

        # Advance sector and reset entry step for envs that just crossed a gate
        self._current_sector[crossed]    = next_gate[crossed]
        self._sector_entry_step[crossed] = self.episode_length_buf[crossed].int()
        self._sector_d_prev = d_current

        # ── Wheel slip penalty (AWD — all 4 wheels) ──────────────────────────────────
        all_wheel_omega = torch.cat([
            self.robot.data.joint_vel[:, self._rear_throttle_ids],
            self.robot.data.joint_vel[:, self._front_throttle_ids],
        ], dim=-1)                                                                 # [N, 4] rad/s
        wheel_vel    = all_wheel_omega.mean(dim=-1) * WHEEL_RADIUS                # [N] m/s
        slip_speed   = torch.abs(wheel_vel - lin_vel_b)                           # [N] m/s
        r_slip       = -self.cfg.slip_weight * slip_speed / INITIAL_VEL_CAP


        # ── Steering magnitude penalty (fifth-root curve) ─────────────────────────────
        # (|steer_angle| / MAX_STEER)^(1/5) is concave: penalises small angles
        # proportionally more than a linear term while saturating toward 1 at full lock.
        steer_norm = torch.abs(self._filtered_steer).clamp(0.0, 1.0) < 0.05
        r_steer_shape = self.cfg.steer_shape_weight * steer_norm


        # ── Ramp contact reward ───────────────────────────────────────────────────────
        # _on_ramp is computed and cached in _get_dones (called before _get_rewards).
        # Only reward if the car is properly flat on the ramp (|roll| < 5°).
        # grav_y = Y-body component of world-up = sin(roll_angle); threshold = sin(5°).
        q = self.robot.data.root_quat_w                                     # [N, 4] (w,x,y,z)
        grav_y = 2.0 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])            # [N]
        # low_roll = grav_y.abs() < 0.0872                                    # sin(5°) ≈ 0.0872
        r_ramp = self.cfg.ramp_reward_weight * self._on_ramp.float() * (lin_vel_b / INITIAL_VEL_CAP).clamp(0.0, 1.0)

        
        # ── Slow-driving penalty ─────────────────────────────────────────────────────
        # Suppressed while on the ramp — the car naturally slows on inclines.
        is_slow = (lin_vel_b < (SLOW_SPEED_FRACTION * self._vel_cap)) & ~self._on_ramp
        self._slow_timer = torch.where(
            is_slow,
            self._slow_timer + 1,
            torch.zeros_like(self._slow_timer),
        )
        r_slow = -self.cfg.slow_penalty_weight * is_slow.float()

        total = (
            r_distance
            + r_forward
            + r_slip
            + r_steer_shape
            + r_slow
            + r_wall
            + r_sector
            + r_ramp
            + r_lap
        )

        # Accumulate per-episode return for fixed→random spawn threshold check.
        self._ep_return_buf += total.detach()

        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ── Wall termination: dual detection ─────────────────────────────────────────
        # Primary: sticky contact flag set by _apply_action every physics sub-step.
        #   Covers contacts from sub-steps 0–2: each sub-step K's contact is stored by
        #   scene.update() after sim.step(K), then read by _apply_action at sub-step K+1.
        wall_terminated = self._wall_ever_touched.clone()

        # Secondary: direct sensor read here in _get_dones.
        #   The decimation loop order is: _apply_action → write → sim.step → scene.update.
        #   After the LAST sub-step (sub-step 3), scene.update() runs and refreshes sensor
        #   data, but there is no sub-step 4 _apply_action to consume it.  Without this
        #   read, sub-step 3 contacts are missed until the NEXT policy step's _apply_action,
        #   causing up to ~66 ms delay.  Reading sensors directly here closes that gap.
        for sensor in self._wall_contact_sensors.values():
            fm = sensor.data.force_matrix_w   # (N, B, M, 3) or None
            if fm is not None:
                h = torch.linalg.vector_norm(fm[..., :2], dim=-1)
                wall_terminated |= h.reshape(self.num_envs, -1).gt(WALL_CONTACT_FORCE_THRESH).any(dim=1)

        # Ramp detection via Z position — reliable and sensor-filter-independent.
        # Flat ground z ~ 0.02 m; ramp raises robot above ramp_ground_z immediately.
        self._on_ramp = self.robot.data.root_pos_w[:, 2] >= 0.001

        # Sticky flag persists until _reset_idx clears it.
        self._wall_ever_touched |= wall_terminated
        wall_terminated = self._wall_ever_touched
        self._wall_terminated = wall_terminated

        # Flip / rollover detection.
        # The Z component of the robot's local up-axis (0,0,1) in world frame equals
        #   up_z = 1 - 2*(qx² + qy²)
        # which is +1 when perfectly upright and -1 when fully inverted.
        # Terminate whenever up_z < 0.3, i.e. the robot has tilted more than ~72°
        # from vertical — at that angle recovery is impossible and physics will
        # produce garbage observations (camera points at sky/ground).
        q = self.robot.data.root_quat_w                              # [N, 4]  (w, x, y, z)
        up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
        flipped = up_z < 0.3

        # Near-stop termination: immediate termination if speed drops below 5 % of vel_cap.
        # Grace period of 30 steps (~1 s) so the car can accelerate from a standing start.
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        stopped = (lin_vel_b < (0.02 * self._vel_cap)) & (self.episode_length_buf > 45)

        terminated = wall_terminated | flipped | stopped
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # ── Episode return history (fixed→random spawn unlock) ────────────────
        # Append the accumulated return for each resetting env, then reset its buffer.
        for eid in env_ids:
            self._ep_return_history.append(float(self._ep_return_buf[eid].item()))
        self._ep_return_buf[env_ids] = 0.0

        # Check unlock condition once per _reset_idx call (cheap — O(1) after deque fill).
        if (
            not self._random_spawn_unlocked
            and len(self._ep_return_history) >= self.cfg.fixed_spawn_history_len
        ):
            mean_return = sum(self._ep_return_history) / len(self._ep_return_history)
            if mean_return >= self.cfg.fixed_spawn_reward_threshold:
                self._random_spawn_unlocked = True
                print(
                    f"[Spawn] Random spawning UNLOCKED! "
                    f"Mean episode return = {mean_return:.1f} >= "
                    f"{self.cfg.fixed_spawn_reward_threshold:.1f}"
                )
        # ─────────────────────────────────────────────────────────────────────

        # ── Curriculum advancement ────────────────────────────────────────────
        # Advance vel_cap whenever any agent has completed a full lap at the current cap.
        # _lap_at_current_vel is set in _get_rewards; reset it here after consuming it.
        if self._lap_at_current_vel and self._vel_cap < VEL_CAP_MAX:
            self._lap_at_current_vel = False
            self._vel_cap = min(self._vel_cap + VEL_CAP_STEP, VEL_CAP_MAX)
            print(f"[Curriculum] vel_cap → {self._vel_cap:.2f} m/s  (full lap completed)")
        # ─────────────────────────────────────────────────────────────────────

        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        SPAWN_SPEED = 2   # m/s

        fx, fy = self.cfg.fixed_spawn_pos
        diff         = self._sector_gates - torch.tensor([[fx, fy]], dtype=torch.float32, device=self.device)
        fixed_nearest_gate = torch.linalg.vector_norm(diff, dim=-1).argmin()  # scalar

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)  # [n]

        spawn_xy = torch.empty(n, 2, device=self.device)
        yaw      = torch.empty(n,    device=self.device)

        # ── Fixed spawn: all agents at (15.15, 6.06, 0.002) ──────────────────
        FIXED_SPAWN_X = 15.15
        FIXED_SPAWN_Y = 6.06
        FIXED_SPAWN_Z = 0.002
        spawn_xy[:] = torch.tensor([[FIXED_SPAWN_X, FIXED_SPAWN_Y]], dtype=torch.float32, device=self.device)
        yaw[:]      = self.cfg.fixed_spawn_yaw
        self._current_sector[env_ids_t] = fixed_nearest_gate

        # ── Random CSV-gate spawning (commented out — re-enable to use multi-spawn curriculum) ──
        # use_random = torch.ones(n, dtype=torch.bool, device=self.device)
        # fixed_mask = ~use_random
        # if fixed_mask.any():
        #     nf = fixed_mask.sum()
        #     spawn_xy[fixed_mask] = torch.tensor([[fx, fy]], dtype=torch.float32, device=self.device).expand(nf, -1)
        #     yaw[fixed_mask]      = self.cfg.fixed_spawn_yaw
        #     self._current_sector[env_ids_t[fixed_mask]] = fixed_nearest_gate
        # if use_random.any():
        #     nr       = use_random.sum()
        #     gate_idx = torch.randint(0, self._n_sectors, (nr,), device=self.device)
        #     spawn_xy[use_random] = self._sector_gates[gate_idx]
        #     gate_tan             = self._sector_tangents[gate_idx]
        #     yaw[use_random]      = torch.atan2(gate_tan[:, 1], gate_tan[:, 0])
        #     self._current_sector[env_ids_t[use_random]] = gate_idx

        default_root_state[:, 0] = spawn_xy[:, 0]
        default_root_state[:, 1] = spawn_xy[:, 1]
        default_root_state[:, 2] = FIXED_SPAWN_Z

        zeros = torch.zeros(n, device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw)
        default_root_state[:, 3:7] = dq

        # Initial physics velocity along spawn heading
        default_root_state[:, 7:]  = 0.0
        default_root_state[:, 7]   = SPAWN_SPEED * torch.cos(yaw)   # world vx
        default_root_state[:, 8]   = SPAWN_SPEED * torch.sin(yaw)   # world vy

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # ── Sector + quick-start + steering state reset ──────────────────────────────
        self._spawn_pos[env_ids]              = spawn_xy
        self._sectors_this_ep[env_ids]        = 0
        self._agent_spawn_vel_cap[env_ids]    = self._vel_cap

        self._sector_entry_step[env_ids] = 0  # fresh start for this episode

        # Signed distance from spawn to NEXT gate, clamped ≤ 0 — prevents a false
        # crossing on the very first step.
        next_gate_r = (self._current_sector[env_ids] + 1) % self._n_sectors       # [n]
        gate_pts_r  = self._sector_gates[next_gate_r]                             # [n, 2]
        gate_tan_r  = self._sector_tangents[next_gate_r]                          # [n, 2]
        d_init      = ((spawn_xy - gate_pts_r) * gate_tan_r).sum(dim=-1)          # [n]
        self._sector_d_prev[env_ids] = d_init.clamp(max=0.0)

        # Seed velocity command to match the initial physics velocity so the integrator
        # doesn't immediately decelerate from 0 on the first step.
        self._current_vel[env_ids]    = SPAWN_SPEED
        self._accel_m_s2[env_ids]     = 0.0
        self._filtered_steer[env_ids] = 0.0
        self._steer_integral[env_ids] = 0.0
        self._prev_steer[env_ids]     = 0.0

        # Clear slow-driving timer, ramp state, and wall-contact flag for the new episode.
        self._slow_timer[env_ids]        = 0
        self._ramp_ascended[env_ids]     = False
        self._wall_ever_touched[env_ids] = False

        # Reset joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
