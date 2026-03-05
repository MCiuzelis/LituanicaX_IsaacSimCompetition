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
    -slip_weight * |rear_wheel_vel − v_fwd| / INITIAL_VEL_CAP (traction loss penalty, same fixed normalisation)
    -steer_weight * max(0, |steer_norm| − deadzone) / (1−dz)  (steering sharpness penalty beyond 20% deadzone)
    -wall_penalty       on wall termination only               (penalty for wall contact)
    +completion_weight * (1 - elapsed/max_steps)              (speed bonus for lap completion)

Termination conditions:
    - Wall contact: any lateral force on robot links above WALL_CONTACT_FORCE_THRESH (near-zero)
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

from collections.abc import Sequence
from collections import deque
import math

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

# ---------------------------------------------------------------------------
# Asset USD paths
# ---------------------------------------------------------------------------
MUSHR_USD = "/home/matasciuzelis/Documents/lituanicaXsim/assets/mushr_nano_v2.usd"

TRACK_USD = "/home/matasciuzelis/Documents/lituanicaXsim/assets/CONES.usd"

WALL_USD = "/home/matasciuzelis/Documents/lituanicaXsim/assets/WALLS.usd"

# ---------------------------------------------------------------------------
# Physical constants — MuSHR nano v2 (Ackermann / RWD)
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
INITIAL_VEL_CAP = 2.0    # m/s — starting velocity ceiling
VEL_CAP_MAX     = 10.0    # m/s — absolute maximum
VEL_CAP_STEP    = 0.1   # m/s — increment per curriculum advancement
VEL_HISTORY_LEN = 100    # completed episodes in rolling mean window
VEL_CAP_REWARD_THRESHOLD = 500.0  # mean episode return required to advance

# ---------------------------------------------------------------------------
# Acceleration limits — derived from friction and rear-weight distribution.
# Net ground friction (rubber wheel × basketball court, multiply combine):
#   μ_s = 0.8, μ_d = 0.7,  g = 9.81 m/s²
# Rear-weight fraction: ≈0.50 at rest, ≈0.40 during braking (weight transfers forward).
#   MAX_ACCEL: μ_s × g × 0.50 ≈ 3.9  → 4.0 m/s²  (avoids rear-wheel spin-up)
#   MAX_DECEL: μ_d × g × 0.40 ≈ 2.75 → 3.0 m/s²  (avoids rear-wheel lockup)
# At 30 Hz (dt≈0.033 s): max Δv per step ≈ 0.13 m/s (accel) / 0.10 m/s (decel).
# ---------------------------------------------------------------------------
MAX_ACCEL = 4.0   # m/s² — max forward acceleration without wheelspin
MAX_DECEL = 3.0   # m/s² — max deceleration without rear-wheel lockup

MAX_STEER    = 0.488   # rad  (~28°) max steering angle

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

# Any lateral contact force above this threshold on a robot link triggers wall termination.
# Set low so any genuine wall touch terminates immediately; ground contacts are mostly
# vertical and therefore rejected by the lateral-force filter.
WALL_CONTACT_FORCE_THRESH = 0.1   # N

# Slow-driving penalty and termination.
# Any step where forward speed < SLOW_SPEED_FRACTION × vel_cap counts toward the
# slow timer.  The timer resets to 0 the moment speed rises above the threshold.
# Terminate if the timer reaches SLOW_TIMEOUT_STEPS (6 s × 30 Hz = 180 steps).
SLOW_SPEED_FRACTION = 0.4    # fraction of vel_cap below which the robot is "too slow"
SLOW_TIMEOUT_STEPS  = 180    # 6 s × 30 Hz

# Robot links monitored for wall contact.
WALL_CONTACT_LINKS = (
    "base_link",
    "front_left_wheel_link",
    "front_right_wheel_link",
    "back_left_wheel_link",
    "back_right_wheel_link",
)


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
    observation_space: int = 2 * POLICY_IMAGE_WIDTH * POLICY_IMAGE_HEIGHT + 1
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
        num_envs=16,
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=100000.0,
                max_depenetration_velocity=100.0,
                max_contact_impulse=0.0,
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
            pos=(15.15, 5, 0.005),
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
            # Rear drive wheels — velocity-controlled (RWD)
            # velocity_limit_sim / effort_limit from WheeledLab MUSHR_SUS_2WD_CFG reference.
            "rear_throttle": ImplicitActuatorCfg(
                joint_names_expr=["back_left_wheel_throttle", "back_right_wheel_throttle"],
                stiffness=0.0,
                damping=1000.0,
                velocity_limit_sim=450.0,
                effort_limit_sim=0.5,
            ),
            # Front wheels — passive (no torque, free-spinning in RWD)
            "front_throttle": ImplicitActuatorCfg(
                joint_names_expr=["front_left_wheel_throttle", "front_right_wheel_throttle"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=0.0,
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
    distance_weight:    float = 0.5   # reward per metre of forward travel
    forward_weight:     float = 4.0

    # Extra one-time penalty specifically for wall contact (on top of collision_penalty).
    wall_penalty:       float = 100

    # Lap completion speed bonus: scales with how quickly the lap was completed.
    completion_weight:  float = 8000.0
    departure_dist:     float = 2.0   # metres from spawn before "departed"
    return_dist:        float = 2.0   # metres from spawn that counts as lap complete

    # Rear-wheel slip penalty: discourages wheel-locking hard braking.
    # slip_speed = |mean_rear_wheel_tangential_vel − body_forward_vel| / MAX_LIN_VEL
    slip_weight:        float = 1.0

    # Steering sharpness penalty: linear penalty beyond a deadzone fraction of MAX_STEER.
    # No penalty for |steer| ≤ steer_deadzone; linearly increasing above that.
    steer_weight:    float = 0.5   # small — just suppresses constant oscillation
    steer_deadzone:  float = 0.2   # fraction of MAX_STEER below which no penalty applies

    # Slow-driving penalty: applied every step the robot is below SLOW_SPEED_FRACTION × vel_cap.
    slow_penalty_weight: float = 2.0

    # Spawn locations (x, y, yaw_rad).  At each reset a point is chosen uniformly at random.
    # Add or remove entries to change how many spawn points exist.
    spawn_locations: list = None  # set in __post_init__ via dataclass default

    def __post_init__(self):
        if self.spawn_locations is None:
            self.spawn_locations = [
                (15.15, 5.0, -1.5708),   # default single spawn — TUNE after first run
                (9.36, -4.2, 0),
                (12.4, 0.13, 3.14),
                (8.39, 8.94, -1.065),
                (12.57, 4.87, 1.92),
            ]

    # Yaw jitter applied on top of each spawn point's yaw (radians, uniform ±jitter)
    init_yaw_jitter: float = 0.1

    # Low-pass filter (EMA) for steering — simulates servo bandwidth (~100 ms).
    # Throttle no longer uses LPF: velocity integration with MAX_ACCEL/MAX_DECEL
    # provides the same rate-limiting effect inherently.
    # alpha = 1 - exp(-dt / tau) where dt ≈ 1/30 s at 30 Hz policy.
    # alpha=0.25 → tau≈108 ms.
    action_lpf_alpha_steer: float = 0.5     # steering EMA coefficient


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
        cfg.observation_space   = _tmp_vision.obs_size

        super().__init__(cfg, render_mode, **kwargs)

        # Rear throttle joint indices (velocity-controlled, RWD)
        self._rear_throttle_ids, _ = self.robot.find_joints([
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
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

        # Low-pass filtered steering [N] — persists across steps.
        self._filtered_steer = torch.zeros(self.num_envs, device=self.device)

        # Curriculum: current velocity ceiling (advances as training improves).
        self._vel_cap: float = INITIAL_VEL_CAP
        # Step counter at last vel_cap advancement — used to rate-limit to one
        # advancement per iteration (64 policy steps).
        self._last_curriculum_step: int = -10000

        # Rolling episode-return history for curriculum advancement check.
        self._ep_return_history: deque = deque(maxlen=VEL_HISTORY_LEN)
        # Per-env cumulative episode return (reset to 0 in _reset_idx).
        self._ep_return_buf = torch.zeros(self.num_envs, device=self.device)

        # Spawn locations tensor [S, 3] — (x, y, yaw_rad) per spawn point.
        _sp = cfg.spawn_locations
        self._spawn_locs = torch.tensor(
            [[x, y, yaw] for x, y, yaw in _sp], dtype=torch.float32, device=self.device
        )   # [S, 3]

        # Lap completion tracking
        self._start_pos    = torch.zeros(self.num_envs, 3, device=self.device)
        self._has_departed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Wall termination flag — set in _get_dones, read in _get_rewards
        self._wall_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

        # 2b. Load the wall boundary with the same scale and orientation as the track.
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

        # The walls are static colliders: enable collisions using triangle mesh geometry.
        # They remain invisible to policy LiDAR because LiDAR targets only TrackMergedMesh.
        wall_root = stage.GetPrimAtPath("/World/Walls")
        if wall_root.IsValid():
            for prim in Usd.PrimRange(wall_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                wall_collision = UsdPhysics.CollisionAPI.Apply(prim)
                wall_collision.GetCollisionEnabledAttr().Set(True)
                wall_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                wall_mesh_collision.GetApproximationAttr().Set("none")

        # Make walls invisible to all renderers (including the policy camera).
        # Walls exist only for contact-force termination — PhysX collision detection
        # is geometry-based and unaffected by UsdGeom visibility.
        if wall_root.IsValid():
            UsdGeom.Imageable(wall_root).MakeInvisible()

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
        # Wall termination uses per-link contact-force sensing.
        for link_name in WALL_CONTACT_LINKS:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/mushr_nano/{link_name}",
                history_length=0,
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

        # 8. Dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        # Rear throttle — velocity target (both rear wheels same speed for open diff)
        rear_vels = self._rear_wheel_ang_vel.unsqueeze(-1).expand(-1, 2)  # [N, 2]
        self.robot.set_joint_velocity_target(rear_vels, joint_ids=self._rear_throttle_ids)

        # Front steering — position target (same angle for simplified RWD model)
        steer_positions = self._steer_tan.unsqueeze(-1).expand(-1, 2)  # [N, 2]
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

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0:1] / self._vel_cap   # forward, normalised by curriculum cap

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

        # Lap completion speed bonus: larger when the lap is finished quickly
        robot_pos_xy    = self.robot.data.root_pos_w[:, :2]
        dist_from_start = torch.norm(robot_pos_xy - self._start_pos[:, :2], dim=-1)

        self._has_departed |= dist_from_start > self.cfg.departure_dist
        lap_completed = (dist_from_start < self.cfg.return_dist) & self._has_departed

        elapsed_fraction = self.episode_length_buf.float() / self.max_episode_length
        r_completion = self.cfg.completion_weight * lap_completed.float() * (1.0 - elapsed_fraction)

        self._has_departed[lap_completed] = False

        # Rear-wheel slip penalty — normalised by INITIAL_VEL_CAP (fixed, same reason as r_forward).
        rear_wheel_omega = self.robot.data.joint_vel[:, self._rear_throttle_ids]  # [N, 2] rad/s
        rear_wheel_vel   = rear_wheel_omega.mean(dim=-1) * WHEEL_RADIUS           # [N] m/s
        slip_speed       = torch.abs(rear_wheel_vel - lin_vel_b)                  # [N] m/s
        r_slip           = -self.cfg.slip_weight * slip_speed / INITIAL_VEL_CAP

        # Steering sharpness penalty: zero below the deadzone, linear above it.
        # Uses the post-LPF steering command (normalised [-1,1]) so it's insensitive
        # to the agent briefly commanding a high angle that the servo hasn't reached.
        steer_abs    = torch.abs(self._filtered_steer)                           # [N] in [0, 1]
        steer_excess = (steer_abs - self.cfg.steer_deadzone).clamp(min=0.0)     # [N] in [0, 0.8]
        steer_excess_norm = steer_excess / (1.0 - self.cfg.steer_deadzone)      # [N] in [0, 1]
        r_steer      = -self.cfg.steer_weight * steer_excess_norm

        # Slow-driving penalty: penalise every step the robot moves below the speed
        # threshold.  The timer is incremented here (same step as reward) so the
        # penalty and the termination counter stay in sync.
        is_slow = lin_vel_b < (SLOW_SPEED_FRACTION * self._vel_cap)
        self._slow_timer = torch.where(
            is_slow,
            self._slow_timer + 1,
            torch.zeros_like(self._slow_timer),
        )
        r_slow = -self.cfg.slow_penalty_weight * is_slow.float()

        # agent_stopped = -300.0 * (lin_vel_b < 0.1).float()

        total = r_distance + r_forward + r_slip + r_steer + r_slow + r_wall + r_completion
        # Accumulate per-env episode return for curriculum advancement check in _reset_idx.
        self._ep_return_buf += total
        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Wall collision: any lateral contact force above threshold on any robot link.
        # Ground contacts are predominantly vertical and therefore rejected.
        wall_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for sensor in self._wall_contact_sensors.values():
            net_forces_w = sensor.data.net_forces_w
            lateral_force = torch.linalg.vector_norm(net_forces_w[..., :2], dim=-1).reshape(self.num_envs, -1)
            has_contact = lateral_force.gt(WALL_CONTACT_FORCE_THRESH).any(dim=1)
            wall_terminated |= has_contact

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

        # Slow-driving termination: _slow_timer is updated in _get_rewards each step.
        # Terminate if the robot has been continuously below the speed threshold for
        # SLOW_TIMEOUT_STEPS steps (6 s at 30 Hz).
        slow_terminated = self._slow_timer >= SLOW_TIMEOUT_STEPS

        # Near-stop termination: immediate termination if speed drops below 5 % of vel_cap.
        # Grace period of 30 steps (~1 s) so the car can accelerate from a standing start.
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]
        stopped = (lin_vel_b < (0.05 * self._vel_cap)) & (self.episode_length_buf > 30)

        terminated = wall_terminated | flipped | slow_terminated | stopped
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # ── Curriculum advancement ────────────────────────────────────────────
        # Log the completed episode returns for terminated envs, then check if
        # the rolling mean has crossed the threshold for a vel_cap increase.
        for eid in env_ids:
            self._ep_return_history.append(float(self._ep_return_buf[eid].item()))
        self._ep_return_buf[env_ids] = 0.0

        # Rate-limit to one vel_cap advancement per iteration (64 policy steps).
        # _reset_idx is called once per policy step for each batch of resets, so
        # without this guard many resets in a single iteration would each trigger
        # the check and advance vel_cap multiple times.
        steps_since_advance = self.common_step_counter - self._last_curriculum_step
        if (
            steps_since_advance >= 64
            and len(self._ep_return_history) >= VEL_HISTORY_LEN // 2
            and self._vel_cap < VEL_CAP_MAX
        ):
            mean_ret = sum(self._ep_return_history) / len(self._ep_return_history)
            # Scale threshold proportionally with vel_cap so agents must actually drive
            # faster (not just survive at the old speed) to earn each advancement.
            # At vel_cap=2.0: threshold=500 (baseline); at vel_cap=4.0: threshold=1000; etc.
            scaled_threshold = VEL_CAP_REWARD_THRESHOLD * (self._vel_cap / INITIAL_VEL_CAP)
            if mean_ret >= scaled_threshold:
                self._vel_cap = min(self._vel_cap + VEL_CAP_STEP, VEL_CAP_MAX)
                self._last_curriculum_step = self.common_step_counter
                # Clear history so the next check uses only episodes collected at the
                # new (higher) vel_cap.  Without this, stale high-return episodes keep
                # the mean above the new threshold and advancement fires every iteration.
                # This is safe because the scaled threshold requires agents to actually
                # perform at the new speed level before advancing again.
                self._ep_return_history.clear()
                print(
                    f"[Curriculum] vel_cap → {self._vel_cap:.2f} m/s  "
                    f"(mean_ep_return={mean_ret:.1f}, threshold={scaled_threshold:.0f})"
                )
        # ─────────────────────────────────────────────────────────────────────

        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Pick a spawn location uniformly at random for each resetting env.
        spawn_idx = torch.randint(0, self._spawn_locs.shape[0], (n,), device=self.device)
        chosen    = self._spawn_locs[spawn_idx]   # [n, 3]: (x, y, yaw)

        # Override XY position (keep Z from default_root_state so suspension sits correctly)
        default_root_state[:, 0] = chosen[:, 0]
        default_root_state[:, 1] = chosen[:, 1]

        # Apply spawn yaw plus optional jitter
        yaw_jitter = (torch.rand(n, device=self.device) - 0.5) * (2.0 * self.cfg.init_yaw_jitter)
        yaw        = chosen[:, 2] + yaw_jitter
        zeros      = torch.zeros(n, device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw)
        default_root_state[:, 3:7] = dq

        # Zero initial velocity
        default_root_state[:, 7:] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._start_pos[env_ids]    = default_root_state[:, :3]
        self._has_departed[env_ids] = False

        # Clear velocity command and steering filter for reset envs.
        self._current_vel[env_ids]    = 0.0
        self._filtered_steer[env_ids] = 0.0

        # Clear slow-driving timer so the new episode starts fresh.
        self._slow_timer[env_ids] = 0

        # Reset joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
