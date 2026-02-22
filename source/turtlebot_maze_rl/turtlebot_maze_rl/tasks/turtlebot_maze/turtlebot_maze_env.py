"""TurtleBot3 Burger — maze navigation environment (DirectRLEnv).

Observation space (38):
    [0:36]  normalized lidar distances (RayCaster, 36 beams, 360°, 10° step)
    [36]    normalized forward linear velocity  (v / 0.22)
    [37]    normalized yaw angular velocity     (ω / 2.84)

Action space (2):
    [0]  linear  velocity command in [-1, 1]  → scaled to [-0.22, +0.22] m/s
    [1]  angular velocity command in [-1, 1]  → scaled to [-2.84, +2.84] rad/s

Reward:
    +alive_weight / max_episode_length              (per step, encourages survival)
    +forward_weight * (v_fwd / 0.22)                (encourages forward motion)
    -clearance_weight * exp(-min_ray / 0.08)        (smooth wall-clearance penalty; ≈0 above 0.20 m)
    -smooth_weight * |ang_vel_norm|                                     (penalizes erratic turning)
    -collision_penalty  on termination                                  (hard penalty for wall collision)
    +completion_weight * (1 - elapsed/max_steps)                        (bonus for completing a lap; faster = larger)
    -backward_weight * clamp(-v_fwd/0.22, 0)                           (penalty for driving in reverse)
    -nospin_weight * |ang_vel_norm| * (1 - clamp(v_fwd/0.22, 0, 1))   (penalty for spinning in place; zero during cornering at speed)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

# ---------------------------------------------------------------------------
# Robot USD path
# ---------------------------------------------------------------------------
TURTLEBOT3_BURGER_USD = (
    "/home/matasciuzelis/Documents/turtlebot3/turtlebot3_description"
    "/urdf/turtlebot3_burger/turtlebot3_burger.usd"
)

# ---------------------------------------------------------------------------
# Track USD path
# ---------------------------------------------------------------------------
TRACK_USD = "/home/matasciuzelis/Documents/turtlebot_maze_rl/Track.usd"

# ---------------------------------------------------------------------------
# Physical constants — TurtleBot3 Burger
# ---------------------------------------------------------------------------
WHEEL_SEPARATION = 0.160   # m  (distance between left and right wheels)
WHEEL_RADIUS     = 0.033   # m  (TurtleBot3 Burger URDF spec)
HALF_SEPARATION  = WHEEL_SEPARATION / 2.0   # 0.080 m

MAX_LIN_VEL = 0.3   # m/s
MAX_ANG_VEL = 2   # rad/s

LIDAR_MAX_RANGE = 3.5   # m  (LDS-01 specification)
LIDAR_NUM_BEAMS = 36    # one beam every 10°

# Collision threshold
COLLISION_DIST  = 0.12   # m — hard termination


# ===========================================================================
# Environment Configuration
# ===========================================================================

@configclass
class TurtleBotMazeEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # Core env parameters
    # ------------------------------------------------------------------
    decimation: int = 4                # policy at ~30 Hz (sim 120 Hz)
    episode_length_s: float = 900.0

    action_space:      int = 2                        # [lin_vel_norm, ang_vel_norm]
    observation_space: int = LIDAR_NUM_BEAMS + 2      # lidar beams + lin_vel + ang_vel
    state_space:       int = 0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    # env_spacing=0.0 keeps all env origins at (0,0,0) so every robot spawns
    # inside the single global Track mesh. The RayCaster BVH is baked from
    # that mesh and is shared across environments, which means all robots
    # receive correct LiDAR readings only when they are within the track bounds.
    # Isaac Lab isolates each environment via collision groups, so robots do not
    # physically interact with each other.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Robot articulation
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=TURTLEBOT3_BURGER_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            # Position matched to Isaac Sim scene (robot spawn inside the track)
            pos=(-6.4105, 3.24511, -0.05931),
            joint_pos={
                "a__namespace_wheel_left_joint":  0.0,
                "a__namespace_wheel_right_joint": 0.0,
            },
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[
                    "a__namespace_wheel_left_joint",
                    "a__namespace_wheel_right_joint",
                ],
                stiffness=0.0,
                damping=100.0,           # PhysX implicit integrator — unconditionally stable
                velocity_limit_sim=15.0, # headroom above max ~13.5 rad/s (0.22/0.033 + turning)
            )
        },
    )

    # ------------------------------------------------------------------
    # Lidar sensor
    # ------------------------------------------------------------------
    lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/a__namespace_base_scan",
        mesh_prim_paths=["/World/Track/Track/Shell1/Mesh"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-0.01, 0.01),                    # single horizontal plane
            horizontal_fov_range=(-180.0, 180.0),                # full 360°
            horizontal_res=360.0 / LIDAR_NUM_BEAMS,              # degrees per beam
        ),
        max_distance=LIDAR_MAX_RANGE,
        ray_alignment="base",
        debug_vis=False,  # enable only for single-env visual checks (crashes with 0 hits)
    )

    # ------------------------------------------------------------------
    # Reward weights
    # ------------------------------------------------------------------
    alive_weight:       float = 1.0
    forward_weight:     float = 2.5
    clearance_weight:   float = 0.5   # smooth exponential wall penalty
    smooth_weight:      float = 0.01
    collision_penalty:  float = 5.0

    # Lap completion speed reward.
    # A bonus of completion_weight * (1 - elapsed/max_steps) is awarded each time
    # the robot returns within return_dist metres of its spawn point after having
    # first travelled at least departure_dist metres away.  Completing the lap
    # in fewer steps yields a larger bonus, incentivising maximum speed.
    completion_weight:  float = 20.0  # max bonus magnitude (awarded on fastest possible lap)
    departure_dist:     float = 0.5   # metres from spawn before the robot is considered "departed"
    return_dist:        float = 0.4   # metres from spawn that counts as completing the lap

    # Anti-reversal penalties
    backward_weight:    float = 3.0   # penalty for any negative linear velocity (driving in reverse)
    # Spin-in-place penalty: |ang_vel_norm| * (1 - clamp(lin_vel_norm, 0, 1))
    # → maximum when spinning with zero forward speed (U-turn in place)
    # → near-zero during legitimate cornering at speed
    # → zero when driving straight
    nospin_weight:      float = 2.0


# ===========================================================================
# Environment Class
# ===========================================================================

class TurtleBotMazeEnv(DirectRLEnv):
    cfg: TurtleBotMazeEnvCfg

    def __init__(self, cfg: TurtleBotMazeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Wheel joint indices (populated after super().__init__ → _setup_scene)
        self._wheel_joint_ids, _ = self.robot.find_joints([
            "a__namespace_wheel_left_joint",
            "a__namespace_wheel_right_joint",
        ])

        # Buffers for differential-drive pre-computation
        self._left_wheel_vel  = torch.zeros(self.num_envs, device=self.device)
        self._right_wheel_vel = torch.zeros(self.num_envs, device=self.device)

        # Lap completion tracking
        self._start_pos    = torch.zeros(self.num_envs, 3, device=self.device)
        self._has_departed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # 1. Spawn robot articulation (one per environment via prim_path wildcard)
        self.robot = Articulation(self.cfg.robot_cfg)

        # 2. Load the track once as a global prim (not inside env_0).
        #    The RayCaster bakes exactly one mesh into a Warp BVH; keeping the track
        #    global ensures the RayCaster path is always a valid, concrete prim.
        #    Scale: 1.2 (user) × 0.01 (cm→m unit conversion) = 0.012
        #    Orientation: 90° around X → quaternion (w, x, y, z)
        track_cfg = sim_utils.UsdFileCfg(
            usd_path=TRACK_USD,
            scale=(0.012, 0.012, 0.012),
        )
        track_cfg.func(
            "/World/Track",
            track_cfg,
            translation=(-7.51, -0.00375, -0.09265),
            orientation=(0.70711, 0.70711, 0.0, 0.0),
        )

        # 3. Apply triangle-mesh collision to the track mesh.
        #    USD files from STEP import often have instanceable=True, making sub-prims
        #    read-only. Disable instancing first, then apply CollisionAPI.
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        track_root = stage.GetPrimAtPath("/World/Track")
        for p in Usd.PrimRange(track_root):
            if p.IsInstanceable():
                p.SetInstanceable(False)
        mesh_prim = stage.GetPrimAtPath("/World/Track/Track/Shell1/Mesh")
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set("none")  # exact triangle mesh

        # 4. Register RayCaster lidar sensor
        self.lidar = RayCaster(self.cfg.lidar_cfg)

        # 5. Clone environments (robot only; track is global)
        self.scene.clone_environments(copy_from_source=False)

        # 6. Register assets / sensors with the scene manager
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lidar"] = self.lidar

        # 7. Dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert normalised [lin_vel, ang_vel] actions → wheel velocities (rad/s)."""
        lin_vel = actions[:, 0] * MAX_LIN_VEL   # m/s
        ang_vel = actions[:, 1] * MAX_ANG_VEL   # rad/s

        # Differential-drive inverse kinematics
        self._left_wheel_vel  = (lin_vel - ang_vel * HALF_SEPARATION) / WHEEL_RADIUS
        self._right_wheel_vel = (lin_vel + ang_vel * HALF_SEPARATION) / WHEEL_RADIUS

    def _apply_action(self) -> None:
        wheel_vels = torch.stack(
            [self._left_wheel_vel, self._right_wheel_vel], dim=-1
        )  # [num_envs, 2]
        self.robot.set_joint_velocity_target(wheel_vels, joint_ids=self._wheel_joint_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Ray hit positions in world frame: [num_envs, num_beams, 3]
        # Sensor origin in world frame:     [num_envs, 3]
        ray_hits = self.lidar.data.ray_hits_w                    # [N, B, 3]
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)          # [N, 1, 3]
        ray_dist = torch.norm(ray_hits - sensor_pos, dim=-1)     # [N, B]

        lidar_obs = torch.clamp(ray_dist / LIDAR_MAX_RANGE, 0.0, 1.0)  # [N, 36]

        # Body-frame velocities
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0:1] / MAX_LIN_VEL   # forward
        ang_vel_b = self.robot.data.root_ang_vel_b[:, 2:3] / MAX_ANG_VEL   # yaw

        obs = torch.cat([lidar_obs, lin_vel_b, ang_vel_b], dim=-1)  # [N, 38]
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # --- Re-compute lidar distances (already computed in _get_dones first) ---
        ray_hits  = self.lidar.data.ray_hits_w
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_dist   = torch.norm(ray_hits - sensor_pos, dim=-1)   # [N, 36]
        min_dist   = ray_dist.min(dim=-1).values                  # [N]

        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0]   # forward (m/s)
        ang_vel_b = self.robot.data.root_ang_vel_b[:, 2]   # yaw (rad/s)

        # Alive reward (normalised per step)
        r_alive = torch.full(
            (self.num_envs,),
            self.cfg.alive_weight / self.max_episode_length,
            device=self.device,
        )

        # Forward progress reward
        r_forward = self.cfg.forward_weight * (lin_vel_b / MAX_LIN_VEL)

        # Smooth wall-clearance penalty: ~-0.5 at d=0, ~-0.003 at d=0.25 m, negligible above 0.30 m
        r_clearance = -self.cfg.clearance_weight * torch.exp(-min_dist / 0.08)

        # Angular smoothness penalty
        r_smooth = -self.cfg.smooth_weight * torch.abs(ang_vel_b / MAX_ANG_VEL)

        # Collision / termination penalty
        r_collision = -self.cfg.collision_penalty * self.reset_terminated.float()

        # Lap completion speed reward.
        # Step 1: measure XY distance from each robot's spawn point.
        robot_pos_xy   = self.robot.data.root_pos_w[:, :2]                         # [N, 2]
        dist_from_start = torch.norm(robot_pos_xy - self._start_pos[:, :2], dim=-1) # [N]

        # Step 2: mark robots that have left the start area.
        self._has_departed |= dist_from_start > self.cfg.departure_dist

        # Step 3: a lap is complete when the robot is back near the spawn point
        # AND had previously departed.
        lap_completed = (dist_from_start < self.cfg.return_dist) & self._has_departed

        # Step 4: scale bonus by remaining time — faster laps earn more reward.
        elapsed_fraction = self.episode_length_buf.float() / self.max_episode_length
        r_completion = self.cfg.completion_weight * lap_completed.float() * (1.0 - elapsed_fraction)

        # Reset departure flag so multi-lap bonuses can accumulate within one episode.
        self._has_departed[lap_completed] = False

        # Backward driving penalty: fires whenever linear velocity is negative.
        r_backward = -self.cfg.backward_weight * torch.clamp(-lin_vel_b / MAX_LIN_VEL, min=0.0)

        # Spin-in-place penalty: scales with angular velocity but is suppressed when the
        # robot is also moving forward at speed, so it does not penalise normal cornering.
        forward_fraction = torch.clamp(lin_vel_b / MAX_LIN_VEL, min=0.0, max=1.0)
        r_nospin = -self.cfg.nospin_weight * torch.abs(ang_vel_b / MAX_ANG_VEL) * (1.0 - forward_fraction)

        return r_alive + r_forward + r_clearance + r_smooth + r_collision + r_completion + r_backward #+ r_nospin

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ray_hits  = self.lidar.data.ray_hits_w
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_dist   = torch.norm(ray_hits - sensor_pos, dim=-1)
        min_dist   = ray_dist.min(dim=-1).values

        terminated = min_dist < COLLISION_DIST
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Default root state + environment origin offset
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Small yaw jitter (±0.2 rad) to diversify initial conditions
        yaw_jitter = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.4  # ±0.2 rad
        zeros      = torch.zeros(len(env_ids), device=self.device)
        dq = quat_from_euler_xyz(zeros, zeros, yaw_jitter)  # [N, 4] (w, x, y, z)
        # Compose with default orientation (simple overwrite — default is identity)
        default_root_state[:, 3:7] = dq

        # Zero initial velocity
        default_root_state[:, 7:] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Record spawn positions and reset lap-tracking state for reset envs
        self._start_pos[env_ids]    = default_root_state[:, :3]
        self._has_departed[env_ids] = False

        # Reset joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
