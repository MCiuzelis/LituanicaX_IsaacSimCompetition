"""TurtleBot3 Burger — maze navigation environment (DirectRLEnv).

Observation space (74):
    [0:72]  normalized lidar distances (RayCaster, 72 beams, 360°, 5° step) — cone track only
    [72]    normalized forward linear velocity  (v / MAX_LIN_VEL)
    [73]    normalized yaw angular velocity     (ω / MAX_ANG_VEL)

Action space (2):
    [0]  linear  velocity command in [-1, 1]  → scaled to [-MAX_LIN_VEL, +MAX_LIN_VEL] m/s
    [1]  angular velocity command in [-1, 1]  → scaled to [-MAX_ANG_VEL, +MAX_ANG_VEL] rad/s

Reward:
    +alive_weight / max_episode_length              (per step, encourages survival)
    +forward_weight * (v_fwd / MAX_LIN_VEL)         (encourages forward motion)
    -clearance_weight * exp(-min_ray / 0.08)        (smooth cone-clearance penalty; ≈0 above 0.20 m)
    -smooth_weight * |ang_vel_norm|                 (penalizes erratic turning)
    -collision_penalty  on termination              (hard penalty for any termination — cone OR wall)
    -wall_penalty       on wall termination only    (additional large one-time penalty; walls = catastrophic)
    +completion_weight * (1 - elapsed/max_steps)    (bonus for completing a lap; faster = larger)
    -backward_weight * clamp(-v_fwd/MAX_LIN_VEL, 0)               (penalty for driving in reverse)
    -nospin_weight * |ang_vel_norm| * (1 - clamp(v_fwd/MAX_LIN_VEL, 0, 1))   (spin-in-place penalty)

Wall system (invisible-to-LiDAR boundary):
    walls_export.usd is loaded at /World/Walls with the same scale and orientation as the cone
    track.  No CollisionAPI is applied — the closed-loop wall mesh, if given a convexHull, would
    span the entire track interior at Z≈0 and compete with GroundPlaneCfg (same issue as the cone
    base mesh in TrackExport.usd).  Instead, all wall meshes are merged into /World/WallsMergedMesh
    and a dedicated wall_lidar RayCaster targets that mesh.  The policy observation LiDAR only
    targets /World/TrackMergedMesh (cones), so the walls are invisible to the policy.  When any
    wall_lidar beam < WALL_CONTACT_DIST the episode terminates, identical to cone termination.
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
TRACK_USD = "/home/matasciuzelis/Documents/turtlebot_maze_rl/TrackExport.usd"

# ---------------------------------------------------------------------------
# Wall USD path  (invisible boundary; no CollisionAPI — see module docstring)
# ---------------------------------------------------------------------------
WALL_USD = "/home/matasciuzelis/Documents/turtlebot_maze_rl/walls_export.usd"

# ---------------------------------------------------------------------------
# Physical constants — TurtleBot3 Burger
# ---------------------------------------------------------------------------
WHEEL_SEPARATION = 0.160   # m  (distance between left and right wheels)
WHEEL_RADIUS     = 0.033   # m  (TurtleBot3 Burger URDF spec)
HALF_SEPARATION  = WHEEL_SEPARATION / 2.0   # 0.080 m

MAX_LIN_VEL = 0.4   # m/s
MAX_ANG_VEL = 2   # rad/s

LIDAR_MAX_RANGE = 2.5   # m  (LDS-01 specification)
LIDAR_NUM_BEAMS = 36    # one beam every 10°

# Termination distance thresholds (same value — both represent approx. robot half-width)
COLLISION_DIST    = 0.13   # m — cone proximity termination (policy LiDAR)
WALL_CONTACT_DIST = 0.1   # m — wall proximity termination (dedicated wall LiDAR)


# ===========================================================================
# Environment Configuration
# ===========================================================================

@configclass
class TurtleBotMazeEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # Core env parameters
    # ------------------------------------------------------------------
    decimation: int = 4                # policy at ~30 Hz (sim 120 Hz)
    # 900 s × 30 Hz = 27 000 steps — long enough for ~18 laps at 0.4 m/s.
    # Avoid large values here: alive_weight is normalised by max_episode_length,
    # so episode_length_s = 12 000 s would make alive_weight ≈ 0 per step.
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
        num_envs=1000,
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
            pos=(2.989, 0.9613 + 5, 0.03),  # small clearance above ground plane to avoid PhysX contact jitter
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
    # Policy LiDAR — sees only the cone track (TrackMergedMesh)
    # ------------------------------------------------------------------
    lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/a__namespace_base_scan",
        mesh_prim_paths=["/World/Track"],   # placeholder; overwritten in _setup_scene
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
    # Wall LiDAR — sees only the wall boundary (WallsMergedMesh).
    # NOT added to observations; used solely for termination detection.
    # The policy cannot observe walls — it learns to avoid them through
    # the termination penalty alone.
    # ------------------------------------------------------------------
    wall_lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/a__namespace_base_scan",
        mesh_prim_paths=["/World/Walls"],   # placeholder; overwritten in _setup_scene
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-0.01, 0.01),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / LIDAR_NUM_BEAMS,
        ),
        max_distance=LIDAR_MAX_RANGE,
        ray_alignment="base",
        debug_vis=False,
    )

    # ------------------------------------------------------------------
    # Reward weights
    # ------------------------------------------------------------------
    alive_weight:       float = 1.0
    forward_weight:     float = 2.5
    clearance_weight:   float = 0.5   # smooth exponential cone-clearance penalty
    smooth_weight:      float = 0.025
    collision_penalty:  float = 100    # fires on any termination (cone OR wall)
    # Wall contact is treated as catastrophic — the robot must navigate by cones alone
    # and should NEVER touch the boundary.  This penalty fires on top of collision_penalty
    # for wall-only terminations: total wall penalty = collision_penalty + wall_penalty.
    # At wall_penalty=200 and typical episode reward ~3 000, wall death costs ~68× more
    # than a cone death (5.0) — effectively teaching the agent that walls must be avoided.
    wall_penalty:       float = 100  # extra penalty added on wall-only termination

    # Lap completion speed reward.
    # A bonus of completion_weight * (1 - elapsed/max_steps) is awarded each time
    # the robot returns within return_dist metres of its spawn point after having
    # first travelled at least departure_dist metres away.  Completing the lap
    # in fewer steps yields a larger bonus, incentivising maximum speed.
    completion_weight:  float = 8000  # max bonus magnitude (awarded on fastest possible lap)
    departure_dist:     float = 0.5   # metres from spawn before the robot is considered "departed"
    return_dist:        float = 0.4   # metres from spawn that counts as completing the lap

    # Anti-reversal penalties
    backward_weight:    float = 3.0   # penalty for any negative linear velocity (driving in reverse)
    # Spin-in-place penalty: |ang_vel_norm| * (1 - clamp(lin_vel_norm, 0, 1))
    # → maximum when spinning with zero forward speed (U-turn in place)
    # → near-zero during legitimate cornering at speed
    # → zero when driving straight
    nospin_weight:      float = 0.2


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

        # Wall termination flag — set in _get_dones, read in _get_rewards to apply wall_penalty
        self._wall_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # 1. Spawn robot articulation (one per environment via prim_path wildcard)
        self.robot = Articulation(self.cfg.robot_cfg)

        # 2. Load the traffic-cone track once as a global prim (not inside env_0).
        #    Scale: X/Z = 0.75 × 0.01 (cm→m) = 0.0075; Y = 1.25 × 0.01 = 0.0125
        #    Orientation: +90° around X cancels the -90° baked into the USD export.
        track_cfg = sim_utils.UsdFileCfg(
            usd_path=TRACK_USD,
            scale=(0.0075, 0.0125, 0.0075),
        )
        track_cfg.func(
            "/World/Track",
            track_cfg,
            translation=(0.0, 5.0, 0.0),
            orientation=(0.70711, 0.70711, 0.0, 0.0),   # +90° around X to cancel Y-up baked rotation
        )

        # 2b. Load the wall boundary with the same scale and orientation as the track.
        #     No CollisionAPI is applied here — see module docstring for why.
        wall_cfg = sim_utils.UsdFileCfg(
            usd_path=WALL_USD,
            scale=(0.0075, 0.0125, 0.0075),
        )
        wall_cfg.func(
            "/World/Walls",
            wall_cfg,
            translation=(0.0, 5.0, 0.0),
            orientation=(0.70711, 0.70711, 0.0, 0.0),
        )

        # 3. Merge all TRACK meshes into /World/TrackMergedMesh for the policy LiDAR.
        #    No physics CollisionAPI on any track mesh — TrackExport.usd contains a
        #    flat ground-level base surface that would fight the GroundPlaneCfg plane.
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, Vt
        stage = omni.usd.get_context().get_stage()
        xform_cache = UsdGeom.XformCache()

        def merge_meshes(root_prim, out_path: str) -> None:
            """Merge all UsdGeom.Mesh prims under root_prim into a single world-space mesh at out_path."""
            pts_acc: list[Gf.Vec3f] = []
            fvc_acc: list[int]      = []
            fvi_acc: list[int]      = []
            offset = 0
            for p in Usd.PrimRange(root_prim):
                if p.IsInstanceable():
                    p.SetInstanceable(False)
                if not p.IsA(UsdGeom.Mesh):
                    continue
                mesh_geom = UsdGeom.Mesh(p)
                pts = mesh_geom.GetPointsAttr().Get()
                fvc = mesh_geom.GetFaceVertexCountsAttr().Get()
                fvi = mesh_geom.GetFaceVertexIndicesAttr().Get()
                if not (pts and fvc and fvi):
                    continue
                world_xform = xform_cache.GetLocalToWorldTransform(p)
                for pt in pts:
                    wp = world_xform.Transform(Gf.Vec3d(pt[0], pt[1], pt[2]))
                    pts_acc.append(Gf.Vec3f(wp[0], wp[1], wp[2]))
                fvc_acc.extend(list(fvc))
                fvi_acc.extend([i + offset for i in fvi])
                offset += len(pts)
            merged = UsdGeom.Mesh.Define(stage, out_path)
            merged.GetPointsAttr().Set(Vt.Vec3fArray(pts_acc))
            merged.GetFaceVertexCountsAttr().Set(Vt.IntArray(fvc_acc))
            merged.GetFaceVertexIndicesAttr().Set(Vt.IntArray(fvi_acc))

        track_root = stage.GetPrimAtPath("/World/Track")
        merge_meshes(track_root, "/World/TrackMergedMesh")
        self.cfg.lidar_cfg.mesh_prim_paths = ["/World/TrackMergedMesh"]

        # 3b. Merge all WALL meshes into /World/WallsMergedMesh for the wall LiDAR.
        #     No CollisionAPI — the wall mesh is a closed-loop boundary; its convexHull
        #     would span the track interior at Z≈0 and fight the ground plane (same bug
        #     as the cone track).  Proximity-based termination via RayCaster is used
        #     instead, identical to how cone collision is handled.
        wall_root = stage.GetPrimAtPath("/World/Walls")
        merge_meshes(wall_root, "/World/WallsMergedMesh")
        self.cfg.wall_lidar_cfg.mesh_prim_paths = ["/World/WallsMergedMesh"]

        # 4. Physics ground plane — the cone track has no floor mesh, so we add
        #    an infinite plane at Z=0 for the robot wheels to rest on.
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        # 5. Register RayCaster sensors (must be created AFTER mesh_prim_paths are set).
        self.lidar      = RayCaster(self.cfg.lidar_cfg)
        self.wall_lidar = RayCaster(self.cfg.wall_lidar_cfg)

        # 6. Clone environments (robot only; track and walls are global)
        self.scene.clone_environments(copy_from_source=False)

        # 7. Register assets / sensors with the scene manager
        self.scene.articulations["robot"]       = self.robot
        self.scene.sensors["lidar"]             = self.lidar
        self.scene.sensors["wall_lidar"]        = self.wall_lidar

        # 8. Dome light
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

        lidar_obs = torch.clamp(ray_dist / LIDAR_MAX_RANGE, 0.0, 1.0)  # [N, 72]

        # Body-frame velocities
        lin_vel_b = self.robot.data.root_lin_vel_b[:, 0:1] / MAX_LIN_VEL   # forward
        ang_vel_b = self.robot.data.root_ang_vel_b[:, 2:3] / MAX_ANG_VEL   # yaw

        obs = torch.cat([lidar_obs, lin_vel_b, ang_vel_b], dim=-1)  # [N, 74]
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # --- Re-compute lidar distances (already computed in _get_dones first) ---
        ray_hits  = self.lidar.data.ray_hits_w
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_dist   = torch.norm(ray_hits - sensor_pos, dim=-1)   # [N, 72]
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

        # Smooth cone-clearance penalty: ~-0.5 at d=0, ~-0.003 at d=0.25 m, negligible above 0.30 m
        r_clearance = -self.cfg.clearance_weight * torch.exp(-min_dist / 0.08)

        # Angular smoothness penalty
        r_smooth = -self.cfg.smooth_weight * torch.abs(ang_vel_b / MAX_ANG_VEL)

        # Base termination penalty — fires for ANY termination (cone or wall)
        r_collision = -self.cfg.collision_penalty * self.reset_terminated.float()

        # Additional wall-only penalty — makes wall contact catastrophic relative to cone contact.
        # Total wall penalty = collision_penalty + wall_penalty ≈ 205.0
        # Total cone penalty = collision_penalty                 ≈   5.0
        r_wall = -self.cfg.wall_penalty * self._wall_terminated.float()

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

        return r_alive + r_forward + r_clearance + r_smooth + r_collision + r_wall + r_completion + r_backward + r_nospin

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Cone proximity (policy LiDAR) ---
        ray_hits   = self.lidar.data.ray_hits_w
        sensor_pos = self.lidar.data.pos_w.unsqueeze(1)
        ray_dist   = torch.norm(ray_hits - sensor_pos, dim=-1)
        cone_terminated = ray_dist.min(dim=-1).values < COLLISION_DIST

        # --- Wall proximity (wall LiDAR, invisible to policy) ---
        wall_hits       = self.wall_lidar.data.ray_hits_w
        wall_sensor_pos = self.wall_lidar.data.pos_w.unsqueeze(1)
        wall_dist       = torch.norm(wall_hits - wall_sensor_pos, dim=-1)
        wall_terminated = wall_dist.min(dim=-1).values < WALL_CONTACT_DIST

        # Cache for _get_rewards so the large wall_penalty can be applied separately
        self._wall_terminated = wall_terminated

        terminated = cone_terminated | wall_terminated
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
