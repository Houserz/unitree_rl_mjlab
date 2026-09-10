"""Dobot Rover velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg as MjlabUniformVelocityCommandCfg

from src.assets.robots import get_dobot_robot_cfg
from src.assets.robots.dobot.dobot_constants import (
  DOBOT_ACTION_SCALE,
  DOBOT_DECIMATION,
  DOBOT_FOOT_GEOM_NAMES,
  DOBOT_FOOT_SITE_NAMES,
  DOBOT_PHYSICS_DT,
)
from src.tasks.velocity.mdp import (
  StratifiedVelocityCommandCfg,
  UniformVelocityCommandCfg as LocalUniformVelocityCommandCfg,
  VelocityCommandBucket,
)
import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def _dobot_rover_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the shared Go2-style velocity task with the Dobot robot layer.

  The rough variant remains private in phase one. The registered flat task removes
  height scan observations and therefore keeps the 47-D actor contract.
  """
  cfg = make_velocity_env_cfg()

  # Case B contact timing: retain a 50 Hz policy period while resolving each
  # policy interval with eight 2.5 ms physics substeps.
  cfg.sim.mujoco.timestep = DOBOT_PHYSICS_DT
  cfg.decimation = DOBOT_DECIMATION

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_dobot_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "link_trunk"

  # ContactMatch resolves patterns in MJCF natural order. Dobot's MJCF, sites, and
  # these constants are deliberately all ordered FL, FR, RL, RR.
  site_names = DOBOT_FOOT_SITE_NAMES
  geom_names = DOBOT_FOOT_GEOM_NAMES

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    # Cover one complete policy interval. Keep synchronized with decimation.
    history_length=DOBOT_DECIMATION,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DOBOT_ACTION_SCALE
  joint_pos_action.use_default_offset = True

  cfg.viewer.body_name = "link_trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  # Keep the effective foot friction equal to the Dobot v2 MuJoCo default instead
  # of applying the shared Unitree task's 0.3--1.6 startup randomization.
  cfg.events["foot_friction"].params["ranges"] = (1.8, 1.8)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("link_trunk",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*_abad": 0.05,
    r".*_thigh_pitch": 0.1,
    r".*_calf_pitch": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_abad": 0.15,
    r".*_thigh_pitch": 0.35,
    r".*_calf_pitch": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_abad": 0.15,
    r".*_thigh_pitch": 0.35,
    r".*_calf_pitch": 0.5,
  }

  # Natural contact order is FL, FR, RL, RR. This pairs FL+RR and FR+RL.
  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (
    "link_trunk",
  )
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("link_trunk",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  # Phase one intentionally keeps the Go2 task semantic: any non-foot ground
  # contact over 10 N terminates the episode. This is not an Isaac collision mapping.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def dobot_rover_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the phase-one 47-D Dobot Rover flat-terrain task."""
  cfg = _dobot_rover_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, MjlabUniformVelocityCommandCfg)

  if not play:
    # Set the initial distribution explicitly. The curriculum runs during reset,
    # but its step-0 stage is not active while common_step_counter is still zero.
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

    # Grow one command axis at a time. Fixed-command probes showed that the base
    # gait is ready by roughly iteration 500, while a single full-range jump at
    # iteration 5000 drives the policy toward a conservative standing solution.
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
      },
      {"step": 12000, "lin_vel_x": (-0.5, 1.2), "lin_vel_y": (-0.5, 0.5)},
      {"step": 24000, "lin_vel_x": (-0.5, 1.4), "lin_vel_y": (-0.5, 0.5)},
      {"step": 36000, "lin_vel_x": (-0.5, 1.6), "lin_vel_y": (-0.5, 0.5)},
      {"step": 48000, "lin_vel_x": (-0.5, 1.8), "lin_vel_y": (-0.5, 0.5)},
      {"step": 60000, "lin_vel_x": (-0.5, 2.0), "lin_vel_y": (-0.5, 0.5)},
      {"step": 84000, "lin_vel_x": (-0.75, 2.0), "lin_vel_y": (-0.5, 0.5)},
      {"step": 96000, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-0.5, 0.5)},
      {"step": 108000, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-0.75, 0.75)},
      {"step": 120000, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
    ]
  else:
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def dobot_rover_flat_kp25_kd13_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """From-scratch 25/1.3 baseline; retain lateral range at ±0.75.

  The 4501-iteration historical run never reached the final ±1.0 stage.
  Remove that unused stage in v1 so a longer run cannot enter it silently.
  """

  cfg = dobot_rover_flat_env_cfg(play=play)
  cfg.scene.entities = {
    "robot": get_dobot_robot_cfg(stiffness=25.0, damping=1.3)
  }
  if not play:
    cfg.scene.num_envs = 4096
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      stage for stage in stages if stage["step"] < 120000
    ]
  return cfg


def dobot_rover_flat_kp25_kd13_lateral_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Kp25 continuation curriculum with dedicated forward-retention buckets.

  The policy is warm-started from the Kp25 ``model_4500`` checkpoint, whose
  training range reached ``±0.75``.  New optimizer state starts at that
  distribution, then widens only the dedicated lateral bucket every 500 PPO
  iterations.  Dedicated medium/high-forward buckets are never diluted.
  """

  cfg = dobot_rover_flat_kp25_kd13_env_cfg(play=play)
  initial_lateral = 1.0 if play else 0.75
  cfg.commands["twist"] = StratifiedVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=True,
    heading_control_stiffness=0.5,
    ranges=LocalUniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-3.141592653589793, 3.141592653589793),
    ),
    buckets=(
      # Forward retention: 50% of command samples remain nearly straight.
      VelocityCommandBucket(
        weight=0.30,
        lin_vel_x=(1.4, 2.0),
        lin_vel_y=(-0.10, 0.10),
        ang_vel_z=(-0.15, 0.15),
        rel_heading_envs=0.10,
      ),
      VelocityCommandBucket(
        weight=0.20,
        lin_vel_x=(0.5, 1.4),
        lin_vel_y=(-0.10, 0.10),
        ang_vel_z=(-0.20, 0.20),
        rel_heading_envs=0.20,
      ),
      # This is the only bucket widened by the lateral curriculum below.
      VelocityCommandBucket(
        weight=0.20,
        lin_vel_x=(-0.40, 0.80),
        lin_vel_y=(-initial_lateral, initial_lateral),
        ang_vel_z=(-0.30, 0.30),
        rel_heading_envs=0.20,
      ),
      VelocityCommandBucket(
        weight=0.10,
        lin_vel_x=(0.50, 1.40),
        lin_vel_y=(-0.35, 0.35),
        ang_vel_z=(-0.60, 0.60),
        rel_heading_envs=0.75,
      ),
      VelocityCommandBucket(
        weight=0.10,
        lin_vel_x=(-1.0, -0.25),
        lin_vel_y=(-0.20, 0.20),
        ang_vel_z=(-0.40, 0.40),
        rel_heading_envs=0.25,
      ),
      VelocityCommandBucket(
        weight=0.10,
        lin_vel_x=(-0.25, 1.0),
        lin_vel_y=(-0.15, 0.15),
        ang_vel_z=(-1.0, 1.0),
        rel_heading_envs=1.0,
      ),
    ),
  )
  if not play:
    # ``common_step_counter`` advances once per environment step: 12,000 steps
    # equal 500 PPO iterations with this run's 24 rollout steps per iteration.
    cfg.curriculum = {
      "lateral_range": CurriculumTermCfg(
        func=mdp.stratified_lateral_range,
        params={
          "command_name": "twist",
          "lateral_bucket_index": 2,
          "stages": [
            {"step": 0, "lin_vel_y": (-0.75, 0.75)},
            {"step": 12000, "lin_vel_y": (-0.85, 0.85)},
            {"step": 24000, "lin_vel_y": (-0.95, 0.95)},
            {"step": 36000, "lin_vel_y": (-1.0, 1.0)},
          ],
        },
      )
    }
  else:
    cfg.curriculum = {}
  return cfg
