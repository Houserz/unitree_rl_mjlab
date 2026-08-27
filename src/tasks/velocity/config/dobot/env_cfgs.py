"""Dobot Rover velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import get_dobot_robot_cfg
from src.assets.robots.dobot.dobot_constants import (
  DOBOT_ACTION_SCALE,
  DOBOT_FOOT_GEOM_NAMES,
  DOBOT_FOOT_SITE_NAMES,
)
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def _dobot_rover_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the shared Go2-style velocity task with the Dobot robot layer.

  The rough variant remains private in phase one. The registered flat task removes
  height scan observations and therefore keeps the 47-D actor contract.
  """
  cfg = make_velocity_env_cfg()

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
    history_length=4,
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

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
