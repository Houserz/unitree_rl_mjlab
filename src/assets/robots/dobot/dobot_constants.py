"""Dobot Rover constants and MJLab entity configuration."""

from dataclasses import replace
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

##
# Stable model interface.
##

DOBOT_JOINT_ORDER = (
  "joint_front_left_abad",
  "joint_front_left_thigh_pitch",
  "joint_front_left_calf_pitch",
  "joint_front_right_abad",
  "joint_front_right_thigh_pitch",
  "joint_front_right_calf_pitch",
  "joint_rear_left_abad",
  "joint_rear_left_thigh_pitch",
  "joint_rear_left_calf_pitch",
  "joint_rear_right_abad",
  "joint_rear_right_thigh_pitch",
  "joint_rear_right_calf_pitch",
)

DOBOT_FOOT_ORDER = ("FL", "FR", "RL", "RR")
DOBOT_FOOT_SITE_NAMES = DOBOT_FOOT_ORDER
DOBOT_FOOT_GEOM_NAMES = tuple(f"{name}_foot_collision" for name in DOBOT_FOOT_ORDER)
DOBOT_EFFORT_LIMITS = (23.0, 23.0, 55.0) * 4

DOBOT_TOTAL_MASS = 17.2352
DOBOT_INIT_ROOT_HEIGHT = 0.41
DOBOT_ACTION_SCALE = 0.25
DOBOT_PHYSICS_DT = 0.0025
DOBOT_DECIMATION = 8

##
# MJCF and assets.
##

DOBOT_XML: Path = SRC_PATH / "assets" / "robots" / "dobot" / "xmls" / "dobot.xml"
assert DOBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Load Dobot mesh assets for an in-memory MjSpec."""
  assets: dict[str, bytes] = {}
  update_assets(assets, DOBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Return a fresh Dobot MjSpec without terrain or XML-defined actuators."""
  spec = mujoco.MjSpec.from_file(str(DOBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuators.
##

# The active controller parameters come from the Isaac Gym Dobot task. Armature,
# passive damping and frictionloss use the tested Dobot MuJoCo deployment baseline;
# they are intentionally not claimed to be an exact dump of Isaac Gym DOF properties.
DOBOT_ACTUATOR_ABAD = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_abad",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=23.0,
  armature=0.000074,
  frictionloss=0.02,
)
DOBOT_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thigh_pitch",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=23.0,
  armature=0.000074,
  frictionloss=0.02,
)
DOBOT_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_pitch",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=55.0,
  armature=0.000074,
  frictionloss=0.02,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, DOBOT_INIT_ROOT_HEIGHT),
  joint_pos={
    ".*front.*_abad": 0.0,
    ".*front.*_thigh_pitch": 0.84,
    ".*front.*_calf_pitch": -1.30,
    ".*rear.*_abad": 0.0,
    ".*rear.*_thigh_pitch": 0.75,
    ".*rear.*_calf_pitch": -1.22,
  },
  joint_vel={".*": 0.0},
)

##
# Collision configuration.
##

# Enable every named primitive collision geom while excluding robot self-collision.
# The visual meshes are unnamed and therefore disabled by CollisionCfg. Contact
# impedance, friction and dimensionality follow the Dobot v2 MuJoCo terrain defaults;
# priority keeps those robot-side values effective against MJLab's terrain geom.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim=3,
  priority=1,
  friction=(1.8, 0.1, 0.01),
  solref=(0.02, 1.0),
  solimp=(0.8, 0.99, 0.001, 0.3, 1.0),
  contype=1,
  conaffinity=0,
)

##
# Final entity config.
##

DOBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DOBOT_ACTUATOR_ABAD,
    DOBOT_ACTUATOR_THIGH,
    DOBOT_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dobot_robot_cfg(
  *,
  stiffness: float | None = None,
  damping: float | None = None,
  identified: bool = False,
) -> EntityCfg:
  """Return a fresh Dobot Rover configuration, optionally with one PD pair.

  The default preserves the established 10/1 actuator baseline.  Supplying a
  pair creates fresh actuator configuration objects, so a PD experiment cannot
  mutate the baseline task through the module-level articulation singleton.
  ``identified=True`` instead loads the per-joint values and PD gains from
  motor_parameters.py for the separate identified task.
  """

  articulation = DOBOT_ARTICULATION
  if identified:
    from . import motor_parameters as motors
    from .identified_actuator import IdentifiedActuatorCfg

    if stiffness is not None or damping is not None:
      raise ValueError("Edit identified PD gains in motor_parameters.py")
    if set(motors.JOINT_PARAMETERS) != set(DOBOT_JOINT_ORDER):
      raise ValueError("motor_parameters must name exactly the 12 Dobot joints")
    if any(len(values) != 4 for values in motors.JOINT_PARAMETERS.values()):
      raise ValueError("Each joint needs armature, viscous damping, friction, bias")
    articulation = replace(
      DOBOT_ARTICULATION,
      actuators=tuple(
        IdentifiedActuatorCfg(
          target_names_expr=(name,),
          stiffness=motors.KP, damping=motors.KD,
          effort_limit=limit,
          armature=motors.JOINT_PARAMETERS[name][0],
          viscous_damping=motors.JOINT_PARAMETERS[name][1],
          frictionloss=motors.JOINT_PARAMETERS[name][2],
          encoder_bias=motors.JOINT_PARAMETERS[name][3],
          delay_steps=motors.DELAY_STEPS,
        )
        for name, limit in zip(DOBOT_JOINT_ORDER, DOBOT_EFFORT_LIMITS, strict=True)
      ),
    )
  if stiffness is not None or damping is not None:
    if stiffness is None or damping is None:
      raise ValueError("stiffness and damping must be supplied together")
    if stiffness <= 0.0 or damping < 0.0:
      raise ValueError("stiffness must be positive and damping must be nonnegative")
    articulation = replace(
      DOBOT_ARTICULATION,
      actuators=tuple(
        replace(actuator, stiffness=stiffness, damping=damping)
        for actuator in DOBOT_ARTICULATION.actuators
      ),
    )
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )
