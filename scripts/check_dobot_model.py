"""Fail-closed static and optional one-step checks for the Dobot Rover model."""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from src.assets.robots.dobot.dobot_constants import (
  DOBOT_ACTION_SCALE,
  DOBOT_DECIMATION,
  DOBOT_EFFORT_LIMITS,
  DOBOT_FOOT_GEOM_NAMES,
  DOBOT_FOOT_SITE_NAMES,
  DOBOT_INIT_ROOT_HEIGHT,
  DOBOT_JOINT_ORDER,
  DOBOT_PHYSICS_DT,
  DOBOT_TOTAL_MASS,
  get_dobot_robot_cfg,
)
from src.tasks.velocity.config.dobot.env_cfgs import dobot_rover_flat_env_cfg

EXPECTED_COLLISION_GEOMS = (
  "trunk_collision",
  "FL_hip_collision",
  "FL_thigh_collision",
  "FL_calf_collision",
  "FL_foot_collision",
  "FR_hip_collision",
  "FR_thigh_collision",
  "FR_calf_collision",
  "FR_foot_collision",
  "RL_hip_collision",
  "RL_thigh_collision",
  "RL_calf_collision",
  "RL_foot_collision",
  "RR_hip_collision",
  "RR_thigh_collision",
  "RR_calf_collision",
  "RR_foot_collision",
)


def _assert_close(actual: float, expected: float, name: str, atol: float = 1e-8) -> None:
  if not math.isclose(actual, expected, abs_tol=atol, rel_tol=0.0):
    raise AssertionError(f"{name}: expected {expected}, got {actual}")


def _check_compiled_model() -> None:
  robot = get_dobot_robot_cfg().build()
  model = robot.spec.compile()

  assert robot.joint_names == DOBOT_JOINT_ORDER
  assert robot.site_names == ("imu", *DOBOT_FOOT_SITE_NAMES)
  collision_geoms = tuple(name for name in robot.geom_names if name.endswith("_collision"))
  assert collision_geoms == EXPECTED_COLLISION_GEOMS

  assert model.nq == 19
  assert model.nv == 18
  assert model.nu == 12
  assert model.nbody == 14
  assert model.ngeom == 30
  assert model.nsite == 5

  total_mass = float(model.body_mass[1:].sum())
  _assert_close(total_mass, DOBOT_TOTAL_MASS, "total_mass", atol=1e-10)

  effort_by_joint = dict(zip(DOBOT_JOINT_ORDER, DOBOT_EFFORT_LIMITS, strict=True))
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = model.joint(joint_id).name
    assert joint_name in effort_by_joint
    assert model.actuator_forcelimited[actuator_id]
    low, high = model.actuator_forcerange[actuator_id]
    expected_effort = effort_by_joint[joint_name]
    _assert_close(float(low), -expected_effort, f"{joint_name}.effort_low")
    _assert_close(float(high), expected_effort, f"{joint_name}.effort_high")
    _assert_close(float(model.actuator_gainprm[actuator_id, 0]), 10.0, f"{joint_name}.kp")
    _assert_close(float(model.actuator_biasprm[actuator_id, 2]), -1.0, f"{joint_name}.kd")

    dof_id = int(model.jnt_dofadr[joint_id])
    _assert_close(float(model.dof_damping[dof_id]), 0.02, f"{joint_name}.passive_damping")
    _assert_close(float(model.dof_armature[dof_id]), 0.000074, f"{joint_name}.armature")
    _assert_close(float(model.dof_frictionloss[dof_id]), 0.02, f"{joint_name}.frictionloss")

  assert model.nkey == 1
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  mujoco.mj_forward(model, data)
  _assert_close(float(data.qpos[2]), DOBOT_INIT_ROOT_HEIGHT, "init_root_height")

  clearances: dict[str, float] = {}
  for geom_name in DOBOT_FOOT_GEOM_NAMES:
    geom_id = model.geom(geom_name).id
    radius = float(model.geom_size[geom_id, 0])
    clearances[geom_name] = float(data.geom_xpos[geom_id, 2]) - radius
  min_clearance = min(clearances.values())
  if min_clearance <= 0.0:
    raise AssertionError(f"Initial foot collision penetrates the plane: {clearances}")

  data.ctrl[:] = model.key_ctrl[0]
  for _ in range(DOBOT_DECIMATION):
    mujoco.mj_step(model, data)
  assert np.isfinite(data.qpos).all()
  assert np.isfinite(data.qvel).all()
  assert np.isfinite(data.sensordata).all()

  print("[PASS] compiled Dobot model")
  print(f"  nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
  print(f"  total mass: {total_mass:.8f} kg")
  print(f"  collision geoms: {len(collision_geoms)}")
  print(f"  foot clearances: {clearances}")


def _check_flat_environment(run_step: bool) -> None:
  cfg = dobot_rover_flat_env_cfg(play=False)
  cfg.scene.num_envs = 1
  cfg.curriculum = {}
  cfg.observations["actor"].enable_corruption = False

  # Keep the reset events that materialize EntityCfg.init_state in simulation.
  # Remove only domain randomization so this check remains deterministic.
  for event_name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)
  assert "reset_base" in cfg.events
  assert "reset_robot_joints" in cfg.events
  reset_pose_range = cfg.events["reset_base"].params["pose_range"]
  reset_pose_range.update(
    {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
  )

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    action_term = env.action_manager.get_term("joint_pos")
    assert tuple(action_term.target_names) == DOBOT_JOINT_ORDER
    _assert_close(float(action_term.scale), DOBOT_ACTION_SCALE, "action_scale")

    default_joint_pos = env.scene["robot"].data.default_joint_pos
    assert default_joint_pos is not None
    expected_default = default_joint_pos[:, action_term.target_ids]

    zero_action = torch.zeros((1, len(DOBOT_JOINT_ORDER)), device=env.device)
    action_term.process_actions(zero_action)
    assert torch.allclose(action_term._processed_actions, expected_default)  # noqa: SLF001

    basis_action = zero_action.clone()
    basis_action[0, 0] = 1.0
    action_term.process_actions(basis_action)
    expected_basis = expected_default.clone()
    expected_basis[0, 0] += DOBOT_ACTION_SCALE
    assert torch.allclose(action_term._processed_actions, expected_basis)  # noqa: SLF001

    obs, _ = env.reset(seed=42)
    assert obs["actor"].shape == (1, 47)
    assert obs["critic"].shape == (1, 74)

    robot = env.scene["robot"]
    expected_root_pos = torch.tensor(
      [[0.0, 0.0, DOBOT_INIT_ROOT_HEIGHT]], device=env.device
    )
    expected_root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    assert torch.allclose(robot.data.root_link_pos_w, expected_root_pos, atol=1e-6)
    assert torch.allclose(robot.data.root_link_quat_w, expected_root_quat, atol=1e-6)
    assert torch.allclose(robot.data.joint_pos, default_joint_pos, atol=1e-6)
    assert torch.allclose(robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel))

    feet_sensor = env.scene["feet_ground_contact"]
    assert isinstance(feet_sensor, ContactSensor)
    resolved_feet = tuple(
      slot.primary_name for slot in feet_sensor._slots if slot.field_name == "found"  # noqa: SLF001
    )
    assert resolved_feet == DOBOT_FOOT_GEOM_NAMES

    _assert_close(env.physics_dt, DOBOT_PHYSICS_DT, "physics_dt")
    assert env.cfg.decimation == DOBOT_DECIMATION
    _assert_close(env.step_dt, DOBOT_PHYSICS_DT * DOBOT_DECIMATION, "policy_dt")

    print("[PASS] Dobot-Rover-Flat environment configuration")
    print("  actor/critic: 47/74")
    print(f"  reset root position: {robot.data.root_link_pos_w[0].tolist()}")
    print(f"  reset joint position: {robot.data.joint_pos[0].tolist()}")
    print(f"  resolved feet: {resolved_feet}")
    print(f"  device: {device}")
    print(f"  physics/policy dt: {env.physics_dt}/{env.step_dt}")

    if run_step:
      step_action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
      next_obs, reward, terminated, truncated, _ = env.step(step_action)
      assert next_obs["actor"].shape == (1, 47)
      assert next_obs["critic"].shape == (1, 74)
      assert torch.isfinite(next_obs["actor"]).all()
      assert torch.isfinite(next_obs["critic"]).all()
      assert torch.isfinite(reward).all()
      assert not terminated.item()
      assert not truncated.item()
      print("[PASS] Dobot-Rover-Flat one-step Warp simulation")
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--runtime",
    action="store_true",
    help="Also reset and step the MuJoCo Warp environment.",
  )
  args = parser.parse_args()
  np.set_printoptions(precision=8, suppress=True)
  _check_compiled_model()
  _check_flat_environment(run_step=args.runtime)
  suffix = "including Warp runtime" if args.runtime else "static and configuration"
  print(f"[PASS] all Dobot migration checks ({suffix})")


if __name__ == "__main__":
  main()
