"""Offline regression check: python scripts/check_dobot_identified.py."""

from dataclasses import replace

import numpy as np
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import joint_pos_rel

from src.assets.robots.dobot import motor_parameters as motors
from src.assets.robots.dobot.dobot_constants import DOBOT_JOINT_ORDER, get_dobot_robot_cfg
from src.tasks.velocity.config.dobot.env_cfgs import dobot_rover_flat_identified_env_cfg


def main():
  robot_cfg = get_dobot_robot_cfg(identified=True)
  robot = robot_cfg.build()
  model = robot.spec.compile()
  assert model.nu == 12
  for name in DOBOT_JOINT_ORDER:
    dof = model.joint(name).dofadr[0]
    np.testing.assert_allclose(
      [model.dof_armature[dof], model.dof_damping[dof], model.dof_frictionloss[dof]],
      motors.JOINT_PARAMETERS[name][:3], rtol=0, atol=1e-12,
    )
  baseline = get_dobot_robot_cfg().build().spec.compile()
  np.testing.assert_allclose(baseline.dof_damping[6:], 0.02)

  # Exercise real actuator initialization and the full action/observation chain.
  cfg = dobot_rover_flat_identified_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.events = {k: v for k, v in cfg.events.items()
                if k in ("reset_base", "reset_robot_joints")}
  env = ManagerBasedRlEnv(cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
  try:
    obs, _ = env.reset(seed=42)
    robot = env.scene["robot"]
    bias = torch.tensor([motors.JOINT_PARAMETERS[n][3] for n in DOBOT_JOINT_ORDER],
                        device=env.device)
    torch.testing.assert_close(robot.data.encoder_bias, -bias.expand(2, -1))
    torch.testing.assert_close(
      joint_pos_rel(env, biased=True),
      robot.data.joint_pos - bias - robot.data.default_joint_pos,
    )
    action = env.action_manager.get_term("joint_pos")
    zeros = torch.zeros((2, 12), device=env.device)
    action.process_actions(zeros)
    action.apply_actions()
    torch.testing.assert_close(robot.data.joint_pos_target,
                               robot.data.default_joint_pos + bias)
    assert obs["actor"].shape == (2, 47) and obs["critic"].shape == (2, 74)

    # A changing target AND velocity distinguishes torque delay from target delay.
    actuator = robot.actuators[0]
    actuator.reset()
    expected_history = []
    for step in range(motors.DELAY_STEPS + 7):
      target = torch.full((2, 1), 0.2 * (step + 1), device=env.device)
      vel = torch.full_like(target, float(step % 3) * 12)
      zero = torch.zeros_like(target)
      cmd = ActuatorCmd(target, zero, zero, zero, vel)
      limit = actuator.cfg.effort_limit
      expected_history.append((motors.KP * target - motors.KD * vel).clamp(-limit, limit))
      torch.testing.assert_close(actuator.compute(cmd),
                                 expected_history[max(0, step - motors.DELAY_STEPS)])

    # Reset only one world: no old torque leaks into it; the other keeps history.
    actuator.reset(torch.tensor([0], device=env.device))
    zero = torch.zeros((2, 1), device=env.device)
    output = actuator.compute(ActuatorCmd(zero, zero, zero, zero, zero))
    torch.testing.assert_close(output[0], zero[0])
    expected_history.append(zero)
    torch.testing.assert_close(output[1], expected_history[-1-motors.DELAY_STEPS][1])

    # At +/-24 rad/s, full motoring torque remains available in either direction.
    actuator.reset()
    target = torch.tensor([[100.0], [-100.0]], device=env.device)
    vel = torch.tensor([[24.0], [-24.0]], device=env.device)
    output = actuator.compute(ActuatorCmd(target, zero, zero, zero, vel))
    torch.testing.assert_close(output, torch.sign(target) * actuator.cfg.effort_limit)

    env.reset(seed=42)
    for _ in range(8):
      obs, reward, _, _, _ = env.step(zeros)
      assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
    env.reset(env_ids=torch.tensor([1], device=env.device))
    torch.testing.assert_close(robot.data.encoder_bias, -bias.expand(2, -1))

    # Reject malformed manual edits before constructing a simulation.
    for change in ({"delay_steps": 4.5}, {"viscous_damping": -1}, {"encoder_bias": float("nan")}):
      try:
        replace(actuator.cfg, **change)
      except ValueError:
        pass
      else:
        raise AssertionError(f"Invalid edit accepted: {change}")
  finally:
    env.close()
  print("PASS: 12-joint mapping, baseline, encoder/action signs, clipped torque delay,")
  print("      full torque at +/-24 rad/s, partial reset, invalid edits, 8 finite steps")


if __name__ == "__main__":
  main()
