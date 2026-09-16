"""CPU regression checks: python scripts/check_dobot_training.py."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from mjlab.tasks.registry import list_tasks
from evaluate_dobot_velocity import SUITES, _force_command
from train import TrainConfig
from src.tasks.velocity.config.dobot.env_cfgs import (
  dobot_rover_flat_env_cfg,
  dobot_rover_flat_kp25_kd13_env_cfg,
  dobot_rover_flat_kp25_kd13_lateral_curriculum_env_cfg,
)
from src.tasks.velocity.mdp.curriculums import commands_vel, stratified_lateral_range


def main() -> None:
  expected_tasks = {
    "Dobot-Rover-Kp25Kd1p3-TaskRandV2",
    "Dobot-Rover-Flat",
    "Dobot-Rover-Flat-Identified",
    "Dobot-Rover-Flat-Kp25Kd1p3-TaskRandV1",
    "Dobot-Rover-Flat-Kp25Kd1p3",
    "Dobot-Rover-Flat-Kp25Kd1p3-LateralCurriculum",
  }
  assert {name for name in list_tasks() if name.startswith("Dobot-")} == expected_tasks
  train = TrainConfig.from_task("Dobot-Rover-Flat-Kp25Kd1p3")
  assert not train.agent.resume and train.warm_start_from is None
  assert train.agent.max_iterations == 4501 and train.env.scene.num_envs == 4096
  assert train.agent.actor.distribution_cfg["init_std"] == 0.4

  baseline = dobot_rover_flat_env_cfg()
  scratch = dobot_rover_flat_kp25_kd13_env_cfg()
  # Building a PD variant must not mutate either the existing or fresh baseline.
  for cfg, kp, kd in ((scratch, 25.0, 1.3), (baseline, 10.0, 1.0),
                      (dobot_rover_flat_env_cfg(), 10.0, 1.0)):
    model = cfg.scene.entities["robot"].build().spec.compile()
    assert model.nu == 12
    assert (model.actuator_gainprm[:, 0] == kp).all()
    assert (model.actuator_biasprm[:, 2] == -kd).all()

  command = SimpleNamespace(cfg=scratch.commands["twist"])
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
  ids = torch.arange(1)
  params = scratch.curriculum["command_vel"].params
  for step, expected in ((12000, 1.0), (12001, 1.2), (60001, 2.0)):
    env.common_step_counter = step
    commands_vel(env, ids, **params)
    assert command.cfg.ranges.lin_vel_x[1] == expected
  env.common_step_counter = 240000
  commands_vel(env, ids, **params)
  assert command.cfg.ranges.lin_vel_y == (-0.75, 0.75)
  assert baseline.curriculum["command_vel"].params["velocity_stages"][-1]["step"] == 120000

  lateral = dobot_rover_flat_kp25_kd13_lateral_curriculum_env_cfg()
  command.cfg = lateral.commands["twist"]
  original_buckets = command.cfg.buckets
  params = lateral.curriculum["lateral_range"].params
  for step, expected in ((12000, 0.75), (12001, 0.85), (24001, 0.95), (36001, 1.0)):
    env.common_step_counter = step
    stratified_lateral_range(env, ids, **params)
    assert command.cfg.buckets[2].lin_vel_y == (-expected, expected)
    assert all(a == b for i, (a, b) in enumerate(
      zip(original_buckets, command.cfg.buckets, strict=True)) if i != 2)

  # Noncontiguous indexing must update both velocity and heading buffers.
  cfg = replace(command.cfg, init_velocity_prob=0.0, rel_standing_envs=0.0,
    buckets=(
      replace(original_buckets[0], weight=0.3, lin_vel_x=(1.0, 1.0),
              heading=(1.0, 2.0), rel_heading_envs=1.0),
      replace(original_buckets[1], weight=0.7, lin_vel_x=(2.0, 2.0),
              heading=(-2.0, -1.0), rel_heading_envs=1.0),
    ))
  sampler = cfg.build(SimpleNamespace(
    num_envs=8192, device="cpu", scene={"robot": object()}))
  torch.manual_seed(42)
  sampled_ids = torch.arange(0, 8192, 2)
  sampler._resample_command(sampled_ids)
  vx = sampler.vel_command_b[sampled_ids, 0]
  heading = sampler.heading_target[sampled_ids]
  first = vx == 1.0
  assert torch.all(first | (vx == 2.0))
  assert abs(first.float().mean().item() - 0.3) < 0.03
  assert torch.all((heading[first] >= 1.0) & (heading[first] <= 2.0))
  assert torch.all((heading[~first] >= -2.0) & (heading[~first] <= -1.0))
  assert torch.all(sampler.heading_target[1::2] == 0.0)
  assert torch.all(sampler.vel_command_b[1::2] == 0.0)
  assert len(SUITES["all"]) == 12
  assert len({s.name for s in SUITES["all"]}) == 12
  fixed_command = torch.tensor([0.0, 0.0, -0.5])
  _force_command(sampler, fixed_command)
  assert torch.all(sampler.vel_command_b == fixed_command)
  assert not sampler.is_heading_env.any() and not sampler.is_standing_env.any()
  assert torch.all(sampler.time_left == 1e9)
  print("[PASS] tasks, scratch defaults, PD isolation, curricula, sampling, evaluation commands")


if __name__ == "__main__":
  main()
