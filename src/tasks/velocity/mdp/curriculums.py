from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import (
  StratifiedVelocityCommandCfg,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


class LateralBucketStage(TypedDict):
  """One range expansion for a stratified lateral-command bucket."""

  step: int
  lin_vel_y: tuple[float, float]


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  # The first reset happens before any episode has run. World-origin model
  # poses must not be scored against the newly assigned terrain origins.
  if env.common_step_counter == 0:
    return torch.mean(terrain.terrain_levels.float())

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def stratified_lateral_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  lateral_bucket_index: int,
  stages: list[LateralBucketStage],
) -> dict[str, torch.Tensor]:
  """Expand only one lateral bucket while retaining all other bucket weights.

  A wide uniform command box made high-forward straight commands rare once
  lateral range reached ``±1``.  This curriculum preserves dedicated forward
  buckets and changes only the range of the lateral-training bucket.
  """

  del env_ids  # The curriculum is global, as are the existing command stages.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(StratifiedVelocityCommandCfg, command_term.cfg)
  if not 0 <= lateral_bucket_index < len(cfg.buckets):
    raise ValueError("lateral_bucket_index is outside the configured buckets")

  lateral_range: tuple[float, float] | None = None
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      lateral_range = stage["lin_vel_y"]
  if lateral_range is not None:
    buckets = list(cfg.buckets)
    buckets[lateral_bucket_index] = replace(
      buckets[lateral_bucket_index], lin_vel_y=lateral_range
    )
    cfg.buckets = tuple(buckets)
  return {}
