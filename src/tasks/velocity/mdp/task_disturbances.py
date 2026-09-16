"""TaskRandV3: mutually exclusive episode-level pushes and horizontal forces."""

import math

import torch

from mjlab.envs.mdp.events import push_by_setting_velocity


def disturbance_delta_v_range(step: int) -> tuple[float, float]:
  """500--2000 PPO iterations at 24 rollout steps/iteration."""
  alpha = min(1.0, max(0.0, (step - 12000) / 36000))
  return 0.05 + 0.10 * alpha, 0.15 + 0.25 * alpha


class TaskRandV3Disturbance:
  """70% force episodes, 30% velocity-push episodes; no body parameter changes.

  Force duration is quantized to policy steps and the actual duration is used
  to convert sampled impulse to force. The 50 N cap can reduce realized delta-v.
  This term owns the trunk external wrench; do not combine with another writer.
  """

  def __init__(self, cfg, env):
    self.asset = env.scene[cfg.params['asset_cfg'].name]
    self.body_ids = cfg.params['asset_cfg'].body_ids
    assert isinstance(self.body_ids, list) and len(self.body_ids) == 1
    self.dt = env.step_dt
    self.ids = torch.arange(env.num_envs, device=env.device)
    self.force_episode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.wait = torch.zeros(env.num_envs, device=env.device)
    self.forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    self.torques = torch.zeros_like(self.forces)

  def reset(self, env_ids=None):
    ids = self.ids if env_ids is None else self.ids[env_ids]
    self.force_episode[ids] = torch.rand(len(ids), device=self.ids.device) < .7
    self.remaining[ids] = 0
    self.forces[ids] = 0
    self.wait[ids] = 3. + torch.rand(len(ids), device=self.ids.device) * torch.where(
      self.force_episode[ids], 4., 5.)
    self.asset.write_external_wrench_to_sim(
      self.forces[ids], self.torques[ids], env_ids=ids, body_ids=self.body_ids)

  def __call__(self, env, env_ids, asset_cfg, mass, push_velocity_range):
    del env_ids
    active = self.remaining > 0
    self.wait[~active] -= self.dt
    self.remaining[active] -= 1
    expired = active & (self.remaining == 0)
    self.forces[expired] = 0
    self.wait[expired] = 3. + torch.rand(int(expired.sum()), device=self.ids.device) * 4.

    ready = (self.remaining == 0) & (self.wait <= 0)
    push_ids = self.ids[ready & ~self.force_episode]
    if len(push_ids):
      push_by_setting_velocity(env, push_ids, push_velocity_range, asset_cfg)
      self.wait[push_ids] = 3. + torch.rand(len(push_ids), device=self.ids.device) * 5.

    force_ids = self.ids[ready & self.force_episode]
    if len(force_ids):
      n = len(force_ids)
      durations = .1 + .3 * torch.rand(n, device=self.ids.device)
      steps = torch.ceil(durations / self.dt).long()
      low, high = disturbance_delta_v_range(env.common_step_counter)
      delta_v = low + (high - low) * torch.rand(n, device=self.ids.device)
      magnitude = (mass * delta_v / (steps * self.dt)).clamp(max=50.)
      angle = 2 * math.pi * torch.rand(n, device=self.ids.device)
      self.forces[force_ids, 0, 0] = magnitude * angle.cos()
      self.forces[force_ids, 0, 1] = magnitude * angle.sin()
      self.remaining[force_ids] = steps
    self.asset.write_external_wrench_to_sim(
      self.forces, self.torques, env_ids=self.ids, body_ids=self.body_ids)
