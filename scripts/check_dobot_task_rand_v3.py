"""V3 disturbance lifecycle checks; --step runs the full terrain environment."""

import argparse
from types import SimpleNamespace

import torch

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.tasks.velocity.mdp.task_disturbances import (
  TaskRandV3Disturbance, disturbance_delta_v_range,
)

TASK = 'Dobot-Rover-Kp25Kd1p3-TaskRandV3'


class FakeAsset:
  def __init__(self, n):
    self.data = SimpleNamespace(root_link_vel_w=torch.zeros(n, 6))
    self.wrench = torch.zeros(n, 1, 3)
    self.pushed = torch.zeros(n, dtype=torch.bool)

  def write_external_wrench_to_sim(self, forces, torques, env_ids, body_ids):
    assert not torques.any() and body_ids == [0]
    self.wrench[env_ids] = forces

  def write_root_link_velocity_to_sim(self, velocity, env_ids):
    self.data.root_link_vel_w[env_ids] = velocity
    self.pushed[env_ids] = True


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--step', action='store_true')
  args = parser.parse_args()
  cfg = load_env_cfg(TASK)
  v2 = load_env_cfg('Dobot-Rover-Kp25Kd1p3-TaskRandV2')
  assert cfg.curriculum == v2.curriculum
  assert cfg.scene.terrain.terrain_generator == v2.scene.terrain.terrain_generator
  assert cfg.observations == v2.observations
  assert 'push_robot' not in cfg.events
  assert 'push_robot' in v2.events
  assert 'task_disturbance' not in load_env_cfg(TASK, play=True).events
  assert load_rl_cfg(TASK).num_steps_per_env == 24
  for step, expected in ((0, (.05, .15)), (12000, (.05, .15)),
                         (30000, (.10, .275)), (48000, (.15, .40)),
                         (108024, (.15, .40))):
    torch.testing.assert_close(torch.tensor(disturbance_delta_v_range(step)),
                               torch.tensor(expected))

  torch.manual_seed(42)
  asset = FakeAsset(1000)
  env = SimpleNamespace(scene={'robot': asset}, num_envs=1000, device='cpu',
                        step_dt=.02, common_step_counter=48000)
  params = dict(cfg.events['task_disturbance'].params)
  params['asset_cfg'] = SimpleNamespace(name='robot', body_ids=[0])
  term = TaskRandV3Disturbance(SimpleNamespace(params=params), env)
  term.reset()
  assert .65 < term.force_episode.float().mean() < .75
  assert term.wait.ge(3).all() and not asset.wrench.any()
  term.wait[:] = 0
  term(env, None, **params)
  force_mask = term.force_episode.clone()
  assert torch.equal(asset.pushed, ~force_mask)
  assert not term.forces[~force_mask].any()
  assert not term.forces[:, :, 2].any()
  magnitudes = term.forces.norm(dim=-1).flatten()
  assert magnitudes.max() <= 50. + 1e-5
  assert term.remaining[force_mask].min() >= 5
  assert term.remaining[force_mask].max() <= 20
  # Integrate every active policy interval; capping may only lower the impulse.
  impulse = torch.zeros(1000, 1, 3)
  for _ in range(20):
    impulse += term.forces * .02
    term(env, None, **params)
  realized = impulse.norm(dim=-1).flatten()[force_mask] / params['mass']
  assert realized.min() >= .15 - 1e-5 and realized.max() <= .4 + 1e-5
  assert not term.forces.any() and not asset.wrench.any()
  assert term.wait.gt(0).all()

  # Reset only a subset while forces are active: clear exactly those environments.
  term.wait[force_mask] = 0
  term(env, None, **params)
  before = term.forces.clone()
  ids = torch.arange(0, 1000, 2)
  term.reset(ids)
  assert not asset.wrench[ids].any() and not term.remaining[ids].any()
  torch.testing.assert_close(term.forces[1::2], before[1::2])
  print('[PASS] inheritance, curriculum, 70/30 split, cap, impulse, expiry, partial reset')

  if args.step:
    cfg.scene.num_envs = 10
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
      obs, _ = env.reset(seed=42)
      assert obs['actor'].shape == (10, 47)
      term = env.event_manager.get_term_cfg('task_disturbance').func
      term.force_episode[:] = torch.arange(10, device=env.device) % 2 == 0
      term.wait[:] = 0
      for _ in range(25):
        obs, reward, _, _, _ = env.step(torch.zeros(10, 12, device=env.device))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(x).all() for x in obs.values())
        assert not term.forces[~term.force_episode].any()
        assert not term.forces[:, :, 2].any()
      env.reset()
      assert not term.forces.any()
      print('[PASS] V3 terrain, both disturbance modes, 25 GPU steps, reset clearing')
    finally:
      env.close()


if __name__ == '__main__':
  main()
