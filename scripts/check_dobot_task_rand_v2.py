"""Check V2 terrain allocation and reward coordinates; --step adds a simulation smoke."""

import argparse
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains.terrain_entity import TerrainEntity
from src.tasks.velocity.mdp.observations import foot_height
from src.tasks.velocity.mdp.rewards import feet_clearance
from src.assets.robots.dobot.dobot_constants import DOBOT_FOOT_GEOM_NAMES

TASK = 'Dobot-Rover-Kp25Kd1p3-TaskRandV2'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--step', action='store_true')
  args = parser.parse_args()
  cfg = load_env_cfg(TASK)
  v1 = load_env_cfg('Dobot-Rover-Flat-Kp25Kd1p3-TaskRandV1')
  assert cfg.events == v1.events and cfg.commands == v1.commands
  assert cfg.curriculum['command_vel'] == v1.curriculum['command_vel']
  assert 'terrain_levels' in cfg.curriculum
  assert cfg.observations['actor'] == v1.observations['actor']
  assert load_env_cfg(TASK, play=True).curriculum == {}
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg.max_init_terrain_level == 0
  terrain_cfg.num_envs = 100
  terrain = TerrainEntity(terrain_cfg, device='cpu')
  model = terrain.spec.compile()
  assert model.nhfield == 15  # 5 rows x (two slopes + one Perlin column).
  assert terrain.terrain_levels.eq(0).all()

  # Sample starts on every generated patch, including highest difficulty.
  robot_model = cfg.scene.entities['robot'].build().spec.compile()
  robot_data = mujoco.MjData(robot_model)
  ground_data = mujoco.MjData(model)
  mujoco.mj_forward(model, ground_data)
  rng = np.random.default_rng(42)
  minimum_clearance = float('inf')
  for origin in terrain.terrain_origins.numpy().reshape(-1, 3):
    for _ in range(32):
      robot_data.qpos[:] = robot_model.key_qpos[0]
      robot_data.qpos[:3] += origin + [*rng.uniform(-.5, .5, 2), .03]
      euler = np.array([*rng.uniform(-np.pi / 60, np.pi / 60, 2),
                        rng.uniform(-np.pi, np.pi)])
      mujoco.mju_euler2Quat(robot_data.qpos[3:7], euler, 'xyz')
      robot_data.qpos[7:] += rng.uniform(-.03, .03, 12)
      mujoco.mj_forward(robot_model, robot_data)
      for name in DOBOT_FOOT_GEOM_NAMES:
        geom = robot_model.geom(name).id
        foot = robot_data.geom_xpos[geom]
        ray_start = foot + [0., 0., 1.]
        distance = mujoco.mj_ray(model, ground_data, ray_start,
          np.array([0., 0., -1.]), None, True, -1, np.array([-1], dtype=np.int32))
        assert distance >= 0.
        clearance = distance - 1. - robot_model.geom_size[geom, 0]
        minimum_clearance = min(minimum_clearance, clearance)
  assert minimum_clearance > 0., minimum_clearance
  print(f'[PASS] 1600 terrain spawn samples, minimum foot clearance {minimum_clearance:.6f} m')
  counts = torch.bincount(terrain.terrain_types, minlength=10)
  assert counts.tolist() == [10] * 10  # 7 flat, 1 slope, 1 inverse, 1 Perlin.
  # Reset origins on slopes are centered on a 2 m flat platform; Perlin uses
  # its maximum elevation, so V1's verified spawn clearance is preserved.
  terrain.update_env_origins(torch.arange(100), torch.ones(100, dtype=torch.bool),
                             torch.zeros(100, dtype=torch.bool))
  assert terrain.terrain_levels.eq(1).all()
  terrain.update_env_origins(torch.arange(100), torch.zeros(100, dtype=torch.bool),
                             torch.ones(100, dtype=torch.bool))
  assert terrain.terrain_levels.eq(0).all()

  # A common world-Z translation must not affect terrain-relative rewards.
  sensors = ('s0', 's1', 's2', 's3')
  robot = SimpleNamespace(data=SimpleNamespace(site_lin_vel_w=torch.ones(2, 4, 3)))
  scene = {'robot': robot}
  for name in sensors:
    scene[name] = SimpleNamespace(cfg=SimpleNamespace(max_distance=2.), data=SimpleNamespace(
      pos_w=torch.tensor([[0., 0., .1], [0., 0., 1.1]]),
      hit_pos_w=torch.tensor([[[0., 0., 0.]], [[0., 0., 1.]]]),
      distances=torch.full((2, 1), .1)))
  env = SimpleNamespace(scene=scene)
  heights = foot_height(env, sensor_names=sensors)
  assert torch.allclose(heights, torch.full((2, 4), .1), atol=1e-6)
  cost = feet_clearance(env, target_height=.1, sensor_names=sensors)
  assert torch.allclose(cost, torch.zeros(2), atol=1e-6)
  print('[PASS] V2 inheritance, terrain allocation, origins, relative clearance')

  if args.step:
    cfg.scene.num_envs = 10
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
      obs, _ = env.reset(seed=42)
      assert env.scene.terrain.terrain_levels.eq(0).all()
      assert obs['actor'].shape == (10, 47)
      assert obs['critic'].shape == (10, 74)
      for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(10, 12, device=env.device))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in obs.values())
      for name in cfg.rewards['foot_clearance'].params['sensor_names']:
        assert env.scene[name].data.distances.ge(0).all(), name
      print('[PASS] 10 environments, 47-D actor, 10 finite simulation steps, ground rays')
    finally:
      env.close()


if __name__ == '__main__':
  main()
