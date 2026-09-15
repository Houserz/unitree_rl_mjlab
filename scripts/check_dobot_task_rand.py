"""CPU task isolation and spawn-clearance check: python scripts/check_dobot_task_rand.py."""

import itertools
import math

import mujoco
import numpy as np

import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.assets.robots.dobot.dobot_constants import DOBOT_FOOT_GEOM_NAMES


def main():
  name = 'Dobot-Rover-Flat-Kp25Kd1p3'
  baseline = load_env_cfg(name)
  cfg = load_env_cfg(name + '-TaskRandV1')
  play = load_env_cfg(name + '-TaskRandV1', play=True)
  assert baseline.events['reset_base'].params['pose_range']['z'] == (0., 0.)
  assert baseline.commands['twist'].rel_standing_envs == 0.05
  assert cfg.events['push_robot'].interval_range_s == (3., 8.)
  assert cfg.events['push_robot'].params == baseline.events['push_robot'].params
  assert 'push_robot' not in play.events
  assert not play.observations['actor'].enable_corruption
  assert cfg.scene.terrain.terrain_type == baseline.scene.terrain.terrain_type == 'plane'
  assert cfg.scene.terrain.terrain_generator is None
  assert cfg.scene.num_envs == baseline.scene.num_envs
  assert cfg.curriculum == baseline.curriculum
  assert cfg.rewards == baseline.rewards
  assert cfg.terminations == baseline.terminations
  command = cfg.commands['twist']
  assert command.resampling_time_range == (2., 8.)
  assert command.rel_standing_envs == 0.1 and command.rel_heading_envs == 0.5
  assert command.init_velocity_prob == 0.0
  agent = load_rl_cfg(name + '-TaskRandV1')
  assert not agent.resume and agent.max_iterations == 4501 and agent.seed == 42

  model = cfg.scene.entities['robot'].build().spec.compile()
  base_model = baseline.scene.entities['robot'].build().spec.compile()
  for field in ('body_mass', 'body_inertia', 'body_ipos', 'dof_armature',
                'dof_damping', 'dof_frictionloss', 'actuator_gainprm',
                'actuator_biasprm', 'geom_friction'):
    np.testing.assert_array_equal(getattr(model, field), getattr(base_model, field))
  data = mujoco.MjData(model)
  geom_ids = [model.geom(n).id for n in DOBOT_FOOT_GEOM_NAMES]
  tilt = math.radians(3.)
  lowest = float('inf')
  # All corners of roll/pitch and twelve joint offset ranges, plus interior samples.
  corners = (np.array(v) * np.array([tilt, tilt] + [.03] * 12)
             for v in itertools.product((-1., 1.), repeat=14))
  rng = np.random.default_rng(42)
  interior = rng.uniform(-1., 1., (4096, 14)) * [tilt, tilt, *([.03] * 12)]
  for values in itertools.chain(corners, interior):
    data.qpos[:] = model.key_qpos[0]
    data.qpos[2] += cfg.events['reset_base'].params['pose_range']['z'][0]
    mujoco.mju_euler2Quat(data.qpos[3:7], np.array([*values[:2], 0.]), 'xyz')
    data.qpos[7:] += values[2:]
    mujoco.mj_forward(model, data)
    clearance = data.geom_xpos[geom_ids, 2] - model.geom_size[geom_ids, 0]
    lowest = min(lowest, float(clearance.min()))
  assert lowest > 0., lowest
  print(f'[PASS] task isolation, play settings, spawn clearance: {lowest:.6f} m')


if __name__ == '__main__':
  main()
