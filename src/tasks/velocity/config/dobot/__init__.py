from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  dobot_rover_flat_env_cfg,
  dobot_rover_flat_identified_env_cfg,
  dobot_rover_flat_kp25_kd13_env_cfg,
  dobot_rover_flat_kp25_kd13_task_rand_v1_env_cfg,
  dobot_rover_kp25_kd13_task_rand_v2_env_cfg,
  dobot_rover_kp25_kd13_task_rand_v3_env_cfg,
  dobot_rover_flat_kp25_kd13_lateral_curriculum_env_cfg,
)
from .rl_cfg import (
  dobot_rover_kp25_kd13_ppo_runner_cfg,
  dobot_rover_kp25_kd13_lateral_ppo_runner_cfg,
  dobot_rover_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Dobot-Rover-Kp25Kd1p3-TaskRandV3",
  env_cfg=dobot_rover_kp25_kd13_task_rand_v3_env_cfg(),
  play_env_cfg=dobot_rover_kp25_kd13_task_rand_v3_env_cfg(play=True),
  rl_cfg=replace(
    dobot_rover_kp25_kd13_ppo_runner_cfg(),
    run_name="kp25_kd1p3_task_rand_v3",
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Dobot-Rover-Kp25Kd1p3-TaskRandV2",
  env_cfg=dobot_rover_kp25_kd13_task_rand_v2_env_cfg(),
  play_env_cfg=dobot_rover_kp25_kd13_task_rand_v2_env_cfg(play=True),
  rl_cfg=replace(
    dobot_rover_kp25_kd13_ppo_runner_cfg(),
    run_name="kp25_kd1p3_task_rand_v2",
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Dobot-Rover-Flat-Kp25Kd1p3-TaskRandV1",
  env_cfg=dobot_rover_flat_kp25_kd13_task_rand_v1_env_cfg(),
  play_env_cfg=dobot_rover_flat_kp25_kd13_task_rand_v1_env_cfg(play=True),
  rl_cfg=replace(
    dobot_rover_kp25_kd13_ppo_runner_cfg(),
    run_name="kp25_kd1p3_task_rand_v1",
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Dobot-Rover-Flat-Identified",
  env_cfg=dobot_rover_flat_identified_env_cfg(),
  play_env_cfg=dobot_rover_flat_identified_env_cfg(play=True),
  rl_cfg=replace(
    dobot_rover_kp25_kd13_ppo_runner_cfg(),
    experiment_name="dobot_rover_identified", run_name="best172",
  ),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
  task_id="Dobot-Rover-Flat",
  env_cfg=dobot_rover_flat_env_cfg(),
  play_env_cfg=dobot_rover_flat_env_cfg(play=True),
  rl_cfg=dobot_rover_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Dobot-Rover-Flat-Kp25Kd1p3-LateralCurriculum",
  env_cfg=dobot_rover_flat_kp25_kd13_lateral_curriculum_env_cfg(),
  play_env_cfg=dobot_rover_flat_kp25_kd13_lateral_curriculum_env_cfg(play=True),
  rl_cfg=dobot_rover_kp25_kd13_lateral_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Dobot-Rover-Flat-Kp25Kd1p3",
  env_cfg=dobot_rover_flat_kp25_kd13_env_cfg(),
  play_env_cfg=dobot_rover_flat_kp25_kd13_env_cfg(play=True),
  rl_cfg=dobot_rover_kp25_kd13_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
