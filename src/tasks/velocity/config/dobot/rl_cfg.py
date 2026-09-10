"""RL configuration for the Dobot Rover velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def dobot_rover_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the Go2 PPO setup with the Dobot action clipping contract."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="dobot_rover_velocity",
    clip_actions=100.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def dobot_rover_kp25_kd13_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO settings for an isolated 25/1.3 PD training run.

  With unchanged action scale, a Gaussian action std of 1.0 would make the
  initial proportional-torque perturbation 2.5 times the established 10/1
  baseline.  Scaling the initial std by ``10 / 25`` preserves its approximate
  magnitude while leaving the learned action contract unchanged.
  """

  cfg = dobot_rover_ppo_runner_cfg()
  cfg.run_name = "kp25_kd1p3_std0p4_scratch"
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  cfg.save_interval = 50
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.4
  return cfg


def dobot_rover_kp25_kd13_lateral_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Fresh-optimizer continuation settings for lateral range expansion."""

  cfg = dobot_rover_kp25_kd13_ppo_runner_cfg()
  cfg.max_iterations = 1500
  cfg.save_interval = 50
  cfg.run_name = "kp25_kd1p3_lateral_from_model4500"
  # The current policy already walks.  Keep updates conservative while the
  # command distribution changes, and warm-start actor/critic only.
  cfg.algorithm.learning_rate = 2.5e-4
  return cfg
