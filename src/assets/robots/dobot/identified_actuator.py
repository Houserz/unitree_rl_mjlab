"""Identified passive dynamics, encoder bias and output delay with ideal PD."""

from dataclasses import dataclass
import math

from mjlab.actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.utils.buffers import DelayBuffer


@dataclass(kw_only=True)
class IdentifiedActuatorCfg(IdealPdActuatorCfg):
  viscous_damping: float
  encoder_bias: float
  delay_steps: int

  def __post_init__(self):
    super().__post_init__()
    for name in ("armature", "frictionloss", "viscous_damping", "stiffness",
                 "damping", "effort_limit"):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(self.encoder_bias):
      raise ValueError("encoder_bias must be finite")
    if type(self.delay_steps) is not int or self.delay_steps < 0:
      raise ValueError("delay_steps must be a nonnegative integer")

  def build(self, entity, target_ids, target_names):
    return IdentifiedActuator(self, entity, target_ids, target_names)


class IdentifiedActuator(IdealPdActuator):
  cfg: IdentifiedActuatorCfg

  def edit_spec(self, spec, target_names):
    super().edit_spec(spec, target_names)
    for name in target_names:
      spec.joint(name).damping = self.cfg.viscous_damping

  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    self.torque_history = DelayBuffer(
      min_lag=self.cfg.delay_steps, max_lag=self.cfg.delay_steps,
      batch_size=data.nworld, device=device,
    )

  def compute(self, cmd):
    # Fixed torque limits only; no assumed motor torque-speed envelope.
    torque = super().compute(cmd)
    self.torque_history.append(torque)
    return self.torque_history.compute()

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    # Each cfg controls one joint; keep the env and joint axes independent.
    for joint_id in self._target_ids_list:
      self.entity.data.encoder_bias[ids, joint_id] = -self.cfg.encoder_bias
    # Native buffer backfills with the first new torque, matching PACE startup.
    self.torque_history.reset(env_ids)
