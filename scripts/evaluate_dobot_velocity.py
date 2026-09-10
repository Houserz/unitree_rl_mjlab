"""Fixed-command, multi-seed evaluation for Dobot velocity policies.

The evaluator deliberately disables command resampling and interval pushes.  It
therefore measures tracking and failure behaviour for named commands rather
than averaging together unrelated command changes from the training sampler.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class Scenario:
  name: str
  command: tuple[float, float, float]


SCENARIOS = (
  Scenario("medium_forward", (1.0, 0.0, 0.0)),
  Scenario("medium_fast_forward", (1.4, 0.0, 0.0)),
  Scenario("high_forward", (2.0, 0.0, 0.0)),
  Scenario("forward_turn", (1.4, 0.0, 0.5)),
  Scenario("forward_lateral", (1.4, 0.35, 0.0)),
  Scenario("reverse", (-1.0, 0.0, 0.0)),
)


@dataclass(frozen=True)
class EvaluateConfig:
  checkpoint_file: str
  """Local checkpoint to evaluate."""

  task_id: str = "Dobot-Rover-Flat"
  num_envs: int = 64
  duration_s: float = 10.0
  settle_s: float = 2.0
  seeds: tuple[int, ...] = (42, 43, 44)
  device: str | None = None
  output_file: str | None = None


def _force_command(command_term: Any, command: torch.Tensor) -> None:
  """Hold one body-frame command across resets and evaluation steps."""

  command_term.vel_command_b[:] = command
  command_term.is_heading_env[:] = False
  command_term.is_standing_env[:] = False
  command_term.time_left[:] = 1.0e9


def _evaluate_scenario(
  *,
  task_id: str,
  checkpoint: Path,
  cfg: EvaluateConfig,
  scenario: Scenario,
  seed: int,
  device: str,
) -> dict[str, float | int | str]:
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = seed
  # A time-out at the end of an otherwise successful fixed window is not a
  # locomotion failure.  The evaluator owns the window length below instead.
  env_cfg.episode_length_s = int(1e9)
  env_cfg.events.pop("push_robot", None)

  # The command is forced below.  Avoid random replacement at episode reset or
  # during the fixed-command window.
  command_cfg = env_cfg.commands["twist"]
  command_cfg.resampling_time_range = (1.0e9, 1.0e9)
  command_cfg.rel_standing_envs = 0.0

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  command_term = wrapped_env.unwrapped.command_manager.get_term("twist")
  assert command_term is not None
  command = torch.tensor(scenario.command, dtype=torch.float32, device=device)
  _force_command(command_term, command)
  obs = wrapped_env.get_observations()

  duration_steps = int(round(cfg.duration_s / wrapped_env.unwrapped.step_dt))
  settle_steps = int(round(cfg.settle_s / wrapped_env.unwrapped.step_dt))
  if duration_steps <= settle_steps:
    raise ValueError("duration_s must be greater than settle_s")

  done_once = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  velocity_sum = torch.zeros(3, device=device)
  xy_error_sum = torch.zeros((), device=device)
  sample_count = 0
  fell_over_count = 0
  illegal_contact_count = 0

  with torch.inference_mode():
    for step in range(duration_steps):
      actions = policy(obs)
      obs, _, dones, _ = wrapped_env.step(actions)

      terminations = wrapped_env.unwrapped.termination_manager
      newly_done = dones.bool() & ~done_once
      fell_over_count += int(
        torch.count_nonzero(newly_done & terminations.get_term("fell_over")).item()
      )
      illegal_contact_count += int(
        torch.count_nonzero(
          newly_done & terminations.get_term("illegal_contact")
        ).item()
      )
      done_once |= newly_done

      # Terminated environments are reset by the vector environment.  Exclude
      # them from post-settle tracking averages rather than counting reset state.
      active = ~done_once
      if step >= settle_steps and active.any():
        robot = wrapped_env.unwrapped.scene["robot"]
        actual = robot.data.root_link_lin_vel_b[:, :3]
        velocity_sum += actual[active].sum(dim=0)
        xy_error_sum += torch.norm(actual[active, :2] - command[:2], dim=1).sum()
        sample_count += int(torch.count_nonzero(active).item())

      _force_command(command_term, command)

  denominator = max(sample_count, 1)
  result: dict[str, float | int | str] = {
    "scenario": scenario.name,
    "seed": seed,
    "command_vx": scenario.command[0],
    "command_vy": scenario.command[1],
    "command_wz": scenario.command[2],
    "mean_vx": float((velocity_sum[0] / denominator).item()),
    "mean_vy": float((velocity_sum[1] / denominator).item()),
    "mean_vz": float((velocity_sum[2] / denominator).item()),
    "mean_xy_error": float((xy_error_sum / denominator).item()),
    "survival_rate": float((~done_once).float().mean().item()),
    "fell_over_count": fell_over_count,
    "illegal_contact_count": illegal_contact_count,
    "post_settle_samples": sample_count,
  }
  wrapped_env.close()
  return result


def run_evaluate(cfg: EvaluateConfig) -> list[dict[str, float | int | str]]:
  configure_torch_backends()
  checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  if cfg.num_envs <= 0 or cfg.duration_s <= 0 or cfg.settle_s < 0:
    raise ValueError("num_envs and duration_s must be positive; settle_s must be nonnegative")
  if not cfg.seeds:
    raise ValueError("At least one seed is required")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  results: list[dict[str, float | int | str]] = []
  for seed in cfg.seeds:
    for scenario in SCENARIOS:
      result = _evaluate_scenario(
        task_id=cfg.task_id,
        checkpoint=checkpoint,
        cfg=cfg,
        scenario=scenario,
        seed=seed,
        device=device,
      )
      results.append(result)
      print(
        f"[RESULT] seed={seed} scenario={scenario.name} "
        f"v=({result['mean_vx']:.3f}, {result['mean_vy']:.3f}) "
        f"e_xy={result['mean_xy_error']:.3f} "
        f"survival={result['survival_rate']:.3f} "
        f"fall={result['fell_over_count']} illegal={result['illegal_contact_count']}"
      )

  if cfg.output_file is not None:
    output_path = Path(cfg.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote evaluation results to {output_path}")
  return results


def main() -> None:
  import src.tasks  # noqa: F401

  run_evaluate(tyro.cli(EvaluateConfig))


if __name__ == "__main__":
  main()
