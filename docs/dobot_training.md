# Dobot 训练入口与历史版本

整理日期：2026-09-10。当前保留 3 个任务；本次整理没有重新完成一轮正式训练。

## 从零完整训练 Kp25/Kd1.3

```bash
conda activate unitree_rl_mjlab
cd /home/houser/code/unitree_all/unitree_rl_mjlab
python scripts/train.py Dobot-Rover-Flat-Kp25Kd1p3 \
  --agent.seed 42 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 4501 \
  --agent.resume False \
  --enable-nan-guard True
```

这就是独立的从零训练入口：随机初始化 actor/critic 和优化器，不需要下载或指定
checkpoint，不传 `--warm-start-from`。默认参数已与命令一致；默认写本地 TensorBoard，
不上传模型。输出在 `logs/rsl_rl/dobot_rover_velocity/<时间>_kp25_kd1p3_scratch_v1/`，
最终编号为 `model_4500.pt`，每 50 轮保留 checkpoint。

`scratch_v1` 采用历史完成过的 4501 轮配方：Kp/Kd=25/1.3、初始动作 std=0.4、
4096 环境、seed=42、24 环境步/轮。std=0.4 是已有训练使用的取值，不是已证明的最优值。
保留 Case B 物理步长 0.0025 s、decimation=8、50 Hz 策略、action scale=0.25、
47/74/12 观测/动作接口及原奖励、10 N 非足接触终止。

| 开始环境步（约训练轮数） | vx 范围（m/s） | vy 范围（m/s） |
|---|---|---|
| 0 | -0.5～1.0 | ±0.5 |
| 12000（500） | -0.5～1.2 | ±0.5 |
| 24000（1000） | -0.5～1.4 | ±0.5 |
| 36000（1500） | -0.5～1.6 | ±0.5 |
| 48000（2000） | -0.5～1.8 | ±0.5 |
| 60000（2500） | -0.5～2.0 | ±0.5 |
| 84000（3500） | -0.75～2.0 | ±0.5 |
| 96000（4000） | -1.0～2.0 | ±0.5 |
| 108000（4500） | -1.0～2.0 | ±0.75 |

课程在环境 reset 时应用，条件是 `common_step_counter > step`；表中轮数是近似值。
角速度训练范围保持 ±1 rad/s。4501 轮覆盖上述课程，完整从零训练不依赖横移续训。

与历史源码的区别：删除 Kp25 课程中尚未执行的 120000 步横移 ±1.0 阶段，
因此延长训练也不会自动进入该阶段。历史 4501 轮没有到达它；这里没有宣称已验证更长训练。
Kp10 历史任务仍保留原始完整阶段表，便于旧实验对照。

## 三个入口

| 任务 | 用途 | 默认轮数 |
|---|---|---|
| `Dobot-Rover-Flat` | Kp10/Kd1 历史对照，保留旧课程 | 10001 |
| `Dobot-Rover-Flat-Kp25Kd1p3` | Kp25/Kd1.3 从零训练，`scratch_v1` | 4501 |
| `Dobot-Rover-Flat-Kp25Kd1p3-LateralCurriculum` | Kp25 横移扩展续训，`lateral_v2` | 1800 |

横移续训从已有 Kp25 策略出发，仅载入 actor/critic，重新初始化优化器和训练轮数。
它保留中速/高速前进采样桶，仅将横移桶从 ±0.75 扩到 ±0.85、±0.95、±1.0。
该入口是扩展候选，不是从零入口的必要步骤。

```bash
RUN=logs/rsl_rl/dobot_rover_velocity/2026-09-04_15-56-33_kp25_kd1p3_std0p4_scratch
python scripts/train.py Dobot-Rover-Flat-Kp25Kd1p3-LateralCurriculum \
  --warm-start-from "$RUN/model_4500.pt" \
  --agent.resume False \
  --enable-nan-guard True
```

`lateral_v2` 修复了分桶航向目标随机采样未写回缓冲区的问题，并将默认轮数从 1500
调整到历史实际运行的 1800。采样修复改变后续训练分布，所以不能将历史 `model_1799`
视为这个修复版本的训练结果。需要精确恢复旧采样时使用下方归档分支。

`--agent.resume True` 是载入完整训练状态，和 actor/critic-only warm-start 不同。
不要同时传 resume 和 warm-start，也不要以 resume 的 checkpoint 轮数推断环境课程计数已恢复。

## 历史取舍

| 阶段 | 结论与处理 |
|---|---|
| `99ef0b4` Dobot 迁移、`c85fa75` v2 参数 | 保留模型迁移和接口基础 |
| `1887f7b` Case B、`5051a99` 渐进课程 | 保留；突然扩大指令范围曾导致前进能力退化 |
| calf-excluded 实验 | 保留归档标签，不合入；减少 reset 不代表步态更好 |
| HighSpeedRetention / SafeRetention | 移除日常任务；已有结果体现高速收益与其他场景退化的取舍 |
| HighSpeedBalancedRetention | 移除日常任务；历史记录未显示全面超过基线，本次未找到独立评估 JSON |
| Kp25 从零 `model_4500` | 作为主要保留的历史训练参考 |
| Kp25 横移 `model_1799` | 保留候选；横移改善，但高速测试出现非法接触 |

下面是已有 3 seed × 64 环境 × 10 s 固定指令 JSON 的汇总，不是本次重新评估结果。

| 历史候选 | vx=1 实际 vx | vx=2 实际 vx | 核心场景失败情况 |
|---|---|---|---|
| Kp10 `model_4500` 的 warm-start `model_0` 对照 | 0.945 | 1.584 | 已测场景零摔倒/非法接触 |
| Kp25 从零 `model_4500` | 0.873 | 1.759 | 已测场景零摔倒/非法接触 |
| Kp25 横移 `model_1799` | 0.916 | 1.797 | 高速平均存活率 97.92%，4 次非法接触 |

Kp25 横移 `model_1799` 在 vy=+1/-1 测试中实际 vy 约 +0.766/-0.817 m/s，
已测横移场景零摔倒/非法接触。Kp10 对照的 JSON 位于
`2026-09-02_12-24-12_high_speed_retention_from_model4500/eval_model_0_3seed_10s.json`。

[dobot_baselines.json](dobot_baselines.json) 记录三个原始模型的完整 run 路径、checkpoint、
保存的 agent 参数与课程、模型/YAML/源码 diff/评估文件 SHA-256。
实际 `env.yaml`、模型和日志继续保存在本地 `logs/`；它们被 Git 忽略，不能用仅克隆仓库代替日志备份。
历史配置以 run 内 `params/{env,agent}.yaml` 和 `git/unitree_rl_mjlab.diff` 为准。

所有结果只说明已测仿真场景表现，不构成完整性能验收或实机验收。

## 统一检查与评估

```bash
python scripts/check_dobot_model.py --runtime
python scripts/check_dobot_training.py

RUN=logs/rsl_rl/dobot_rover_velocity/2026-09-04_15-56-33_kp25_kd1p3_std0p4_scratch
python scripts/evaluate_dobot_velocity.py \
  --task-id Dobot-Rover-Flat-Kp25Kd1p3 \
  --checkpoint-file "$RUN/model_4500.pt" \
  --suite all \
  --output-file logs/validation/kp25_model4500_all.json
```

`--suite core` 保持原来的 6 个场景（默认）；`lateral` 为 4 个横移场景；
`yaw` 为正/反纯转向；`all` 合计 12 个。默认 3 seed、64 环境、10 s，前 2 s 为过渡期。
输出包含实际线速度、XY 误差、实际角速度、角速度绝对误差、存活率和首次终止原因。
速度误差按尚未终止的环境样本统计，应与存活率及 `post_settle_samples` 一起阅读；
零有效样本时输出的零值不代表完美跟踪。固定指令测试关闭推扰，不等于动态切换或推扰验收。

Play 示例（显式指定任务，避免 Kp10/Kp25 混用）：

```bash
python scripts/play.py Dobot-Rover-Flat-Kp25Kd1p3 \
  --checkpoint-file "$RUN/model_4500.pt" --num-envs 4 --viewer viser
tensorboard --logdir logs/rsl_rl/dobot_rover_velocity --port 6006
```

## Git 回退与验证

- 原分支 `train/case-b-gradual-curriculum` 保留在 `5051a99`。
- 整理前快照：`archive/dobot-before-cleanup-2026-09-10`，提交 `0b92c93`。
- 整理分支：`maintenance/dobot-training-cleanup`。
- 实验归档标签：`archive/dobot-calf-excluded-stage2-2026-09-10`、
  `archive/dobot-gradual-probe-2026-09-10`。原实验分支也保留。
- 模型、训练日志及 `nohup.out` 未删除；缓存和终端日志已加入忽略规则。没有推送远端。

本次验证：原模型/环境检查通过（47/74 观测、12 动作）；CPU 回归检查通过；
Kp25 从零入口完成 1 轮 × 16 环境的冒烟训练，无 checkpoint 加载，生成的
`model_0` actor/critic 张量均有限；正反纯转向各完成 0.5 s × 2 环境评估，
新增角速度字段有限且响应方向正确。没有执行正式 4501 轮训练。
冒烟产物在 `logs/rsl_rl/dobot_rover_velocity/2026-09-10_20-54-25_cleanup_scratch_smoke_20260910/`，
转向输出在 `logs/validation/cleanup_yaw_smoke_20260910.json`。它们只验证运行流程。

需要查看旧实验时，可使用独立 worktree，避免覆盖当前源码：

```bash
git worktree add --detach ../unitree_rl_mjlab_history archive/dobot-before-cleanup-2026-09-10
```
