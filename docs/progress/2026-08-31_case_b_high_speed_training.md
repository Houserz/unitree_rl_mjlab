# Dobot Case B 高速训练进展

更新时间：2026-08-31

## 当前状态

- 仓库：`/home/houser/code/unitree_all/unitree_rl_mjlab`
- 分支：`train/case-b-gradual-curriculum`
- HEAD：`5051a99 feat: add gradual Dobot velocity curriculum`
- 该分支尚未推送；个人远端为 `personal -> git@github.com:Houserz/unitree_rl_mjlab.git`。
- tracked worktree 干净；`nohup.out` 和 `__pycache__` 为保留的未跟踪文件。
- MJLab 核心仓库 `/home/houser/code/unitree_all/mjlab_src` 未修改。

## 当前有效配置

| 参数 | 值 |
| --- | --- |
| physics dt | `0.0025 s` |
| decimation | `8` |
| policy frequency | `50 Hz` |
| ground `solref` | `(0.01, 1.0)`，即 Case B |
| actor / critic / action | `47 / 74 / 12` |
| illegal contact | 保留，非脚部接触力阈值 `10 N` |
| linear tracking reward | weight `1.0`，std `0.5`，未修改 |

## 已完成工作与结论

1. 回放诊断确认：高速 illegal contact 主要来自小腿下端外缘，尤其后腿；很多事件与同腿脚掌瞬时穿入地面相关，并不等同于整机摔倒。
2. 比较了 A/B/C/D 物理参数，最终采用 Case B。排除小腿或删除 illegal termination 会产生小腿接触利用，因此当前仍保留 termination。
3. 旧课程在约 5000 轮同时把 `vx`、`vy` 扩到全范围，策略随后通过停止运动规避风险。
4. 单变量渐进前进 probe 成功：`vx max` 分级提高后，`vx=2.0` 实际约 `1.51~1.53 m/s`，固定场景存活率 100%，未出现 illegal/fall。
5. 已完成从头渐进训练：

   `logs/rsl_rl/dobot_rover_velocity/2026-08-30_21-41-32_case_b_gradual_curriculum_4096e_6501i_seed42`

   关键 checkpoint：

   | checkpoint | 固定 `vx=2.0` 实际速度 | 结论 |
   | --- | ---: | --- |
   | `model_2500` | `1.619 m/s` | 高速能力良好 |
   | `model_4500` | `1.575 m/s` | 当前最佳综合候选 |
   | `model_5000` | `-0.052 m/s` | 横移扩到 `±1.0` 后纯前进塌缩 |
   | `model_6500` | `0.027 m/s` | 最终模型未恢复高速前进 |

   `model_4500` 还达到：`vx=-1 -> -0.848 m/s`、`vy=1 -> 0.635 m/s`、组合指令 `1.5/0.5/0.5 -> 1.285/0.322/0.447`，已测固定场景没有 fall/illegal。

## 原因判断

- 高速能力丢失发生在横移范围从 `±0.75` 扩到 `±1.0` 时，不是前进或后退课程本身失败。
- 全范围均匀采样时，`vx>1.8` 且 `|vy|<0.1` 的概率约 `0.67%`；纯高速前进样本被横移、转向和组合指令稀释。
- 当前线速度奖励为指数形式。`vx_cmd=2`、实际接近零时奖励约 `exp(-16)=1.1e-7`，简单放大整体权重难以提供恢复梯度，并会同时改变横移权衡。

## 下一步建议

1. 从 `model_4500.pt` 恢复，先把横移上限固定在 `vy=±0.75`，保持 reward、physics 和 illegal termination 不变，巩固训练 300~500 轮。
2. 更完整的方案是分层采样，例如：25% 纯高速前进、15% 后退、15% 横移、15% yaw、25% 组合、5% 站立。
3. 每 50 轮做固定指令、多 seed 评测，记录实际速度、存活率、fall/illegal、脚掌穿入、力矩和 target clamp。
4. 若修复采样后仍遗忘前进，再从同一个 `model_4500` 做 A/B：新增仅对纯前进生效的非饱和 Huber/Smooth-L1 辅助项；不要先全局放大现有 XY tracking reward。

## 实机边界

- `model_4500.pt` SHA-256：`6651f301bb4230d88c8c2ed881bcc04944dff5ad1f88ae49ae29f118a3e247e8`。
- 它是当前最佳仿真候选，但 `hardware_ready=false`，不应直接上机。
- 主要阻塞：策略为含 gait phase 的 47-D contract，而现有 Rover 部署链是不同语义的 48-D contract；训练目录 `policy.onnx` 对应最终 `model_6500`，不是 `model_4500` 的可信精确导出。
- 后续必须依次完成：精确 ONNX 导出与 PyTorch parity、版本化 47-D adapter/manifest/bundle、Dobot MuJoCo Sim2Sim、state-only shadow/read-only preflight，最后才进行人工分阶段低速实机测试。

## 关键代码

- 课程配置：`src/tasks/velocity/config/dobot/env_cfgs.py`
- reward 配置：`src/tasks/velocity/velocity_env_cfg.py`
- 线速度 reward 实现：`src/tasks/velocity/mdp/rewards.py`
- 配置/runtime 检查：`scripts/check_dobot_model.py`

