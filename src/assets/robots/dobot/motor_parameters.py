"""Dobot 辨识参数：修改本文件后重新启动训练/play 即可生效。

只用于 Dobot-Rover-Flat-Identified；原有任务不读取这组参数。
每行依次为 armature [kg m²], viscous_damping [Nm s/rad],
frictionloss [Nm], encoder_bias [rad]；编码器角度 = 仿真角度 - bias。
"""

# Source: C 组（bias 固定为 0，Kp/Kd=25/1.3）
#   训练目录: /home/houser/dynamic_identification/pace-sim2real-mjlab/logs/pace/dobot_all/26_09_12_15-32-04-335564
#   参数文件: best_trajectory_params.pt（该目录缺少 best_params.pt），最佳代索引 146
#   对比文档: /home/houser/下载/dobot_bias_comparison_20260912.md（第 3 节表格的 C 列）
# 精度说明: 原始目录不在本机，以下数值取自对比文档表格（统一六位有效数字）；
#   若拿到原始 best_trajectory_params.pt / dobot_bias_comparison_20260912.csv，可用其更高精度值替换。
# 注意: C 组的 bias 范围 [0, 0] 是人为固定假设，不代表已标定出真实零位。

KP = 25.0
KD = 1.3  # PD derivative gain, separate from passive viscous_damping below.
# No DC torque-speed envelope; fixed 23/23/55 Nm limits live in dobot_constants.py.
# This differs from the identification model's assumed 20 rad/s DC envelope.
DELAY_STEPS = 5  # truncate(5.685327); 5 × 0.0025 s = 12.5 ms, output torque delay.

JOINT_PARAMETERS = {
  "joint_front_left_abad": (0.00823688, 0.379518, 0.189661, 0.0),
  "joint_front_left_thigh_pitch": (0.0022037, 0.428069, 0.121918, 0.0),
  "joint_front_left_calf_pitch": (0.0319597, 0.498574, 0.699978, 0.0),
  "joint_front_right_abad": (0.00918881, 0.398799, 0.1453, 0.0),
  "joint_front_right_thigh_pitch": (0.0031537, 0.433033, 0.110813, 0.0),
  "joint_front_right_calf_pitch": (0.0303951, 0.412071, 0.699919, 0.0),
  "joint_rear_left_abad": (0.0120249, 0.437784, 0.144832, 0.0),
  "joint_rear_left_thigh_pitch": (0.00476253, 0.443508, 0.121348, 0.0),
  "joint_rear_left_calf_pitch": (0.0309388, 0.360797, 0.504975, 0.0),
  "joint_rear_right_abad": (0.0144851, 0.440689, 0.150869, 0.0),
  "joint_rear_right_thigh_pitch": (0.00548627, 0.449345, 0.181384, 0.0),
  "joint_rear_right_calf_pitch": (0.0321416, 0.380179, 0.506328, 0.0),
}
