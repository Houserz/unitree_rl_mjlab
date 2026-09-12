"""Dobot 辨识参数：修改本文件后重新启动训练/play 即可生效。

只用于 Dobot-Rover-Flat-Identified；原有任务不读取这组参数。
每行依次为 armature [kg m²], viscous_damping [Nm s/rad],
frictionloss [Nm], encoder_bias [rad]；编码器角度 = 仿真角度 - bias。
"""

# Source: /home/houser/dynamic_identification/pace-sim2real-mjlab/logs/analysis/all_best172_heldout_chirp30_20260910/params_best172.pt
# SHA256: 4bbf628f1465d3e9542c05e982717ad33721420943322b0bec4f9d7a013c9e99
# Generation: 172; full float32 values from params_best172.pt.

KP = 25.0
KD = 1.3  # PD derivative gain, separate from passive viscous_damping below.
# No DC torque-speed envelope; fixed 23/23/55 Nm limits live in dobot_constants.py.
# This differs from the identification model's assumed 20 rad/s DC envelope.
DELAY_STEPS = 4  # truncate(4.588862419128418); 4 × 0.0025 s = 10 ms, output torque delay.

# bias过大有外八的参数
JOINT_PARAMETERS = {
  "joint_front_left_abad": (0.006461928132921457, 0.32331764698028564, 0.2500119209289551, -0.2541595995426178),
  "joint_front_left_thigh_pitch": (2.1026849026384298e-06, 0.3511424660682678, 0.1608039140701294, 0.09434741735458374),
  "joint_front_left_calf_pitch": (0.032027434557676315, 0.3804265558719635, 0.9999680519104004, -0.13673032820224762),
  "joint_front_right_abad": (0.005926887504756451, 0.39852631092071533, 0.164006769657135, 0.1763002574443817),
  "joint_front_right_thigh_pitch": (7.467097475455375e-06, 0.33984819054603577, 0.16879802942276, 0.17658069729804993),
  "joint_front_right_calf_pitch": (0.030530080199241638, 0.3226298391819, 0.890254557132721, -0.13198478519916534),
  "joint_rear_left_abad": (0.007560172583907843, 0.42799273133277893, 0.1811484694480896, -0.1861141324043274),
  "joint_rear_left_thigh_pitch": (0.0001131162207457237, 0.343362957239151, 0.19853812456130981, -0.19383877515792847),
  "joint_rear_left_calf_pitch": (0.030819520354270935, 0.30422425270080566, 0.5740265846252441, 0.1773476004600525),
  "joint_rear_right_abad": (0.011033033952116966, 0.35105255246162415, 0.2545524835586548, 0.2480875849723816),
  "joint_rear_right_thigh_pitch": (2.132487225026125e-06, 0.3669545352458954, 0.23991501331329346, -0.11972358822822571),
  "joint_rear_right_calf_pitch": (0.031806182116270065, 0.321434885263443, 0.5955274105072021, 0.23019498586654663),
}
