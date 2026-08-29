# Dobot Rover asset provenance

## Source

- Repository: `/home/houser/code/dobot_rover_original`
- Commit: `03f25d36d3e6ea15e34ddbd5c665550781b18ba2`
- License: BSD 3-Clause, copied to `doc/license/dobot-rover-license`

Geometry and inertia were cross-checked from:

- `dobot_rl_gym/resources/robots/dobot/mujoco/dobot_quad.xml`
- `dobot_rl_gym/resources/robots/dobot/urdf/dobot_quad_ros.urdf`

Active control parameters come from:

- `dobot_rl_gym/legged_gym/envs/dobot/dobot_config.py`

The referenced `dobot_rl_gym` files were unmodified at copy time. The source repository
is read-only for this migration.

## Dobot quad v2 parameter update

The active MJLab asset additionally adopts robot parameters from the local Dobot quad
v2 package at `/home/houser/下载/dobot_quad_v2`:

- Kinematic hip/thigh offsets, link masses, centers of mass, and inertias come from
  `urdf/dobot_quad.urdf` and were cross-checked against
  `mujoco/scene_dobot_quad.xml`.
- Each fixed `0.05 kg` foot is fused into its `0.377 kg` calf, giving four symmetric
  `0.427 kg` calf bodies and a total robot mass of `17.2352 kg`. This deliberately
  avoids the generated v2 MJCF's asymmetric rear-right foot body and duplicated
  `0.05 kg` mass.
- The calf collision cylinder remains at radius/half-length `0.02/0.125 m`, centered
  at local `z=-0.12 m`. The foot sphere changes to radius `0.028 m`, centered at
  local `z=-0.245 m`; the four MJLab foot sites move with the sphere centers.
- Contact friction, `solimp`, and `condim` follow the v2 MuJoCo terrain defaults:
  `(1.8, 0.1, 0.01)`, `(0.8, 0.99, 0.001, 0.3, 1)`, and `3`, respectively.
  The active Case B setting uses a harder `solref=(0.01, 1)` instead of the v2
  default `(0.02, 1)`.

The MJLab robot self-collision setting remains unchanged (`contype=1`,
`conaffinity=0`). The v2 terrain include, torque motors, deployment sensors, ready
keyframe, `0.002 s` timestep, and solver defaults are not imported. MJLab continues
to provide terrain, 12 position actuators, task sensors, and the existing initial
pose. The active Case B timing is `0.0025 s` physics with decimation `8`, preserving
the `0.02 s` policy period.

## Selected control baseline

- Position residual scale: `0.25 rad`
- Policy/physics timing: `0.02 s / 0.0025 s`, decimation `8`
- PD gains: `Kp=10`, `Kd=1`
- Effort limits: abad/thigh/calf `23/23/55 Nm`
- Passive joint damping: `0.02`
- Armature: `0.000074`
- Friction loss: `0.02`

The active controller values come from the Isaac Gym task. The passive joint dynamics
are the tested Dobot MuJoCo deployment baseline and are not presented as an exact dump
of Isaac Gym's compiled DOF properties.

## Local MJCF adaptations

- Removed the terrain include; terrain is provided by MJLab `SceneCfg`.
- Changed `meshdir` to the repository-local `xmls/assets` directory.
- Removed the XML motor actuator block; MJLab creates exactly 12 position actuators.
- Replaced the deployment sensor block with the Go2-compatible `imu_ang_vel`,
  `imu_lin_vel`, `imu_lin_acc`, and `root_angmom` built-in sensors.
- Named the existing 17 primitive collision geoms without adding duplicate geometry.
- Added `FL`, `FR`, `RL`, and `RR` sites at the foot-sphere centers.
