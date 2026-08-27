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

## Selected dynamics baseline

- Position residual scale: `0.25 rad`
- Policy/physics timing: `0.02 s / 0.005 s`, decimation `4`
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
