# Extended Domain Randomization config for Anymal C rough terrain.
# Inherits from the baseline and widens randomization ranges to improve
# sim-to-real transfer robustness.
#
# Changes vs baseline:
#   - Friction: fixed (0.8/0.6) → randomized (0.2, 1.5) static / (0.1, 1.2) dynamic
#   - Mass:     ±5 kg          → ±10 kg
#   - CoM:      ±0.05m x/y     → ±0.15m x/y, ±0.05m z
#   - Push:     ±0.5 m/s       → ±1.5 m/s, push interval tightened
#   - Joint friction/armature: (new) → randomized per joint to cover actuator wear

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import AnymalCRoughEnvCfg


@configclass
class AnymalCRoughEnvCfg_ExtendedDR(AnymalCRoughEnvCfg):
    """Anymal C rough terrain with extended domain randomization for sim-to-real robustness."""

    def __post_init__(self):
        super().__post_init__()

        # --- Friction: wide range covering dry concrete to wet metal ---
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.2, 1.5),
                "dynamic_friction_range": (0.1, 1.2),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
            },
        )

        # --- Mass: ±10 kg covers payload variation and model error ---
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-10.0, 10.0),
                "operation": "add",
            },
        )

        # --- Center of mass: wider shift to cover payload mounting variation ---
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (-0.05, 0.05)},
            },
        )

        # --- Motor friction: ±20% per joint to cover actuator wear and model mismatch ---
        self.events.joint_stiffness_and_damping = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.0, 0.05),
                "armature_distribution_params": (0.0, 0.01),
                "operation": "add",
                "distribution": "uniform",
            },
        )

        # --- Push: stronger and more frequent pushes ---
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
        )


@configclass
class AnymalCRoughEnvCfg_ExtendedDR_PLAY(AnymalCRoughEnvCfg_ExtendedDR):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
