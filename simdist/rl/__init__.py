import gymnasium as gym

from simdist import rl

gym.register(
    id="Go2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2:Go2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{rl.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2:Go2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{rl.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)
