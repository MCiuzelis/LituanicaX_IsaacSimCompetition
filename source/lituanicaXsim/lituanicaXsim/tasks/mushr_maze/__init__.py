"""MuSHR nano v2 RC car maze navigation task — gym registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Mushr",
    entry_point=f"{__name__}.mushr_maze_env:MushrMazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mushr_maze_env:MushrMazeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MushrMazePPORunnerCfg",
    },
)
