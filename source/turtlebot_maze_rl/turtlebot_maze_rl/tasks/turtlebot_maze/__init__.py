"""TurtleBot3 maze navigation task — gym registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-TurtleBot-Maze-Direct-v0",
    entry_point=f"{__name__}.turtlebot_maze_env:TurtleBotMazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turtlebot_maze_env:TurtleBotMazeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurtleBotMazePPORunnerCfg",
    },
)
