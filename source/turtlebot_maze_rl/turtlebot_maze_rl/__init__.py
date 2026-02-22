"""TurtleBot3 Maze RL — Isaac Lab external extension."""

import os

# Convenience: advertise the extension directory so Isaac Lab can find it.
TURTLEBOT_MAZE_RL_EXT_DIR = os.path.abspath(os.path.dirname(__file__))

# Import tasks sub-package so gym.register() calls are executed on import.
from . import tasks  # noqa: F401, E402
