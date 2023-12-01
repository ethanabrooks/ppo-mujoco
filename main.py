import gymnasium as gym
import pyglet

pyglet.options["headless"] = True

import miniworld  # noqa: F401, E402

env = gym.make("MiniWorld-OneRoom-v0")

print("Worked!")
