import gymnasium as gym
import pyglet

pyglet.options["headless"] = True

import miniworld  # noqa: F401, E402

env = gym.make("MiniWorld-OneRoom-v0")
print(env.reset())
print(env.step(0))

print("Worked!")
