#!/usr/bin/env python3

import gymnasium as gym


env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="human")

observation, info = env.reset(seed=42)


for _ in range(1000):

    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    if terminated or truncated:
        observation, info = env.reset()
env.close()