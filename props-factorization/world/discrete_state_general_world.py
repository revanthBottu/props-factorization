import gymnasium as gym
from gymnasium import spaces
import numpy as np
from world.base_world import BaseWorld


class DiscreteStateGeneralWorld(BaseWorld):
    def __init__(
        self,
        gym_env_name,
        render_mode,
        max_traj_length=20,
        env_kwargs=None,
    ):
        super().__init__(gym_env_name)
        assert render_mode in ["human", "rgb_array", None, False]
        
        if gym_env_name == "maze-sample-3x3-v0":
            self.env = gym.make(
                gym_env_name,
                enable_render=render_mode,
            )
        else:
            self.env = gym.make(
                gym_env_name, render_mode=render_mode, **(env_kwargs if env_kwargs else {})
            )
        self.steps = 0
        self.accu_reward = 0
        self.max_traj_length = max_traj_length
        self.gym_env_name = gym_env_name
        self.render_mode = render_mode
        self.env_kwargs = env_kwargs


    def reset(self, seed=None):
        try:
            state, _ = self.env.reset(seed=seed)
        except TypeError:
            state, _ = self.env.reset()
        self.steps = 0
        self.accu_reward = 0

        if self.gym_env_name == "maze-sample-3x3-v0":
            state = state[1] * 3 + state[0]
        return state

    def step(self, action):
        self.steps += 1
        state, reward, done, truncated, _ = self.env.step(action)
        self.accu_reward += reward

        if self.steps >= self.max_traj_length or truncated:
            done = True

        if self.gym_env_name == "maze-sample-3x3-v0":
            state = state[1] * 3 + state[0]
        return state, reward, done

    def get_accu_reward(self):
        return self.accu_reward
