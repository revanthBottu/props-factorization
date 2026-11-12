import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NimEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, initial_sticks=10):
        super().__init__()
        self.initial_sticks = initial_sticks
        self.current_sticks = self.initial_sticks

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=1, 1=2, 2=3 sticks
        self.observation_space = spaces.Discrete(11)  # 0-10 sticks

        # Rendering setup
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sticks = self.initial_sticks
        self.terminated = False
        self.truncated = False
        info = {}
        return self.current_sticks, info

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        # Validate action
        sticks_to_remove = action  # Convert action to 1-3 sticks

        # Invalid action check
        if (
            sticks_to_remove < 1
            or sticks_to_remove > 3
            or sticks_to_remove > self.current_sticks
        ):
            self.terminated = True
            reward = -1
            return self.current_sticks, reward, True, False, {"message": "Invalid move"}

        # Agent's move
        self.current_sticks -= sticks_to_remove

        # Check if agent lost
        if self.current_sticks == 0:
            self.terminated = True
            reward = -1
            return 0, reward, True, False, {}

        # # Environment's move (random valid action)
        # possible_actions = list(range(1, min(3, self.current_sticks) + 1))
        # env_action = self.np_random.choice(possible_actions)

        # Environment's move (optimal strategy)
        env_action = (self.current_sticks + 3) % 4
        if env_action == 0:
            possible_actions = list(range(1, min(3, self.current_sticks) + 1))
            env_action = self.np_random.choice(possible_actions)
        self.current_sticks -= env_action

        # Check if environment lost
        if self.current_sticks == 0:
            self.terminated = True
            reward = 1
            return 0, reward, True, False, {}

        # Game continues
        reward = 0
        return self.current_sticks, reward, False, False, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Sticks remaining: {self.current_sticks}")

    def close(self):
        pass


gym.register(
    id="Nim-v0",
    entry_point=NimEnv,
)
