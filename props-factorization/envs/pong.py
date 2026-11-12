import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class PongEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        # Game parameters
        self.screen_width = 800
        self.screen_height = 600
        self.paddle_width = 10
        self.paddle_height = 80
        self.ball_size = 10
        self.paddle_speed = 5
        self.ball_speed = 5
        self.max_steps = 1000

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # [up, down, noop]
        self.observation_space = spaces.Box(
            low=np.array([
                0,  # paddle1_y min
                # 0,  # paddle2_y min
                0,  # ball_x min
                0,  # ball_y min
                -self.ball_speed,  # ball_dx min
                -self.ball_speed   # ball_dy min
            ]),
            high=np.array([
                self.screen_height - self.paddle_height,  # paddle1_y max
                # self.screen_height - self.paddle_height,  # paddle2_y max
                self.screen_width - self.ball_size,       # ball_x max
                self.screen_height - self.ball_size,      # ball_y max
                self.ball_speed,   # ball_dx max
                self.ball_speed    # ball_dy max
            ]),
            dtype=np.float32
        )

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset positions
        self.paddle1_y = self.screen_height // 2 - self.paddle_height // 2
        self.paddle2_y = self.screen_height // 2 - self.paddle_height // 2
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        self.ball_dx = self.np_random.choice([-self.ball_speed, self.ball_speed])
        self.ball_dy = self.np_random.uniform(-self.ball_speed, self.ball_speed)
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.paddle1_y + self.paddle_height / 2,
            # self.paddle2_y,
            self.ball_x,
            self.ball_y,
            self.ball_dx,
            self.ball_dy
        ], dtype=np.float32)

    def step(self, action):
        # Move player paddle
        if action == 0:  # Up
            self.paddle1_y = max(0, self.paddle1_y - self.paddle_speed)
        elif action == 1:  # Down
            self.paddle1_y = min(self.screen_height - self.paddle_height, 
                               self.paddle1_y + self.paddle_speed)

        # Move AI paddle
        if self.paddle2_y + self.paddle_height/2 < self.ball_y:
            self.paddle2_y += self.paddle_speed
        else:
            self.paddle2_y -= self.paddle_speed
        self.paddle2_y = np.clip(self.paddle2_y, 0, self.screen_height - self.paddle_height)

        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball collisions with walls
        if self.ball_y <= 0 or self.ball_y >= self.screen_height - self.ball_size:
            self.ball_dy *= -1

        # Ball collisions with paddles
        reward = 0
        terminated = False
        if self.ball_dx < 0:  # Moving left
            if (self.ball_x <= self.paddle_width and
                self.paddle1_y <= self.ball_y <= self.paddle1_y + self.paddle_height):
                self._handle_paddle_hit(self.paddle1_y)
                reward += 1  # Reward for hitting paddle
        else:  # Moving right
            if (self.ball_x >= self.screen_width - self.paddle_width - self.ball_size and
                self.paddle2_y <= self.ball_y <= self.paddle2_y + self.paddle_height):
                self._handle_paddle_hit(self.paddle2_y)

        # Check for episode termination
        if self.ball_x < 0:
            terminated = True
        elif self.ball_x > self.screen_width - self.ball_size:
            terminated = True

        # Check for truncation
        truncated = False
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _handle_paddle_hit(self, paddle_y):
        self.ball_dx *= -1
        # Calculate reflection angle based on where ball hit paddle
        hit_pos = (self.ball_y - paddle_y) / self.paddle_height
        self.ball_dy = self.ball_speed * (2 * hit_pos - 1)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Pong")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((0, 0, 0))

        # Draw paddles
        pygame.draw.rect(canvas, (255, 255, 255),
                         (0, self.paddle1_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(canvas, (255, 255, 255),
                         (self.screen_width - self.paddle_width, self.paddle2_y,
                         self.paddle_width, self.paddle_height))

        # Draw ball
        pygame.draw.ellipse(canvas, (255, 255, 255),
                            (self.ball_x, self.ball_y, self.ball_size, self.ball_size))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None


gym.register(
    id="Pong-v0",
    entry_point=PongEnv,
    max_episode_steps=1000,
    reward_threshold=3.0,
)


if __name__ == "__main__":
    env = PongEnv(render_mode="human")
    observation, _ = env.reset()
    env.render()
    
    try:
        step = 0
        while True:
            action = 2  # Default: no operation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()
                
            # Get keyboard input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action = 0
            elif keys[pygame.K_DOWN]:
                action = 1

            # Step the environment
            observation, reward, terminated, truncated, _ = env.step(action)
            step += 1
            # print(step, observation, reward, terminated, truncated)
            print(reward)
            
            if terminated or truncated:
                observation, _ = env.reset()
                step = 0
            
            env.render()
    finally:
        env.close()
