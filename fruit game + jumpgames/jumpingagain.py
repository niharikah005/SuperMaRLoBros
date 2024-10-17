import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

class JumpGameEnv(gym.Env):
    def __init__(self):
        super(JumpGameEnv, self).__init__()

        # 0 for not jump, 1 for jump
        self.action_space = spaces.Discrete(2)

        # state space
        self.observation_space = spaces.Dict({
            'obstacle_distance': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            'obstacle_type': spaces.Discrete(2),  
            'speed': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            'player_height': spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            'time_elapsed': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'player_energy': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'score': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            'boulder_size': spaces.Box(low=1, high=5, shape=(1,), dtype=np.float32),
            'hole_depth': spaces.Box(low=1, high=10, shape=(1,), dtype=np.float32),
        })

        # pygame part
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Vidhi's Jumping Game")
        self.clock = pygame.time.Clock()

        self.state = None
        self.steps = 0

    def reset(self, seed=None):
        # Reset the state 
        self.state = {
            'obstacle_distance': np.array([5.0], dtype=np.float32),
            'obstacle_type': 0,  # Start with a boulder
            'speed': np.array([5.0], dtype=np.float32),
            'player_height': np.array([0.0], dtype=np.float32),
            'time_elapsed': np.array([0.0], dtype=np.float32),
            'player_energy': np.array([100.0], dtype=np.float32),
            'score': np.array([0.0], dtype=np.float32),
            'boulder_size': np.array([3.0], dtype=np.float32),
            'hole_depth': np.array([2.0], dtype=np.float32)
        }
        self.steps = 0
        return self.state, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Update state 
        if action == 1:  # Jump
            self.state['player_height'] = np.array([3.0], dtype=np.float32)
            if self.state['obstacle_type'] == 1:  # Hole
                reward = -100.0
                terminated = True  # Episode ends if jumped into a hole
            else:  # Boulder
                reward = 10.0  # Successful jump over a boulder
        else:  # No jump
            self.state['player_height'] = np.array([0.0], dtype=np.float32)
            if self.state['obstacle_type'] == 0:  # Boulder
                reward = -10.0  # Collided with a boulder

        # Update other state variables
        self.state['time_elapsed'] += np.array([1.0], dtype=np.float32)
        self.state['obstacle_distance'] -= self.state['speed']  # obstacle tocome closer
        self.steps += 1

        if self.steps >= 1000 or terminated:
            truncated = True  # End episode after n steps or if condition met

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # empty screen
        self.screen.fill((135, 206, 250))  # Sky blue 

        # road
        road_height = 100
        pygame.draw.rect(self.screen, (50, 50, 50), (0, self.screen_height - road_height, self.screen_width, road_height))  # gray road

        #
        player_height = float(self.state['player_height'])
        obstacle_distance = float(self.state['obstacle_distance'])
        score = float(self.state['score'])
        time_elapsed = float(self.state['time_elapsed'])

        # player
        player_x = 50
        player_y = self.screen_height - road_height - int(player_height)
        pygame.draw.rect(self.screen, (0, 0, 255), (player_x, player_y, 50, 50))  # Blue player

        # obstacle
        obstacle_x = self.screen_width - int(obstacle_distance)
        obstacle_y = self.screen_height - road_height
        obstacle_width = 50
        obstacle_height = 50
        if self.state['obstacle_type'] == 0:  # Boulder
            pygame.draw.rect(self.screen, (0, 255, 0), (obstacle_x, obstacle_y, obstacle_width, obstacle_height))  # Green boulder
        else:  # Hole
            pygame.draw.rect(self.screen, (255, 0, 0), (obstacle_x, obstacle_y, obstacle_width, obstacle_height))  # Red hole

        # HUD
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {int(score)}", True, (0, 0, 0))  
        self.screen.blit(score_text, (10, 10))

        ''' 
        time_text = font.render(f"Time: {int(time_elapsed)}", True, (0, 0, 0))  
        self.screen.blit(time_text, (10, 50))
        '''

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()

# Instantiate the environment
env = JumpGameEnv()

# Define and train 
model = PPO("MultiInputPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=10000)

model.save("ppo_jump_game")
model = PPO.load("ppo_jump_game", device='cpu')

# Evaluate the model using the evaluate_policy from imports
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)

# print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test 
obs = env.reset()[0]
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _info = env.step(action)
    print(f"Step: {env.steps}, Obs: {obs}, Action: {action}, Reward: {reward}")  
    env.render()
    if terminated or truncated:
        done = True
    # Add a delay to slow down the rendering (bec my screens slow  or its slow on my screen)
    pygame.time.delay(50)

env.close()
