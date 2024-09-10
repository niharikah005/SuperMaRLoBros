import cv2
import pygame
import random
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3 import PPO
from gymnasium.wrappers import frame_stack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

#constants
SCREEN_WIDTH, SCREEN_HEIGHT = 480, 480
REGION_SIZE = 175
GROUND_HEIGHT = 50
GROUND_COLOR = (0,255,0)
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 40
BOULDER_COLOR = (255,0,0)
HOLE_COLOR = (0,0,0)
PLAYER_COLOR = (0,0,255)
OBSTACLE_HEIGHT = 35
OBSTACLE_WIDTH = 20
OBSTACLE_VELOCITY = 5
WAVELENGTH = 10
FPS = 30
clock = pygame.time.Clock()


# player class:
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.jump_strength = 20
        self.give_gravity = 2
        self.y_vel = 0
        self.in_air = False
    def draw(self, window):
        self.rect.x = self.x
        self.rect.y = self.y
        pygame.draw.rect(window, PLAYER_COLOR, self.rect)
    def jump(self):
        if not self.in_air:
            self.y_vel += self.jump_strength
            self.y -= self.y_vel
            self.in_air = True
    def gravity(self):
        if self.in_air:
            self.y_vel -= self.give_gravity
            self.y -= self.y_vel

            if self.y >= SCREEN_HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT:
                self.y = SCREEN_HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT
                self.y_vel = 0
                self.in_air = False


# obstacle class:
class Obstacle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.rect = pygame.Rect(self.x, self.y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)
    def move(self):
        self.x -= OBSTACLE_VELOCITY
    def draw(self,window):
        self.rect.x = self.x
        self.rect.y = self.y
        pygame.draw.rect(window, self.color, self.rect)


# ground class:
class Ground:
    def __init__(self):
        self.height = GROUND_HEIGHT
        self.rect = pygame.Rect(0, SCREEN_HEIGHT - self.height, SCREEN_WIDTH, self.height)
    def draw(self, window):
        pygame.draw.rect(window, GROUND_COLOR, self.rect)


# utils:
def spawn_obstacles():
    _last_obstacle_x = SCREEN_WIDTH
    _obstacle_list = []
    for _ in range(WAVELENGTH):
        _obstacle_type = random.choice([0,1])
        if _obstacle_type == 0:
            _obstacle = Obstacle(_last_obstacle_x, SCREEN_HEIGHT - GROUND_HEIGHT - OBSTACLE_HEIGHT, BOULDER_COLOR)
        elif _obstacle_type == 1:
            _obstacle = Obstacle(_last_obstacle_x, SCREEN_HEIGHT - GROUND_HEIGHT, HOLE_COLOR)
        _obstacle_list.append(_obstacle)
        _last_obstacle_x +=  random.randint(PLAYER_WIDTH*3, PLAYER_WIDTH*4)
    return _obstacle_list


def check_for_collision(_obstacle, agent):
    if _obstacle.y == SCREEN_HEIGHT - GROUND_HEIGHT - OBSTACLE_HEIGHT:
        return agent.rect.colliderect(_obstacle.rect)
    elif _obstacle.y == SCREEN_HEIGHT - GROUND_HEIGHT:
        return (agent.rect.y + agent.rect.height == _obstacle.y
                and _obstacle.rect.x < agent.rect.x + agent.rect.width
                and agent.rect.x < _obstacle.rect.x + _obstacle.rect.width)


def nearest_enemy(obstacle_list, agent):
    _min_distance = float('inf')
    for _obstacle in obstacle_list:
        if _obstacle.x > agent.x + agent.rect.width:
            _min_distance = min(_min_distance,(_obstacle.x - (agent.x + agent.rect.width)))
    return _min_distance


def update_screen(window, agent, obstacle_list, ground):
    window.fill((0,0,0))
    ground.draw(window)
    agent.draw(window)
    for _obstacle in obstacle_list:
        _obstacle.draw(window)
    pygame.display.flip()
    # clock.tick(FPS)


# custom env:
class JumpEnv(gym.Env):
    def __init__(self):
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._agent = Agent(80, SCREEN_HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT)
        self._ground = Ground()
        self._obstacle_list = []
        self.truncated = False
        self.terminated = False
        self._steps = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=((REGION_SIZE+GROUND_HEIGHT)*REGION_SIZE,), dtype=np.float32)

    def step(self, action):
        self.truncated = False
        self.terminated = False
        reward = 0
        if action == 1 and not self._agent.in_air:
            self._agent.jump()
        self._agent.gravity()
        self._steps += 1

        for _obstacle in self._obstacle_list:
            _obstacle.move()
        obs = self.get_obs()
        _min_distance = nearest_enemy(self._obstacle_list, self._agent)
        # rewards:
        for _obstacle in self._obstacle_list:
            if check_for_collision(_obstacle, self._agent):
                reward -= 150
                self.terminated = True
                return obs, reward, self.terminated, self.truncated, {}
        
        if self._agent.in_air and _min_distance >= 30:
            reward -= 1
        else:
            reward += 2   

        # termination:
        if self._steps >= 500:
            reward += 100
            self.truncated = True
        
        # draw:
        update_screen(self._screen, self._agent, self._obstacle_list, self._ground)
        return obs, reward, self.terminated, self.truncated, {}
    
    def reset(self, seed=None):
        self.terminated = False
        self.truncated = False
        self._steps = 0
        self._agent = Agent(80, SCREEN_HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT)

        # spawn enemies
        self._obstacle_list = spawn_obstacles()
        return self.get_obs(), {}
    
    def get_obs(self):
        # Define the region around the agent
        forward_offset = REGION_SIZE // 2
        x = max(0, self._agent.x + self._agent.rect.width*2 - forward_offset)
        y = SCREEN_HEIGHT - GROUND_HEIGHT - REGION_SIZE
        region = pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)
        region = region[y:y+REGION_SIZE+GROUND_HEIGHT, x:x+REGION_SIZE, :] # takes a region_size * region_size part into consideration
        grayscale_region = np.mean(region, axis=2).astype(np.float32)  # Use float32 for normalization
        normalized_region = grayscale_region / 255.0
        flattened_obs = normalized_region.flatten()
        
        cv2.imshow('Agent view', normalized_region)
        cv2.waitKey(1)
        return flattened_obs
    def close(self):
        pygame.quit()
        cv2.destroyAllWindows()

log_dir = './logs/'
env = JumpEnv()
# check_env(env)
model = PPO('MlpPolicy', env, tensorboard_log=log_dir, verbose=1)
model.learn(total_timesteps=40000)
model.save('using_PPO_new')
# model.load('using_PPO')
# obs, _ = env.reset()
# for _ in range(5):
#     truncated = False
#     terminated = False
#     total_reward = 0
#     while not truncated and not terminated:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, _ = env.step(action)
#         total_reward += reward
#         if truncated or terminated:
#             print(total_reward)
#             print('terminated')
#             obs, _ = env.reset()
env.close()
    
