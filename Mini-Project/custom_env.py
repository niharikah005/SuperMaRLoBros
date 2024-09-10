import gymnasium 
from gymnasium import spaces
import numpy as np
import pygame
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env

FPS = 30

class BoulderHole(gymnasium.Env):
    def __init__(self):
        super(BoulderHole, self).__init__()
        
        pygame.init()

        self.width, self.height = 800, 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

        self.floor_y = self.height - 100

        self.player_size = 50
        self.player_x = 100
        self.player_y = self.floor_y - self.player_size
        self.player_velocity = 0
        self.gravity = 1
        self.jump_power = -15
        self.is_jumping = False

        self.obstacle_width = 30  
        self.obstacle_height = 20  
        self.hole_width = self.player_size - 10
        self.obstacles = []
        self.obstacle_speed = 6
        self.spawn_timer = 0

        self.points = []
        self.score = 0
        self.count = 0 # number of obstacles successfully crossed
 
        self.safe_distance = 100

        self.observation_space = gymnasium.spaces.Dict({
            'obstacle_distance': spaces.Box(low=0, high=self.width, shape=(1,), dtype=np.float32),
            'obstacle_type': spaces.Discrete(2),  # 0 is boulder, 1 is hole
            'y_dist_to_obstacle': spaces.Box(low=-self.height, high=self.height, shape=(1,), dtype=np.float32),
            'speed': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            'player_height': spaces.Box(low=0, high=self.height, shape=(1,), dtype=np.float32),
            'time_elapsed': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            'score': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
            'boulder_size': spaces.Box(low=0, high=self.obstacle_height, shape=(1,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(2)
        
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.player_y = self.floor_y - self.player_size
        self.player_velocity = 0
        self.is_jumping = False
        self.obstacles = []
        self.points = []
        self.score = 0
        self.spawn_timer = 0
        self.count = 0

        observation = self._get_observation()
        info = {} # extra info, not needed
        return observation, info

    def step(self, action):
        reward = 0.1  # small reward for staying alive
        terminated = False
        truncated = False

        # for jumping
        if action == 1 and not self.is_jumping:
            self.player_velocity = self.jump_power
            self.is_jumping = True

        self.player_velocity += self.gravity
        self.player_y += self.player_velocity

        if self.player_y >= self.floor_y - self.player_size:
            self.player_y = self.floor_y - self.player_size
            self.is_jumping = False

        if self.obstacles:
            next_obstacle = self.obstacles[0]
            distance_to_obstacle = next_obstacle['x'] - self.player_x

            # reward for being on ground close to obstacle
            if 0 < distance_to_obstacle <= 100 and not self.is_jumping:
                reward += 7 

            # penalty for jumping too far from obstacle
            if distance_to_obstacle > 100 and action == 1:
                reward -= 10  

        self.spawn_timer += 1
        if self.spawn_timer > random.randint(30, 70):
            self._spawn_obstacle()
            if random.random() < 0.5:
                self._spawn_points()
            self.spawn_timer = 0

        obstacle_reward, terminated = self._move_obstacles()
        reward += obstacle_reward

        coin_reward = self._move_points()
        reward += coin_reward

        # 30 obstacles crossed and 15 points collected
        if self.score >= 15 and self.count >= 30:
            truncated = True
            print("insane performance ong")

        observation = self._get_observation()
        info = {}

        self.render()
        return observation, reward, terminated, truncated, info


    def _get_observation(self):
        time_elapsed = pygame.time.get_ticks() / 1000  

        if self.obstacles:
            next_obstacle = self.obstacles[0]
            distance_to_obstacle = next_obstacle['x'] - self.player_x
            obstacle_type = 0 if next_obstacle['type'] == 'boulder' else 1
            y_dist_to_obstacle = self.player_y - next_obstacle['y'] 
            obstacle_size = self.obstacle_height if obstacle_type == 0 else self.hole_width  # height for boulder, width for hole
        else:
            distance_to_obstacle = self.width
            obstacle_type = 0
            y_dist_to_obstacle = 0
            obstacle_size = 0

        return {
            'obstacle_distance': np.array([distance_to_obstacle], dtype=np.float32),
            'obstacle_type': obstacle_type,
            'y_dist_to_obstacle': np.array([y_dist_to_obstacle], dtype=np.float32),
            'speed': np.array([self.obstacle_speed], dtype=np.float32),
            'player_height': np.array([self.player_y], dtype=np.float32),
            'time_elapsed': np.array([time_elapsed], dtype=np.float32),
            'score': np.array([self.score], dtype=np.float32),
            'boulder_size': np.array([obstacle_size], dtype=np.float32),
        }

    def _spawn_obstacle(self):
        if random.random() < 0.7:  
            self.obstacles.append({'x': self.width, 'y': self.floor_y - self.obstacle_height, 'type': 'boulder'})
        else:
            self.obstacles.append({'x': self.width, 'y': self.floor_y, 'type': 'hole'})

    def _spawn_points(self):
        coin_x = self.width
        for obstacle in self.obstacles:
            if abs(obstacle['x'] - coin_x) < self.safe_distance:
                coin_x = obstacle['x'] + self.safe_distance

        coin_y = self.floor_y - 15
        self.points.append({'x': coin_x, 'y': coin_y})


    def _move_obstacles(self):
        reward = 0
        terminated = False
        for obstacle in self.obstacles[:]:
            obstacle['x'] -= self.obstacle_speed

            if obstacle['x'] < -self.obstacle_width:
                self.obstacles.remove(obstacle) 
            else:
                self.distance_to_obstacle = obstacle['x'] - self.player_x

                # successful jump over the obstacle
                if self.player_x > obstacle['x'] + self.obstacle_width and self.is_jumping:
                    reward += 30
                    self.obstacles.remove(obstacle)
                    self.count += 1
                    print("jumped over obstacle")

                # collision penalty
                elif (self.player_x + self.player_size > obstacle['x'] and self.player_x < obstacle['x'] + self.obstacle_width and self.player_y + self.player_size >= obstacle['y']):
                    terminated = True 
                    reward -= 80
                    print("termination due to collision")

        return reward, terminated


    def _move_points(self):
        reward = 0
        for point in self.points[:]:
            point['x'] -= self.obstacle_speed
            if point['x'] < -20:
                self.points.remove(point)
            elif (self.player_x + self.player_size > point['x'] and self.player_x < point['x'] + 20 and
                  self.player_y + self.player_size >= point['y']):
                self.points.remove(point)
                self.score += 1
                reward += 5 # reward for collecting points 

        return reward

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                break

        self.screen.fill(self.WHITE)
        
        pygame.draw.line(self.screen, self.BLACK, (0, self.floor_y), (self.width, self.floor_y), 2)

        pygame.draw.rect(self.screen, self.BLACK, (self.player_x, self.player_y, self.player_size, self.player_size))
        
        for obstacle in self.obstacles:
            if obstacle['type'] == 'boulder':
                pygame.draw.rect(self.screen, self.RED, (obstacle['x'], obstacle['y'], self.obstacle_width, self.obstacle_height))
            elif obstacle['type'] == 'hole':
                pygame.draw.rect(self.screen, self.GREEN, (obstacle['x'], self.floor_y, self.hole_width, 10))
        
        for point in self.points:
            pygame.draw.circle(self.screen, self.GREEN, (point['x'], point['y']), 10)

        pygame.display.flip()
        self.clock.tick(FPS*4)

    def close(self):
        pygame.quit()

# env = BoulderHole()
# obs = env.reset()

# # check_env(env)

# terminated = False
# for i in range(10):
#     while not terminated:
#         action = env.action_space.sample()
#         obs, reward, terminated, info = env.step(action)
#         env.render()
    
#     obs = env.reset()
#     terminated = False

# env.close()

env = BoulderHole()
obs = env.reset()

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir)

eval_callback = EvalCallback(env, best_model_save_path='logs/', log_path='logs/', eval_freq=1000, deterministic=True, render=False)

if check_env(env): print("env works omg")

model = model.learn(total_timesteps=40000, callback=eval_callback)

model.save('model40k')

# model = model.load('model40k', env=env)

obs, info = env.reset()
for _ in range(10):
    terminated = False
    truncated = False
    total_reward = 0
    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
    print(total_reward)
    obs, info = env.reset()

env.close()