import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DQN
import pygame
import random, os

BG = (135, 206, 250)  
ROAD_COLOR = (128, 128, 128)  
WIDTH = 640
HEIGHT = 480
GROUND_HEIGHT = 50
PLAYER_X = WIDTH // 16
PLAYER_Y = HEIGHT - GROUND_HEIGHT - (HEIGHT // 16)
PLAYER_WIDTH = WIDTH // 16
PLAYER_HEIGHT = HEIGHT // 16
GROUND_COLOR = (0, 255, 0)
obstacle_vel = 4
wave_length = 10
clock = pygame.time.Clock()
FPS = 30

class Player:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (0, 0, 255)  
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.y_vel = 0
        self.gravity = 1
        self.jump_strength = 15
        self.ground = HEIGHT - GROUND_HEIGHT - height
        self.on_ground = True
        self.action_locked = False
        self.is_jumping = False
        self.successful_jumps = []

    def draw(self, window):
        self.rect.x = self.x
        self.rect.y = self.y
        pygame.draw.rect(window, self.color, self.rect)

    def jump(self):
        if self.on_ground and not self.action_locked and not self.is_jumping:
            self.y_vel = -self.jump_strength
            self.on_ground = False
            self.action_locked = True
            self.is_jumping = True

    def apply_gravity(self):
        if not self.on_ground:
            self.y_vel += self.gravity
            self.y += self.y_vel

            if self.y >= self.ground:
                self.y = self.ground
                self.y_vel = 0
                self.on_ground = True
                self.action_locked = False
                self.is_jumping = False

        return np.array([self.y - PLAYER_Y / 45], dtype=np.float32)

    def keep_down(self):
        self.apply_gravity()

    def create_state(self, window):
        pygame.draw.rect(window, ROAD_COLOR, pygame.Rect(0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT))

    def is_jumping_over(self, obstacles):
        return (self.is_jumping and 
                self.rect.x + self.rect.width > obstacles.rect.x 
                and self.rect.x < obstacles.rect.x + obstacles.rect.width 
                and self.rect.y + self.rect.height < obstacles.rect.y)

class Enemy:
    def __init__(self, x, y, vel):
        self.x = x
        self.y = y
        self.vel = vel
        self.width = 0.5 * PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        if y == HEIGHT - GROUND_HEIGHT:
            self.color = (139, 69, 19) 
        else:
            self.color = (0, 255, 0) 
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, window):
        self.rect.x = self.x
        self.rect.y = self.y
        pygame.draw.rect(window, self.color, self.rect)

    def move(self):
        self.x -= self.vel

def collisions(obstacles, player):
    if obstacles.y == HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT:
        return player.rect.colliderect(obstacles.rect)
    elif obstacles.y == HEIGHT - GROUND_HEIGHT:
        return ((player.rect.y + player.rect.height == obstacles.rect.y) 
                and (obstacles.rect.x < player.rect.x + player.rect.width) 
                and (player.rect.x < obstacles.rect.x + obstacles.rect.width))

def remove_enemies(obstacles):
    for obstacle in obstacles[:]:
        obstacle.move()
        if obstacle.rect.x + obstacle.rect.width < 0:
            obstacles.remove(obstacle)
    return obstacles

def spawn_enemies(obstacles, last_obs_x):
    type = random.choice([0, 1])
    min_distance = random.randint(PLAYER_WIDTH * 3, PLAYER_WIDTH * 4)
    obs_x = last_obs_x + min_distance
    if type == 0:
        obstacle = Enemy(obs_x, HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT, obstacle_vel)
    else:
        obstacle = Enemy(obs_x, HEIGHT - GROUND_HEIGHT, obstacle_vel)
    obstacles.append(obstacle)
    last_obs_x = obs_x
    return obstacles, last_obs_x

def updates(window, player, obstacles):
    window.fill(BG)
    player.create_state(window)
    player.draw(window)

    for obstacle in obstacles:
        obstacle.draw(window)
    pygame.display.flip()

def find_nearest_enemy(obstacles, agent):
    obstacle = [obstacle for obstacle in obstacles if obstacle.x > agent.x]
    if obstacle:
        closest_obstacle = min(obstacle, key=lambda obs: obs.x)
        min_distance = closest_obstacle.x
        return min_distance, closest_obstacle
    else:
        return None, None

class JumpGameEnv(gym.Env):
    def __init__(self):
        super(JumpGameEnv, self).__init__()
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.state = None
        self.steps = 0
        self.player_pos = (PLAYER_X, PLAYER_Y)
        self.obstacles = []
        self.last_obs = WIDTH // 2
        self.min_distance = 0.0
        self.successful_jumps = 0.0
        self.score = 0  

        pygame.display.set_caption("Vidhi's Jump Game")
        self.clock = pygame.time.Clock().tick(FPS)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            'player_y': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'player_y_vel': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'on_ground': spaces.Discrete(2),
            'is_jumping': spaces.Discrete(2),
            'nearest_enemy': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'nearest_enemy_height': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'successful_jumps': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'just_cleared_enemy': spaces.Discrete(2)
        })

    def reset(self, seed=None):
        self.agent = Player(PLAYER_X, PLAYER_Y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.steps = 0
        self.jumping = False
        self.jump_height = 0.0
        self.jump_velocity = 0.0
        self.successful_jumps = 0
        self.score = 0 #to reset
        self.last_obs = WIDTH // 2
        self.obstacles = []
        info = {}
        self.min_distance = 0.0

        for _ in range(wave_length):
            self.obstacles, self.last_obs = spawn_enemies(self.obstacles, self.last_obs)
        self.min_distance, obstacle = find_nearest_enemy(self.obstacles, self.agent)

        self.state = {
            'player_y': np.array([self.agent.y / HEIGHT], dtype=np.float32),
            'player_y_vel': np.array([self.agent.y_vel / self.agent.jump_strength], dtype=np.float32),
            'on_ground': 1 if self.agent.on_ground else 0,
            'is_jumping': 1 if self.agent.is_jumping else 0,
            'nearest_enemy': np.array([max(self.min_distance / WIDTH, 0.0) if self.min_distance else 1.0], dtype=np.float32),
            'nearest_enemy_height': np.array([-1.0 if obstacle and obstacle.y == (HEIGHT - GROUND_HEIGHT) else 1.0], dtype=np.float32),
            'successful_jumps': np.array([self.successful_jumps / 100.0], dtype=np.float32),
            'just_cleared_enemy': 1 if any(self.agent.is_jumping_over(obstacle) for obstacle in self.obstacles) else 0
        }

        return self.state, info

    def render(self, mode="render_human"):
        updates(self.window, self.agent, self.obstacles)
        
        # Display the score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        
        pygame.display.flip()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        # if action is to jump and if the conditions are right
        if action == 1 and self.agent.on_ground and not self.agent.is_jumping:
            self.agent.jump()
            reward += 1
        
        self.agent.keep_down()
        
        #nearest enemy and update reward
        self.min_distance, obstacle = find_nearest_enemy(self.obstacles, self.agent)
        reward += (1 / self.min_distance) * 100 if self.min_distance else 0
        
        # Penalize if jumping over an enemy too far
        if self.agent.is_jumping and self.state['nearest_enemy'] * WIDTH > PLAYER_WIDTH:
            reward -= 10
        
        # Reward for clearing an enemy (succesfull jump)
        if self.state['just_cleared_enemy']:
            reward += 100
            self.successful_jumps += 1
            self.score += 10  

        #remove enemy when out of screen and spwan new one
        self.obstacles = remove_enemies(self.obstacles)
        self.obstacles, self.last_obs = spawn_enemies(self.obstacles, self.last_obs)
        
        self.state = {
            'player_y': np.array([self.agent.y / HEIGHT], dtype=np.float32),
            'player_y_vel': np.array([self.agent.y_vel / self.agent.jump_strength], dtype=np.float32),
            'on_ground': 1 if self.agent.on_ground else 0,
            'is_jumping': 1 if self.agent.is_jumping else 0,
            'nearest_enemy': np.array([max(self.min_distance / WIDTH, 0.0) if self.min_distance else 1.0], dtype=np.float32),
            'nearest_enemy_height': np.array([-1.0 if obstacle and obstacle.y == (HEIGHT - GROUND_HEIGHT) else 1.0], dtype=np.float32),
            'successful_jumps': np.array([self.successful_jumps / 100.0], dtype=np.float32),
            'just_cleared_enemy': 1 if any(self.agent.is_jumping_over(obstacle) for obstacle in self.obstacles) else 0
        }

        for obstacle in self.obstacles:
            if collisions(obstacle, self.agent):
                reward -= 200
                terminated = True
                self.render()
                pygame.display.flip()
                pygame.time.wait(10)
                return self.state, np.float32(reward), terminated, truncated, {}

        # Reward for surviving, penalty for jumping too far
        reward += 0.1
        self.steps += 1

        done = False
        if len(self.obstacles) == 0:
            truncated = True
            reward += 500

        self.render()
        clock.tick(FPS*4)
        info = {}
        
        print(f"Step: {self.steps}, Reward: {reward}, Score: {self.score}")  # Print reward and score for each step
        
        return self.state, np.float32(reward), terminated, truncated, info

    def close(self):
        pygame.quit()

env = JumpGameEnv()

log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

model = DQN('MultiInputPolicy', env, target_update_interval=100, exploration_fraction=0.45, verbose=1, tensorboard_log=log_dir)

eval_callback = EvalCallback(env, best_model_save_path='logs/', log_path='logs/', eval_freq=1000, deterministic=True, render=False)

model.learn(total_timesteps=65000)
model.save('dqn_check')

env = JumpGameEnv()
model.load('dqn_check', env=env)

obs, info = env.reset()
for _ in range(5):
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Score: {env.score}")  
    obs, info = env.reset()

env.close()
