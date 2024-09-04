import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
import pygame
import random

# constants
BG = (255, 255, 255)
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
FPS = 60


# player class
class Player:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 0, 0)
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

    
    def check_landing(self):
        return (self.rect.y + self.rect.width <= self.ground + 1)

    def keep_down(self):
        self.apply_gravity()

    def create_state(self, window):
        pygame.draw.rect(window, GROUND_COLOR, pygame.Rect(0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT))

    def is_jumping_over(self, obstacles):
        return (self.is_jumping and 
                self.rect.x + self.rect.width > obstacles.rect.x 
                and self.rect.x < obstacles.rect.x + obstacles.rect.width 
                and self.rect.y + self.rect.height < obstacles.rect.y)

# obstacles class
class Enemy:
    def __init__(self, x, y, vel):
        self.x = x
        self.y = y
        self.vel = vel
        self.width = 0.5*PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.color = (0, 0, 255)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, window):
        self.rect.x = self.x
        self.rect.y = self.y
        pygame.draw.rect(window, self.color, self.rect)

    def move(self):
        self.x -= self.vel

# extra functions
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
    type = random.choice([0,1])
    min_distance = random.randint(PLAYER_WIDTH * 4, PLAYER_WIDTH * 5) # can be improved, fine for now
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
    obstacle = [obstacle for obstacle in obstacles if obstacle.rect.x > agent.rect.x]
    if obstacle:
        closest_obstacle = min(obstacle, key=lambda obs: obs.rect.x)
        min_distance = closest_obstacle.rect.x
        return min_distance, closest_obstacle
    else:
        return None, None
    
def find_passed_enemy(obstacles, agent):
    obstacle = [obstacle for obstacle in obstacles if obstacle.rect.x + obstacle.rect.width < agent.rect.x]
    closest_obstacle = max(obstacle, key=lambda obs: obs.x) if obstacle else None
    if closest_obstacle:
        return True
    else:
        return False


class JumpGameEnv(gym.Env):
    def __init__(self):
        super(JumpGameEnv, self).__init__()
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.state = None
        self.steps = 0
        self.player_pos = (PLAYER_X, PLAYER_Y) 
        self.obstacles = []
        self.last_obs = WIDTH//2
        self.min_distance = 0.0
        self.steps = 0
        self.jumping = 0
        self.successful_jumps = 0.0

        pygame.display.set_caption("Jump Game")
        self.clock = pygame.time.Clock().tick(FPS)
        '''Define action space: 0 for not jump, 1 for jump'''
        self.action_space = gym.spaces.Discrete(2)
        '''Define state space: a dict with 8 values (can be integers, floats, etc.)'''
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
        self.last_obs = WIDTH//2
        self.obstacles = []
        self.jumping = 0
        self.steps = 0
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
        '''Update the Pygame window with the current game state'''
        

    def step(self, action): # can be improved
        reward = 0
        terminated = False
        truncated = False
        if action == 1 and self.agent.on_ground and not self.agent.is_jumping:
            self.agent.jump()
            self.steps += 1

        self.agent.keep_down()
        self.min_distance, obstacle = find_nearest_enemy(self.obstacles, self.agent)

        for obstacle in self.obstacles:
            if collisions(obstacle, self.agent):
                reward -= 300
                terminated = True
                updates(self.window, self.agent, self.obstacles)
                pygame.display.flip()
                pygame.time.wait(10)
                return self.state, np.float32(reward), terminated, truncated, {}

        if (self.agent.is_jumping 
            and self.min_distance <= self.agent.rect.x + self.agent.rect.width + 40 
            and self.agent.rect.y + PLAYER_HEIGHT  >= HEIGHT - GROUND_HEIGHT - 20
            and not self.jumping):
            print('success')
            reward += 100
            self.jumping = True
        # elif self.agent.is_jumping and self.min_distance > self.agent.rect.x + self.agent.rect.width + 15 and not self.jumping:
        #     reward -= 0.01
        if find_passed_enemy(self.obstacles, self.agent) and self.agent.check_landing() and self.jumping: 
            print('good jump')
            reward += 30
            self.jumping = False
            self.successful_jumps += 1
        
        
        self.obstacles = remove_enemies(self.obstacles)
        self.obstacles, self.last_obs = spawn_enemies(self.obstacles, self.last_obs)

        self.state = {
            'player_y': np.array([self.agent.y / HEIGHT], dtype=np.float32),
            'player_y_vel': np.array([self.agent.y_vel / self.agent.jump_strength], dtype=np.float32),
            'on_ground': 1 if self.agent.on_ground else 0,
            'is_jumping': 1 if self.agent.is_jumping else 0,
            'nearest_enemy': np.array([max(self.min_distance - self.agent.rect.x / WIDTH, 0.0) if self.min_distance else 1.0], dtype=np.float32),
            'nearest_enemy_height': np.array([-1.0 if obstacle and obstacle.y == (HEIGHT - GROUND_HEIGHT) else 1.0], dtype=np.float32),
            'successful_jumps': np.array([self.successful_jumps / 100.0], dtype=np.float32),
            'just_cleared_enemy': 1 if any(self.agent.is_jumping_over(obstacle) for obstacle in self.obstacles) else 0
        }

        truncated = False
        if self.steps >= 100:
            reward += 100
            truncated = True

        updates(self.window, self.agent, self.obstacles)
        clock.tick(FPS)
        info = {}
        return self.state, np.float32(reward), terminated, truncated, info


    def close(self):
        # Clean up resources (if needed)
        pygame.quit()

# Instantiate the environment
env = JumpGameEnv()
model = DQN('MultiInputPolicy', env, learning_rate=0.001, target_update_interval=100, exploration_fraction=0.85, verbose=1)
model.learn(total_timesteps=3500)
model.save('dqn_check')
# env = JumpGameEnv()
# model.load('dqn_check', env=env)
# # Test the environment
obs, info = env.reset()
for _ in range(5):
    terminated = False
    truncated = False
    total_reward = 0
    while not terminated and not truncated:
        action, _ = model.predict(obs)  # Randomly sample an action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(total_reward)
    obs, info = env.reset()

env.close()

# make the checks for each jump, not each frame, and change some of the rewards
