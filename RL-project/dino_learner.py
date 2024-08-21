import pygame 
import random
import cv2 as cv
import numpy as np
import time
import os
import torch
from torch import nn
from collections import deque
import torch.nn.functional as F

'''
create a game in which there are two blocks, one that spawns randomly and the other 
that must jump over it. train a CNN over multiple epochs for this state and use its knowledge
upon the target network to obtain the q-values. 
The reward function will be simple: if the two blocks collide then give a negative reward or else ekkp giving
a small positive reward, make sure that the negative reward is much much larger than the positive reward
'''


WIDTH, HEIGHT = 640, 640
window = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
FPS = 60
BG = (0,0,255)
GROUND_HEIGHT = 100
GROUND_COLOR = (165, 42, 42)
PLAYER_X = WIDTH // 16
PLAYER_Y = HEIGHT - GROUND_HEIGHT - (HEIGHT // 16)
PLAYER_WIDTH = WIDTH // 16
PLAYER_HEIGHT = HEIGHT // 16
wave_length = 10
enemy_vel = 4   


class DinoCNN(nn.Module):
    def __init__(self):
        super(DinoCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*62*62, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.maxpool(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class ReplayMemory():
    def __init__(self, max_size):
        self.memory = deque([], maxlen=max_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNAgent():
    def __init__(self, gamma=0.9,action_space=2, epsilon_start=1.0, epsilon_decay=100, epsilon_end=0.01, lr=0.0001, 
                 batch_size=32, replay_capacity=10000):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = epsilon_decay
        self.batch_size = batch_size
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = DinoCNN()
        self.target_network = DinoCNN()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.memory = ReplayMemory(replay_capacity)
        self.steps_done = 0

    def select_action(self, state):
        self.epsilon = max(self.epsilon_end, self.epsilon - 1/self.decay)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # adds a batch dimension
            with torch.no_grad():
                return self.policy_network(state).argmax(1).item()
            
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

class Env():
    def __init__(self, screen):
        self.screen = screen

    def reset_game(self):
        global player, enemies, last_enemy_x
        player = Player(PLAYER_X, PLAYER_Y, PLAYER_WIDTH, PLAYER_HEIGHT)
        enemies = []
        last_enemy_x = WIDTH

        if len(enemies) == 0:
            for i in range(wave_length):
                min_distance = random.randint(PLAYER_WIDTH * 3, PLAYER_WIDTH * 4)
                enemy_x = last_enemy_x + min_distance
                enemy = Enemy(enemy_x, HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT, enemy_vel)
                enemies.append(enemy)
                last_enemy_x = enemy_x
         
    def get_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1,0,2))
        return self.preprocess_frame(frame)

    def preprocess_frame(self,frame):
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        resized = cv.resize(gray, (128,128), interpolation=cv.INTER_AREA)
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)
    
    def step(self, action, player, enemies, done):
        # Define the game step based on the action
        if action == 1:  
            player.jump()
        player.keep_down()
        # Move enemies
        for enemy in enemies[:]:
            # Check for collisions
            if collisions(enemy, player):
                reward = -10  # Large negative reward for collision
            else:
                reward = 1  # Default reward for surviving
            remove_enemies(enemy)

        if len(enemies) == 0:
            done = True
        # Create the new state
        updates(window, player)
        new_state = self.get_frame()

        return new_state, reward, done
    
def dqn_train(agent, env, num_episodes=1000, target_update=10):
    pygame.init()

    for episode in range(num_episodes):
        env.reset_game()
        state = env.get_frame()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            new_state, reward, done = env.step(action, player, enemies, done)
            agent.memory.append((state, action, reward, new_state, done))
            state = new_state
            total_reward += reward
            agent.optimize()
            clock.tick(FPS)

        if episode % target_update == 0:
            agent.update_target_network()
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    pygame.quit()


def dqn_test(agent, env, episodes=10, step_count=1000):
    pygame.init()

    for episode in range(episodes):
        env.reset_game()
        state = env.get_frame()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < step_count:
            steps += 1
            action = agent.select_action(state)
            new_state, reward, done = env.step(action, player, enemies, done)
            total_reward += reward
            state = new_state
            updates(window, player)
            clock.tick(FPS)

            if done:
                print(f"Test Episode {episode + 1}/{episodes} finished after {step_count} steps with total reward {total_reward}")
                break

    pygame.quit()


class Player:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.x_vel = 0
        self.y_vel = 0
        self.width = width
        self.height = height
        self.color = (255,0,0)
        self.GRAVITY = 1
        self.ground =  pygame.Rect(0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height) 
        self.fall_count = 0
        self.flag = False
        self.jump_cooldown = 10
        self.last_jump_time = time.time()

    def draw(self, window):
        self.flag = False
        self.rect.x += self.x_vel 
        self.rect.y += self.y_vel
        pygame.draw.rect(window, self.color, self.rect)

    def jump(self):
        current_time = time.time()
        if current_time - self.last_jump_time >= self.jump_cooldown:
            self.flag = True
            self.fall_count = 0
            self.y_vel -= self.GRAVITY * 10
            self.last_jump_time = current_time

    def keep_down(self):
        if self.rect.bottom != self.ground.top:
            self.fall_count += 1
            if self.y < HEIGHT - GROUND_HEIGHT:
                self.y_vel += 0.8 
        elif not self.flag:
            self.y_vel = 0

    def create_state(self, window):
        pygame.draw.rect(window, GROUND_COLOR, self.ground)

class Enemy():
    def __init__(self, x, y, velocity):
        self.x = x
        self.y = y
        self.x_vel = 0
        self.y_vel = 0
        self.color = (0,255,0)
        self.rect = pygame.Rect(self.x, self.y, 10, PLAYER_HEIGHT)
        self.enemy_velocity = velocity
        
    def move(self):
        self.rect.x -= self.enemy_velocity
    
    def draw(self, window):
        self.move()
        pygame.draw.rect(window, self.color, self.rect)

    

def collisions(enemy, player):
    if enemy.rect.colliderect(player.rect):
        return True
    return False


def remove_enemies(enemy):
    if enemy.rect.x < 0:
        enemies.remove(enemy)


def updates(window, player):
    window.fill(BG)
    player.create_state(window)
    for enemy in enemies:
        enemy.draw(window)
    player.draw(window)
    pygame.display.update()

player = Player(PLAYER_X, PLAYER_Y, PLAYER_WIDTH, PLAYER_HEIGHT)
agent = DQNAgent()
env = Env(window)
enemies = []
last_enemy_x = WIDTH


dqn_train(agent, env)
dqn_test(agent, env)

pygame.quit()
