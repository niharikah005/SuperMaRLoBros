import pygame 
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
from collections import deque
import torch.nn.functional as F

WIDTH, HEIGHT = 640, 640
window = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FPS = 30
BG = (0, 0, 255)
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
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

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
    def __init__(self, gamma=0.99, action_space=2, epsilon_start=1.0, epsilon_decay=1000, epsilon_end=0.01, lr=0.0003, 
                 batch_size=64, replay_capacity=100000):

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = epsilon_decay
        self.batch_size = batch_size
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = DinoCNN().to(self.device) 
        self.target_network = DinoCNN().to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.memory = ReplayMemory(replay_capacity)
        self.steps_done = 0

    def select_action(self, state):
        self.epsilon = max(self.epsilon_end, self.epsilon - 1/self.decay)
        if random.random() > self.epsilon:
            state_tensor = self.dict_to_tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.policy_network(state_tensor).argmax(1).item()
            return action
        else:
            return random.randint(0, self.action_space - 1)

    def dict_to_tensor(self, state):
        return torch.FloatTensor([
            state['player_y'],
            state['player_velocity'],
            state['on_ground'],
            state['is_jumping'],
            state['nearest_enemy_distance'],
            state['nearest_enemy_height'],
            state['successful_jumps'],
            state['just_cleared_enemy']
        ])
    
    def get_q_values(self, state):
        state_tensor = self.dict_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_network(state_tensor).squeeze().tolist()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([self.dict_to_tensor(s) for s in states]).to(self.device)
        next_states = torch.stack([self.dict_to_tensor(s) for s in next_states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

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

    def keep_down(self):
        self.apply_gravity()

    def create_state(self, window):
        pygame.draw.rect(window, GROUND_COLOR, pygame.Rect(0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT))

    def is_jumping_over(self, enemy):
        return (self.is_jumping and 
                self.rect.right > enemy.rect.left and 
                self.rect.left < enemy.rect.right and 
                self.rect.bottom < enemy.rect.top)

class Enemy:
    def __init__(self, x, y, vel):
        self.x = x
        self.y = y
        self.vel = vel
        self.width = 0.5*PLAYER_WIDTH
        self.height = PLAYER_HEIGHT
        self.color = (0, 255, 0)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, window):
        self.rect.x = self.x
        self.rect.y = self.y
        pygame.draw.rect(window, self.color, self.rect)

    def move(self):
        self.x -= self.vel

def collisions(enemy, player):
    return player.rect.colliderect(enemy.rect)

def remove_enemies(enemies):
    for enemy in enemies[:]:
        enemy.move()
        if enemy.rect.x + enemy.rect.width < 0:
            enemies.remove(enemy)
    return enemies

def spawn_enemies(enemies, last_enemy_x):
    if len(enemies) < wave_length:
        min_distance = random.randint(PLAYER_WIDTH * 4, PLAYER_WIDTH * 5) # can be improved, fine for now
        enemy_x = last_enemy_x + min_distance
        enemy = Enemy(enemy_x, HEIGHT - GROUND_HEIGHT - PLAYER_HEIGHT, enemy_vel)
        enemies.append(enemy)
        last_enemy_x = enemy_x
    return enemies, last_enemy_x

def updates(window, player, enemies):
    window.fill(BG)
    player.create_state(window)
    player.draw(window)

    for enemy in enemies:
        enemy.draw(window)
    pygame.display.update()

class Env():
    def __init__(self, screen):
        self.screen = screen
        pygame.display.set_caption("Dino Game")
        self.last_enemy_x = WIDTH
        self.successful_jumps = 0

    def reset_game(self):
        self.player = Player(PLAYER_X, PLAYER_Y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.enemies = []
        self.last_enemy_x = WIDTH
        self.successful_jumps = 0

        # initial enemies
        for _ in range(wave_length):
            self.enemies, self.last_enemy_x = spawn_enemies(self.enemies, self.last_enemy_x)

    def get_state(self): # can be improved
        nearest_enemy = min(self.enemies, key=lambda e: e.x - self.player.x) if self.enemies else None
        return {
            'player_y': self.player.y / HEIGHT,
            'player_velocity': self.player.y_vel,
            'on_ground': float(self.player.on_ground),
            'is_jumping': float(self.player.is_jumping),
            'nearest_enemy_distance': (nearest_enemy.x - self.player.x) if nearest_enemy else 1.0,
            'nearest_enemy_height': nearest_enemy.height if nearest_enemy else 0,
            'successful_jumps': self.successful_jumps ,
            'just_cleared_enemy': float(any(self.player.is_jumping_over(enemy) for enemy in self.enemies))
        }

    def get_nearest_enemy_distance(self):
        if not self.enemies:
            return float('inf')
        return min(enemy.x - self.player.x for enemy in self.enemies if enemy.rect.x > self.player.rect.x)

    def step(self, action): # can be improved
        reward = 0
        
        if action == 1 and self.player.on_ground and not self.player.is_jumping:
            self.player.jump()
            reward += 1
        
        

        self.player.keep_down()
        
        state = self.get_state()
        reward += (1/ self.get_nearest_enemy_distance()) * 100
        if state['just_cleared_enemy']:
            print("jumped over")
            reward += 100
            self.successful_jumps += 1
        
        self.enemies = remove_enemies(self.enemies)
        self.enemies,self.last_enemy_x = spawn_enemies(self.enemies, self.last_enemy_x)

        for enemy in self.enemies:
            if collisions(enemy, self.player):
                reward -= 200
                done = True
                updates(self.screen, self.player, self.enemies)
                pygame.display.flip()
                pygame.time.wait(10)
                return state, reward, done

        reward += 0.1

        done = False
        if len(self.enemies) == 0:
            done = True
            reward += 500

        updates(self.screen, self.player, self.enemies)
        pygame.display.flip()

        return state, reward, done

def dqn_train(agent, env, num_episodes=1000, target_update=100):
    print("Startin training...")
    pygame.init()
    # clock = pygame.time.Clock()
    steps = 0
    best_reward = float('-inf')

    for episode in range(num_episodes):
        print(f"episode {episode}")
        env.reset_game()
        state = env.get_state()
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if not pygame.display.get_init():
                print("Pygame display closed")
                return

            action = agent.select_action(state)
            new_state, reward, done = env.step(action)
            agent.memory.append((state, action, reward, new_state, done))
            state = new_state
            total_reward += reward
            loss = agent.optimize()

            steps += 1
            # clock.tick(FPS)

            if steps % target_update == 0:
                agent.update_target_network() # continuous update

        if total_reward > best_reward:
            best_reward = total_reward

        if episode % 10 == 0:
            print(f"episode: {episode}, total reward: {total_reward}, best reward: {best_reward}, loss: {loss} , epsilon: {agent.epsilon}")
    
        pygame.time.wait(100)
    print(f"episode: {episode}, total reward: {total_reward}, best reward: {best_reward}, loss: {loss} , epsilon: {agent.epsilon}")
    pygame.quit()

if __name__ == "__main__":
    try:
        env = Env(window)
        agent = DQNAgent()
        dqn_train(agent, env, num_episodes=1000)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.quit()