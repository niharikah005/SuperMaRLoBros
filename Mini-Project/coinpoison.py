import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv
from collections import namedtuple, deque

GRID_SIZE = 40
GRID_WIDTH = 17
GRID_HEIGHT = 17
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT
BG_COLOR = (30, 30, 30)
PLAYER_COLOR = (0, 255, 0)
COIN_COLOR = (255, 215, 0)
POISON_COLOR = (255, 0, 0)

# for coin block
class Coin(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill(COIN_COLOR)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x * GRID_SIZE, y * GRID_SIZE)

# poison block
class Poison(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill(POISON_COLOR)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x * GRID_SIZE, y * GRID_SIZE)

# player
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((GRID_SIZE, GRID_SIZE))
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.rect.topleft = (0, 0)
        self.score = 0

    def move(self, dx, dy):
        new_rect = self.rect.move(dx * GRID_SIZE, dy * GRID_SIZE)
        if 0 <= new_rect.left < SCREEN_WIDTH and 0 <= new_rect.top < SCREEN_HEIGHT:
            self.rect = new_rect
        else:
            self.score -= 1 # hit the boundary

class GameEnv:
    min_reward = -100  
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("coin&poison")
        self.font = pygame.font.Font(None, 36)
        self.reset()

    def reset(self):
        self.player = Player()
        self.coins = pygame.sprite.Group()
        self.poisons = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)
        self.spawn_objects()
        self.done = False
        self.last_score = self.player.score
        self.coins_collected = 0  
        return self.get_frame()

    def spawn_objects(self):
        avoid_positions = {(self.player.rect.x // GRID_SIZE, self.player.rect.y // GRID_SIZE)} # to avoid spawning coins and poison over each other and player

        for _ in range(10):  # spawn 10 poisons
            x, y = self.random_position(avoid_positions)
            poison = Poison(x, y)
            self.poisons.add(poison)
            self.all_sprites.add(poison)
            avoid_positions.add((x, y))

        for _ in range(20): # spawn 20 coins
            x, y = self.random_position(avoid_positions)
            coin = Coin(x, y)
            self.coins.add(coin)
            self.all_sprites.add(coin)
            avoid_positions.add((x, y))

    def random_position(self, avoid_positions):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in avoid_positions:
                return x, y

    def step(self, action):
        if action == 0:
            self.player.move(-1, 0)
        elif action == 1:
            self.player.move(1, 0)
        elif action == 2:
            self.player.move(0, -1)
        elif action == 3:
            self.player.move(0, 1)

        self.all_sprites.update()
        collected_coins = pygame.sprite.spritecollide(self.player, self.coins, True)
        collected_count = len(collected_coins)
        self.coins_collected += collected_count
        self.player.score += collected_count 
        reward = 5 * collected_count # basically each coin collected gives +5 reward

        if self.player.score < -100: # score goes too low
            self.done = True
            reward = -100  

        if not self.coins and not pygame.sprite.spritecollideany(self.player, self.poisons):
            self.done = True
            reward = 100 # all coins collected

        if pygame.sprite.spritecollideany(self.player, self.poisons):
            self.done = True
            reward = -20 # hit poison  

        #print(f"Player Score: {self.player.score}, Reward: {reward}, Done: {self.done}")

        state = self.get_frame()
        return state, reward, self.done


    def calculate_reward(self):
        if pygame.sprite.spritecollideany(self.player, self.poisons):
            self.done = True
            return -10
        elif len(self.coins) == 0:
            self.done = True
            return 100
        else:
            return 10 * len(pygame.sprite.spritecollide(self.player, self.coins, False))

    def get_frame(self):
        self.screen.fill(BG_COLOR)  
        self.all_sprites.draw(self.screen) 
        score_text = self.font.render(f"Score: {self.player.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()  
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        return self.preprocess_frame(frame)

    def preprocess_frame(self, frame): 
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        resized = cv.resize(gray, (128, 128), interpolation=cv.INTER_AREA)
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit() 

        self.get_frame() 

    def close(self):
        pygame.quit()

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=state_dim[0], out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        
        self._to_linear = 64 * 16 * 16  

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.batch_size = 64
        self.target_update = 10

        self.q_network = QNetwork(state_dim, action_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_network = QNetwork(state_dim, action_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(10000)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        q_values = self.q_network(state)
        return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        q_values = self.q_network(state_batch).gather(1, action_batch)
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values)

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_dqn(agent, env, num_episodes=200):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, next_state, reward)
            agent.train()
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward} | Coins Collected: {env.coins_collected}")

        if episode % agent.target_update == 0:
            agent.update_target_network()

def test_agent(agent, env, num_episodes=4):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        print(f"Testing Episode {episode + 1}")

        while not done:
            env.render() 
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            state = next_state

        print(f"Testing Episode {episode + 1} Complete | Total Reward: {total_reward} | Coins Collected: {env.coins_collected}")

env = GameEnv()

state_dim = (1, 128, 128)
action_dim = 4
agent = DQNAgent(state_dim, action_dim)
train_dqn(agent, env)

test_agent(agent, env)

env.close()
