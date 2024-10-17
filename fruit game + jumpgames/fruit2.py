import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# --- Game Environment ---

class Field:
    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.clear_field()

    def clear_field(self):
        self.body = np.zeros((self.height, self.width))

    def update(self, fruits, player):
        self.clear_field()
        for fruit in fruits:
            if not fruit.out_of_field:
                self.body[fruit.y:fruit.y + fruit.height, fruit.x:fruit.x + fruit.width] = 1
        self.body[player.y, player.x:player.x + player.width] = 3

class Fruit:
    def __init__(self, field, height=1, width=1, x=None, y=0, speed=1):
        self.field = field
        self.height = height
        self.width = width
        self.x = x if x is not None else np.random.randint(0, self.field.width - self.width)
        self.y = y
        self.speed = speed
        self.out_of_field = False
        self.is_caught = 0

    def move(self):
        self.y += self.speed
        self.out_of_field = self.y >= self.field.height

    def check_caught(self, player):
        if self.y == player.y and player.x <= self.x < player.x + player.width:
            self.is_caught = 1
        elif self.y == player.y:
            self.is_caught = -1
        else:
            self.is_caught = 0

class Player:
    def __init__(self, field, height=1, width=3):
        self.field = field
        self.height = height
        self.width = width
        self.x = field.width // 2 - width // 2
        self.y = field.height - 1
        self.dir = 0
        self.colour = "blue"

    def move(self):
        self.x = (self.x + self.dir) % self.field.width
        self.dir = 0

    def set_action(self, action):
        self.dir = -1 if action == 1 else 1 if action == 2 else 0

class Environment:
    def __init__(self):
        self.reset()

    def get_state(self):
        return self.field.body.flatten() / 2  # becomes nowas [0, 1]

    def reset(self):
        self.game_tick = 0
        self.score = 0
        self.fruits = []
        self.field = Field()
        self.player = Player(self.field)  #bring new fruits or like to keep bringing iin new fruits
        self.spawn_fruit()
        self.field.update(self.fruits, self.player)
        return self.get_state()

    def spawn_fruit(self):
        if len(self.fruits) < 1:
            self.fruits.append(Fruit(field=self.field))
            self.next_spawn_tick = self.game_tick + 20

    def step(self, action=None):
        self.game_tick += 1
        if self.game_tick >= getattr(self, 'next_spawn_tick', 0) or not self.fruits:
            self.spawn_fruit()

        if action is not None:
            self.player.set_action(action)
        self.player.move()

        reward, in_field_fruits = 0, []
        for fruit in self.fruits:
            fruit.move()
            fruit.check_caught(self.player)
            if fruit.is_caught == 1:
                reward = 1
                self.score += reward
            elif fruit.is_caught == -1:
                reward = -1
                self.score += reward
            if not fruit.out_of_field:
                in_field_fruits.append(fruit)
        self.fruits = in_field_fruits

        self.field.update(self.fruits, self.player)

        done = self.score <= -5 or self.score >= 5
        return self.get_state(), reward, done, self.score

    def render(self, screen):
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, pygame.Color(self.player.colour),
                         pygame.Rect(self.player.x * DRAW_MUL, self.player.y * DRAW_MUL,
                                     self.player.width * DRAW_MUL, self.player.height * DRAW_MUL))
        for fruit in self.fruits:
            pygame.draw.rect(screen, pygame.Color("green" if fruit.is_caught == 1 else "red"),
                             pygame.Rect(fruit.x * DRAW_MUL, fruit.y * DRAW_MUL,
                                         fruit.width * DRAW_MUL, fruit.height * DRAW_MUL))
        pygame.display.flip()

# --- DQN Agent ---

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, replay_buffer_size=10000, batch_size=64, gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size, self.action_size = state_size, action_size
        self.memory = deque(maxlen=replay_buffer_size)
        self.epsilon = epsilon_start
        self.q_network, self.target_network = QNetwork(state_size, action_size, hidden_size), QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma, self.batch_size = gamma, batch_size
        self.epsilon_end, self.epsilon_decay = epsilon_end, epsilon_decay
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            return torch.argmax(self.q_network(torch.FloatTensor(state).unsqueeze(0))[0]).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, actions, rewards, next_states, dones = torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(dones)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# --- Main Loop ---

def main():
    global DRAW_MUL
    DRAW_MUL = 70  # cell in pixels
    pygame.init()
    screen = pygame.display.set_mode((8 * DRAW_MUL, 8 * DRAW_MUL))
    clock = pygame.time.Clock()
    env = Environment()
    agent = DQNAgent(len(env.get_state()), 3)  #  left, right, stay

    num_episodes = 500
    for episode in range(num_episodes):
        state = env.reset()
        total_reward, done = 0, False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

            env.render(screen)
            clock.tick(20)  

        print(f"Episode {episode + 1}: Score {total_reward}")

        if episode % 10 == 0:
            agent.update_target_network()

    pygame.quit()

if __name__ == "__main__":
    main()
