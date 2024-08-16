import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 4x4 maze with 0 as free path and 1 as obstacle
maze = np.array([
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])

# Hyperparameters
ENV_ROWS, ENV_COLS = 4, 4
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_MEMORY_SIZE = 10000
BATCH_SIZE = 32

# Q-network model
class DQN(nn.Module):
    def __init__(self, in_states, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, out_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Replay memory
class ReplayMemory():
    def __init__(self, max_size):
        self.memory = deque([], maxlen=max_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Agent class
class MazeSolver_DQN():
    def __init__(self):
        self.model = DQN(in_states=16, out_actions=4)
        self.target_model = DQN(in_states=16, out_actions=4)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(MAX_MEMORY_SIZE)
        self.epsilon = 1.0
        self.steps = 0

    def get_state(self, row, col):
        state = np.zeros((ENV_ROWS, ENV_COLS))
        state[row, col] = 1
        return state.flatten()

    def get_reward(self, row, col):
        if maze[row, col] == 1:
            return -10  # Hit an obstacle
        elif (row, col) == (ENV_ROWS-1, ENV_COLS-1):
            return 100  # Goal
        else:
            return -1  # Normal move

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Random action
        else:
            with torch.no_grad():
                return self.model(torch.FloatTensor(state)).argmax().item()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_q_values(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        # Select Q-values for the taken actions
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q = next_q_values.max(1)[0].detach()
        target_q = rewards + (DISCOUNT_FACTOR * max_next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def terminal_state(self, row, col):
        # Check if agent has hit an obstacle or reached the goal
        if (row, col) == (ENV_ROWS - 1, ENV_COLS - 1):
            return True, "goal"  # Reached the goal
        elif maze[row, col] == 1:
            return True, "obstacle"  # Hit an obstacle
        return False, "none"  # Not a terminal state

    def train(self, episodes):
        for episode in range(episodes):
            row, col = 0, 0  # Start at the top-left corner
            state = self.get_state(row, col)
            total_reward = 0

            for step in range(100):  # Limit the steps per episode
                action = self.select_action(state)

                # Move based on the action
                if action == 0 and col > 0: col -= 1  # Left
                if action == 1 and col < ENV_COLS-1: col += 1  # Right
                if action == 2 and row > 0: row -= 1  # Up
                if action == 3 and row < ENV_ROWS-1: row += 1  # Down

                reward = self.get_reward(row, col)
                next_state = self.get_state(row, col)
                done = (row, col) == (ENV_ROWS-1, ENV_COLS-1)  # Check if reached the goal

                # Update memory and Q-values
                self.update_memory(state, action, reward, next_state, done)
                self.update_q_values()

                state = next_state
                total_reward += reward
                if done:
                    break

            # Decay epsilon for exploration-exploitation balance
            self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)

            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

            # Sync the target model every few steps
            if episode % 20 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def test(self, episodes=10):
        for episode in range(episodes):
            row, col = 0, 0
            state = self.get_state(row, col)
            steps = 0
            total_reward = 0
            action_store = []
            all_actions = []

            print(f"episode: {episode + 1}")
            terminated = False
            while not terminated and steps < 100:
                action = self.select_action(state)
                action_store.append(action)

                if action == 0 and  col > 0:
                    col -= 1
                elif action == 1 and col < ENV_COLS - 1:
                    col += 1
                elif action == 2 and row > 0:
                    row -= 1
                elif action == 3 and row < ENV_ROWS - 1:
                    row += 1

                terminated, terminal_type = self.terminal_state(row, col)
                reward = self.get_reward(row, col)
                total_reward += reward
                state = self.get_state(row, col)

                steps += 1

                if terminated:
                    if terminal_type == "goal":
                        print("goal reached")
                    elif terminal_type == "obstacle":
                        print("obstacle hit!")
                    break
            all_actions.append(action_store)
            print(f"total reward = {total_reward}")
        return all_actions

if __name__ == "__main__":
    solver = MazeSolver_DQN()
    solver.train(500)  # Train for 500 episodes
    actions = solver.test()
    print(actions[0])
