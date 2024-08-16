import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime

# Define maze
maze = np.array([
    [1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 1.0]
])

ACTIONS = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
rows, cols = maze.shape
visited_mark = 0.8
p_mark = 0.5
epsilon = 0.1

class MazeSolve(object):
    def __init__(self, maze, player=(0, 0)):
        self.ogmaze = maze
        self.target = (4, 4)
        self.free = [(r, c) for r in range(rows) for c in range(cols) if self.ogmaze[r, c] == 1.0]
        self.free.remove(self.target)
        if player not in self.free:
            raise ValueError("Invalid starting position for player")
        self.reset(player)

    def reset(self, player):
        self.player = player
        self.maze = np.copy(self.ogmaze)
        self.state = (player[0], player[1], 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()
        self.visited.add(player)

    def update_state(self, action):
        rat_row, rat_col, mode = self.state
        #print(f"Before action {action}, position: {self.state}")

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))

        valid_actions = self.valid_actions()
        if action not in valid_actions:
            mode = 'invalid'
        else:
            mode = 'valid'
            if action == 'L':
                rat_col -= 1
            elif action == 'D':
                rat_row += 1
            elif action == 'R':
                rat_col += 1
            elif action == 'U':
                rat_row -= 1

        if 0 <= rat_row < rows and 0 <= rat_col < cols:
            if self.maze[rat_row, rat_col] == 0.0:
                mode = 'blocked'
        else:
            mode = 'invalid'

        self.state = (rat_row, rat_col, mode)
        #print(f"After action {action}, position: {self.state}")


    def get_reward(self):
        p_row, p_col, mode = self.state
        if (p_row, p_col) == self.target:
            return 1.0
        if mode == 'blocked':
            return -1.0
        if (p_row, p_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        #print(f"Action: {action}, Reward: {reward}, Total Reward: {self.total_reward}, Status: {status}")
        return envstate, self.total_reward, status

    def observe(self):
        canvas = self.draw_env()
        return canvas.reshape((1, -1))

    def draw_env(self):
        canvas = np.copy(self.maze)
        row, col, _ = self.state
        canvas[row, col] = p_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        p_row, p_col, _ = self.state
        if (p_row, p_col) == self.target:
            return 'win'
        return 'ongoing'

    def valid_actions(self):
        row, col, _ = self.state
        actions = []
        if col > 0: actions.append('L')
        if col < cols - 1: actions.append('R')
        if row > 0: actions.append('U')
        if row < rows - 1: actions.append('D')
        #print(f"Valid actions: {actions}")
        return actions

class Experience(object):
    def __init__(self, model, max_len=100, discount=0.95):
        self.model = model
        self.max_len = max_len
        self.discount = discount
        self.memory = deque([], maxlen=max_len)
        self.batch_size = 32

    def append(self, episode):
        self.memory.append(episode)

    def predict(self, envstate):
        if isinstance(envstate, torch.Tensor):
            envstate_tensor = envstate
        elif isinstance(envstate, np.ndarray):
            envstate_tensor = torch.from_numpy(envstate).float()
        else:
            raise TypeError("Expected input to be a NumPy array or PyTorch tensor")

        if len(envstate_tensor.shape) == 1:
            envstate_tensor = envstate_tensor.unsqueeze(0)
        return self.model(envstate_tensor).detach().numpy()

    def get_data(self, data_size=10):
        if len(self.memory) == 0:
            return np.zeros((1, self.model.out.out_features)), np.zeros((1, self.model.out.out_features))
        data_size = min(len(self.memory), data_size)
        env_size = self.memory[0][0].shape[1]
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.model.out.out_features))
        for i, j in enumerate(np.random.choice(range(len(self.memory)), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, ACTIONS[action]] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

def qtrain(model, maze, **opt):
    global epsilon
    n_ep = opt.get('n_ep', 100)
    maxlen = opt.get('max_len', 250)
    data_size = opt.get('data_size', 50)
    name = opt.get('name', 'model')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    mazesolve = MazeSolve(maze)
    experience = Experience(model, max_len=maxlen)

    win_history = []
    hsize = mazesolve.maze.size // 2
    win_rate = 0.0
    total_rewards = []

    for episode in range(n_ep):
        print('episode: ', episode)
        loss = 0.0
        rat_cell = random.choice(mazesolve.free)
        mazesolve.reset(rat_cell)
        game_over = False
        total_reward = 0
        game_status = ''

        envstate = mazesolve.observe()
        while not game_over and game_status!='lose':
            valid_actions = mazesolve.valid_actions()
            if not valid_actions:
                break

            prev_envstate = envstate
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                if len(experience.memory) < experience.batch_size:
                    action = np.random.choice(valid_actions)
                else:
                    prev_envstate_tensor = torch.from_numpy(prev_envstate.astype(np.float32))
                    q = experience.predict(prev_envstate_tensor)
                    action = int(np.argmax(q))  
                    action = list(ACTIONS.keys())[action]  # to map index to action
            
            envstate_next, reward, game_status = mazesolve.act(action)
            envstate_next_tensor = torch.from_numpy(envstate_next.astype(np.float32))

            episode = [prev_envstate, action, reward, envstate_next, game_status]
            experience.append(episode)
            total_reward += reward

            if len(experience.memory) >= experience.batch_size:
                inputs, targets = experience.get_data(data_size=data_size)
                if inputs.size == 0 or targets.size == 0:
                    continue

                inputs_tensor = torch.from_numpy(inputs.astype(np.float32))
                targets_tensor = torch.from_numpy(targets.astype(np.float32))

                model.train()
                optimizer.zero_grad()
                outputs = model(inputs_tensor)
                loss = criterion(outputs, targets_tensor)
                loss.backward()
                optimizer.step()
        if game_status == 'lose':
                print(f"terminated becuz of loss at episode {episode}.")
                game_over = True
        if game_status == 'win':
            print(f"terminated cuz of win on ep {episode}")
        print(total_reward)
        total_rewards.append(total_reward)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        if win_rate > 0.9:
            epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize: #and completion_check(model, mazesolve):
            print("Reached 100% win rate at episode: %d" % (episode,))
            break

    torch.save(model.state_dict(), name + ".pt")
    print('files: %s' % (name + ".pt",))
    return total_rewards


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return x.numpy()

def build_model(maze, lr=0.001):
    maze_size = maze.size
    num_actions = len(ACTIONS)
    h1_nodes = maze_size
    model = DQN(maze_size, h1_nodes, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

input_size = rows * cols
output_size = len(ACTIONS)
model = DQN(input_size, output_size, output_size) 

total_rewards = qtrain(model, maze)

plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
