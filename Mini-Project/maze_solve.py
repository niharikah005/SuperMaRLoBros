import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame

pygame.init()

# 1 is blocked and 0 is free to move on
maze = np.array([
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0]
])

rows, cols = maze.shape # 5, 5
p_mark = 0.5

epsilon = 1.0  # high initial exploration rate
epsilon_min = 0.01  # min epsilon
epsilon_decay = 0.995 # factor to decay epsilon by

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
    
# for replay mem
class Memory():
    def __init__(self, max_len):
        self.memory = deque([], maxlen=max_len)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    
class Mazesolve():
    def __init__(self):
        self.model = DQN(in_states=25, out_actions=4) # policy network
        self.target_model = DQN(in_states=25, out_actions=4) # target network
        self.memory = Memory(max_len=10000) # create mem 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss = nn.MSELoss() # mean sqred error loss
        self.epsilon = epsilon
        self.minreward = -1.5*rows*cols # if reward < minreward, then terminate
        self.gamma = 0.95 # discount factor
        self.batch_size = 64 
        self.steps = 0 

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self):
        canvas = np.copy(self.maze)
        row, col, _ = self.state
        canvas[row, col] = p_mark
        return canvas.flatten()

    def update_state(self, action):
        p_row, p_col, mode = self.state
        #print(f"Before action {action}, position: {self.state}")

        valid_actions = self.valid_actions()
        if action not in valid_actions:
            mode = 'invalid'
        else:
            mode = 'valid' 
            if action == 0:
                p_col -= 1
            elif action == 1:
                p_row += 1
            elif action == 2:
                p_col += 1
            elif action == 3:
                p_row -= 1

        if 0 <= p_row < rows and 0 <= p_col < cols:
            if self.maze[p_row, p_col] == 1.0: # trying to move onto blocked square
                mode = 'blocked'
        else:
            mode = 'invalid'

        self.state = (p_row, p_col, mode)
        if mode == 'valid': # to keep track of visited squares
            self.visited.add((p_row, p_col))
        #print(f"action taken: {action}, position: {self.state}")

    def get_reward(self):
        p_row, p_col, mode = self.state
        if (p_row, p_col) == self.target:
            return 100.0
        if mode == 'blocked':
            return -10.0
        if (p_row, p_col) in self.visited:
            return -1.5
        if mode == 'invalid':
            return -2.0
        return -0.75
    
    def add_to_mem(self, state, action, reward, next_state, game_status): 
        self.memory.add((state, action, reward, next_state, game_status))
    
    def valid_actions(self):
        row, col, _ = self.state
        actions = []
        
        # not going out of boundary and target loc not blocked
        if col > 0 and self.maze[row, col - 1] != 1.0:  
            actions.append(0)
        
        if row < rows - 1 and self.maze[row + 1, col] != 1.0: 
            actions.append(1)
        
        if col < cols - 1 and self.maze[row, col + 1] != 1.0:  
            actions.append(2)
        
        if row > 0 and self.maze[row - 1, col] != 1.0:  
            actions.append(3)
        
        return actions
    
    def action_select(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  
        else:
            with torch.no_grad():
                return self.model(torch.FloatTensor(state)).argmax().item()

    def game_status(self):
        p_row, p_col, _ = self.state
        if (p_row, p_col) == self.target:
            return 'win'
        if self.total_reward < self.minreward or self.steps > 3*rows*cols: # reward is lesser than minreward or player has made too many redundant steps
            return 'lose'
        return 'not_over'

    def qval_update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, game_status = [], [], [], [], []
        

        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action) 
            rewards.append(reward)
            next_states.append(next_state)
            game_status.append(done)
            #print(states,actions,rewards,next_states,game_status)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        game_status = torch.FloatTensor(game_status)

        q_vals = self.model(states)
        next_q_vals = self.target_model(next_states)

        curr_q = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q = next_q_vals.max(1)[0].detach()
        target_q = rewards + (1 - game_status) * self.gamma * max_next_q
        loss = self.loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self): # to reset the env after every run
        self.state = (0, 0, 'start')
        self.total_reward = 0
        self.steps = 0
        self.visited = set()
        self.maze = np.copy(maze)
        self.target = (rows-1, cols-1)
        self.update_target_model()
        return self.get_state()
    
    def terminal(self,state):
        row,col,status = state
        if (row,col) == self.target: # reached target loc
            return True,'win'
        if self.maze[row,col] == 1.0: # on a blocked square
            return True,'lose'
        if self.total_reward < self.minreward or self.steps > 3*rows*cols:
            return True,'lose'
        return False,'not_over'
    
    def train(self,num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state = self.reset()
            done = False
            total_reward = 0
            while not done: 
                action = self.action_select(state)
                self.update_state(action)
                reward = self.get_reward()
                next_state = self.get_state()
                done, status = self.terminal(self.state)
                self.add_to_mem(state,action,reward,next_state,done)
                self.qval_update()
                state = next_state
                total_reward += reward
                self.steps += 1
                
            rewards.append(total_reward)
            self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
            if episode % 10 == 0:
                print(f"ep: {episode}, total reward: {total_reward}, epsilon: {self.epsilon}")
            if episode % 10 == 0: # update target model every 10 episodes
                self.update_target_model()
            
        return rewards
    
    def test(self, num_episodes):
        for episode in range(num_episodes):
            state = self.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.action_select(state)
                self.update_state(action)
                reward = self.get_reward()
                total_reward += reward
                _, status = self.terminal(self.state)
                                
                if status == 'win':
                    print("won")
                    break
                if status == 'lose':
                    print("lost")
                    break
                
                state = self.get_state()
            print(f"ep: {episode}, total reward: {total_reward}")

maze_solver = Mazesolve()
rewards = maze_solver.train(300)
maze_solver.test(10)

plt.plot(rewards)
plt.title('Rewards')
plt.show()

def visualize_path(maze_solver, fps=3):
    block_size = 100
    margin = 5
    screen_size = ((cols * block_size) + (cols + 1) * margin, (rows * block_size) + (rows + 1) * margin)
    
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("maze visualizer")

    colors = {
        'free': (255, 255, 255),
        'blocked': (0, 0, 0),
        'start': (0, 255, 0),
        'target': (255, 0, 0),
        'path': (0, 0, 255)
    }

    def draw_maze(state, visited):
        screen.fill((0, 0, 0)) 
        
        # to draw maze
        for r in range(rows):
            for c in range(cols):
                color = colors['free'] if maze[r, c] == 0.0 else colors['blocked']
                pygame.draw.rect(screen, color, [(margin + block_size) * c + margin,(margin + block_size) * r + margin,block_size,block_size])

        # for drawing visited cells
        for v_r, v_c in visited:
            pygame.draw.rect(screen,colors['path'],[(margin + block_size) * v_c + margin,(margin + block_size) * v_r + margin,block_size,block_size])
        
        # player
        p_r, p_c, _ = state
        pygame.draw.rect(screen, colors['start'], [(margin + block_size) * p_c + margin,(margin + block_size) * p_r + margin,block_size,block_size])
        
        # to draw the target
        t_r, t_c = maze_solver.target
        pygame.draw.rect(screen, colors['target'], [(margin + block_size) * t_c + margin, (margin + block_size) * t_r + margin, block_size, block_size])

        pygame.display.flip()

    clock = pygame.time.Clock()

    for episode in range(1):
        state = maze_solver.reset()
        done = False
        maze_solver.visited.add((0, 0))
        while not done:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            draw_maze(maze_solver.state, maze_solver.visited)
            action = maze_solver.action_select(state)
            maze_solver.update_state(action)
            state = maze_solver.get_state()
            done, status = maze_solver.terminal(maze_solver.state)
            if status == 'win':
                print("Reached the target!")
                done = True  
            draw_maze(maze_solver.state, maze_solver.visited)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

visualize_path(maze_solver)