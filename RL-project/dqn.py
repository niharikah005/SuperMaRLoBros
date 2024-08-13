import pygame
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque

maze = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 0, 10]
])

ENV_ROWS, ENV_COLS = 4, 4

# define the model:
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__() 
        # inherit the nn.Module
        self.fc1 = nn.Linear(in_states, h1_nodes)
        # first fully connected vectorized linear input -> hidden layer
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, activation):
        activation = F.relu(self.fc1(activation))
        # move the values forward using the reLU function, non-linear
        activation = self.out(activation)
        return activation
    
# Define memory for the experience replay:
class ReplayMemory():
    def __init__(self, max_len):
        self.memory = deque([], maxlen=max_len)
        # creation of a deque object called memory

    def appending(self, transition):
        self.memory.append(transition)
        # this is the (state, action, reward, new_state, terminated) tuple

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)

# the actual neural net:
class MazeSolver_DQL():
    # define the hyperparameters:
    learning_rate = 0.001   # the learning rate for game
    discount_factor = 0.8   # the dicount factore for the game
    network_sync_rate = 10  #no of steps the agent takes before syncing the policy and the target nns
    replay_memory_size = 1000   # size of replay memory
    mini_batch_size = 32    # size of training data sampled from the memory
    row_index, col_index = 0, 0
    reward = 0

    # Neural Network:
    loss_fn = nn.MSELoss()  # loss function, Mean Square Error
    optimizer = None    # NN optimizer, initialize later.

    ACTIONS = ['L', 'D', 'R', 'U']  # all the actions inside a list

    def terminal_state(self):
        if maze[self.row_index][self.col_index] == 0:
            return False
        else:
            return True
        # checks if state is terminal or not

    def get_reward(self, position, goal_position, maze):
        # dynamic reward function.
        if position == goal_position:
            return 1
        elif maze[position] == 1:
            return -0.1
        else:
            return -0.01

    def get_start(self):
        self.row_index = np.random.randint(ENV_ROWS)
        self.col_index = np.random.randint(ENV_COLS)
        if self.terminal_state():
            return self.get_start()
        else:
            return self.row_index,self.col_index

    def get_next_loc(self, action_index):
            new_row_index  =self.row_index
            new_col_index = self.col_index
            if self.ACTIONS[action_index] == "L" and new_col_index > 0:
                new_col_index -= 1
            if self.ACTIONS[action_index] == "R" and new_col_index < 3:
                new_col_index += 1
            if self.ACTIONS[action_index] == "U" and new_row_index > 0:
                new_row_index -= 1 
            if self.ACTIONS[action_index] == "D" and new_row_index < 3:
                new_row_index += 1
            return new_row_index, new_col_index # changes the row and col according to the action chosen
    
    def truncate_check(self, step_count):
        if step_count >= 120:
            return True
        
    def take_step(self, step_count, action, state):
        state[self.row_index * 4 + self.col_index] = 0
        self.row_index, self.col_index = self.get_next_loc(action)
        state[self.row_index * 4 + self.col_index] = 1
        reward = self.get_reward((self.row_index, self.col_index), (ENV_ROWS-1, ENV_COLS-1), maze)
        terminated = self.terminal_state()
        truncated = self.truncate_check(step_count)
        return state, reward, terminated, truncated
    
    def env(self, num_states):
        state = np.zeros(num_states)
        return state

    def train(self, episodes):
        num_states = ENV_ROWS * ENV_COLS
        nums_actions = len(self.ACTIONS)
        epsilon = 1 # 100% random
        memory = ReplayMemory(self.replay_memory_size)  
        self.row_index, self.col_index =  self.get_start()

        # create policy and target neural networks:
        policy_nn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=nums_actions)
        target_nn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=nums_actions)

        # copy the weights and biases of the policy into the target nn:
        target_nn.load_state_dict(policy_nn.state_dict())

        print('Policy (random, before training): ')
        self.print_dqn(policy_nn)

        # policy optimizer, here we are using 'adam':
        self.optimizer = torch.optim.Adam(policy_nn.parameters(), lr=self.learning_rate)

        # store number of rewards obtained in each episode: 
        rewards_per_episode = np.zeros(episodes)

        # list to keep track of epsilon decay:
        epsilon_decay = [] 

        # track the number of steps taken, used to sync the target with the policy network
        step_count = 0

        for i in range(episodes):
            state = self.env(num_states)  # initialize the states to zero
            self.row_index, self.col_index = self.get_start()   # obtain random start for each episode
            state[self.row_index * 4 + self.col_index] = 1 # initialize the position of the player to 1, encoding
            terminated = False  # use this variable to check if the agent reaches a terminal state
            truncated = False   # used to stop the episode if a certain number of unrealistic steps have been taken

            # Agent will travel the maze until it reaches a terminal state or the number of steps goes beyond allowed
            while (not terminated and not truncated):
                if random.random() < epsilon:
                    action = random.randrange(len(self.ACTIONS))
                else:
                    with torch.no_grad():
                        action = policy_nn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # execute the action:
                new_state, self.reward, terminated, truncated = self.take_step(step_count, action, state)
                # save the experience into the memory:
                memory.appending((state, action, new_state, self.reward, terminated))

                # go to the new state:
                state = new_state
                # increment the step counter:
                step_count += 1

            # keep track of the rewards:
            if self.reward == 1:
                rewards_per_episode[i] = 1
            
            # check if enough exp has been collected and some reward has been collected:
            if len(memory) >= self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                for batch in mini_batch:
                    print(f"{batch[1]}, {batch[3]}")
                self.optimize(mini_batch, policy_nn, target_nn)

                # decay the epsilon:
                epsilon = max(epsilon - 1/episodes, 0)  # limit the decay to zero
                epsilon_decay.append(epsilon)   # store the epsilon history as a list

                # copy policy nn to target nn after certain number of steps:
                if step_count > self.network_sync_rate:
                    target_nn.load_state_dict(policy_nn.state_dict())
                    step_count = 0

        torch.save(policy_nn.state_dict(), "maze_solver.pt")

    # optimize the network:
    def optimize(self, mini_batch, policy_nn, target_nn):
        num_states = policy_nn.fc1.in_features

        current_q_list = []
        target_q_list = []
        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # crashed with a black block or reached reward:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():   # does not change the gradients
                    target = torch.FloatTensor(reward 
                                               + self.discount_factor
                                               * target_nn(self.state_to_dqn_input(new_state, num_states)).max()
                                               )
                    
        # get the current set of Q values:
        current_q = policy_nn(self.state_to_dqn_input(state, num_states))
        current_q_list.append(current_q)

        # get the target set of q-values:
        target_q = target_nn(self.state_to_dqn_input(state, num_states))
        # put the q-value just calculated into the specific action from the target_nn list
        target_q[action] = target
        target_q_list.append(target_q)

        # compute loss for the whole mini_batch:
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # optimize the model:
        self.optimizer.zero_grad()  # clears gradients
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state: int, num_states) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor
    
    def test(self, episodes):
        nums_states = ENV_ROWS * ENV_COLS
        num_actions = 4

        # load the learned policy in the test function:
        policy_nn = DQN(in_states=nums_states, h1_nodes=nums_states, out_actions=num_actions)
        policy_nn.load_state_dict(torch.load("maze_solver.pt", weights_only=True)) # load the network
        policy_nn.eval()    # switch the network to eval mode

        print("policy trained")
        self.print_dqn(policy_nn)

        step_count = 0
        for i in range(episodes):
            state = self.env(nums_states)
            terminated = False
            truncated = False

            while (not terminated and not truncated):
                # select the best action:
                with torch.no_grad():
                    action = policy_nn(self.state_to_dqn_input(state, nums_states)).argmax().item()

                # execute the action:
                state, self.reward, terminated, truncated, = self.take_step(step_count, action, state)
                step_count += 1
    
    def print_dqn(self, dqn):
        num_states = dqn.fc1.in_features

        for s in range(num_states):
            q_values = ""
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+" "
            q_values = q_values.rstrip()

            # map the best action to L D R U:
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            print(f"{s:02}, {best_action}, [{q_values}]", end=" ")
            if (s+1) % 4 == 0:
                print()

if __name__ == "__main__":
    maze_solver = MazeSolver_DQL()
    maze_solver.train(1000)
    maze_solver.test(10)

# --------------------------------------------------------------------------------------------------------------------- #

"""
pygame.init()
WIDTH, HEIGHT = 770, 770
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
FPS = 60
BG = pygame.Color(0,0,0)
CellSize = WIDTH // 11
RowCount = 11
ColCount = 11
created = False
count = 0
path_place = 0

def final(window):
    reward = pygame.Rect(5*CellSize, 0, CellSize, CellSize)
    pygame.draw.rect(window, pygame.Color(0,255,0), reward)

def terminals(window):
    for rows in range(1,4):
        for columns in range(1,4):
            if maze[rows][columns] == 1:
                pygame.draw.rect(window, pygame.Color(255,255,255), (columns*CellSize, rows*CellSize, CellSize, CellSize))

def create_player(window,created,orig_row_index, orig_col_index):
    row_index, column_index = orig_row_index, orig_col_index
    player = pygame.Rect(column_index*CellSize, row_index*CellSize, CellSize, CellSize)
    pygame.draw.rect(window, pygame.Color(0,0,255), player)

path = []
path = shortest_path(9,3)

def updates(window,orig_row_index, orig_col_index):
    window.fill(BG)
    final(window)
    terminals(window)
    create_player(window,created,orig_row_index, orig_col_index)
    pygame.display.update()

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if count % (FPS) == 0:
        if path_place < len(path):
            orig_row_index, orig_col_index = path[path_place]
            path_place += 1
    count += 1

    updates(WIN,orig_row_index, orig_col_index)
    

pygame.quit()
"""