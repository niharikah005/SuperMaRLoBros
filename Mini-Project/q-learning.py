# warehouse robot has to learn the shortest path to a certain location 
# black and green squares are terminal states, green is the goal and black means robot crashed
# goal: maximize total reward and minimize total punishments

# libraries
import numpy as np
import pygame
pygame.init()

# environment

# shape of environment
env_rows = 11
env_columns = 11

# 3d numpy arr to hold curr q values
q_values = np.zeros((env_rows, env_columns, 4)) 
# the 4 is for the number of possible actions available

# actions   0       1        2       3
actions = ['up', 'right', 'down', 'left']

# rewards
rewards = np.full((env_rows, env_columns), -100)
rewards[0,5] = 100   # the green box (goal) 

# aisle locations (white squares with -1 reward)
aisles = {} # for holding the locs
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

# set reward for white boxes (aisles)
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1

# training
'''
new episode begins with randon non terminal state (white box)
choose action using epsilon greedy policy
perform action, move to next state(location)
calc td value after recieving reward for new state
update q value for current state using bellman equation
if terminal ep reached, repeat else choose another action and continue
'''

# terminal state
def terminal_state(curr_row_index, curr_col_index):
    if rewards[curr_row_index][curr_col_index] == -1:
        return False
    else:
        return True
# all boxes other than white are terminal states
    
# choose random initial state (non terminal)
def get_starting_location():
    while True:
        curr_row_index = np.random.randint(env_rows)
        curr_col_index = np.random.randint(env_columns)
        if not terminal_state(curr_row_index, curr_col_index):
            return curr_row_index, curr_col_index

# choose action with epsilon greedy policy (if randomly chosen value smaller than epsilon, choose max q value)
def next_action(curr_row_index, curr_col_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[curr_row_index, curr_col_index])
    else:
        return np.random.randint(4)
    
# next loc based on action chosen(robot should remain inside grid)
def next_location(curr_row_index, curr_col_index, action_index):
    new_row_index = curr_row_index
    new_col_index = curr_col_index
    if actions[action_index] == 'up' and curr_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and curr_col_index < env_columns - 1:
        new_col_index += 1
    elif actions[action_index] == 'down' and curr_row_index < env_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and curr_col_index > 0:
        new_col_index -= 1
    return new_row_index, new_col_index

# shortest path from any loc 
def get_shortest_path(start_row_index, start_col_index):
    if terminal_state(start_row_index, start_col_index):
        return []
    else:
        curr_row_index, curr_col_index = start_row_index, start_col_index
        short_path = []
        short_path.append([curr_row_index, curr_col_index])
        while not terminal_state(curr_row_index, curr_col_index): # not yet on terminal state
            action_index = next_action(curr_row_index, curr_col_index, 1) # greedy action
            curr_row_index, curr_col_index = next_location(curr_row_index, curr_col_index, action_index) # new loc based on action
            short_path.append([curr_row_index, curr_col_index]) # append new loc
    return short_path
    
# actual training
epsilon = 0.9 # for choosing action with help of epsilon-greedy
discount_factor = 0.9 # for future rewards
learning_rate = 0.9 # how fast agent learns

# 1000 training eps
for episode in range(1000):
    row_index, col_index = get_starting_location() # random non terminal state

    while not terminal_state(row_index, col_index):
        action_index = next_action(row_index, col_index, epsilon) # choose action
        old_row_index, old_col_index = row_index, col_index
        row_index, col_index = next_location(row_index, col_index, action_index) # update loc
        reward = rewards[row_index, col_index]
        old_q_value = q_values[old_row_index, old_col_index, action_index]
        td = reward + (discount_factor * np.max(q_values[row_index, col_index])) - old_q_value
        new_q_value = old_q_value + (learning_rate * td) # bellman eqn, update q value
        q_values[old_row_index, old_col_index, action_index] = new_q_value

# to visualize the path
pygame.init()
screen_size = 600
cell_size = screen_size // env_rows
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("path visualization")

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 128)
yellow = (255, 255, 0)

def draw_grid():
    for row in range(env_rows):
        for col in range(env_columns):
            color = white
            if rewards[row, col] == -100:
                color = black
            elif rewards[row, col] == 100:
                color = green
            pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size), 0)
            pygame.draw.rect(screen, blue, (col * cell_size, row * cell_size, cell_size, cell_size), 1)

def visualize_path(path):
    for loc in path:
        pygame.draw.rect(screen, yellow, (loc[1] * cell_size, loc[0] * cell_size, cell_size, cell_size), 0)
        pygame.display.update()
        pygame.time.wait(200)

start_positions = [(9, 9), (2, 7), (7, 5)]
for start_row, start_col in start_positions:
    path = get_shortest_path(start_row, start_col)
    draw_grid()
    visualize_path(path)
    pygame.time.wait(1000)

pygame.quit()