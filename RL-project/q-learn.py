# import libs
import numpy as np

# create the environment
environment_rows = 11
environment_cols = 11
possible_actions = 4

q_values = np.zeros((environment_rows, environment_cols, possible_actions)) # the last arg is for the total actions possible in a single 
# state

# define actions
actions = ["left", "right", "up", "down"]

# make a rewards table for each state:
rewards = np.full((environment_rows,environment_cols), -100) # the first arg is the dimensions, the second one is the value to be filled.

rewards[0,5] = 100 # the final state/ terminal state of reward

aisles = {} # dictionary to easily create free-space

aisles[1] = [i for i in range(1,environment_cols-1)]
aisles[2] = [1,7,9]
aisles[3] = [i for i in range(1,8)]
aisles[3].append(9)
aisles[4] = [3,7]
aisles[5] = [i for i in range(0,environment_cols)]
aisles[6] = [5]
aisles[7] = [i for i in range(1,10)]
aisles[8] = [3,7]
aisles[9] = [i for i in range(0,environment_cols)]

# assign -1 to all row,cols present in aisles:
for rows in range(1,10):
    for columns in aisles[rows]:
        rewards[rows][columns] = -1


# TRAIN MODEL: 
# state: choose a random non-terminal state for the agent
# action: the action will be chosen from the actions list, they will be chosen using the epsilon-greedy policy
# use the action to transition to the next state
# recieve reward for the current state and calculate the TD value and error
# update the q-value and continue
# if reach a terminal state, go to first step, else go to second step.


def terminal_state(row_index, col_index):
    if rewards[row_index][col_index] == -1:
        return False
    else:
        return True
    # checks if state is terminal or not

def get_start():
    row_index = np.random.randint(environment_rows)
    col_index = np.random.randint(environment_cols)
    if terminal_state(row_index, col_index):
        return get_start()
    else:
        return row_index,col_index
    
def epsilon_greedy_action(row_index, col_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[row_index, col_index]) # chooses the action number that has the highest q-value
    else:
        return np.random.randint(4)
    # chooses a random action value if the toss provides a number larger than epsilon, epsilon is the hyperparameter here.

def get_next_loc(row_index, col_index, action_index):
    new_row_index  =row_index
    new_col_index = col_index
    if actions[action_index] == "left" and new_col_index > 0:
        new_col_index -= 1
    if actions[action_index] == "right" and new_col_index < 10:
        new_col_index += 1
    if actions[action_index] == "up" and new_row_index > 0:
        new_row_index -= 1 
    if actions[action_index] == "down" and new_row_index < 10:
        new_row_index += 1
    return new_row_index, new_col_index # changes the row and col according to the action chosen


# define a function that returns the shortest path from any location to the final reward-position:
# returns a list of actions
def shortest_path(start_row_index, start_col_index):
    if terminal_state(start_row_index, start_col_index):
        return []
    else:
        current_row, current_col = start_row_index, start_col_index
        path = []
        path.append([current_row, current_col])
        while not terminal_state(current_row, current_col): # checks if the location is terminal or not
            action_no = epsilon_greedy_action(current_row, current_col, 1.0) # decides the action using epsilo-greedy policy (best path)
            current_row, current_col = get_next_loc(current_row, current_col, action_no) # gets the loc using the action
            path.append([current_row, current_col]) # appends the new location 
    return path 

# start with the q-learning model:

epsilon = 0.9 # used in the epsilon-greedy policy for action-choosing
discount_factor = 0.9 # used in the discounting of reward in TD learning
learn_rate = 0.9 # decides how quickly the AI learns

# run thru 1000 episodes:
for episode in range(1000):
    row_index, col_index = get_start()

    while not terminal_state(row_index, col_index):
        # take action:
        action_value = epsilon_greedy_action(row_index, col_index, epsilon)
        old_row_index, old_col_index = row_index, col_index # store the old indices
        row_index, col_index = get_next_loc(old_row_index, old_col_index, action_value)

        # recieve the reward, calculate the temporal difference:
        reward = rewards[row_index, col_index]
        old_q_value = q_values[old_row_index, old_col_index, action_value]
        temporal_D = reward + (discount_factor * np.max(q_values[row_index, col_index]) - old_q_value)

        # update the q_value for the previous state and action pair:
        new_q_value = old_q_value + learn_rate * temporal_D
        q_values[old_row_index, old_col_index, action_value] = new_q_value


# by the above method, the agent updated the q-values and now the epsilon greedy policy when run on max greediness will be able to 
# easily find the state that has the smallest q-value, in this way, the agent will be able to find the shortest path no matter what the
# position is.

# run the shortest path function for a value:
print(shortest_path(0,0)) # dies not work since its a terminal state
print(shortest_path(9,0)) 
