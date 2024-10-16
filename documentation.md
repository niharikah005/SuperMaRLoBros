# Documentation for the Project: SuperMaRLoBros

## This was a project done by ayush, niharikah, and vidhi

## This project was supported by Vedant and Labhansh as Mentors
## would not be possible without their help

### Project Work-Flow:

1. Learning Phase: 
Here we sent around two weeks learning about the bascis of python, pygame, and Reinforcement Learning.
We used online documentation of Pygame and FreeCodeCamp to learn about pygame and its various functionalities
We used an online YouTube video curated by Google's DeepMind branch of AI for Reinforcement Learning
This was a playlist on various topics, all taught by David Silver, head of Developers of AlphaZero and the likes
We also studied the workings of Neural Network, the cornerstone of our project, from a coursera course taught by the 
esteemed Andrew Ng.

2. Projects:
We created various projects to solidify our learning of pygame and Reinforcement Learning
Our first project was to create a simple game using pygame to understand basic window creation and sprite updation

Our next project was to implement one of the Reinforcement Learning Algorithms from scratch,
The Deep Q-Network or DQN is what we implemented in order to train the AI to clear a maze on its own

Then we also created a mini-jump game in which there is an agent and the agent must jump over the incoming obstacles
This was also achieved using the DQN Algorithm

3. Understanding the StableBaselines3 documentation:

We then read a few papers regarding the various other algorithms which depended on the value function or the policy function

We then trained some other environments from the OpenAIs gym in order to test which algorithm works best for small timesteps and simple rewards. Finally we decided that the policy based algorithm PPO (Proximal Policy Optimization) is the way forward.

We then dove into the documentations of the Reinforcement Learning Algorithms Library: StableBaselines3

SB3 is a vast library that offers all the Well-known and well-researched algorithms like DQN, DDPG, A2C, PPO, etc

4. Creating the actual Platformer game:

As known, we used the pygame library in python.
We created the player and the enemies as ADTs and then inherited them to create the various objects required for the functioning of the game.

We encountered some issues with the sprite updates and collisions which were then later on discussed and fixed

5. Training and tuning of the agent (on-going):

We then started the training of the agent using the StableBaselines3 Algorithm PPO.
We have been able to successfully train the agent in clearing the stage, however, its not as refined as humans do to time and hardware issues (slow training speed, not able to visualize on cloud GPUs. Inefficient GPU usage, etc) and hence are still trying to train it further.

6. Further prospects:

We aim to now train an actual Robot into solving a real-life 2D maze.
Further updates will be put up as we continue with RnD.

