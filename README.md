# Our Project
Super MaRLo Bros: Making the use of Reinforcement Larning  algorithms to train an agent to clear a level in a super-mario type environment
# Table of contents
- [About the Project](#About-the-Project)
  - [Tech Stack](#Tech-Stack)
  - [File Structure](#File-Structure)
- [Requirements](#Requirements)
  - [Prerequisites and installlation](#Prerequisites-and-installlation)
  - [Execution](#Execution)
- [Theory and Approach](#Theory-and-Approach) 
-  [Results and Demo](#Results-and-Demo)
-  [Future Work](#Future-Work)
-  [Contributors](#Contributors)
-  [Acknowledgements and Resources](#Acknowledgements-and-Resources)

# About the Project
The aim of reinforcement learning (RL) is to develop algorithms that enable agents to learn optimal behaviors through interactions with their environment. By receiving feedback in the form of rewards or penalties based on their actions, agents can improve their decision-making over time to maximize cumulative rewards. This approach is widely used in various applications, including robotics, game playing, and autonomous systems, where the agent learns to make a series of decisions that lead to successful outcomes in complex environments.

We have first created the environment from scratch using the pygame module on python. The game consists of a level which has stationary and moving obstacles that the agent must interact with in order to learn.
The model has been trained using the StableBaselines3 algorithms library that contains various RL algorithms

We have used both MLP network and CNN network in the processing of the observation (the environment) and have compared the results below. The algorithm first intended for use was the Double-Q-Network (DQN), but we later shifted to the Proximal-Policy-Optimisation (PPO) algorithm.

## Tech Stack
- [Python](https://www.python.org/)
- [Numpy](https://numpy.org/doc/#)
- [Tensorflow](https://www.tensorflow.org/)
- [Pytorch](https://pytorch.org/)
- [Pygame](https://www.pygame.org/news)
- [Gym](https://www.gymlibrary.dev/content/environment_creation/) (for the model-environment)
- [StableBaseline3](https://stable-baselines3.readthedocs.io/en/master/)

## File Structure



# Requirements:

## Prerequisites and installation

- Download Python on your device if not already present. 
 Refer [here](https://www.python.org/downloads/) for the setup.
- You can use any code editor.
- All installations mentioned are made using pip hence install pip.
- To install pip , follow this [link](https://www.geeksforgeeks.org/how-to-install-pip-on-windows/)

navigate to the project directory

- create a virtual environment to install the modules in a safe manner:

* Windows:
```
python -m venv env
env\Scripts\activate
```

* Linux:
```
python3 -m venv env
source env/bin/activate
```
  
- To install the requirements 
```
pip install -r requirements.txt 
```

## Execution

- Clone the repository
```
git clone https://github.com/niharikah005/SuperMaRLoBros 
```



# Theory and Approach

* For our project, Super MaRLo Bros, we have implemented a reinforcement learning architecture using model-free algorithm to train an AI agent capable of playing the game. This approach allows the agent to learn optimal strategies through trial and error, improving its performance over time without requiring a predefined model of the game environment. The use of these algorithms enables us to effectively handle the dynamic nature of gameplay, where the agent learns from its interactions and adapts its strategies accordingly.

In reinforcement learning (RL), a model-free algorithm is an approach where the agent learns to make decisions without having an explicit model of the environment. This means the agent does not try to predict the environment's behavior (e.g., state transitions or rewards); instead, it directly learns from its interactions with the environment to make decisions that maximize rewards.

Key Characteristics of Model-Free Algorithms:

No Environment Model: The agent doesn't learn or assume how the environment works internally, such as how actions affect the state transitions or rewards.
Direct Policy or Value Learning: The agent focuses on learning a policy (a mapping from states to actions) or value functions (estimating the long-term reward of being in a certain state or taking a certain action).
Data-Driven: The agent relies on real experience (observed state transitions, rewards) to improve its decision-making.

Types of Model-Free Algorithms:
Value-Based Methods:

These methods focus on estimating the value function, which tells how good it is to be in a certain state or to take a certain action.
Example:
Q-learning: The agent learns an estimate of the Q-value (action-value function), which predicts the expected reward of taking an action in a given state and following the optimal policy afterward.

Policy-Based Methods:

These methods directly learn the policy (the function that maps states to actions) without needing to estimate value functions.
Example:
REINFORCE: This algorithm uses policy gradients to update the policy directly based on the rewards received.
Actor-Critic Methods:

These combine both value-based and policy-based methods by having two components: an actor (which decides actions) and a critic (which evaluates how good the actions are).
Example:
Proximal Policy Optimization (PPO): A popular actor-critic algorithm that combines stability and performance.


We have used a **Policy function approximation algorithm** called **Proximal Policy Optimization (PPO)**. The reason we are using PPO is that it offers a balance between simplicity, efficiency, and performance. PPO is particularly popular because it addresses the instability issues seen in policy gradient methods by introducing a way to limit the magnitude of policy updates, ensuring more stable learning.

PPO achieves this by clipping the probability ratio between the new and old policies during training, preventing overly large updates that can harm performance. This results in smoother and more reliable training compared to older algorithms like vanilla policy gradient or trust region policy optimization (TRPO), while still being computationally efficient and easy to implement.

Additionally, PPO performs well across a variety of continuous and discrete action-space problems, making it a flexible choice for reinforcement learning tasks such as robotic control, game playing, and navigation in complex environments.


* Explanation for why PPO is better than standard Policy algorithms

### **1. Proximal Policy Optimization (PPO) Formula**

The **PPO objective** is given by:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

Where:

## Probability Ratio

The term `r_t(θ)` represents the ratio between the probability of taking action `a_t` in state `s_t` under the updated policy `π_θ`, and the probability of taking the same action under the previous policy `π_{θ_{old}}`. This ratio plays a key role in evaluating how much the policy has changed between updates.

- **Advantage Estimate**: `Â_t` denotes the advantage estimate at time step `t`. The advantage function evaluates how much better (or worse) taking action `a_t` is compared to the expected average action in that state. This estimate helps steer the policy towards actions with better outcomes.

- **Hyperparameter `ε`**: The hyperparameter `ε` controls the extent of policy changes during updates, ensuring stability in the training process by limiting the size of these changes.

- **Clipping Mechanism**: The function `clip(r_t(θ), 1 - ε, 1 + ε)` applies a clipping mechanism to the probability ratio, constraining it within the range `[1 - ε, 1 + ε]`. This prevents excessively large updates that could destabilize training.


### Explanation:
- **First term** This is the basic policy gradient term, where the advantage is scaled by the likelihood ratio \( r_t(\theta) \).
- **Second term**: This term prevents the policy from being updated too aggressively by clipping the ratio to a range around 1. This ensures that the update does not deviate too much from the old policy.
- **min**: PPO uses the minimum between the clipped and unclipped terms to ensure that the final objective discourages updates that deviate too much from the old policy.

The clipping mechanism makes PPO more stable compared to standard policy gradient methods, preventing drastic policy changes and improving training reliability.


Papers referred about Reinforcement Learning:

[REINFORCEMENT_LEARNING_1](https://arxiv.org/abs/2304.00026)

[REINFORCEMENT_LEARNING_2](https://ar5iv.labs.arxiv.org/html/1708.05866)


# RESULTS


### Agent Output
![](vid.mp4)
[Download the video](vid.mp4)


# Future Work

* Working on a robot that can detect its boundaries and move perfectly. 

# Contributors
- [Vidhi Rohira](https://github.com/vidhirohira)
- [Niharika Hariharan](https://github.com/niharikah005) 
- [Ayush Bothra](https://github.com/ayushbothra)

# Acknowledgements

 - [COC VJTI]() ProjectX 2023
 - Our mentors [Vedant Mehra](https://github.com/extint) and [Labhansh Naik](https://github.com/lbhnsh) for their valuable mentorship throughout the project. 
 
# Resources

 - [Reinforcement Learning course by David Silver](https://youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&si=MCeubNLwrGNftvAw)
 - [Nueral Networks Course from Coursera](https://www.coursera.org/account/accomplishments/verify/S2C8DYZN8C54?utm_source=ios&utm_medium=certificate&utm_content=cert_image&utm_campaign=sharing_cta&utm_product=course)
 - [A Comprehensive Guide to Convolutional Neural Networks](https://www.v7labs.com/blog/convolutional-neural-networks-guide)
 
