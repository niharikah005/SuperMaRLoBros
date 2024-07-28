## notes for lecture 8

Learning the model from experience and use planning to construct a value function or policy.

basically, trying to integrate learning and planning into a single architecture. This will give good results.

Model: it is the states understanding of how a state transistions to other states and how and when rewards are given. Its also called as a parametrised MDP. The state transitions and rewards are considered to be independent in this case.

This form of learning is called model-based learning. In this case, the agent is able to image the environment and looks ahead for different state-action pairs without actually carrying them out.

Advantages: 
- Can efficiently learn models in cases where the value function is hard to estimate, using supervised learning methods.

- Can reason about model uncertainty

Disadvantages: 
- learn the model then approx the value function using that model. Hence, two sources of approx error in this case.

# Model learning:

This is a special case where supervised learning methods can be used.

we will have our sample space of state,actions,immediate reward,..... Then we will use two methods:

1. regression to understand association of S,A with R
2. density estimation to understand association of S,A with S prime.

Then we will find the parameter that minimizes the losses in these methods.

## Tabel lookup model:

Count visits N(s,a) to each state-action pair.

Probability value = (1/N) * summ 1 (S = s, A = a, S+1 = s prime)

Reward = (1/N) * summ 1 (S = s, A=a) * Rt

## sample-based planning:

first, we learn the model MDP.
second, we sample state-action-rewards triples from the model only.
third, we apply model-free learing on these samples i.e Q-learning, sarsa, MC etc.

these are really efficient.

## Planning with inacurate model:

The Model-based RL is going to be as good as the approximated model.

that is why:
1. when model is wrong, use model-free RL
2. reason explicitly about model uncertainty.

## The dyna architecture:

It is a mix of model-based and model-free learning.

the agent creates the value function from both the real-life sampling (MC, TD) and also thru the model estimated.

we initialize the q-values and the model values as well.

we then choose an action thru the epsilon greedy policy and then execute it

depending on the planning steps (n):
then we use some prev state and some action taken in that state, run the model to find the reward and next state and update the q-value for that state in this way as well.

## Simulation-based search:

Uses forward search: a method in which a single state is chosen and then DP is applied to only that state, other states are not accounted for, this decreases the amount of data needed for calculations.

Applies model-free RL to these simulations.

we will keep chosing the episodes, and simulate them using model-free RL.

1. Simple MC search:

for each action a belonging to A:

simulate K episodes from the current state:

run MC on each of these states and obtain the q-value.

select the action with the maaximum value.
continue.

2. MC Tree Search:

Here, we choose each action for the current state, we take the mean of all the states from that action. Then we choose the max of these actions as our next action, this way we can improve the policy.

3. TD search:

same as MC search but we use sarsa instead of MC.
uses bootstraping which is good.

## dyna-2:

In this case the agent stores two feature weights:
long-term memory which stores real life experience using say, TD learning.

short-term memory which does a tree-search to certain experiences to update the policy.