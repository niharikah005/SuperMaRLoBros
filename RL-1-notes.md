# RL course notes:  

1. Difference between RL and supervised and unsupervised learning:  

- There is no supervisor or large datasets, there is only a ***reward signal***.  

- The Feedback may be delayed but almost never instantaneous.

- The decision making process should not take too much time for optimal management. Hence, time matters here.

- The agent's actions affect the data that it will recieve

2. How Reinforcement learning works:  

- There is a reward system, in which the **Reward** *Rt* is a simgular scalar quantity which tells how good the agent did at time interval t  

- The agent's job is to maximise this scalar quantity  

- Reinforcement learning works on the reward hypothesis: *All goals can be defined as a set of cumulative rewards and their maximization* 

- thus, the most important thing in RL becomes deciding how to reward the agent in the best way to make it do the task as we expect it  
  to

3. The Agent and Environment:  

- The Agent is some algorithm that will order some action which then goes to the environment, the environment then decides what reward to give to  
  the agent for the action/ state produced by it.

- The *history* Ht is the cumulation of the rewards, observations and actions up to time t.

- The algorithm that we have to create is actually a mapping from the history to the next action. This is done much more efficiently if there  
  were a summary of the history values that could be quantified in a scalar quantity. This summary is usually called the *state*.

- The state is formally known as the function of history up to time t,  
  therefore, St = f(Ht) for the time t.

  It has three main definitions of itself :

  1. The environment state S(e)t: This state is basically the summary of 
  information used by the environment for decidind the next Observation and reward for the agent. This value is usually not visible to the agent and even if it were available, it might contaion some irrelevant information.

  2. The agent state S(a)t : This is the summary of information of the  
  values present in the agent. These help us in deciding the next state/action for the agent and is what RL algorithms use for their working.
  It can be any function of the history i.e. S(a)t = f(Ht).

  3. The information state or markov state (St) : It is the summary of all the useful states of the history.
  Any state can be called a markov state if P(St+1 | St) = P(St+1 | S1,S2,.....,St).
  This basically means that the current state is enough for the determination of the future states and the total history is not needed.

- Types of environment:   
  1. fully observable environment: In this case the agent state can fully match the environment state because they are visible to the agent. Thus Agent state = environment state = information state.
  This is called a Markov Decision Process (MDP)

  2. partially observable state: In this case, the environment state is not completely visible. And so the agent state can never be equal to the environment state. However, we have different ways to make the state now: first is to remember all the history, second is to create probabilistic beliefs that this state might be equal to the environment state at a given moment t. Lastly, we can create a recurring neural network, where we create a pseudo-policy that is the linear transformation of the current observation and the previous state. This is called as Partially Observable Markov Decision Process (POMDP)

- Major components of RL Algorithm: there are three main components:

1. Policy : agents behaviour function.

2. value function: how good is each state/action or both

3. Model: agents representation of the environment

- Policy: these are the functions from state to action: a = p(s)  
  they can also be stochastic(probabilistic seems to be the closest definition for this) wherein we can give certain bayesian probability in order to decide the next action. This helps the agent to explore other behaviours as well/

- value function: this is the prediction of the future reward.
  this is used to evaluate the goodness of the action.
  its basically checking the cumulative rewards possible for some expected value E(p) based upon the chosen state.
  it also has a discounted value that will decide how far ahead into the future it looks.

- Model: it helps in predicting what the environment is like, it consists of two parts:  
  1. Transition: this tells what will be the next state of the model depending on the current state and action.
  2. Rewards: This tells us about the next immediate reward. 

- Types of RL agent: based on the above three, there are three main types:
  1. Value-based: it works on the value function, then the policy is implied that it will try to maximize the value.
  2. Policy-based: it works on the policy generated, it will not store the value function in this case.
  3. actor-critic: This one merges both the value and policy-based systems and is a little complex.
  4. model- free: this will have no model and only the value function and policy systems
  5. model-based: this will have the model, it might have or not have the other two systems.

- The two fundamental problems in RL:
1. learning: here, the agent knows nothing about the environment and it will learn to optimize the policy by interacting with it.
2. planning: here, the agent knows everything about the environment and will interact with the model in order to plan its actions.

it depends on the type of problem to see how well any of these two forms would be   
