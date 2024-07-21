# Reinforcement Learning 

RL is used in many different fields like CS, Neuoroscience, Maths, Economucs, etc

> type of machine learning

## The RL Problem

### How is RL different from others

- only reward signal, no supervisor

    doesn't have any right or wrong action, only trial and error

- delayed feedback

    whether a decision was good or bad will be found out after certain no of steps

- time is important

    what happens at one step will impact the next step, long term consequences, data is non iid (independent and identically distributed) 

    agent can influence and take actions in environment  

### Rewards

R<sub>t</sub> is a scalar feedback given to indicate how well agent is doing at step _t_. Cumulative rewards should be maximized
 
> example for how rewards can be given:
> - time based (shortest time) : reward can be negative to encourage lesser time
> - highscore based : like the dinosaur game, a positive reward can be given for  a higher score

_All goals can be defined as a set of cumulative rewards and their maximization_

Maybe better at times to sacrifice immediate reward to get more long term reward

### Agent and Environment

Agent takes actions at each step based on observations of the environment, gets some reward signal

No control over environment except through actions which can influence it

![](images\agent-environment.png)

### History and State

The time series/sequence of the actions, observations and rewards is basically the data and is called history. History determines what happens next

H<sub>t</sub> = A<sub>1</sub>, O<sub>1</sub>, R<sub>1</sub>, ..... , A<sub>t</sub>, O<sub>t</sub>, R<sub>t</sub>

> all observable variables up to time _t_

**Goal**: build mapping from one history to pick next action

Based on action, environment decides  observation/reward

_**State**_ - information used to determine what happens next

Is a function of history

S<sub>t</sub> = _f_(H<sub>t</sub>)

1. Environment state S<sup>e</sup><sub>t</sub> :
    - info used within environment, necessary to determine what happens next from POV of environment

    - usually not visible to agent, even if visible, is irrelevant to agent

2. Agent state S<sup>a</sup><sub>t</sub> :

    - info used to pick next action

    - used to actually build the RL algo

    - any function of history, S<sup>a</sup><sub>t</sub> = _f_(H<sub>t</sub>)

    3. Information/ **Markov state** 

    - contains all useful information from history

    - a state S<sub>t</sub> is _Markov_ iff

    P[S<sub>t+1</sub> | S<sub>t</sub>
    ] = P[ S<sub>t+1</sub> | S<sub>1</sub>, ... , S<sub>t</sub> ]

    (probability of next state given current state is same as probability of next state given all previous states)

    - > The future is independent of the past given the present

    - current state is enough, full history not need

---

- Fully observable environments:

    - agent can directly observe environment state

    - agent state = environment state = information state

    - called a _Markov Decision Process_ (**MDP**)

- Partially observable environments:

    - agent can indirectly observe environment state

    - agent state ≠ environment state

    - called a _partially observable Markov Decision Process_ (**POMDP**)

    - agent has to contruct state representation, can be done in ways like:

        - remember complete history

        - build probabilistic beliefs of environment 

        - use RNNs (linear transformations)

### Major Components of RL agent:

- Policy : behaviour function of agent

- Value function : estimation of how good is each state and/or action

- Model : representation of the environment from agent's POV

#### Policy

- agents behaviour, map from state to action

> examples:
>
>Deterministic (no randomness)
>
>Stochastic (probability is involved so there is randomness)

#### Value function

- prediction of total future reward, used to determine goodness/badness of state/action

- depends on policy 

- involves a discount factor to take into consideration future rewards

#### Model

- agent's model of reality

- predicts what environment will be like

- transitions predict next state based on current state

- rewards predict the immediate next reward

### Types of Models

1. Value based:

    implicit policy 

2. Policy based: 

    no value function as such

3. Actor Critic:

    both policy and value function


>- Model free 
>
>   no model, only value function and policy
>
>- Model based
>
>has a model, might have value function and policy

### Fundamental problems in RL

**Learning** : environment initially unknown, agent interacts and improves policy

**Planning** : model of environment known, agent performs computations with model and improves policy


### Exploration vs Exploitation

_Exploration_ : find out more abt environment

_Exploitation_ : exploit known info to increase reward

> Tradeoff between both is necessary