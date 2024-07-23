## Lec 2 notes

- MPs: MP or Markov Process, is basically the case in which the agent can see the environment state and decide according to that.  
  Almost all RL problems can be formalized as MPs.

  Any MP can be described as a markov chain, wherein each chain is a markov process i.e. the current state will define the next states according to a certain probability. This creates the tuple (S,P), where S is the state the agent is in and P is the transition probability that decides which next state the agent will choose.

  Now, if we take all these chains then we can create a transition state matrix, in which each position will give us a probability of how the state from the current row will convert to one of the states present in the columns. Now, we can sample the transitions and allow the agent to understand how the environment behaves and what will be the best route in order to achieve maximization of rewards.

- MRPs: MRP, or Markov Reward Processes, are the decision chains the algorithm makes in order to maximize the reward quantity. It is therefore a tuple of (S,P,R,(gamma)), where R is the immediate reward the current state will bring and gamma is the discount factor.
  The main aim is to calculate the cumulative reward Gt = Rt+(gamma)Rt+1+......+(sigma)k=>{0,inf} (gamma)^k*Rt+k+1; and maximize this value.
  here, the reason of using discounts is to avoid infinite loops and to make it mathematically more convinient, however, for terminable processes we can still use a discount value = 1. 0<=gamma<=1; 0 is for short-sightedness and 1 is for far-sightedness.

- value function: this is used to decide which of the samples taken will provide the highest Gt value. basically, we take all the samples chosen from the environment state markov matrix and then calculate the Gt value and multiply that with the expectation value. this can be between 0 and 1 inclusive and is used to deecide how much of the model should be changed depending on the reward. In most cases, we will consider all the rewards in the future so the expectation value is usually chosen to be one. Then we simply chose the path that has the max Gt score and continue sampling from that state and repeat the process. Fo multiple probabilities we will take the average of them and then return to the value function.

- Bellman equation for MRPs: the bellman equation is basically a recursive way of denoting the value function. it is depicted as the following:

    [v(s) = E(Rt+1 + (gamma)*v(s+1))]

it is used in single look-ahead backup diagrams (they resemble trees) 
these can be properly expressed using matrices as well

- The bellman equation is a linear equation, so it can be solved in a simple manner. the answer to the equation is as follows:
  v = R(reward) + (gamma) x P(transition probability) x v
  v(Identity + (gamma)*P) = R
  v = (Identity + (gamma)*P)^-1 * R

  but the computational complexity of this is O(n^3) so this is not very efficient.

- MDPs: MDP or Markov Decision Process is similar to MRP, but in the tuple there is also the action involved. Now, both the transition  
  probability and the reward amount depends on the action taken. Basically, for which action the agent chooses to make, the samples will change and so both the cumulative reward and the probability must also be different.

  for this, we need a mapping of actions and their probability for each of the states. The thing that achieves that is called as the policy.
  P(a|s) = (Probability)[At = a|St = s], the policy also obeys that markow property, in the sense that it will always depend on the current state only. therefore, this is called as stationary policy.

- value function: we will now define two different value functions: state and action-value function.

1. The state value function finds the cumulative reward Gt from state St given that the policy P is being followed

2. The action value function finds the cumulative reward Gt from the state St given that policy P is being followed and action At took place from that state.

- Bellman expectation equation: This is basically the baellman equation but considering tht we a following a policy P, there is no other difference. We can do the same using the action to create the action value function, which will be used in further cases.
  v(a|s) = E[Rt+1 + (gamma)q(at+1|st+1)|s,a]

  All of this is stochastic (we have probability, environment has randomness).

- Optimal Action Value function: This is nothing but the action-value function that is the maximum over all the action-values policies. same can be said for the state value function.
An MDP can be called solved if we find this value consistently over agent runs.

optimal policy: An optimal policy is such that it will always be at least as good as the other policies for all the states. for **any** MDP, there is at least one optimal policy that will be better than all other policies. an example of that would be a policy that follows the maximum possible action value function for all actions performed. In fact, all optimal policies must achieve the optimal function in one way or another.

it can be depicted as follows: P* = 1 iff a = argmax(q*(s|a)) or 0 for all others

this can be found using the bellman optimallity equation: which is basically finding all the q-values for the possible actions that the agent may take and then choosing the one with the max q-value. thats it. then just use this q-value in your policy by providing that route with the probability of one and the agetn will then always choose that possibility.  

The bellman optimality equation is non-linear, so we need to use iterative(recursive) methods to solve for it. those methods are:
1. Value iteration
2. Policy iteration
3. Q-learning
4. Sarsa