## Lec 4 notes:

# Model free prediction: 

- Here, we are not given the MDP but we have to still solve the problem. Then, we need to create the value function according to the interaction of the agent with the environment.

This is called as model free prediction.

there are three main ways:

# Monte-carlo method: 

this goes all the way to the end of the interaction and then estimate the value function.

its idea is as follows: value = mean return

disadv: can only apply MC method to episodic MDPs i.e. MDPs that are divided into sections and each section must terminate too.

we sample a set of state actions and reward from the episode under a certain policy, then we calculate the Gt value and then when we are finding the value function: we use the mean return instead of the expected value, since there can be no expectation if the environment is not known.

to do this, we use two different methods:

1. first-visit MC method: In this method, we keep a counter that works over all episodes and is incremented the first time step we find a state s to be visited in that episode. we then find the Gt for that episode and append this to the cumulative return. this is then averaged over all the episodes to get the value function. This works when we do this calculation for a large number of samples thru all the eps.

2. every-visit MC method: the only difference between this and the first-visit method is that here we include all instances of a state as we work thru an episode, not just the first instance.

- Incremental mean: the mean can be counted incrementally.

Incremental monte-carlo: we update the mean V(s) after each episode. 
the N is incremented acc to the method chosen while at the end of the episode the V(s) is updated as follows: V(s)n = V(S) + (Gt - V(s))/N

if we wish to forget the old episodes, for non-stationary problems i.e. problems where things in the MDP change, we can use a constant value to forget older episodes.