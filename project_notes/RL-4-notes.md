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

#  Temporal-difference :

This learns from incomplete episodes, by *bootstraping*, we guess from a certain limit about the reward. 

We use the same equation as in MC, the difference being that we use the Immediate reward + discounted value of next state to estimate (just as in the bellman expectancy equation), about the value function from that state. This allows us to quickly reach the target without having to actually perform the process to reach it since we are already iterating for that value. 

This new estimate is called the TD target and the difference between this value and the previous value-function of the state is called as TD error.

the advantages of this approach is that it allows us to learn before knowing the outcome, and we can also learn for non-terminating environements as well. also, since we do not need the final reward to decide the value function, we are able to learn even without knowing the
full sequeance of actions.

thus, MC has no bias (coz no estimation) but due to the usage of the Gt function, it has high variance. On the other hand, TD has some bias (which causes it to misbehave in certain cases) but has very low variance since we use a constant function for our value-estimation instead of the non-stationary rewards. 

when only using some part of the episode, the MC will return the answer that is the min RMS of that part but the TD will return an answer that will actually fit the MDP of that part.

TD works more efficiently for markov processes while MC works better for non markov processes

If you provide TD the n-step prediction chance, it is called as 
TD(lambda), the  smallest one is TD(0) which is what we do for normal TD
claculation. Going for the max range will turn it into a MC problem

but how to know which n is the best? for that we can take an average over all the n step lambda values and we call this the lambda algorithm.
for a single check, multiply by a weight of 1-lambda where lambda is the avg of the Goals found

then G is multiplied by this weight. G  = (1-lambda).lambda^n-1 x G