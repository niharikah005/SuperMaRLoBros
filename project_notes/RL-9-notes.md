# Exploration vs Exploitation

Exploitation: using the info obtained greedily.

Exploration: To check other areas of the MDP (environment) to get new data.

# Different approaches:

1. Random exploration: eg, epsilon-greedy

2. Optimism in the face of uncertainty: pick the option that has greater potential.

3. Information state space: Consider agents info as part of the state.

look-ahead to see how information helps reward. It is computationally more difficult than the other two.

# Different ways of exploration:

State-action exploration:

1. systematically explore state space.
eg, pick different actions than the optimal ones. each time S is visited.

2. Parameter exploration:

Parametrise the policy: pick different parameter each time.

adv: consistent exploration
disadv: does not know about the state action space. so we do not know about the prev states and the way the model behaves.

# Multi-armed bandit:

It is a tuple (A,R) where A is a known set of actions A (arms) and R is an unknown prob distribution over the reward.

At each step an Action is chosen and the reward is given. The aim is to max the cumulative reward.

1. Action-value: Its the prob of the reward being given over an action being chosen.

2. Optimal value: Its the max of the action-value.

3. Regret: its the loss in the amount of reward we got for some action compared to the optimal value.

4. Total regret: summ of all the regrets.

5. we will now minimize the total regret. 

We can also imagine the regret as a function of gaps and counts, where a gap is the difference between the current action-value and the optimal value. The count is simply the number of times we selected that action.

therefore: L = expected(summ v - q(A)) when A was chosen
             = summ(expected(N(count)*(v-q(a)))) for some action A
             = summ(expected(N*delta)) where delta is the gap. 

This algo ensures that the counts are small for large gaps. However, the gaps are unknown.

The normal greedy has a linear total regret because it may get stuck in sub-optimal actions.

Optimal-greedy has a linear total regret because even though it assigns the max reward to all actions, due to a few unlucky bouts, the optimal actions value may decrease and the greedy algo may not pick it up again.
However, this does encourage exploration

The decaying-greedy algo needs the optimal value to find all the gaps, then it slowly decays the greedy probability over time, so it gets more and more exploiting, thus, it has logarithmic total regret.

## optimism in the face of uncertainty:

we should choose the value that has not been checkd alot but has a high reward chance. 

we will pick the action with the highest confidence (less uncertain about it) and use that action, however, it should be worse than the optimal q-value minus the current q-value. This way, we guarrantee exploration of parts we have not checked at all.

The confidence is inversely proportional to the number of times we have gone thru a certain action.

Hoeffdings inequality uses a series that works for all distributions,

we will find the probability of whether the action taken was actually worse than the expected value by a certain amount u, this can be bound to the series e^ (-2tu^2) . we can use the UCB on this inequality.

This also has logarithmic asymptotic total regret.

## probability matching:

we update the policy depending on basically picks those actions more that have the chance of being the best. It can be a little difficult tho.

Thomson sampling: does probability matching from samples. Pick one which is the best and then continue with that. uses Bayes law to compute that.
Its one of the few algos that are able to atain the lower bound of the UCB.

# Info gain:

This one takes into account tha amount of info that we obtain when we pick an action and decides whether that info is enough given the amount of times the agent will be able to repeat that action.

We can keep transitioning from the states as we take the actions and note them in the transition matrix.

If we use the same method but with distributions then we can call this Bayesian RL, in this case we can use Beta functions that update the distribution based on the count of success and failures (bernoulli method). This helps in solving the MDP that got created due to the Info gain structure. This one looks ahead for this.


# applying the same thing to MDPs:

we can do that by adding states inthe action-value functions.