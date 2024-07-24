## lecture 5 notes:

# Model-free control: 

This idea builds up on the idea of model-free prediction in the sense that now, we use the two methods, MC and TD, to find the optimal value functions to solve the MDP.

this is useful in all cases where the MDP is either unknown or is so large (too many states, P matrix too big) that we can use it only by sampling it.

# Two methods of learning:

1. On-policy learning: This type of learning is when the agent tries to evaluate the policy (using policy eval techniques) while sampling events from the policy.

2. Off-policy learning: This type of learning is when the agent tries to learn about the policy thru some other examples of an agent using the policy i.e. it samples from someone else's behaviour.

# Learning using MC:

There are two issues with doing this: 

1. When evaluating the policy, we ususally use the bellman expectancy equation to find the value functions, however, if we want to be truly model free, we cannot do such a thing.

2. Because MC takes the approach of selective sampling, not all the states are sampled, which affects the exploring efficiency of this method.

The solution to the first issue is to use the q-value instead of the state-value, in this way, we can just maximise the q-value only and not have to worry about the state-value function at all.

The solution to the second issue is to use a method called epsilon-greedy expression, in which use decide between two choices - first is to choose the greedy action with a probability of 1-(epsilon) and the second is to choose the other actions at random with a probability of epsilon.

now, we can try to estimate the policy even if we run MC for each episode and update the policy then. Hence, we should try to do that as long as it satisfies certain conditions. These conditions are called as Greedy in the Limit Infinite Exploration or GLIE. 

It basically states that any process will be good if :

1. All the state,action pairs are checked infinitely and

2. The policy converges to a greedy policy.

# learning using TD:

there is only one difference: since TD bootstraps the value fnc, we can use even more steps in this case, and so the frequency of the updates will also increase.

The idea is called as sarsa wherein the first state provides the s,a, the reward r is immediate and then we go to the next state s dash, a. hence, sarsa. The first state and action are already known but the next action is sampled using the policy. apart from this, we update it in the same way as in the MC method after *every time-step*

The algo converges to the optimal thing, but we need to be careful about the step-size (altho in real word it works anyway)

# n-step sarsa:

here, we check the returns after a certain number of steps.

ex, if n = 1, then we look ahead one step, so normal sarsa.
this can go al the way to inf, which is full calc or MC method.

for each n, the discount func in the bellman eqn we use is inc by one power. until we cant go beyond the total number of states.

# sarsa lambda:

similar to TD(lambda), we have sarsa lambda, wherein we multiply the Gt value (or the q-value in this case) by a some weight lambda.

for the first step, its 1-lambda (for convergence to one), and then we keep multiplying by lambda each n-value. This helps to average over all the returns that we accumulated over-time

unfortunately, this is not viable for on-line learning, since we have to look ahead in time.

so we use eligibility traces, wherein, we can now look backwards and compute the q-value instead of looking forward. The eligibility trace blames the states or actions that were either most recent or most frequent for achieving the goal.

Et(s,a) = (gamma)(lambda)Et-1(s,a) + 1(s,a)
so now Q = Q + const * (TD error) * Trace calculated

the reason we have a decay rate is to decide the updates according to it to increase explorability of the policy. At the same time, the steps near to the goal should be greedily chosen, hence the GLIE policy is used here.


# off-policy learning:

reasons: 

1. learn from humans or other agents
2. re-use the old policies
3. find the optimal policy while learning all the policies
4. learn about different policies while learning one policies.

note: MC is really bad for off-policy, hence, use TD.

we sample using the TD, then we use the other policy we had and compare it with our working policy to see how better/worse we were to it. Then we multiply this amount with the TD error to normalise it in accordance with this comparision (importance sampling). and then we add this to update our policy.

# Q-learning:

this is one of the best methods for offline learning, the reason being that we directly compare current policy with the expected rewards from the policy that we are observing.

hence: Q(current) = Q(current) + const * ((bellman expectancy for Q(observe) - Q(current)))

lastly, we can try to act greedily as well, this can be done by maxing over all the q values for the observing policy. this helps in improving both the current and the observed policies.

hence we can say that:

using bellman expectancy eqn: policy eval using DP or TD learning
using bellman expectancy eqn for q-values: policy iteration or SARSA
using bellman optimality eqn: value iteration or Q-learning