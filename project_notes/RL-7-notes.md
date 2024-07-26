## Policy Gradient Methods.

# Policy Based Learning:

adv:
- better convergence properties.
- effective in continuous action spaces
- learns stochastic policies

policy based learning using stochastic properties is good because value based may sometimes not be able to differentiate between two states if the features match. In such a case, probabilistic determination of the action is much more fruitful.

# policy objective funcs:

- creating one using the start value:

J = V(s1) where s1 is the starting state. for a policy pi having parameters theta.

- creating one using the average value:

Jav = summ d(s) V(s) over s, where d is the stationary distribution function.
useful in continuous environments.

creating one using avg reward per stim-step:

JavR = summ d(s) summ pi(s) * R(s,a)

# policy optimization:

this is done to find the parameter theta that will get the maximum reward.

methods like hill-climbing, simplex, genetic algorithms are used when the gradient is unknown.

methods like gradient descent, quasi-newton methods, conjugate gradient are used when gradient is known.

# policy gradient:

In this case, we use the gradient ascent.
therefore: delta theta = step-size * gradient of J-func.

then we try to maximize its ascent.

Finite difference: In this case, for each dimension k from [\1,n],  partial diff J wrt theta k approx = J(theta + epsilon * uk) - J(theta)/ep

where uk is the unit vector with one in the kth direction and zero elsewhere.

# score function:

We use the likelihood policy here:

gradient pi = pi * grad(pi) / pi
            = pi * grad(log(pi))

where grad(log(pi)) is called the score function.
this helps in computing policy gradient analytically.

- Softmax policy:

we weigh actions using linear transformations of the features.
we say that the probability of action given a state is proportional to the exponentiation of these linear transformations.

hence, the score function just becomes the difference of the weighed transformation and the avg of transformations over all actions.

- Gaussian policy:

in this case, we take the mean of alot of weighed linear transformations and consider this as a parameter mu(s), there can also be some variance sigma^2 that can be const or varying. we then pick the action from the normal (gaussian) distribution of these two parameters.

thus the score function becomes (action taken - mu(s)) * feature vector /variance.

# One step MDP:

starting state s, one step forward, reward R, terminate.

JavR = E(r) = summ d(s) summ pi(s) * R
grad(JavR)  = summ d(s) summ pi() * grad(log(pi(s))) * R

# policy gradient theorem:

This transforms the liklihood ratio into the multi-step MDP approach

we replace the instantaneous reward into a long-term value Q.
it applies to J = J1, JavR and (1/1-discount) * JavV.

grad J = expected[\ grad(log(pi(s,a))) * Q(s,a)]

# MC policy Gradient: (REINFORCE)

for each episode:
    for t = 1 to T-1:
        theta = theta + grad(log(pi(st,at))) * vt

return theta

where theta is the parameter that we are optimizing.

# actor-critic:

the critic estimates the q-value and the actor tries to fit itself in the q-value.

# Reducing the variance using a baseline:

A good baseline is the value function.

so, the adv function = Q-value - value-function.
then use that in the parameter optimization function.

the critic should actually estimate both the q and the v value.

if we use TD learning, then the TD error would be enough for calc the advantage. That can be used. This would need only the parameter V.

for online application, we can use TD learning with elegibility traces. 