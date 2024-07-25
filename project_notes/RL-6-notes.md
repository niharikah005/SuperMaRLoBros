## notes of lec 6

# Value function approximation:

The Value func is mapped to each state. but for larger MDPS, the number of states is too large, so we need an approximator. To do this, we use a set of weights along side the state to determine the value function for that policy. This is the approximating function.

We update the parameter using MC or TD learning.

There are three ways of func approx:

1. Using the state-value function: This takes in the state and the parameter fits it to some function, then we obtain the value function approximation.

2. Using the action-value function: In this one, we take the state and the action as the inputs and then the approximator returns a q-value for that specific state-action pair.

3. Using action-out methods: In this case, only the state is passed. But all the actions that are possible for this state are chosen one by one and their respective q-value is provided as the output.

This approximation is done using gradient descent values, 

grad(J(w)) = vector of partial differentials of J wrt all parameters.

where J(w) = expected MS error of policy v(s) - approx v(s)
we find the grad(J) and then delta w = -(1/2) * step-size * grad(J)

stochastic grad samples the gradient, hence we dont need the expected value in that case.

the approx v(s) can be found using the feature vector x(S), which is basically a vector of all features that can define that state well.
then we can multiply this vector by some weight to decide its importance in defining the state and use this inplace of the approx v(s)

thus, update = step-size * error in value-func * feature value

now, for the policy v(s) that can only be known if we have the MDP. But if we dont, then we should use learning methods like MC and TD.

for MC, replace the policy value-func by Gt of that episode. this is high variance, no bias learning.

for TD, replace the policy value-func by the bellman expectancy eqn for one step. This is small variance, small bias learning.

for TD(lambda) choose the n-steps and multiply by respective weights to obtain the value-func. This is intermediate of MC and TD.

for backward TD(lambda) the trace will increment its depreciated value by the features that are present.

the same things can be done using q-values.

# Batch method:

This is only slightly different than the above methods in the sense that now we create a data-set of state,value pairs (value of policcy only) and we sample from this data-set only, updating it in the same way as earlier. 

The least squares solution is also a new function that operates on the dataset only, but its pretty much similar to the J(w) func discussed earlier.

The earlier method converges to the min of the least squares solution. Its called experience replay

This can be used in deep Q-network (NN) as the approximator.