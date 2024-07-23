## Lecture 3 notes:

- Dynamic programming: it is a really important part of RL, allowing us to compute cumulative values by looking ahead and dividing problems into solvable parts, it has three important parts:

1. Optimal substructure: When we use DP to divide the problem, we make sure that the smaller problems actually help in optimizing for the bigger picture. In other words, by solving the smaller problems, we are able to optimally solve for the larger ones as well.

2. Overlapping sub-problems: In order for the smaller problems to be useful for solving the larger ones, we must make sure that they recurr in later cases so as to allow for solving for other subproblems as well. eg, in trying to find the shortest distance, we can divide it into halves and try to find the shortest distance among them, then when we merge these answers, the smallest half will recurr for all cases until the whole distance has been found. these are the two requirements of DP for RL.

both of which are satisfies by the bellman equation. The value function is similar for the second point.

DP helps us in solving the MDP, it helps in the planning part in two ways:

1. prediction: we use DP to create an algo that will output the value function when the input is the complete MDP and the policy

2. control: we can also use DP to create an algo that will output the optimal value function given the complete MDP and thus also provide for the optimal policy (implied).

- Bellman expectation equation: This is used in order to find the value function of the policy.

we choose some arbitrary value function and then keep doing the bellman calculation for many iterations for every state and then get the original value function for the policy. This way of using all states  is called as synchronous backup.

to do this we first find the expectation value for the actions then we put the next iterations value using the state function. This way we get one single value function for the next state using the current state.

so, vk+1 = R(pi) + gamma*(transition probability)(pi) * vk(s)

we use the transition state matrix because we want to avg over all the states. This is the sub-problem that we optimally find, then we keep iterating over next state and then over the next and so on.

- policy iteration: this uses two steps:

1. evaluate the policy by the above methods 
2. update/improve the policy by acting greedily once we have obtained the policy.

this method always converges to the optimal policy, but more iterations might be requiared.

for greedly choosing a policy, we choose the policy that gives us the max q-value. so, pi = argmax(all actions) q-value(all policies)

for the new policy, if we run it for one iteration, then hop back to the old policy, and if the value function thru this is at least as good as the previously found value function, then we can say that the policy has improved.

however after some time, this improvement might stop and we may reach a point where it is equal to the previous value function, then, we can check if the value function is equal to the max of all the value functions found previously, i.e. compare its value with the bellman optimality equation, if it holds true, then it is the optimal policy and we can then say that the MDP has been solved.

we can also try to shorten the number of iterations by either checking if the new value function differes from the prev one only by a slight amount OR by putting a limit of iterations. The most extreme case of this would be to optimise the policy after a single iteration, this is called as value iteration method.

