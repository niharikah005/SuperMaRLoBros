# Optimality in games:

1. the policy of the player will be pi^i for the ith player.

2. the policy of the other players (fixed) will be pi^-i

3. Best response : pi^i(pi^-i)

4. Nash equilibrium: This equilibrium suggests the use of a certain combined policy pi^i * pi^-i such that both the player and the others would have the most optimal policy and would choose not to deviate from that policy. 

# Single agent type RL: 

Best response is going to be the optimal policy for the game in which the opponents can be considered a part of the environment.

## Self-playing RL: 

this is when the agent controls all the opponents, here if we keep solving the MDP (hopefully) then we will achieve the nash equilibrium.

## Two-player zero-sum games:

The rewards for the two players are equal and opposite to each other. R1 + R2 = 0.


Perfect info games: markov

imperfect information: partially observed.

## Minimax:

for zero-sum games, we can minimise the value func for player 2 while maximizing it for the 1st player. It is a joint policy. It gives the nash equilibrium.

It becomes very big, so we sample the minimax searches and then create a value function for it.

It is used a lot for chess and checkers and such games.

## Binary-linear Value function:

In chess, we can create a binary feature vector that keeps a weight for the things that exist on the board, we then take a dot product of this with the vector that stores the values for each of the features. Finally, we send the answer thru the logistic (s-shaped) curve for obtaining the probability that such a thing is good for us.

# Self-play TD:

if we use MC: we take the difference in probability of who was more likely to win and who won and then shift the parameter towards that value by a little bit (grad ascent).

if we use TD: its the same as above, except we now use the value function for taking that difference.

for TD lambda: we simply find the G lambda value and use that in the error.

delta w = const (win Gt - lost Gt) * (grad of v wrt w).

# after states:

If the game has a known set of rules then we can try to find the states that will most likely succeed the current state using those rules. This is called the after-state. for this, q(s,a) = v(succ(s,a)) where succ is the successor state. 

We then choose the action that maxs the value for white while mins the value for black (from those max values.)

# Simple TD:

1. We find the value function using TD learning.
2. Use minimax search on these value functions.

## TD Root:

In this case we play an action, we do a deep search, then we minimax on it simultaneously. Finally, we obtain the node that follows the minimax algo, then we go back to the root and update its value function (something like this I hope)

## TD leaf:

In this case, we do the same things as in TD root, but this time, we update the leaf node before the search and keep moving up with the same updates.

## TD strap :
uses all the minimax values instead of just the leaf. 

# simulation based:

This uses the idea of many-armed bandit for each child in the MC tree search algo, it also converges to the min-max value, most useful for cases where its hard to create a value function.

# Smooth UCT search:

does a MCTS to info state space of the game tree.
uses ficticious play- averaging over all possibilities that the opponent could make (MC type).
extract avg strategy from node's(opp) action counts: N(s,a)/N(s) as the policy.

action is picked as follows: A = UCT(S) with eta prob OR the above policy for all the actions in the state space, with 1-eta prob.

almost converged to the nash equilibrium.