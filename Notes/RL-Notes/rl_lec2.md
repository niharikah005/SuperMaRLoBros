# MDPS

## Markov Processes

MDP's formally describe an environment for RL, environment is fully observable and current state characterizes process

Almost all RL problems can be formalised as MDPs

### Markov Property

P[S<sub>t+1</sub> | S<sub>t</sub>
    ] = P[ S<sub>t+1</sub> | S<sub>1</sub>, ... , S<sub>t</sub> ]

> The future is independent of the past given the present

### State Transition Probability and Matrix

tells the probability of transitioning from one state to next state

current state characterizes everything 

P<sub>ss'</sub> = P[ S<sub>t+1</sub> = s' | S<sub>t</sub> = s ]

state transition matrix defines probabilities of transitions from state s to successor states s'

![](images/state-transition-matrix.png)

> each row tells probabilities from one starting state to any other state in the column

### Markov Process/Chain

sequence of random states S<sub>1</sub>, S<sub>2</sub>, ... with markov property

> Defn : Tuple (S,P)
>
> S -> finite set of states
>
> P -> state transition probability matrix

## Markov Reward Process

MRP is a Markov chain with value (judgements in form of rewards)

> Defn : Tuple (S,P,R,γ)
>
> S -> finite set of states
>
> P -> state transition probability matrix
>
> R -> reward function, R<sub>s</sub> = E [R<sub>t+1</sub>| S<sub>t</sub> = s ]
>
> γ -> discount factor ∈ [0, 1]

start in state s at time t, at time t+1  reward recieved

what we mainly care abt is the overall reward that certain sequence gets(maximize it)

### Return

G<sub>t</sub> is total discounted reward from time step _t_ onwards

<p>
    G<sub>t</sub> = R<sub>t+1</sub> + &gamma; R<sub>t+2</sub> + &hellip; = &sum;<sub>k=0</sub><sup>&infin;</sup> &gamma;<sup>k</sup> R<sub>t+k+1</sub>
</p>

γ closer to 0: near sightedness (cares more about short term reward)

γ closer to 1: far sightedness (cares more about long term/delayed reward)

> why discount:
>
> mathematically convenient, prevents infinite returns in cyclic processes

### Value Function

total reward/ expected return starting from state _s_

> Defn : state value function v(_s_) of MRP is expected return starting from state _s_
>
> v(_s_) = E [G<sub>t</sub> | S<sub>t</sub> = _s_]

### Bellman Equation 

v(s) = E [R<sub>t+1</sub> +  γ * v(S<sub>t+1</sub>)| S<sub>t</sub> = _s_]

expected/immediate reward now + discounted value at next state

can be expressed better using matrices and vectors

![](images/bellman-eqn.png)
immediate reward + gamma * transition matrix * value of state ended up in

every state has one entry

eqn is linear, can be solved directly but only possible for small MRPs
![](images/solving-bellman.png)

## Markov Decision Process

an MRP with decisions (can take actions)

![](images/defn-mdp.png)

### Policy

distribution over actions given states, defines behavior of agent

π(a|s) = P [A<sub>t</sub> = a | S<sub>t</sub> = s]

policy is the same, independent of time step in, only depends on current state (called stationary policy)

### Value Function

- state value function: v<sub>π</sub> (_s_)

    expected total return starting from state _s_, given that policy π is followed

- action value function: q<sub>π</sub> (_s_, _a_)

    expected total return starting from state _s_, following policy π after taking action _a_

### Optimal Value Function

best possible performance in MDP

- optimal state-value function v<sub>∗</sub>(s) -> maximum state-value function over all policies

    v<sub>∗</sub>(s) = max v<sub>∗</sub>(s)

- optimal action-value function v<sub>∗</sub>(s) -> maximum action-value function over all policies

    q<sub>∗</sub>(s,a) = max q<sub>∗</sub>(s,a)

MDP can be considered to be solved when we know q<sub>∗</sub>(s,a)  (basically the optimal value function)

### Optimal Policy

a certain policy is better than another policy if over all states, that certain policy's value function is greater

![](images/optimal-policy.png)

![](images/finding-optimal.png)

### Bellman Optimality Equation

>for v<sub>∗</sub> :

finds all q<sub>*</sub> values and chooses the max out of them 

v<sub>∗</sub>(s) = max q<sub>∗</sub>(s, a)

>for q<sub>∗</sub> :

![](images/bellman-optimal-q.png)

since this is non linear, cant use matrix and stuff to solve it, need to use iterative methods like q-learning, value iteration, policy iteration, sarsa