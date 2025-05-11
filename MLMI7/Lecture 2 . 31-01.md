Lecture by Miri Zilka

***topics***: *value functions, the Bellman equation, optimality, and both model-based (dynamic programming) and model-free (Monte Carlo) reinforcement learning methods.*

# 1. Value Functions

## 1.1 Motivation and Definitions
In reinforcement learning (RL), value functions quantify how good it is for an agent to be in a certain state (or to perform a specific action in that state), with respect to a given policy. A policy $\pi$ describes the agent’s way of selecting actions in each state.

### State-Value Function $V^\pi(s)$:
$$V^\pi(s) = \mathbb{E}_\pi[R_t \mid s_t = s].$$
Here, $R_t$ denotes the return starting at time $t$, often expressed as a (discounted) sum of future rewards:

$$R_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1},$$
where $\gamma \in [0,1]$ is the discount factor and $T$ is the final time step for finite-horizon tasks (or $T \to \infty$ in infinite-horizon tasks).

Intuitively, $V^\pi(s)$ measures the long-term desirability of state $s$, assuming the agent continues following policy $\pi$.

### Action-Value Function $Q^\pi(s,a)$:
$$Q^\pi(s,a) = \mathbb{E}_\pi[R_t \mid s_t = s, a_t = a].$$
This function measures the expected return if the agent starts in state $s$, takes action $a$ immediately, and then follows $\pi$ thereafter.

The difference between $V^\pi(s)$ and $Q^\pi(s,a)$ is that $Q^\pi(s,a)$ explicitly captures the choice of the first action $a$, while $V^\pi(s)$ implicitly averages over whichever actions $\pi$ would take in $s$.

# 2. The Bellman Equation

## 2.1 Recursive Structure of Value Functions
Value functions satisfy elegant recursive relationships known as the Bellman equations. These equations exploit the fact that the value of a state under a policy $\pi$ equals its immediate reward plus the discounted value of successor states (again under $\pi$).

For a state $s$,

$$V^\pi(s) = \mathbb{E}_\pi[R_t \mid s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1} \mid s_t = s\right].$$

Rewriting part of the future returns, one step at a time, and then grouping terms shows:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma V^\pi(s')].$$

$p(s', r \mid s, a)$ is the environment’s model: the probability of transitioning to next state $s'$ and receiving reward $r$ given that the agent was in $s$ and took action $a$.

We first average over the action $a$ selected by $\pi$; then, for that action, we average over all possible $(s', r)$ transitions and rewards, weighting by $p(s', r \mid s, a)$.

**Intuition:** the expected long-term value from $s$ equals the expected immediate reward plus the discounted value of whatever next state we land in, according to $\pi$.

This Bellman equation captures how each state’s value depends on the values of its successor states, creating a recursive dependency among all states.

# 3. Optimal Value Functions

## 3.1 Defining Optimality
Among all policies, an optimal policy $\pi^*$ achieves the highest possible value function for every state. Formally,

$$V^*(s) = \max_\pi V^\pi(s).$$

Similarly, for action values,

$$Q^*(s,a) = \max_\pi Q^\pi(s,a).$$

Optimal state-value function $V^*(s)$ gives the maximum achievable expected return from state $s$.

Optimal action-value function $Q^*(s,a)$ gives the maximum achievable expected return when taking action $a$ in state $s$ and then following the best possible policy afterward.

## 3.2 Bellman Optimality Equation
The Bellman equation for the optimal value function expresses the relationship between $V^*(s)$ and its successors under the best choice of action:

$$V^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V^*(s')].$$

An equivalent form, for the optimal action-value function, is:

$$Q^*(s,a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma \max_{a'} Q^*(s', a')].$$

**Key point:** the $\max$ over actions moves inside the expectation, meaning we always pick the action that yields the highest expected return.

For a finite Markov Decision Process (MDP) with $|S|$ states and $|A|$ actions, $\{V^*(s) \mid s \in S\}$ or $\{Q^*(s,a) \mid s \in S, a \in A\}$ can be found by solving these non-linear equations simultaneously.

# 4. From Optimal Value Functions to Optimal Policies
Once $V^*$ or $Q^*$ has been computed, an optimal policy $\pi^*$ is easily extracted:

### From $V^*$:
$$\pi^*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V^*(s')].$$

### From $Q^*$:
$$\pi^*(s) = \arg\max_a Q^*(s,a).$$

In other words, the agent should choose the action that maximizes the optimal value or action-value in each state.

# 5. Approximations and the Online Nature of RL
- **Tabular case:** If states are discrete and not too numerous, we can store $V(s)$ or $Q(s,a)$ in simple tables.
- **Non-tabular case:** For large or continuous state spaces, we must approximate value functions using function approximators (e.g., neural networks, linear functions, etc.).
- **Online RL:** Reinforcement learning’s iterative, online nature focuses computational resources on states that the agent visits most often, thereby devoting more precision to important (frequently encountered) regions of the state space.

# 6. Dynamic Programming (Model-Based RL)

## 6.1 When the Model is Available
Dynamic programming (DP) methods require knowledge of the exact transition probabilities $p(s', r \mid s, a)$ and reward function. They then use value functions to organize the search for good policies.

### 6.1.1 Policy Evaluation
**Goal:** Compute $V^\pi(s)$ for a given policy $\pi$.

1. Initialize $V_0(s)$ arbitrarily (e.g., zeros).
2. Repeatedly update:
   $$V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) [r + \gamma V_k(s')].$$
3. Stop once the values change less than some small $\theta$:
   $$\max_s |V_{k+1}(s) - V_k(s)| < \theta.$$

Each iteration is a full backup of every state’s value, based on the current estimate of successor state values.

### 6.1.2 Policy Improvement
To improve a policy $\pi$ once we know its value function $V^\pi$, we pick actions greedily w.r.t. $V^\pi$:

$$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V^\pi(s')].$$

This new policy $\pi'$ is guaranteed to be at least as good as $\pi$. If it is not strictly better, then $\pi$ is already optimal.

### 6.1.3 Policy Iteration
Policy iteration alternates between these two steps (policy evaluation and policy improvement):

1. **Policy Evaluation:** Calculate $V^\pi$ for the current $\pi$.
2. **Policy Improvement:** Update $\pi$ by choosing actions that are greedy w.r.t. $V^\pi$.
3. Repeat until convergence, i.e., until the policy no longer changes.

### 6.1.4 Value Iteration
An alternative, more streamlined DP approach is value iteration:

1. Initialize $V_0(s)$ arbitrarily.
2. Repeatedly apply the “improvement” step:
   $$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma V_k(s')].$$
3. Convergence again occurs when $V_{k+1}$ and $V_k$ are sufficiently close.

Value iteration can be seen as merging policy evaluation and improvement into a single step: each update is a greedy backup according to the current estimate $V_k$.

## 6.2 Asynchronous Dynamic Programming
In large state spaces, repeatedly sweeping over all states can be computationally excessive.

Asynchronous DP updates the value of states in any order. Some states might get many updates, others fewer, as long as every state is updated infinitely often in the limit.

This can be more flexible in practice while retaining convergence guarantees.

## 6.3 Generalized Policy Iteration (GPI)
GPI conceptualizes any interplay of policy evaluation and policy improvement steps—regardless of whether each step is partial or complete—as eventually driving the value function and policy to optimality.

The policy is repeatedly nudged toward greediness, while the value function is repeatedly nudged toward consistency with that policy.

Over time, they converge to a single solution (the optimal value function and the optimal policy).

# 7. Model-Free Reinforcement Learning

## 7.1 When the Model is Not Available
In many tasks, the agent does not know the transition probabilities or reward structure in advance, nor can it easily learn a perfect model. Model-free methods learn value functions (and hence optimal policies) directly from experience:

- **Real experience** (agent acting in the real environment), or
- **Simulated experience** (agent interacting with a simulator).

**Crucial benefit:** Even without a model, the agent can still learn to make optimal decisions by trial-and-error and statistical averaging over rewards observed.

# 8. Monte Carlo Methods

## 8.1 Key Idea
Monte Carlo (MC) methods estimate the value of states or state-action pairs by running episodes and averaging sample returns.

- **Episode:** A sequence of states, actions, and rewards from a start state to a terminal state.
- **Returns:** The discounted sum of rewards observed after a particular visit to a state (or state-action pair).
- **Estimation:** By averaging these observed returns over many visits, MC produces an unbiased estimate of the true value.

### 8.1.1 Monte Carlo Prediction
To estimate $V^\pi(s)$ for a given policy $\pi$:

1. Initialize $V(s)$ arbitrarily.
2. Collect episodes following $\pi$.
3. For each state $s$ in each episode, record the return that follows that visit to $s$.
4. Average all those returns to get $V(s)$.

As more and more returns are averaged, the estimate converges to the true value $V^\pi(s)$.

### 8.1.2 Monte Carlo Action-Value Estimation
To estimate $Q^\pi(s,a)$, the same averaging approach is used, but we track state-action pairs and average returns after each pair $(s,a)$ is encountered in an episode.

# 9. Monte Carlo Control

## 9.1 On-Policy vs. Off-Policy
- **On-policy:** The same policy is used for both generating data (i.e., deciding what actions to take while learning) and being improved (i.e., whose value we estimate and improve greedily).
- **Off-policy:** We learn about a target policy (the one we truly care about, often greedy w.r.t. our current estimates) while following a possibly different behavior policy that ensures adequate exploration of actions.

## 9.2 On-Policy Control with $\varepsilon$-Greedy Exploration
To find an optimal policy purely by sampling:

1. Maintain an action-value table $Q(s,a)$.
2. Generate episodes by following an $\varepsilon$-greedy version of $\pi$ w.r.t. $Q$. This ensures all actions continue to be explored.
3. Estimate $Q(s,a)$ by the average of returns observed.
4. Improve the policy after each episode by choosing the action that maximizes $Q$-values in each state (still using $\varepsilon$-greedy to maintain exploration).
5. Repeat until convergence.

## 9.3 Off-Policy Methods
Off-policy Monte Carlo control (using methods like importance sampling) allows learning a greedy target policy $\pi$ while generating episodes under a different policy $\mu$. Key requirements:

- $\mu$ (behavior policy) must keep exploring all state-action pairs that $\pi$ might prefer.
- Over many episodes, correct weighting (via importance sampling ratios) ensures unbiased estimation of the returns relevant to $\pi$.

One standard approach is:

$$Q(s,a) \leftarrow Q(s,a) + W \frac{R - Q(s,a)}{C(s,a)},$$

where $W$ is the product of the inverse of the probabilities under $\mu$. Once an action is encountered that deviates from the target policy, we stop updating to avoid infinite variance issues. Although more complex, off-policy approaches can be more data-efficient if you can gather experiences from a broader behavior policy.

# 10. Summary
- **Value Functions:**
  - $V^\pi(s)$ measures how good it is to be in state $s$ under policy $\pi$.
  - $Q^\pi(s,a)$ measures how good it is to take action $a$ in state $s$ under $\pi$.
- **Bellman Equation:**
  - Recursively expresses a state’s value in terms of rewards and the values of successor states.
  - Bellman optimality equations do so for the best achievable values.
- **Optimal Policies:**
  - $V^*(s)$ and $Q^*(s,a)$ define the maximum possible returns.
  - Extracting $\pi^*$ is straightforward once $V^*$ or $Q^*$ is known.
- **Dynamic Programming:**
  - Requires a model.
  - Policy iteration alternates policy evaluation and improvement.
  - Value iteration merges them, iterating a single Bellman optimality backup.
- **Model-Free Reinforcement Learning:**
  - Learns directly from sampled experiences without needing transition probabilities.
  - Monte Carlo methods estimate values by averaging returns over complete episodes.
- **Monte Carlo Methods:**
  - On-policy: evaluate/improve the same policy that generates behavior (must remain sufficiently exploratory).
  - Off-policy: learn an optimal policy from data generated by a different (behavior) policy.

# 11. Next Topics
- **Temporal-Difference (TD) Learning:** Methods that combine ideas from dynamic programming and Monte Carlo, updating value estimates from incomplete episodes and bootstrapping off current estimates.

These notes provide a thorough grounding in how value functions, Bellman equations, and both DP and Monte Carlo methods fit into the broader reinforcement learning landscape. By internalizing these concepts—especially the recursive nature of value functions and the interplay between policy evaluation and improvement—you gain a deep intuition for why RL agents can learn optimal behaviors even in complex, uncertain environments.