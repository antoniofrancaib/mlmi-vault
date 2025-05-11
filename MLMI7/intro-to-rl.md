- **Introduction to Reinforcement Learning**
    
    - Agent-Environment Interaction
    - The Reward Signal and Decision Making
- **Markov Decision Processes (MDPs)**
    
    - Definition and Components
    - The Role of MDPs in RL
    - The Difference Between Planning and Learning
- **Policies and Value Functions**
    
    - Deterministic vs. Stochastic Policies
    - State-Value and Action-Value Functions
    - The Bellman Equations
- **Solving MDPs with Dynamic Programming**
    
    - Policy Evaluation and Improvement
    - Value Iteration (VI)
    - Policy Iteration (PI)
    - Comparing VI and PI
- **Model-Free Learning**
    
    - The Need for Learning Without a Model
    - Monte Carlo Methods
    - Temporal-Difference Learning (TD)
- **Temporal-Difference Control Algorithms**
    
    - SARSA: On-Policy Learning
    - Expected SARSA: A Smoothed Update Rule
    - Q-Learning: Off-Policy Learning
    - Comparing SARSA, Expected SARSA, and Q-Learning
- **Exploration vs. Exploitation**
    
    - The Role of Exploration in RL
    - ϵ-Greedy Strategy and Soft Policies
- **Advanced Topics in RL**
    
    - Function Approximation and Deep RL
    - Policy Gradient Methods
    - RL in High-Dimensional Spaces

# Introduction to Reinforcement Learning

Reinforcement learning (RL) is a paradigm of machine learning concerned with how an agent can learn to make decisions through interaction with an environment in order to achieve a goal. The central concept in RL is the continuous interaction between an agent and its environment.

At each discrete time step $t$, the interaction proceeds as follows:
- The agent observes the current state $S_t$ of the environment.
- Based on $S_t$, the agent selects and executes an action $A_t$.
- The environment responds by transitioning to a new state $S_{t+1}$ and providing a reward $R_{t+1}$ to the agent.

![[Pasted image 20250307191356.png]]

### Markov Decision Processes (MDPs)
To rigorously analyze and design reinforcement learning algorithms, we typically assume that the underlying problem can be modeled as a **Markov Decision Process (MDP)**. An MDP formally captures the dynamics of sequential decision-making under uncertainty and reward.

An MDP is defined as a tuple; $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. 

Let's formally introduce each element:

- **State Space ($\mathcal{S}$)**: The set of all possible states. A state $s \in \mathcal{S}$ represents a complete description of the environment (and agent) at a given moment. $\mathcal{S}$ can be finite or infinite.

- **Action Space** ($\mathcal{A}$): The set of all possible actions the agent can take. Typically, we assume this set is fixed for simplicity, but sometimes actions may depend on the current state.
    
- **Transition Probabilities** ($P$): The state transition probabilities, defining the dynamics of the environment. For any states $s, s' \in \mathcal{S}$ and action $a \in \mathcal{A}$, is the probability that the next state is $s'$ given that the current state is $s$ and action $a$ is taken, 

$$
P(s' \mid s, a) = \Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\},
$$

- **Reward Function** ($R$): Defines the expected immediate reward obtained after transitioning between states by taking a certain action:
$$
R(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']
$$

- **Discount Factor** ($\gamma$): A parameter $\gamma \in [0,1]$ determining the relative value of future rewards. It captures the intuition that immediate rewards are typically more valuable than future rewards. Formally, the agent seeks to maximize the cumulative discounted rewards, also known as the **return**:
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.
$$

Typically, an MDP specification might also include an initial state distribution (for episodic tasks) and a set of terminal states. However, the core definition remains the quintuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

To reiterate, an MDP is a fully specified environment for RL problems under the Markov assumption. If the agent knows the MDP (knows $P$ and $R$), then the reinforcement learning problem reduces to a planning problem: the agent could, in principle, compute an optimal policy by solving the MDP using dynamic programming (because with $P$ and $R$ known, it becomes a classic planning problem for which algorithms like value iteration and policy iteration can be used). 

However, in most RL scenarios, the agent does not initially know $P$ or $R$. It might know $\mathcal{S}$ and $\mathcal{A}$ (what states and actions exist), or it might not even know that. The agent must learn from experience to behave well, without having a complete model of the MDP. This difference is often described as learning (RL) versus planning (classical dynamic programming). 

Nonetheless, MDPs provide the theoretical underpinning: we measure an RL algorithm’s performance in terms of how well it does in the underlying MDP. MDPs also allow us to define what it means to be optimal. Specifically, we can define the concept of a value function on states (and state-action pairs) and derive the Bellman equations, which characterize optimal behavior. That’s our next topic.

##### Policies

A **policy** ($\pi$) defines the agent's behavior by specifying how actions are chosen in each state. Formally, a policy is defined as:

$$
\pi : \mathcal{S} \times \mathcal{A} \to [0,1], \quad \pi(a \mid s) = \Pr\{ A_t = a \mid S_t = s \}
$$

Policies can be:

- **Deterministic**: The agent always takes the same action from a given state.
- **Stochastic**: The agent selects actions according to a probability distribution.

The ultimate goal in RL is finding an **optimal policy** $\pi^*$ that maximizes expected returns:

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi} [G_t]
$$

##### Value Functions 

Much of reinforcement learning theory and algorithms revolve around the concept of value functions. A value function tells us how good it is to be in a state (or to take a certain action in a state) in terms of expected future rewards. 

Intuitively, the value of a state is the total amount of reward an agent can expect to accumulate starting from that state (now and in the future), if it behaves optimally or according to a particular policy. Value functions are useful because they allow the agent to plan ahead in a sense: by knowing the value of subsequent states, the agent can select actions that lead to states of high value, thus obtaining high reward. The two key types of value functions are:

- **State-Value Function ($V$)**: Given a policy $\pi$ (a way of behaving), the state-value function $V^\pi(s)$ is defined as the expected return (cumulative discounted reward) when starting in state $s$ and following policy $\pi$ thereafter. This value function essentially assigns a numerical score to each state, reflecting the desirability of that state under policy $\pi$. Formally, 
$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s\right],
$$


- **Action-Value Function ($Q$)**: Similarly, the action-value function (or Q-value) $Q^\pi(s,a)$ is defined for state-action pairs. $Q^\pi(s,a)$ is the expected return if the agent starts in state $s$, takes action $a$, and thereafter follows policy $\pi$. Formally,
$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right].
$$

State-value and action-value are closely related: In fact, given $Q^\pi$, you can get $V^\pi$ by taking the expected value over the policy's action choice:

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s,a).
$$

This just says: in state $s$, policy $\pi$ chooses action $a$ with probability $\pi(a|s)$. The value of $s$ under $\pi$ is the weighted average of the Q-values of $(s,a)$ for actions $a$ chosen by $\pi$.

##### Bellman Equations
Now, why are these value functions so useful? There are two main reasons: 

The first one is their **relation to optimal policy**. If you somehow knew the true values $Q^*(s,a)$ for all state-action pairs – where $Q^*$ denotes the action-value function for the optimal policy – then acting greedily with respect to these values would give you an optimal policy. Specifically, an optimal policy $\pi^*$ would choose in each state $s$ an action $a$ that maximizes $Q^*(s,a)$. Because $Q^*(s,a)$ already accounts for the future, picking the action with highest $Q^*$ is the best immediate decision. This means the Q-value function alone can dictate optimal decisions. Similarly, $V^*(s)$ (the optimal state-value) is the expected return from $s$ if you behave optimally; and $V^*(s) = \max_a Q^*(s,a)$.

The second one is that value functions satisfy important **recursive equations** due to the Markov property and the definition of return. These are the Bellman equations. They allow us to relate the value of a state to the values of its successor states, which forms the basis for many solution methods (both analytical and learning algorithms).

Let’s derive the Bellman equation for $V^\pi$ (the state-value function for a given policy $\pi$). Consider a particular state $s$. Under policy $\pi$, the agent will take some action $a$ according to $\pi(a|s)$. Then it will receive an immediate reward $R_{t+1}$ (with expectation $r(s,a)$) and transition to a next state $s'$ with probability $P(s'|s,a)$. From that state $s'$, the value is $V^\pi(s')$ (because from $s'$ onward the agent still follows $\pi$). So the expected return from $s$ can be broken down into two parts: the one-step reward plus the discounted value of the next state. Taking expectation over both the random action (as per $\pi$) and the random next state (as per $P$), we get:

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right].
$$

This is the Bellman expectation equation for the policy $\pi$. In words, it says: the value of state $s$ under policy $\pi$ equals the expected immediate reward plus the expected discounted value of the next state, where the expectation is over the action choice of $\pi$ and the environment’s stochastic transition. It’s a self-consistency condition – $V^\pi$ must satisfy this for all states $s$. Written more succinctly using expectation notation:

$$
V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right].
$$

The above is an example of a Bellman equation. It is essentially a linear equation (or system of equations, one per state) for the values $V^\pi(s)$. If the MDP is finite and $\pi$ is fixed, this set of equations can be solved in principle (e.g., by iterative methods or matrix inversion) to find $V^\pi$. Many RL algorithms like policy evaluation (part of policy iteration) are based on iteratively applying this Bellman update to converge to $V^\pi$. There is a similar Bellman equation for the action-value function $Q^\pi$. It can be derived by considering the first step: from $(s,a)$, you get reward $R_{t+1}$ and next state $s'$, then follow policy $\pi$. So:

$$
Q^\pi(s,a) = \mathbb{E}\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a\right] = r(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s').
$$

And if we substitute the expression for $V^\pi(s')$ (expanding it one step further), we could also express $Q^\pi$ in terms of $Q^\pi$ of next state-action pairs (this gets into Bellman operator concepts, but let's keep it simple). Now, the more exciting part is the Bellman optimality equations. These equations characterize the value function $V^*$ of an optimal policy $\pi^*$, without needing to know $\pi^*$ upfront. They are non-linear (because of a max operator), but they are the cornerstone of understanding optimal solutions. For optimal state-value $V^*(s)$, we can reason as follows: if you are in state $s$, you will choose whatever action maximizes your expected return. If that action $a$ leads to a next state $s'$, you will then get optimal return from $s'$, which is $V^*(s')$. So the Bellman optimality equation for $V^*$ is:

$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right].
$$

This equation says: the optimal value of state $s$ equals the maximum (over actions) of the expected immediate reward plus discounted value of the next state, assuming optimal play onward. Notice this is a kind of self-consistency: if $V^*$ is the true optimal value function, plugging it in on the right, the equation should hold. Conversely, any $V$ that satisfies this Bellman optimality equation must be the optimal value function. This equation is non-linear due to the max, but there are standard algorithms (like value iteration) that can solve it iteratively by turning it into an update:

$$
V_{\text{new}}(s) \leftarrow \max_{a} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_{\text{old}}(s')\right].
$$

Repeatedly applying that update will converge to $V^*$. Once you have $V^*$, you can extract an optimal policy $\pi^*$ by choosing for each state $s$ any action $a$ that achieves the argmax in the above equation. There is also a Bellman optimality equation for the optimal action-value function $Q^*(s,a)$:

$$
Q^*(s,a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a')\right].
$$

This is a one-step lookahead: the Q-value of $(s,a)$ under optimal behavior equals the expected reward plus discounted value of the next state, assuming optimal action $a'$ is taken next. Compare this to the non-optimal $Q^\pi$ equation we gave earlier – the difference is we replace "follow policy $\pi$" with "take the best possible next action". The Bellman optimality equations are at the heart of RL because they give us a way to compute or approximate optimal values and hence optimal policies. Many RL algorithms can be viewed as trying to estimate $Q^*$ or $V^*$ without knowing $P$ and $R$ explicitly, by sampling transitions through interaction. For example, Q-learning (a famous RL algorithm) is essentially an iterative update to approximate the Bellman optimality equation for $Q^*$, using actual experienced transitions (samples of $s, a, r, s'$). Another class of algorithms, policy iteration, uses the Bellman expectation equation for a given policy (policy evaluation) and then improves the policy by acting greedily w.r.t. the value function (policy improvement); it can be shown that this converges to satisfy the Bellman optimality equation. It’s worth noting why we use discounting ($\gamma < 1$) in many cases: it ensures that the infinite sum of rewards converges and places progressively less weight on rewards further in the future (which often aligns with practical considerations like uncertainty in long-term predictions). In episodic problems that always terminate, you can set $\gamma = 1$ and consider undiscounted total reward (the episode will end so the sum is finite). The theory still holds, but sometimes even in episodic tasks a discount less than 1 is used to encourage focusing on nearer-term rewards (or to approximate a situation where there is a small probability the episode might continue indefinitely). Finally, a side concept: optimality and uniqueness – for finite MDPs, there is at least one optimal policy $\pi^*$. All optimal policies share the same value function $V^*$ (and $Q^*$). The Bellman optimality equation typically has a unique solution for $V^*$ (assuming some mild conditions), so solving it yields the value of states under any optimal policy. However, there could be multiple policies that achieve those values (multiple ways to be optimal). As an example, in some states there may be two or more actions that are equally good and lead to equally good outcomes; any of those actions could be chosen by an optimal policy. To recap this section: value functions ($V$ and $Q$) are predictions of future reward. They satisfy Bellman equations that relate the value of a state to the values of successor states. The Bellman optimality equation characterizes the value function of the optimal policy. These equations provide both conceptual insight (they tell us what optimality means in a recursive way) and practical algorithms (many RL methods work by trying to satisfy these equations from data). We’ve kept the explanations intuitive, but mathematically, you can see that RL algorithms often involve solving these equations approximately, since we usually don't know $P$ and $R$ to solve them exactly. In Chapter 2, we will dive into specific algorithms (like dynamic programming, Monte Carlo, Temporal-Difference learning) that use these principles to compute or estimate value functions and ultimately find optimal policies.
