Lecture by Miri Zilka
***topics***: *reinforcement learning as learning from interaction, the Markov assumption, markov decision processes (MDPs), value functions, bellman optimality equations*

# 2. Different Learning Frameworks
## Supervised Learning
- Trains on labeled data provided by an external teacher.
- **Objective:** Minimize prediction errors on labeled examples.
- **Example:** Classifying images, where each image comes with a “correct” label.

## Unsupervised Learning
- Finds hidden structure in unlabeled data.
- **Objective:** Discover patterns or estimate underlying distributions.
- **Example:** Clustering items based on similarity without any pre-existing labels.

## Reinforcement Learning
- Learns via interaction with an environment.
- There are no direct labels for correct actions.
- Instead, an RL agent tries actions and observes rewards, aiming to maximize long-run reward.

### Key Distinction:
- Unlike supervised learning, the agent in RL is not directly told what actions to take.
- Unlike unsupervised learning, RL has an explicit objective to maximize reward over time (rather than just discovering structure).

# 3. Learning from Interaction
## 3.1 Definition and Core Idea
Reinforcement Learning concerns learning what actions to take in different situations to maximize cumulative reward.

Crucially, the agent is not told which actions are best. It must discover those actions through trial and error—interacting with the environment.

## 3.2 Influence of Actions on Future Outcomes
A unique aspect of RL is that the agent’s actions affect future states and future rewards.

Choosing an action does more than yield an immediate result—it changes the subsequent path the agent will take in the environment.

## 3.3 Approximate Solutions in Practice
Real-world environments are often continuous and high dimensional.

Exact solutions can be computationally or practically infeasible. Therefore, RL methods often rely on approximations that can handle the scale and complexity of real problems.

# 4. The Exploration–Exploitation Dilemma
The agent must balance two competing objectives:

- **Exploration:** Trying out new actions to discover potentially better rewards.
- **Exploitation:** Leveraging existing knowledge about which actions currently yield the highest reward.

### Tension:
- Exploiting too early or too often might mean missing out on better actions not yet discovered.
- Exploring too much may waste resources and time on suboptimal actions.

# 5. Examples of Reinforcement Learning Applications
- **Robotics:** Training a robot to navigate a space or manipulate objects.
- **Game Playing:** Systems such as AlphaGo that master board or video games.
- **Dialogue Systems:** Interactive AI that improves responses to user queries over time.
- **Online Advertising:** Ad placement strategies to maximize clicks or conversions.
- **Self-Driving Cars:** Learning driving policies that maximize safety and comfort.
- **Healthcare:** Optimizing medication dosages over time for better patient outcomes.

(All these illustrate how RL shapes behavior under uncertainty, with actions feeding back into future states and rewards.)

# 6. Key Properties of Reinforcement Learning
- **Interaction:** The agent learns by directly engaging with its environment.
- **Planning:** Since actions influence future states, the agent typically needs to plan ahead (or learn to plan) while dealing with uncertainty.
- **Goals:** There is an overarching reward signal that the agent tries to maximize.
- **Uncertainty:** The environment can have stochastic or partially known dynamics.
- **Experience-Based Learning:** RL methods often use experiences (state, action, reward transitions) to refine the agent’s policy over time.

# 7. The Agent–Environment Interface

![[agent-env-interface.png]]


At each time step, the agent observes a state $\mathbf{s}_t$ from the environment and takes an action $\mathbf{a}_t$.

The environment responds with a reward $\mathbf{r}_t$ and a new state $\mathbf{s}_{t+1}$.

A policy $\pi_t$ describes how the agent chooses actions given the state, i.e., $\pi_t(a|s) = p(a_t = a | s_t = s)$.

# 8. Main Elements in Reinforcement Learning
## Policy ($\pi$)
- A function mapping states to (probabilities of) actions.
- Intuitively, it defines the agent’s behavior—given the current situation, what should the agent do?

## Reward
- A scalar feedback signal that indicates how good or bad a state or action outcome is, relative to the agent’s objective.
- This is the primary driver of learning. Maximizing this cumulative reward is the ultimate goal.

## Value Function
- While the reward is the immediate outcome, the value function estimates the long-term desirability of states (or state–action pairs).
- It predicts future reward that can be accrued if one follows a particular policy starting from a given state.

## Model
- A model describes how the environment behaves—i.e., the transition dynamics $p(s', r | s, a)$.
- Model-based RL uses such a model to plan ahead.
- Model-free RL learns directly from trials (experience) without an explicit model of transitions and rewards.

# 9. Example: Robot Picking Boxes
- **States:** The positions of the robot and the boxes, whether a box is on the shelf or not, etc.
- **Actions:** Move to a certain location, pick up a box, place a box on a shelf, etc.
- **Reward:** Could be a positive reward each time a box is successfully placed on the shelf, and potentially costs (negative rewards) for collisions or time spent.

This example highlights how we choose:

- Which real-world details go into the state (e.g., location of boxes, battery level)?
- Which actions the robot can execute (move left, right, forward, or pick up, put down, etc.)?
- What reward signal best represents the task objective (maximize boxes shelved, minimize collisions, etc.)?

# 10. Abstraction in Reinforcement Learning
RL posits that we can reduce an agent’s sensors, memory, and controls to just:

- **State:** A formal representation capturing all relevant information.
- **Action:** The possible moves or decisions at each step.
- **Reward:** A numerical measure of success or failure.

This abstraction is powerful because it decouples the learning algorithm from the finer implementation details of sensors and actuation.

# 11. Goals and the Reward Hypothesis
**Reward Hypothesis:** “All of what we mean by goals and purposes can be well thought of as the maximization of the expected cumulative reward.”

This means that if you can define one scalar reward signal that correctly represents your objectives, the RL agent’s aim to maximize this reward effectively pursues the intended goal.

# 12. Task Types in Reinforcement Learning
## Episodic Tasks
- Interaction terminates after a finite number of steps (an episode).
- **Example:** Playing a board game. Once the game ends, a new episode can begin.

## Continuing Tasks
- Interaction goes on indefinitely; no natural termination.
- **Example:** Controlling a power grid, or running an ongoing service. There is no final time step.

# 13. Return
The return $R_t$ is the sum of future rewards from time $t$ onward:

### Episodic:
$$R_t = r_{t+1} + r_{t+2} + \dots + r_T$$
where $T$ is the final time step in the episode.

### Continuing:
$$R_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$$
Here, $0 \leq \gamma < 1$ is the discount factor that downweights future rewards.

### Discount Factor $\gamma$
- $\gamma = 0$: Myopic agent; cares only about immediate reward.
- $\gamma \to 1$: Far-sighted agent; places nearly as much value on future rewards as on immediate ones.
- For continuing tasks, $\gamma$ must be strictly less than 1 to ensure the infinite sum converges.

# 14. The Markov Property
A state $s$ is Markov if it contains all relevant information about the environment’s dynamics going forward. Formally:

$$p(s_{t+1} = s', r_{t+1} = r | s_0, a_0, r_1, \dots, s_{t-1}, a_{t-1}, r_t, s_t, a_t) = p(s_{t+1} = s', r_{t+1} = r | s_t = s, a_t = a).$$

This means that once you know the current state, no additional history is needed to predict the next state and reward.

### Why Markov Matters
If the environment is Markov, then a policy $\pi$ can be defined simply as $\pi(a|s)$, without needing the entire history.

This allows RL algorithms to treat each state as a complete summary of the past, simplifying learning and decision-making.

# 15. Markov Decision Process (MDP)
An MDP is a formal framework that encapsulates:

- States ($S$)
- Actions ($A$)
- Transition probabilities $p(s' | s, a)$
- Reward function (often described by $r(s, a, s')$ or its expectation)

The agent interacts with the environment according to:

$$p(s_{t+1}, r_{t+1} | s_t, a_t).$$

### Key quantities:
- **Transition Probability**
  $$p(s' | s, a) = \sum_{r \in R} p(s', r | s, a).$$
- **Expected Reward**
  $$r(s, a, s') = \frac{\sum_{r \in R} r p(s', r | s, a)}{p(s' | s, a)} = \sum_{r \in R} r p(r | s, a, s').$$

# 16. Value Functions
**Motivation:** Immediate rewards alone are insufficient to gauge how good it is to be in a particular state, because some states can yield larger future rewards than others.

### State-Value Function $V_\pi(s)$
$$V_\pi(s) = \mathbb{E}_\pi[R_t | s_t = s].$$
This is the expected return (sum of discounted future rewards) when starting in state $s$ and following policy $\pi$ thereafter.

### Action-Value Function $Q_\pi(s, a)$
$$Q_\pi(s, a) = \mathbb{E}_\pi[R_t | s_t = s, a_t = a].$$
This is the expected return when starting in state $s$, taking action $a$, and then continuing with policy $\pi$.

# 17. Bellman Equation
The value function must satisfy a powerful recursive property. For any policy $\pi$:

$$V_\pi(s) = \mathbb{E}_\pi[R_t | s_t = s] = \mathbb{E}_\pi[r_{t+1} + \gamma R_{t+1} | s_t = s] = \sum_{a \in A} \pi(a | s) \sum_{s' \in S} \sum_{r \in R} p(s', r | s, a) [r + \gamma V_\pi(s')].$$

### Interpretation:
- From state $s$, the agent picks an action $a$ according to $\pi$.
- It then lands in state $s'$ with some probability, collecting reward $r$.
- The value of the original state is the expected immediate reward plus the discounted value of the next state.

# 18. Optimal Value Functions
We say a policy $\pi$ is better than $\pi'$ if, for all states, $V_\pi(s) \geq V_{\pi'}(s)$. There exists at least one optimal policy $\pi^*$ such that:

$$V_{\pi^*}(s) = V^*(s) = \max_\pi V_\pi(s),$$
and similarly for action values:

$$Q_{\pi^*}(s, a) = Q^*(s, a) = \max_\pi Q_\pi(s, a).$$

# 19. Bellman Optimality Equations
### State-Value Form
$$V^*(s) = \max_a \mathbb{E}[r_{t+1} + \gamma V^*(s_{t+1}) | s_t = s, a_t = a].$$
Expanding the expectation:

$$V^*(s) = \max_{a \in A} \sum_{s'} \sum_r p(s', r | s, a) [r + \gamma V^*(s')].$$

### Action-Value Form
$$Q^*(s, a) = \sum_{s'} \sum_r p(s', r | s, a) [r + \gamma \max_{a'} Q^*(s', a')].$$

### Interpretation:
- The optimal value of a state (or action) is the best immediate reward plus discounted future returns you can achieve, assuming you choose actions optimally thereafter.
- These equations are non-linear but can be solved in the finite-state case by iterative methods (e.g., dynamic programming).
- Once $V^*(s)$ (or $Q^*(s, a)$) is known, we can retrieve an optimal policy by always selecting the action:

$$\pi^*(s) = \argmax_{a} Q^*(s, a).$$

# 20. Optimality and Approximation
### Tabular Case:
- If the state space is small and discrete, we can store value functions in tables and solve the Bellman equations exactly (e.g., via dynamic programming).

### Non-Tabular Case:
- For large or continuous state spaces, we must approximate value functions (and policies) using function approximators (e.g., linear models or neural networks).

### Online Focus:
- RL operates often in an online setting, continuously updating estimates of the value function/policy based on new experiences.
- This naturally allocates more computational effort to frequently visited states, allowing practical solutions to very large problems.

# 21. Summary of Lecture
- **RL vs. Supervised/Unsupervised:** RL uniquely focuses on learning from interaction, aiming to maximize cumulative reward.
- **Main Elements:** Policy, Reward, Value Function, Model.
- **State and Markov Property:** A proper state representation encapsulates all relevant info for predicting the next state and reward.
- **Value Functions:** Estimate expected returns from a state or state–action pair. They satisfy the Bellman equation, expressing values recursively.
- **Optimality:** The Bellman optimality equations define how to compute the maximum possible value for each state/action in an MDP.
