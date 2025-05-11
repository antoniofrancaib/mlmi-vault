

# 1. Monte Carlo Off-Policy Methods

Monte Carlo (MC) methods learn value functions and policies directly from experience by averaging complete returns from episodes. In off-policy settings, two separate policies are involved:

- **Target Policy ($\pi$):**
  This is the policy we aim to improve. In many cases, it is defined as the greedy policy with respect to the current action-value function $\mathbf{Q}$. That is, for any state $\mathbf{s}$,

  $$
  \pi(s) = \arg\max_a \mathbf{Q}(s, a)
  $$

  The target policy is “what we want to do” eventually.

- **Behaviour Policy ($\mu$):**
  This is the policy that actually generates the data (episodes). For off-policy methods to converge to the optimal policy, the behaviour policy must have “coverage” of the target policy. In practice, this means:

  - The behaviour policy is soft: it must select every action with a non-zero probability.
  - It must visit every state-action pair infinitely often so that each $\mathbf{Q}(s, a)$ is updated sufficiently over time.

## Off-Policy Monte Carlo Control Algorithm

The algorithm (often referred to as Algorithm 5 in the provided notes) operates as follows:

### Initialization:

- $\mathbf{Q}(s, a)$ is initialized arbitrarily for all state–action pairs.
- A counter $\mathbf{C}(s, a)$ is set to zero. This counter accumulates importance-sampling weights.
- The target policy $\pi$ is set to be greedy with respect to $\mathbf{Q}$.

### Episode Generation:

- An episode is generated using the soft behaviour policy $\mu$, ensuring that all actions (that could be chosen by $\pi$) are sampled with nonzero probability.

### Backward Update:

- The return $\mathbf{R}$ is accumulated backwards from the terminal state.
- For each time step $t$ (from the end of the episode to the beginning), the following updates occur:
  - The cumulative return is updated:
    $$
    \mathbf{R} \leftarrow \gamma \mathbf{R} + r_{t+1}
    $$
  - The cumulative weight $\mathbf{C}(s_t, a_t)$ is incremented by $\mathbf{W}$, which is the product of the importance-sampling ratios from time $t$ onward.
  - The action-value function is updated using the weighted difference between the observed return and the current $\mathbf{Q}(s_t, a_t)$:
    $$
    \mathbf{Q}(s_t, a_t) \leftarrow \mathbf{Q}(s_t, a_t) + \frac{\mathbf{W}}{\mathbf{C}(s_t, a_t)} (\mathbf{R} - \mathbf{Q}(s_t, a_t))
    $$
  - The target policy is updated to remain greedy with respect to $\mathbf{Q}$.
  - If the action taken does not agree with the target policy, the loop breaks. Otherwise, $\mathbf{W}$ is updated by multiplying with the inverse probability $\frac{1}{\mu(a_t|s_t)}$.

This procedure ensures that even though the data are generated off-policy (using $\mu$), the updates steer $\mathbf{Q}$ toward the values under the greedy target policy.

# 2. Summary of Monte Carlo Methods

- **Model-Free Learning:**
  Monte Carlo methods learn directly from experience without needing a model of the environment’s dynamics.

- **Episode-Based Updates:**
  They require complete episodes to compute the full return and average these returns over many episodes.

- **On-Policy vs. Off-Policy:**
  - On-policy methods use the same policy for both generating behavior and learning.
  - Off-policy methods allow a separation between the behavior policy (which can be exploratory) and the target policy (which is typically greedy).

# 3. Temporal-Difference (TD) Learning

TD methods blend the advantages of Monte Carlo methods and dynamic programming. They update estimates based partly on other learned estimates—a process known as bootstrapping.

## 3.1. TD Prediction

TD prediction uses the immediate reward and the current estimate of the next state’s value to update the value estimate. The standard update rule is:

$$
\mathbf{V}(s_t) \leftarrow \mathbf{V}(s_t) + \alpha (r_{t+1} + \gamma \mathbf{V}(s_{t+1}) - \mathbf{V}(s_t))
$$

where:

- $\alpha$ is the step-size (learning rate),
- $\gamma$ is the discount factor, and
- $\delta_t = r_{t+1} + \gamma \mathbf{V}(s_{t+1}) - \mathbf{V}(s_t)$ is the TD error.

## 3.2. TD Error and Backups

- **TD Error ($\delta_t$):**
  This quantifies the discrepancy between the predicted value and the better-informed estimate incorporating the reward and the next state’s value. It can be viewed as the "surprise" or "error" in the current prediction.

- **Backup:**
  Updating $\mathbf{V}(s_t)$ using $\delta_t$ is known as making a backup. Unlike Monte Carlo methods that wait until the episode ends, TD updates happen immediately at the next time step.

# 4. SARSA: On-Policy TD Control

SARSA is an on-policy TD control algorithm that updates the action-value function $\mathbf{Q}(s, a)$ using the quintuple:

$$
(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})
$$

Its update rule is:

$$
\mathbf{Q}(s_t, a_t) \leftarrow \mathbf{Q}(s_t, a_t) + \alpha (r_{t+1} + \gamma \mathbf{Q}(s_{t+1}, a_{t+1}) - \mathbf{Q}(s_t, a_t))
$$

### Key Points:

- **On-Policy Nature:**
  The policy used to generate actions (usually an $\epsilon$-greedy strategy) is the same one that is improved over time.

- **Balancing Exploration and Exploitation:**
  The $\epsilon$-greedy mechanism ensures sufficient exploration while gradually converging toward the greedy (optimal) policy.

- **Convergence Conditions:**
  SARSA converges to an optimal policy provided that every state–action pair is visited infinitely often and that the policy eventually becomes greedy (for example, by decaying $\epsilon$ appropriately).

# 5. Q-learning: Off-Policy TD Control

Q-learning is an off-policy algorithm where the update rule uses the maximum estimated value of the next state, regardless of the action actually taken:

$$
\mathbf{Q}(s_t, a_t) \leftarrow \mathbf{Q}(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a'} \mathbf{Q}(s_{t+1}, a') - \mathbf{Q}(s_t, a_t))
$$

### Key Insights:

- **Off-Policy Advantage:**
  Q-learning directly approximates the optimal action-value function, independent of the policy being followed. This decouples the exploration policy from the target policy.

- **Simplified Convergence Analysis:**
  Since the algorithm uses the maximization step over actions, the analysis is often more straightforward—the only requirement is that every state–action pair continues to be updated.

- **Comparison with SARSA:**
  In environments with high risk (e.g., cliff-walking), SARSA tends to be more conservative as it incorporates the actual behavior, while Q-learning, by always taking the max, may overestimate the value in risky regions.

# 6. Expected SARSA

Expected SARSA provides an alternative update by replacing the sample of the next action with the expectation under the current policy:

$$
\mathbf{Q}(s_t, a_t) \leftarrow \mathbf{Q}(s_t, a_t) + \alpha (r_{t+1} + \gamma \sum_{a'} \pi(a'|s_{t+1}) \mathbf{Q}(s_{t+1}, a') - \mathbf{Q}(s_t, a_t))
$$

### Advantages:

- **Reduced Variance:**
  By taking an expectation rather than a single sample, Expected SARSA can reduce the variance inherent in the update, often leading to more stable learning.

- **Flexibility:**
  This method can be used in both on-policy and off-policy settings. However, it may involve a higher computational cost because it requires summing over all possible actions.

# 7. Planning and Learning with Tabular Methods

Planning and learning both aim to determine optimal policies by estimating value functions, but they differ in the source of their “experience.”

## 7.1. Model-Free vs. Model-Based Methods

- **Model-Free:**
  Methods such as Monte Carlo and TD learning directly learn from real (or simulated) experience without an explicit model.

- **Model-Based:**
  Planning methods rely on a model of the environment that can simulate transitions. The model might be learned from experience (as in Dyna) or given a priori.

## 7.2. The Role of Backups

Regardless of whether real or simulated experience is used, the core idea is to improve the value function through backup operations. Both planning and learning apply similar TD updates—the difference lies in whether the “experience” comes from interaction with the actual environment or from a simulated model.

# 8. Dyna Architecture

The Dyna framework integrates planning, acting, and learning into a single process, combining the strengths of direct reinforcement learning with model-based planning.

## 8.1. Dyna-Q

Dyna-Q extends Q-learning by simultaneously updating:

- **Direct RL:**
  Updating $\mathbf{Q}(s, a)$ from real experience.
- **Model Learning:**
  Learning a model (typically a table) that predicts the next state and reward for a given state–action pair.
- **Planning:**
  Using the learned model to simulate additional experiences. These simulated transitions are then used to perform further Q-learning style updates.

## 8.2. Advantages of the Dyna Architecture

- **Efficiency:**
  By planning with simulated experience, the agent can improve its policy with fewer actual interactions.
- **Responsiveness:**
  The agent continuously incorporates the latest observations while also planning in the background.
- **Unified Framework:**
  The same update mechanism (TD backup) is used for both real and simulated experience, simplifying the algorithm design.

# 9. Prioritized Sweeping

Not all state–action backups are equally valuable. Prioritized Sweeping is a method designed to focus computational resources on the updates that are expected to be most useful.

## 9.1. Motivation

- **Non-uniform Importance:**
  In many cases, the value of certain state–action pairs changes significantly while others remain nearly constant. It is inefficient to update all pairs uniformly.

- **TD Error as a Priority Metric:**
  The absolute TD error,

  $$
  P = |r + \gamma \max_{a'} \mathbf{Q}(s', a') - \mathbf{Q}(s, a)|
  $$

  serves as a measure of how “urgent” an update is. A larger TD error indicates that the current estimate is far from the new information provided by the transition.

## 9.2. The Algorithm

- **Maintain a Priority Queue:**
  Each state–action pair is inserted into a priority queue with a priority based on its TD error.

- **Backup Ordering:**
  At each iteration, the algorithm extracts the state–action pair with the highest priority and performs a backup.

- **Backward Propagation:**
  When a backup is performed, all predecessors (state–action pairs that might lead to the updated state) are also reconsidered, with their priorities updated based on the new TD errors.

This method focuses the updates where they are most needed, often resulting in a more rapid convergence to the optimal solution.

# 10. Final Summary

- **Monte Carlo Methods:**
  - Learn value functions from complete episodes.
  - Off-policy methods require careful handling of target vs. behaviour policies and importance sampling.

- **Temporal-Difference Learning:**
  - Combines sampling and bootstrapping.
  - Updates occur every time step using the TD error.

- **SARSA and Q-learning:**
  - SARSA (on-policy) updates using the action actually taken, ensuring that the policy used to generate behavior is directly updated.
  - Q-learning (off-policy) updates using the maximum value of the next state, decoupling the target from the behaviour policy.
  - Expected SARSA averages over all possible next actions to reduce variance.

- **Planning and Learning:**
  - Both approaches aim to estimate value functions via backups.
  - Model-based methods simulate experiences to supplement real interactions.

- **Dyna Architecture:**
  - Integrates model learning, planning, and RL in one framework.
  - Uses simulated experiences to boost learning efficiency.

- **Prioritized Sweeping:**
  - Orders updates based on the urgency (TD error) of state–action pairs.
  - Focuses computational effort on the most impactful updates for faster convergence.

These advanced topics form the backbone of many modern reinforcement learning algorithms. A deep understanding of these concepts provides the intuition needed to tackle more complex environments and further innovations such as function approximation and deep reinforcement learning.