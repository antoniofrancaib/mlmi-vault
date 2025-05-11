**Function Approximation in RL**

# 1. Introduction and Motivation

## Why Function Approximation?
In many real-world RL problems, the state space is either very large or continuous. Representing the value function as a table (one value per state) is infeasible because:
- The number of states, $|\mathbf{S}|$, is enormous.
- Many states will be visited rarely, and many transitions cannot be enumerated.

Instead, we approximate the value function using a parameterized function, where a weight vector $\mathbf{\theta} \in \mathbb{R}^n$ (with $n \ll |\mathbf{S}|$) defines the function. Importantly, changing one component of $\mathbf{\theta}$ affects the value estimates of many states simultaneously. This "generalization" is key to making learning scalable and efficient.

---

# 2. Representing the Value Function

## Parameterization Overview
- **Traditional approach**: A lookup table for every state.
- **Approximation approach**: Represent the value function $V^\pi(s)$ approximately as $\hat{V}^\pi(s, \mathbf{\theta})$.
- **Key idea**: Use a compact parameter vector $\mathbf{\theta}$ such that the number of parameters is far smaller than the number of states.
- **Generalization**: A single weight change influences many states—this is beneficial when data is sparse.

---

# 3. Back-ups as Input-Output Pairs

## The Concept of a Backup
- In RL, learning typically involves "backing up" information from subsequent states or rewards.
- Different types of backups:
  - **Monte Carlo (MC) backup**: $s_t \to R_t$, where $R_t$ is the complete return from time $t$.
  - **Temporal-Difference (TD) backup**: $s_t \to r_{t+1} + \gamma \hat{V}(s_{t+1}, \mathbf{\theta})$.
  - **Dynamic Programming (DP) backup**: $s \to \mathbb{E}_\pi\left[r + \gamma \hat{V}(s{\prime}, \mathbf{\theta}) \mid s\right]$.

Each backup provides a training example—a pair consisting of an input (state) and a target output (return or expected return). This perspective allows us to view value estimation as a supervised learning problem where the goal is to match these input-output pairs.

---

# 4. Prediction Objective

## Minimizing Prediction Error
Since it is impossible to perfectly predict the value in every state, we weight the errors by a distribution $d(s)$, which reflects how much we care about each state. Typically, $d(s)$ is chosen as the occupancy frequency under the target policy $\pi$.

**Mean Squared Value Error (MSVE):**

$$
\text{MSVE}(\mathbf{\theta}) = \sum_{s} d(s) \left( V^\pi(s) - \hat{V}(s, \mathbf{\theta}) \right)^2
$$

- **Intuition**: States that are visited more frequently (or that are deemed more important) are given more weight.
- **Design choice**: The distribution $d(s)$ often comes from the fraction of time the policy $\pi$ spends in each state.

---

# 5. On-Policy Distribution

## How $d(s)$ is Determined
- Let $h(s)$ denote the probability that an episode starts in state $s$.
- Let $e(s)$ denote the average number of time steps spent in $s$ during an episode.

The recurrence relation is:

$$
e(s) = h(s) + \sum_{\hat{s}} e(\hat{s}) \sum_{\hat{a}} \pi(\hat{a} \mid \hat{s}) \, p(s \mid \hat{s}, \hat{a})
$$

Solving this system gives us $e(s)$ for all states, and the state distribution is defined as:

$$
d(s) = \frac{e(s)}{\sum_{s{\prime}} e(s{\prime})}
$$

- **Intuition**: $d(s)$ reflects the expected number of visits to $s$ over an episode, normalized to form a probability distribution.

---

# 6. Stochastic Gradient Descent (SGD)

## Using SGD for Function Approximation
Since $\hat{V}(s, \mathbf{\theta})$ is differentiable with respect to $\mathbf{\theta}$, we can use SGD to minimize the MSVE. For a single sample $s_t$ with true value $V^\pi(s_t)$, the weight update is:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha \left( V^\pi(s_t) - \hat{V}(s_t, \mathbf{\theta}_t) \right) \nabla \hat{V}(s_t, \mathbf{\theta}_t)
$$

- **Key points**:
  - $\alpha$ is the step-size parameter.
  - The update is "stochastic" because it is based on a single sample drawn according to $d(s)$.

This method adjusts $\mathbf{\theta}$ in the direction that reduces the squared error for the observed sample.

---

# 7. Target Output: Noisy Estimates

## Real-World Considerations
In practice, the true value $V^\pi(s_t)$ is unavailable. Instead, we use a noisy estimate $U_t$ (which might be derived from MC, TD, or DP backups). The general SGD update becomes:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha \left( U_t - \hat{V}(s_t, \mathbf{\theta}_t) \right) \nabla \hat{V}(s_t, \mathbf{\theta}_t)
$$

- **Requirement**: $U_t$ must be an unbiased estimator of $V^\pi(s_t)$ (i.e., $\mathbb{E}[U_t] = V^\pi(s_t)$) to ensure convergence.

---

# 8. Gradient Monte Carlo Algorithm

## Algorithm Overview
The Gradient Monte Carlo method uses complete episodes to update the value function:
1. **Input**: Policy $\pi$ and a differentiable approximator $\hat{V}(s, \mathbf{\theta})$.
2. **Initialize**: The parameter vector $\mathbf{\theta}$.
3. **Loop**:
   - Generate an episode $s_0, a_0, r_1, \ldots, r_T, s_T$ following $\pi$.
   - For each time step $t$ in the episode, update:
     $$
     \mathbf{\theta} \leftarrow \mathbf{\theta} + \alpha \left( R_t - \hat{V}(s_t, \mathbf{\theta}) \right) \nabla \hat{V}(s_t, \mathbf{\theta})
     $$
   - **Intuition**: Use the complete return $R_t$ as the target for the value at $s_t$.

---

# 9. Semi-Gradient Methods

## Bootstrapping and Bias
When using TD or DP methods, the target depends on the current estimate $\hat{V}(s_{t+1}, \mathbf{\theta}_t)$, making the update only a partial gradient of the true error. These updates are known as semi-gradient methods:
- They use only a part of the gradient (ignoring the dependency of the target on $\mathbf{\theta}$).
- Although this introduces bias, these methods work well in practice and are computationally efficient.

---

# 10. Linear Function Approximation

## Special Case: Linear Methods
A particularly important case is when the function approximation is linear:

$$
\hat{V}(s, \mathbf{\theta}) = \mathbf{\theta}^T \phi(s) = \sum_{i} \theta_i \phi_i(s)
$$

- **Feature vector $\phi(s)$**: Each component $\phi_i(s)$ represents a feature function mapping from state to a real number.
- **Key advantage**: With linearity, the gradient is simply:
  $$
  \nabla \hat{V}(s, \mathbf{\theta}) = \phi(s)
  $$
- **Unique optimum**: The linear setting often guarantees a single optimum in the parameter space.

---

# 11. Semi-Gradient TD Update with Linear Approximation

## Deriving the Update Rule
For a TD update using a linear approximator, the update becomes:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha \left( r_{t+1} + \gamma \mathbf{\theta}_t^T \phi(s_{t+1}) - \mathbf{\theta}_t^T \phi(s_t) \right) \phi(s_t)
$$

This can be rearranged as:

$$
\mathbf{\theta}_{t+1} = \mathbf{\theta}_t + \alpha \left( r_{t+1}\phi(s_t) - \phi(s_t)(\phi(s_t) - \gamma \phi(s_{t+1}))^T \mathbf{\theta}_t \right)
$$

**Steady-State Analysis:**
In expectation (assuming the process has reached a steady state), we can write:

$$
\mathbb{E}[\mathbf{\theta}_{t+1} \mid \mathbf{\theta}_t] = \mathbf{\theta}_t + \alpha (b - A\mathbf{\theta}_t)
$$

where
- $b = \mathbb{E}[r_{t+1} \phi(s_t)]$
- $A = \mathbb{E}\left[\phi(s_t)(\phi(s_t) - \gamma \phi(s_{t+1}))^T\right]$

At convergence, $b - A\mathbf{\theta} = 0$ so that the fixed point satisfies $\mathbf{\theta} = A^{-1}b$.

---

# 12. Least-Squares TD (LSTD)

## Concept and Advantages
Least-Squares TD (LSTD) directly computes the fixpoint $\mathbf{\theta} = A^{-1}b$ by estimating the matrices $A$ and $b$ from data rather than relying on incremental updates.
- **Data Efficiency**: LSTD typically uses available data more efficiently.
- **Asymptotic Convergence**: With appropriate decreasing step sizes, TD with linear function approximation converges to the TD fixpoint.

**Trade-Off:**
- **Computational Complexity**: The LSTD algorithm has complexity $O(n^2)$ compared to the $O(n)$ complexity per update for semi-gradient TD methods.
- **No Step-Size Tuning**: LSTD does not require the careful tuning of a step-size parameter, which can simplify learning in some contexts.
- **Adaptation Issues**: Because LSTD "remembers" past data (it never forgets), it can be less adaptive when the target policy changes.

---

# 13. LSTD Algorithm Outline

## Algorithm Steps:
1. **Inputs**: Policy $\pi$, feature mapping $\phi(s)$ (with $\phi(\text{terminal}) = 0$).
2. **Initialize**: $A_{d-1}^{-1}$ (an initial inverse matrix) and $\hat{b} = 0$.
3. **For each episode**:
   - Start with an initial state $s$ and obtain its feature $\phi(s)$.
   - **For each step**:
     - Select action $a \sim \pi(\cdot \mid s)$.
     - Observe reward $r$ and next state $s{\prime}$ (with feature $\phi(s{\prime})$).
     - Compute an intermediate vector $v = A_{d-1}^{-1} (\phi(s) - \gamma \phi(s{\prime}))$.
     - Update the inverse matrix $A_{d-1}^{-1}$ using a formula that adjusts for the new observation.
     - Update the vector $\hat{b}$ with $r\phi(s)$.
     - Compute the new weight vector $\mathbf{\theta} = A^{-1} \hat{b}$.
     - Move to the next state and repeat.
4. **Repeat until convergence**.

This procedure refines the estimates for $A$ and $b$ until the fixpoint is reached.

---

# 14. Properties and Trade-Offs of LSTD

## Key Characteristics:
- **Computational Complexity**: $O(n^2)$ per update, which may become burdensome for a very high number of features.
- **Data Efficiency**: LSTD is more data-efficient than incremental methods like semi-gradient TD.
- **Parameter Sensitivity**: LSTD does not require a step-size parameter; however, when integrating with policy improvement (such as $\epsilon$-greedy methods), setting $\epsilon$ becomes important:
  - Too small an $\epsilon$ can cause unstable behavior (inversion sequence variation).
  - Too large an $\epsilon$ can slow down learning.
- **Memory and Adaptation**: LSTD’s tendency to "never forget" past data means it might not adapt quickly if the target policy changes over time (a common scenario in generalized policy iteration).

---

# 15. Summary and Concluding Thoughts

## Key Takeaways:
- **Generalization is Essential**: For large or continuous state spaces, parameterized function approximation is crucial.
- **Framework**: The value function is represented as $\hat{V}(s, \mathbf{\theta})$ where $\mathbf{\theta}$ is updated via variations of stochastic gradient descent (SGD).
- **Semi-Gradient Methods**: Although they use a biased update (because of bootstrapping), they are widely used and effective, especially with linear approximators.
- **Linear Function Approximation**: Provides a clear, tractable case where the gradient simplifies and a unique optimum exists.
- **LSTD**: Offers a data-efficient alternative to iterative methods, directly solving for the fixpoint but at the cost of higher computational complexity and reduced adaptability.

Understanding these concepts not only lays the groundwork for effective value function approximation in RL but also provides insights into the trade-offs between computational efficiency, data efficiency, and learning stability. These advanced ideas form the foundation for more complex topics in RL and decision making.