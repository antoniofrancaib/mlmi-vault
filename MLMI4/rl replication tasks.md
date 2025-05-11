# 1. MuJoCo Continuous Control Tasks
These experiments follow the protocol of Finn et al. (2017a) and involve two variations of the Half-Cheetah task.

## 1.1. Cheetah Direction Task

### Task Description:
The agent (a Cheetah robot) must run in a specified direction—either forward or backward—where the desired direction is randomly chosen for each task.

### Reward Function:
The instantaneous reward is the agent’s speed in the desired direction. If $v_x$ is the forward velocity:

$$
r_t =
\begin{cases} 
v_x, & \text{if the desired direction is forward} \\
- v_x, & \text{if the desired direction is backward}
\end{cases}
$$

### Adaptation Equations:

#### Inner Loop (Context Update):
Given an initial context $\phi_0$ (typically $\phi_0 = 0$), for task $T_i$ the context parameters are updated as:

$$
\phi_i = \phi_0 + \alpha \nabla_{\phi} \tilde{J}_{T_i}(\tau_i^{\text{train}}, \pi_{\phi_0}, \theta)
$$

where $\tilde{J}_{T_i}$ is the policy gradient–based objective computed on rollouts $\tau_i^{\text{train}}$.

#### Outer Loop (Meta-Update):
After collecting test rollouts $\tau_i^{\text{test}}$ with the updated policy $\pi_{\phi_i, \theta}$, the shared parameters are updated via:

$$
\theta \leftarrow \theta + \beta \nabla_{\theta} \frac{1}{N} \sum_{i=1}^{N} \tilde{J}_{T_i}(\tau_i^{\text{test}}, \pi_{\phi_i}, \theta)
$$

where $N$ is the number of tasks in the meta-batch.

### Experimental Settings:

- **Rollout Length:** 200 timesteps per trajectory.
- **Meta-Batch Size:** 40 tasks per outer update.
- **Context Parameters:** 50 parameters are adapted for each task.
- **Inner-Loop Learning Rate:** $\alpha = 10$ (set high to strengthen the policy update signal in RL).

### Visual Representation:
Figure 5 in the paper shows the performance curves for the Cheetah Direction task:

- **Subfigure (a):** Plots the average return (over 40 test tasks) against the number of gradient updates. It highlights that after one gradient update, CAVIA outperforms MAML, and both methods continue to improve with further updates.

## 1.2. Cheetah Velocity Task

### Task Description:
The agent is required to run at a target velocity. For each task $T_i$, the target velocity $v_{\text{target}, i}$ is sampled uniformly from the interval:

$$
v_{\text{target}, i} \sim U(0.0, 2.0)
$$

### Reward Function:
The instantaneous reward penalizes the deviation from the target velocity:

$$
r_t = -|v_t - v_{\text{target}, i}|
$$

### Adaptation Equations:
The inner and outer loop updates follow the same structure as in the Cheetah Direction task:

#### Inner Loop:
$$
\phi_i = \phi_0 + \alpha \nabla_{\phi} \tilde{J}_{T_i}(\tau_i^{\text{train}}, \pi_{\phi_0}, \theta)
$$

#### Outer Loop:
$$
\theta \leftarrow \theta + \beta \nabla_{\theta} \frac{1}{N} \sum_{i=1}^{N} \tilde{J}_{T_i}(\tau_i^{\text{test}}, \pi_{\phi_i}, \theta)
$$

### Observations:

- CAVIA initially outperforms MAML after one gradient update.
- In the Cheetah Velocity task, MAML eventually catches up after three gradient updates.

### Visual Representation:
Figure 5 also contains a subfigure for the Cheetah Velocity experiment:

- **Subfigure (b):** Shows the performance comparison between CAVIA and MAML (again averaged over 40 test tasks) over up to three gradient updates.

# 2. 2D Navigation Task
This additional RL experiment is designed to study both performance and the interpretability of the learned task embeddings in a simpler environment.

### Task Description:
An agent navigates in a 2D plane towards an unknown goal position $g_i = (g_{x,i}, g_{y,i})$ where:

$$
g_{x,i}, g_{y,i} \sim U(-0.5, 0.5)
$$

### Reward Function:
At each timestep, the agent receives a reward based on its Euclidean distance to the goal:

$$
r_t = -\|s_t - g_i\|_2
$$

where $s_t$ is the agent’s current position.

### Network Architecture & Training Details:

- **Policy Network:** A two-layer neural network (each layer with 100 units and ReLU activations) along with a linear value function approximator.
- **Context Parameters:** For this task, 5 context parameters are appended to the input.
- **Sampling:**
  - 20 tasks are sampled for both the inner and outer loops.
  - Testing is performed on 40 unseen tasks.
- **Training Regime:**
  - Training is conducted for 500 iterations with one inner loop gradient update per task.
  - At test time, the best-performing policy is adapted with two gradient updates.
- **Learning Rate Sensitivity:**
  - A sensitivity analysis is also performed to evaluate the impact of different inner-loop learning rates.

### Visual Representations:
Figure 6 in the supplementary material contains three panels:

- **Figure 6(a):** Compares the overall performance (average cumulative reward) of CAVIA and MAML on the 2D Navigation task.
- **Figure 6(b):** Plots performance for various inner-loop learning rates after two gradient updates at test time.
- **Figure 6(c):** Visualizes the learned context parameter embeddings (with two context parameters for interpretability), showing that one parameter encodes the goal’s $y$-coordinate and the other encodes the $x$-coordinate.

# 3. Algorithmic Pseudocode for RL
Although not a result per se, the supplementary material includes the pseudocode that outlines the exact training procedure for the RL experiments.

## Algorithm 2: CAVIA for RL
This pseudocode details:

1. Sampling a batch of tasks $T = \{T_i\}_{i=1}^{N}$.
2. For each task:
   - Collecting a training rollout $\tau_i^{\text{train}}$ using the initial policy $\pi_{\phi_0}, \theta$.
   - Performing an inner loop update on the context parameters:

     $$
     \phi_i = \phi_0 + \alpha \nabla_{\phi} \tilde{J}_{T_i}(\tau_i^{\text{train}}, \pi_{\phi_0}, \theta)
     $$

   - Collecting a test rollout $\tau_i^{\text{test}}$ using the adapted policy $\pi_{\phi_i}, \theta$.
   - Applying the outer loop update to $\theta$ based on the aggregated performance across tasks.

# Summary of Replicable Figures for RL Experiments

## Figure 5:
Contains performance curves for both MuJoCo tasks:
- **(a)** Cheetah Direction Task: Shows average return over up to three gradient updates.
- **(b)** Cheetah Velocity Task: Illustrates performance improvement and the eventual catch-up of MAML after three updates.

## Figure 6:
Focuses on the 2D Navigation task:
- **(a):** Overall performance comparison between CAVIA and MAML.
- **(b):** Sensitivity analysis for different inner-loop learning rates.
- **(c):** Visualization of learned context parameter embeddings (demonstrating a disentangled representation of the goal’s $x$ and $y$ coordinates).

## Algorithm 2 (Supplementary Material):
Provides the detailed pseudocode for executing the RL experiments with CAVIA.
