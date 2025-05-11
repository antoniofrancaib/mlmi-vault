Lecture by Florian Fischer

# 1. Introduction to Deep Reinforcement Learning

**Definition:**
Deep reinforcement learning is a branch of RL where one (or more) of the following is approximated using a neural network:

- The value function (estimating how good a state or state–action pair is)
- The policy (mapping states to actions)
- The model of the environment (estimating transitions and rewards)

**Advantages:**

- **Nonlinear function approximation:** Neural networks can represent complex, non-linear relationships.
- **Predefined architectures instead of manual feature design:** The architecture implicitly defines feature hierarchies, reducing the need for hand-crafted features.

**Challenges:**

- **Lack of interpretability:** The learned approximations are often “black-box” models.
- **Local optima:** Neural network training may settle in a locally optimal solution.
- **Data requirements:** Training deep models typically requires large amounts of data, which can be hard to obtain in an RL context.

# 2. Deep Representations and Neural Networks

## 2.1. Deep Representations

**Composition of Functions:**
A deep representation is built as a composition of multiple functions. Each layer transforms its input into a new representation. Because these transformations are differentiable, we can compute gradients through the entire composition using the chain rule. This is the backbone of training via backpropagation.

## 2.2. Deep Neural Networks

**Architecture Overview:**
A neural network transforms an input vector $\mathbf{x}$ into an output $\mathbf{y}$ through a sequence of layers. For layer $i$ (where $i=0,1,\dots,m$):

$$
\mathbf{h}_0 = g_0(\mathbf{W}_0 \mathbf{x}^T + \mathbf{b}_0)
$$

$$
\mathbf{h}_i = g_i(\mathbf{W}_i \mathbf{h}_{i-1}^T + \mathbf{b}_i) \quad \text{for } 0 < i < m
$$

$$
\mathbf{y} = g_m(\mathbf{W}_m \mathbf{h}_{m-1}^T + \mathbf{b}_m)
$$

Here, $g_i$ are differentiable activation functions (like tanh or sigmoid), and $\mathbf{W}_i, \mathbf{b}_i$ are parameters to be learned.

**Loss Functions:**

- **Regression:** Minimise the mean squared error (MSE):
  $$
  L = \|\mathbf{y}^* - \mathbf{y}\|^2
  $$

- **Classification:** Minimise the cross-entropy loss:
  $$
  L = -\sum_i \mathbf{y}_i^* \log \mathbf{y}_i
  $$

Stochastic gradient descent (SGD) and its variants are typically used for optimization.

## 2.3. Weight Sharing

**Recurrent Neural Networks (RNNs):**
Share weights between time steps. This allows the same transformation to be applied across sequential data, making RNNs powerful for temporal modeling.

![[weight-sharing-rnn.png]]
**Convolutional Neural Networks (CNNs):**
Share weights across local regions. This parameter sharing captures local spatial patterns, which is especially useful in image-related tasks (e.g., processing pixels in Atari games).
![[weight-sharing-cnn.png]]

# 3. Value-Based Deep Reinforcement Learning

## 3.1. Q-Networks

Q-networks approximate the action-value function $Q(\mathbf{s}, \mathbf{a})$ using neural networks. There are two common architectural choices:

- **Input: $(\mathbf{s}, \mathbf{a})$ Pair:**
  The network takes both the state $\mathbf{s}$ and action $\mathbf{a}$ as input and produces a scalar $Q(\mathbf{s}, \mathbf{a})$.

- **Input: State $\mathbf{s}$ Only:**
  The network outputs a vector of Q-values for all possible actions $[Q(\mathbf{s}, \mathbf{a}_1), \dots, Q(\mathbf{s}, \mathbf{a}_k)]$.

![[q-nets-architectures.png]]
## 3.2. Deep Q-Network (DQN)

**TD Error and Semi-Gradient Update:**
The DQN is trained to minimize the mean squared temporal-difference (TD) error:

$$
\text{TDE} = \left(r + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}'; \theta) - Q(\mathbf{s}, \mathbf{a}; \theta)\right)^2
$$

Here, $r$ is the reward, $\gamma$ is the discount factor, and $\theta$ represents the network parameters. Note that this update uses a semi-gradient because the target depends on $\theta$ only indirectly.

**Divergence Issues:**
The algorithm can diverge because:

- **Correlated states:** Successive states in an episode are highly correlated.
- **Non-stationary targets:** The target value $r + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}'; \theta)$ changes as the network parameters are updated.

# 4. Techniques to Stabilize DQN

## 4.1. Experience Replay

**Idea:**
Instead of learning from sequentially correlated data, store experiences $(\mathbf{s}, \mathbf{a}, r, \mathbf{s}')$ in a replay buffer. During training, randomly sample mini-batches from this buffer.

**Benefit:**
This randomisation breaks the correlation between samples, stabilizing the training process.

## 4.2. Target Networks

**Concept:**
Maintain a separate network with parameters $\theta^-$ that is used to compute the target:

$$
\text{TDE} = \left(r + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}'; \theta^-) - Q(\mathbf{s}, \mathbf{a}; \theta)\right)^2
$$

**Benefit:**
By updating $\theta^-$ only periodically, the targets become more stationary, reducing the risk of divergence.

## 4.3. Atari Case Study

**End-to-End Learning:**
In the seminal Atari DQN paper, the state $\mathbf{s}$ is represented as a stack of raw pixels from the last 4 frames, and the actions correspond to joystick/button positions. The network learns directly from pixels to predict Q-values.

**Reward Signal:**
The reward is given by the change in score at each step. This setup demonstrates the capability of deep RL to handle high-dimensional sensory inputs.

# 5. Advanced Improvements in Q-Networks

## 5.1. Prioritized Experience Replay

**Motivation:**
Not all experiences are equally informative. Instead of uniform random sampling, prioritize experiences based on their TD error:

$$
\delta = \left| r + \gamma \max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}'; \theta^-) - Q(\mathbf{s}, \mathbf{a}; \theta) \right|
$$

**Effect:**
Sampling experiences with higher TD error more frequently helps the network learn from mistakes and improve faster.

## 5.2. Double DQN

**Problem:**
Standard DQN tends to overestimate Q-values because of the max operator in the target.

**Solution:**
Double DQN decouples action selection from action evaluation by using two networks:

- The current network $\theta$ selects the action:
  $$
  \mathbf{a}^* = \arg\max_{\mathbf{a}'} Q(\mathbf{s}', \mathbf{a}'; \theta)
  $$

- The target network $\theta^-$ evaluates the selected action:
  $$
  \text{TDE} = \left(r + \gamma Q(\mathbf{s}', \mathbf{a}^*; \theta^-) - Q(\mathbf{s}, \mathbf{a}; \theta)\right)^2
  $$

**Variant:**
Clipped Double Q-learning further refines this approach to reduce overestimation even more.

## 5.3. Dueling Q-Networks

**Architecture:**
The network is split into two streams that combine to form the Q-value:

$$
Q(\mathbf{s}, \mathbf{a}) = V(\mathbf{s}) + A(\mathbf{s}, \mathbf{a})
$$

- **Value stream $V(\mathbf{s})$:** Estimates the overall value of the state.
- **Advantage stream $A(\mathbf{s}, \mathbf{a})$:** Estimates the relative benefit of each action in that state.

**Intuition:**
In many states, the choice of action has little effect (e.g., driving on an empty road). The value stream captures this “inherent” state quality, while the advantage stream focuses on situations where the choice of action matters (e.g., when obstacles appear).

**Additional Constraints:**
To ensure identifiability, constraints such as centering the advantage function (subtracting its mean) or enforcing non-positivity may be applied.

# 6. Regularization: Dropout Q-Functions

**Dropout Technique:**
During training, randomly “drop” units from the network (i.e., set them to zero). This prevents overfitting by ensuring that the network does not become overly reliant on any particular set of neurons.

**Multiple Q-Networks:**
Train several Q-networks with dropout. Averaging their predictions can lead to better generalisation.

**Particularly Useful When:**
The update-to-data ratio is high—meaning the model is updated many times using a limited set of experiences. Dropout (often combined with layer normalization) can help stabilize learning in these settings.

# 7. Asynchronous Deep Reinforcement Learning

## 7.1. Asynchronous Methods (e.g., A3C)

**Parallelism:**
Multiple agents (or threads) are run in parallel. Each thread interacts with its own copy of the environment while sharing network parameters.

**Benefits:**

- **Decorrelates Data:**
  Parallel agents produce a diverse set of experiences without the need for a replay buffer.
- **On-Policy Learning:**
  Since all data comes from the current policy, on-policy methods (like REINFORCE or actor–critic variants) can be effectively applied.

**Actor–Critic Methods:**
Asynchronous advantage actor–critic (A3C) extends the one-step actor–critic method using asynchronous updates and n-step returns to provide more robust value estimates.

## 7.2. Policy Approximation

**Policy-Gradient Methods:**
Neural networks can approximate the policy directly (e.g., in the REINFORCE algorithm). This allows for continuous action spaces or when the policy cannot be easily derived from a value function.

# 8. The Deadly Triad of Reinforcement Learning

**Components:**

- **Bootstrapping:**
  Updating estimates based partly on other learned estimates (as in TD learning).
- **Off-Policy Training:**
  Learning about one policy while following another.
- **Function Approximation:**
  Using approximators (e.g., neural networks) to estimate value functions.

**Problem:**
The combination of these three elements can lead to instability and divergence (often referred to as “soft-divergence”) because errors can compound.

**Mitigation:**
Techniques like target networks, Double DQN, dueling architectures, and careful design of the learning process all aim to reduce the instability introduced by the deadly triad.

# 9. Model-Based Deep Reinforcement Learning

## 9.1. The Dyna-Q Framework

**Idea:**
Instead of solely learning a policy or value function, the agent also learns a model of the environment—i.e., transition probabilities and rewards—using a neural network.

**Challenge:**

- **Compounding Errors:**
  Even small errors in the model can accumulate over long planning horizons. When planning, trajectories that diverge from those actually executed can lead to completely wrong reward predictions.

**Recent Advances:**
Newer methods use ensemble techniques to better capture epistemic uncertainty (i.e., uncertainty due to limited data), helping to mitigate compounding errors during planning.

# 10. Summary and Key Takeaways

- **Neural Networks in RL:**
  They can be used to approximate value functions, policies, or models. However, naïve applications may lead to convergence issues because of biased estimates and correlated data.

- **Stabilisation Techniques:**
  Methods such as experience replay, target networks, Double DQN, dueling architectures, and dropout Q-functions are critical to improving stability.

- **The Deadly Triad:**
  The combination of bootstrapping, off-policy updates, and function approximation is the primary source of instability in deep RL. Mitigation strategies are essential for robust learning.

- **Model-Based Challenges:**
  While model-based methods promise more efficient planning, the difficulty lies in accurately approximating the transition dynamics without error propagation.

- **Parallel and Asynchronous Learning:**
  These approaches provide alternatives to experience replay by decorrelating data through parallelism and enabling on-policy updates.