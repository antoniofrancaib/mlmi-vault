# Part I: Foundations of Supervised Learning and Regression

## 1. Setting Up the Learning Problem

**1.1. Goal: Learning a mapping from inputs to outputs.**  
The fundamental objective in supervised learning is to approximate or learn an unknown underlying function $g:X \rightarrow Y$ that maps inputs from an input space $X$ to outputs in an output space $Y$. Typically, $X \subseteq \mathbb{R}^D$ and $Y \subseteq \mathbb{R}^K$ for some dimensions $D$ and $K$.

**1.2. Training Data Representation:**  
We are provided with a finite set of observations, the training dataset, denoted by $D$. This dataset consists of $n$ input-output pairs:

$$D = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)}) \mid i = 1, \ldots, n\}$$

Here, each $\mathbf{x}^{(i)} \in X$ is an input vector (often called a feature vector or predictor), and each $\mathbf{y}^{(i)} \in Y$ is the corresponding observed output (often called a label, target, or response). For simplicity in initial discussions, $Y$ might be $\mathbb{R}$ (regression) or a discrete set (classification), represented numerically. The notes sometimes use $y_i$ or $\mathbf{y}_{\text{obs}}$ for observed outputs, especially when they are vectors.

**1.3. The Parameterized Function:**  
We postulate a family of functions, parameterized by a vector $\theta \in \Theta \subseteq \mathbb{R}^P$, denoted as $f(\mathbf{x}; \theta)$. The goal is to select a *specific parameter vector $\theta^*$* from the parameter space $\Theta$ such that the function $f(\mathbf{x}; \theta^*)$ best approximates the true underlying mapping $g$, based on the evidence provided by the training data $D$. The choice of the **functional form of $f$** (the model architecture) is a critical design decision.

**1.4. General Deep Learning Goal:**  
In the context of deep learning, $f(\mathbf{x}; \theta)$ is typically a complex, highly non-linear function represented by a neural network. The core task remains learning this parameterized mapping from input vectors $\mathbf{x}$ to output vectors $\mathbf{y}$ (where $\mathbf{y} = f(\mathbf{x}; \theta)$ represents the prediction). The structure of $f$ is designed to be flexible enough to capture intricate relationships within the data.

## 2. Loss Functions: Measuring Prediction Error

**2.1. Purpose:**  
To quantify the discrepancy between the predicted output $\hat{\mathbf{y}} = f(\mathbf{x}^{(i)}; \theta)$ and the observed output $\mathbf{y}^{(i)}$ for a single data point, we introduce a loss function (or cost function) $L: Y \times Y \rightarrow \mathbb{R}_{\geq 0}$. $L(\mathbf{y}, \hat{\mathbf{y}})$ measures the "cost" or "error" incurred when predicting $\hat{\mathbf{y}}$ while the true value is $\mathbf{y}$.

**2.2. Simple Loss Example:**  
The notes mention $L(\mathbf{y}, \hat{\mathbf{y}}) = \mathbf{y} - \hat{\mathbf{y}}$. Mathematically, this represents the *signed error*. However, loss functions are typically non-negative and often symmetric or penalize larger errors more severely. This form is unusual as a standalone loss; it's more common as a residual used within other loss functions like squared error.

**2.3. Regression Loss (Least Squares):**  
A standard and mathematically convenient loss function for regression (where $Y = \mathbb{R}$) is the squared error or L2 loss:

$$L(y^{(i)}, \hat{y}^{(i)}) = (y^{(i)} - \hat{y}^{(i)})^2 = (y^{(i)} - f(\mathbf{x}^{(i)}; \theta))^2$$

Alternative notation used: $L(y, y_{\text{obs}}) = (y - y_{\text{obs}})^2$. This loss penalizes large errors quadratically and is differentiable everywhere.

**2.4. Total Loss over Dataset:**  
The overall performance of the function $f(\mathbf{x}; \theta)$ on the entire training dataset $D$ is typically measured by aggregating the individual losses. A common aggregation is the *sum (or average)* of the losses over all training examples:

$$L(\theta; D) = \sum_{i=1}^n L(y^{(i)}, f(\mathbf{x}^{(i)}; \theta))$$

This total loss (or empirical risk) $L(\theta; D)$ is a function of the parameters $\theta$.

**2.5. Classification Loss (Cross-Entropy):**  
For classification tasks with $K$ classes, where $y^{(i)}$ represents the true class label (e.g., an integer $l \in \{1, \ldots, K\}$), the model typically outputs a vector of scores or logits $\mathbf{y} \in \mathbb{R}^K$

**Softmax Normalization:** To convert these logits into a probability distribution over the $K$ classes, the softmax function is applied elementwise:

$$z_k = \text{softmax}(\mathbf{y})_k = \frac{\exp(y_k)}{\sum_{k'=1}^K \exp(y_{k'})}$$

The resulting vector $\mathbf{z} = (z_1, \ldots, z_K)$ satisfies $z_k \geq 0$ for all $k$ and $\sum_{k=1}^K z_k = 1$, thus forming a valid probability distribution. $z_k$ can be interpreted as the model's estimated probability that the input $\mathbf{x}$ belongs to class $k$.

**Cross-Entropy Loss:** The cross-entropy loss measures the dissimilarity between the predicted probability distribution $\mathbf{z}$ and the true distribution (which is typically represented as a one-hot vector where the true class $l$ has probability 1 and others 0). It is equivalent to the *negative log-likelihood of the true class $l$ under the predicted distribution*:

$$L(\mathbf{y}, l) = -\log(z_l)$$

Substituting the softmax formula gives:

$$L(\mathbf{y}, l) = -\log\left(\frac{\exp(y_l)}{\sum_{k'=1}^K \exp(y_{k'})}\right) = -\left(y_l - \log\sum_{k'=1}^K \exp(y_{k'})\right)$$

This is the standard loss function for multi-class classification trained with maximum likelihood.

**2.6. Overall Objective:**  
The central goal of the training process is to find the parameter vector $\theta^*$ that minimizes the total loss function:

$$\theta^* = \arg\min_{\theta \in \Theta} L(\theta; D)$$

This optimization problem is typically solved using iterative methods based on gradients.

# Part II: Constructing the Model Function $f(\mathbf{x}; \theta)$

## 3. Simple Function Families

Before deep networks, simpler parametric families were common. These serve as *conceptual building blocks*. Assume $x$ is a scalar for simplicity here, though extension to vectors is direct.

**3.1. Linear Function:** Represents a hyperplane (or a line in 1D).

$$f(x; \theta) = \theta_0 + \theta_1 x$$

Parameters $\theta = (\theta_0, \theta_1)$ represent the intercept and slope.

**3.2. Quadratic Function:** Allows capturing simple curvature.

$$f(x; \theta) = \theta_0 + \theta_1 x + \theta_2 x^2$$

Parameters $\theta = (\theta_0, \theta_1, \theta_2)$.

**3.3. Nonlinear Components:** Incorporating known nonlinearities can be beneficial if the data exhibits such patterns.

$$f(x; \theta) = \theta_0 + \theta_1 x + \theta_2 \sin x$$

Parameters $\theta = (\theta_0, \theta_1, \theta_2)$. Useful for periodic data.

## 4. Basis Functions Approach

A more systematic way to construct flexible functions is to use a linear combination of pre-defined (or learned) basis functions $\phi_k: X \rightarrow \mathbb{R}$.

**4.1. General Form:**

$$f(x; \theta) = \sum_{k=1}^M \theta_k \phi_k(x)$$

Here, the parameters $\theta = (\theta_1, \ldots, \theta_M)$ are the coefficients of the linear combination. The complexity and flexibility of $f$ depend on the choice and number $M$ of basis functions $\phi_k$. Note that a bias term $\theta_0$ can be included by setting $\phi_0(x) = 1$.

**4.2. Common Basis Function Choices ($\phi_k(x)$):**

- **Polynomials:** $\phi_k(x) = x^k$ (for scalar $x$). For vector $\mathbf{x}$, monomials of varying degrees can be used (e.g., $x_1^2, x_1 x_2, x_2^3$). Leads to polynomial regression.
- **Sinusoids:** $\phi_k(x) = \sin(kx)$ or $\cos(kx)$ (or frequencies $\omega_k$). Useful for periodic signals (Fourier series).
- **Exponentials (Gaussian Basis Functions):** Also known as Radial Basis Functions (RBFs).

$$\phi_k(x) = \exp\left(-\frac{\|\mathbf{x} - \mu_k\|^2}{2\sigma_k^2}\right)$$

These are localized functions centered at $\mu_k$ with a width controlled by $\sigma_k$. The centers $\mu_k$ and widths $\sigma_k$ can be fixed or learned.

- **Sigmoids and ReLUs:** These nonlinear functions, central to neural networks, can also be viewed as basis functions, especially when their parameters (like shift/scale) are learned.

**4.3. Translated Basis Functions:**  
Instead of fixed basis functions, we can allow transformations, such as translation, within the basis function itself, making the basis function adaptive.

$\phi(x; p_k)$ where $p_k$ are parameters specific to the $k$-th basis function. A common case is translation:

$$\phi_k(x) = \phi(x - c_k)$$

where $c_k$ is a learnable center (parameter). This allows the basis functions (e.g., Gaussians, sigmoids) to position themselves optimally to model localized features in the data.

## 5. Neural Networks as Function Approximators

Neural networks provide a powerful framework for learning the basis functions implicitly through layers of transformations.

**5.1. Shallow Network (Single Hidden Layer):**  
This is the simplest form of a multi-layer perceptron (MLP). It comprises an input layer, one hidden layer, and an output layer.

![[shallow-net.png]]

**Hidden Unit Computation:** For each hidden unit $k$ (where $k = 1, \ldots, M$, $M$ being the width of the hidden layer), a pre-activation $z_k$ is computed as an affine transformation of the input $\mathbf{x} \in \mathbb{R}^D$, followed by a nonlinear activation function $\phi$:

$$z_k = \mathbf{a}_k^T \mathbf{x} + b_k = \sum_{j=1}^D a_{kj} x_j + b_k$$

$$h_k = \phi(z_k) = \phi(\mathbf{a}_k^T \mathbf{x} + b_k)$$

Here, $\mathbf{a}_k \in \mathbb{R}^D$ is the weight vector and $b_k \in \mathbb{R}$ is the bias for the $k$-th hidden unit. The collection of $h_k$ forms the hidden layer activation vector $\mathbf{h} \in \mathbb{R}^M$.

**Output Computation:** The final output $y \in \mathbb{R}$ (for regression) is computed as an affine transformation of the hidden layer activations:

$$y = \mathbf{c}^T \mathbf{h} + c_0 = \sum_{k=1}^M c_k h_k + c_0$$

Here, $\mathbf{c} \in \mathbb{R}^M$ is the output weight vector and $c_0 \in \mathbb{R}$ is the output bias.

**Parameters:** The full parameter set is $\theta = \{\mathbf{a}_k, b_k\}_{k=1}^M \cup \{\mathbf{c}, c_0\}$.

**Interpretation:** Each hidden unit $h_k$ can be seen as computing a learned basis function $\phi(\mathbf{a}_k^T \mathbf{x} + b_k)$, and the output layer computes a linear combination of these basis functions.

**5.2. ReLU Activation Function:**  
The Rectified Linear Unit (ReLU) is a popular choice for $\phi$:

$$\phi(t) = \max(0, t)$$

It is computationally efficient and helps mitigate the vanishing gradient problem. Its derivative is simple (0 for $t < 0$, 1 for $t > 0$, undefined/subgradient at $t = 0$).

**Piecewise Linearity:** Functions built with ReLU activations are continuous and piecewise linear. Each hidden unit $\phi(\mathbf{a}_k^T \mathbf{x} + b_k)$ introduces a "hinge" or "kink" along the hyperplane $\mathbf{a}_k^T \mathbf{x} + b_k = 0$. The composition of these results in a function that is linear within different regions of the input space.

**5.3. Universal Approximation Theorem:**  
Theorems like the Cybenko (1989) or Hornik (1991) theorem state, roughly, that a shallow neural network with a sufficient number of hidden units ($M$) and a suitable "squashing" activation function (like sigmoid, although later extended to others including ReLU under certain conditions) can approximate any continuous function $g: K \rightarrow \mathbb{R}$ arbitrarily well on a compact set $K \subseteq \mathbb{R}^D$. That is, for any $\epsilon > 0$, there exists a set of parameters $\theta$ such that $|f(\mathbf{x}; \theta) - g(\mathbf{x})| < \epsilon$ for all $\mathbf{x} \in K$.

**Implication:** Theoretically, a single hidden layer is sufficient for representation power, but it might require an exponentially large number of units ($M$) in practice.

**5.4. Network Expressivity: Regions and Convex Polytopes:**  
With ReLU activations, the input space $\mathbb{R}^D$ is partitioned by the hyperplanes $\mathbf{a}_k^T \mathbf{x} + b_k = 0$ associated with each hidden unit $k$.

- **1D Input:** Each ReLU unit $\max(0, a_k x + b_k)$ creates a single "kink" at $x = -b_k / a_k$. A sum of these creates a continuous piecewise linear function where the segments meet at these kinks.
- **Higher-Dimensional Input:** The $M$ hyperplanes partition $\mathbb{R}^D$ into multiple regions. Within each region, the activation pattern (which ReLUs are active, i.e., $\mathbf{a}_k^T \mathbf{x} + b_k > 0$) is fixed. Since the output $y$ is a linear combination of these (now effectively linear or zero) activations, the overall function $f(\mathbf{x}; \theta)$ behaves linearly within each region. These regions are convex polytopes (or polyhedra).

**Zaslavsky's Theorem:** Provides bounds on the maximum number of regions $R$ created by $M$ hyperplanes in $\mathbb{R}^D$. The number of regions grows combinatorially, $R \leq \sum_{i=0}^D \binom{M}{i}$. For large $M$, $R$ can be roughly $O(M^D)$. This indicates the potential for high expressivity.

**5.5. Deep Networks: Composition of Functions:**  
Deep networks utilize multiple hidden layers, composing functions layer by layer. Let $F_l(\cdot; \theta_l)$ represent the transformation performed by layer $l$.

**Structure:**

$$\mathbf{h}_1 = F_1(\mathbf{x}; \theta_1)$$
$$\mathbf{h}_2 = F_2(\mathbf{h}_1; \theta_2)$$
$$\ldots$$
$$y = F_L(\mathbf{h}_{L-1}; \theta_L)$$

**Hierarchical Features:** The intuition is that layers progressively transform the representation, potentially **learning features at different levels of abstraction** (e.g., edges -> textures -> parts -> objects in image recognition).

**Necessity of Nonlinearity:** If each $F_l$ were a purely linear (affine) transformation, $F_l(\mathbf{z}) = A_l \mathbf{z} + \mathbf{b}_l$, their composition would be equivalent to a single affine transformation:

$$y = A_L(\ldots(A_1 \mathbf{x} + \mathbf{b}_1)\ldots) + \mathbf{b}_L = A' \mathbf{x} + \mathbf{b}'$$

where $A' = A_L \ldots A_1$ and $\mathbf{b}'$ is a combination of transformed bias terms. Thus, without nonlinear activation functions $\phi$ applied within each layer (or between layers), a deep linear network has no more representational power than a shallow linear network. Activation functions are crucial for the depth advantage.

**5.6. Typical Deep Network Layer Structure:**  
A standard feedforward layer applies an affine transformation followed by an elementwise activation function:

$$\mathbf{h}_l = \phi(A_l \mathbf{h}_{l-1} + \mathbf{b}_l)$$

where $\mathbf{h}_0 = \mathbf{x}$, $A_l$ is the weight matrix for layer $l$, $\mathbf{b}_l$ is the bias vector, and $\phi$ is the activation function (e.g., ReLU). The final layer might use a different activation $\phi'$ (e.g., identity for regression, softmax for classification):

$$y = \phi'(A_L \mathbf{h}_{L-1} + \mathbf{b}_L)$$

## 6. Neural Network Notation and Terminology

Precision in terminology is important.

**6.1. Parameters:**

- **Weights:** Coefficients multiplying inputs or activations. Represent connection strengths. Examples: $\theta_k$ (basis function coefficient), $a_{kj}$ (input-to-hidden weight), $c_k$ (hidden-to-output weight), matrix entries in $A_l$. Sometimes referred to as "slopes".
- **Biases:** Additive constants applied before activation. Allow shifting the activation function. Examples: $\theta_0$, $b_k$, $c_0$, vector entries in $\mathbf{b}_l$. Sometimes referred to as "Y-offsets" or intercepts. 

**6.2. Activation Stages:**

- **Pre-activations:** The result of the affine transformation within a neuron/unit before the nonlinearity is applied. Example: $z_k = \mathbf{a}_k^T \mathbf{x} + b_k$ or the vector $\mathbf{z}_l = A_l \mathbf{h}_{l-1} + \mathbf{b}_l$.
- **Activations:** The output of a neuron/unit after the nonlinearity is applied. Example: $h_k = \phi(z_k)$ or the vector $\mathbf{h}_l = \phi(\mathbf{z}_l)$.

**6.3. Network Architecture Terms:**

- **Fully Connected (Dense):** Every neuron in layer $l-1$ is connected to every neuron in layer $l$. Represented by dense weight matrices $A_l$.
- **Feedforward:** Information flows unidirectionally from input to output without cycles. Standard MLPs, CNNs (in their basic form). Contrast with Recurrent Neural Networks.
- **Shallow vs. Deep:** Shallow networks typically have one hidden layer. Deep networks have multiple (two or more) hidden layers. Deep networks are often argued to represent complex functions more efficiently (with fewer parameters) than shallow ones, particularly for hierarchical data.

## 7. Vector and Matrix Representations

Linear algebra provides the natural language for describing network operations, especially when dealing with vector inputs/outputs and layers of neurons.

**7.1. Handling Vector Inputs/Outputs:**  
Consider a single layer mapping $\mathbf{x} \in \mathbb{R}^D$ to $\mathbf{y} \in \mathbb{R}^K$, with $M$ hidden units.

**Hidden Layer Computation (Matrix Form):** Let $A \in \mathbb{R}^{M \times D}$ contain the weights $\mathbf{a}_k^T$ as rows, and $\mathbf{b} \in \mathbb{R}^M$ contain the biases $b_k$. The pre-activations are $\mathbf{z} = A \mathbf{x} + \mathbf{b}$, and activations are $\mathbf{h} = \phi(\mathbf{z})$ (elementwise $\phi$).

**Output Layer Computation (Matrix Form):** Let $C \in \mathbb{R}^{K \times M}$ contain output weights as rows, and $\mathbf{c}_0 \in \mathbb{R}^K$ contain output biases. The final output is $\mathbf{y} = C \mathbf{h} + \mathbf{c}_0$.

**Combined Formula (Shallow Network):** $\mathbf{y} = C \phi(A \mathbf{x} + \mathbf{b}) + \mathbf{c}_0$.

**7.2. Example: Multiple Outputs (1 input, 4 hidden, 2 outputs):**  
Let $x \in \mathbb{R}$ (scalar input, $D = 1$). Hidden layer has $M = 4$. Output layer has $K = 2$.

**Hidden Layer:** $a_k \in \mathbb{R}^1$ (just $a_k$), $b_k \in \mathbb{R}$.

$$h_k = \phi(a_k x + b_k) \text{ for } k = 1, \ldots, 4$$

$$\mathbf{h} = [h_1, h_2, h_3, h_4]^T \in \mathbb{R}^4$$

**Output Layer:** Output weight matrix $C \in \mathbb{R}^{2 \times 4}$, bias vector $\mathbf{c}_0 = [c_{10}, c_{20}]^T \in \mathbb{R}^2$.

$$y_1 = \sum_{k=1}^4 C_{1k} h_k + c_{10} = c_{11} h_1 + c_{12} h_2 + c_{13} h_3 + c_{14} h_4 + c_{10}$$

$$y_2 = \sum_{k=1}^4 C_{2k} h_k + c_{20} = c_{21} h_1 + c_{22} h_2 + c_{23} h_3 + c_{24} h_4 + c_{20}$$

In matrix form: $\mathbf{y} = C \mathbf{h} + \mathbf{c}_0$.

**7.3. Batching:**  
To leverage parallel hardware (GPUs/TPUs), computations are performed on mini-batches of $B$ samples simultaneously. If $\mathbf{x}^{(i)}$ is the $i$-th sample, we form an input matrix $X \in \mathbb{R}^{D \times B}$ where the $i$-th column is $\mathbf{x}^{(i)}$. A function $f: \mathbb{R}^D \rightarrow \mathbb{R}^K$ applied to a batch becomes $F: \mathbb{R}^{D \times B} \rightarrow \mathbb{R}^{K \times B}$. For example, a linear layer $A \mathbf{x} + \mathbf{b}$ becomes $A X + \mathbf{b} \mathbf{1}^T$, where $\mathbf{1}^T$ is a row vector of ones, effectively adding the bias vector $\mathbf{b}$ to each column (sample) of $A X$. (Broadcasting handles this implicitly in frameworks).

**7.4. Broadcasting:**  
Libraries like NumPy, PyTorch, TensorFlow, JAX automatically handle operations between arrays of compatible but different shapes. For example, adding a bias vector $\mathbf{b} \in \mathbb{R}^M$ to a matrix of pre-activations $Z \in \mathbb{R}^{M \times B}$ is understood as adding $\mathbf{b}$ to each column of $Z$. This simplifies coding and maintains efficiency. The vmap transform in JAX provides a more explicit functional way to achieve batching.

# Part III: Training Neural Networks

## 8. Optimization Objective

The goal is to find parameters $\theta^*$ that minimize the total loss (empirical risk) over the training data $D$:

$$\theta^* = \arg\min_{\theta \in \Theta} L(\theta) = \arg\min_{\theta \in \Theta} \sum_{i=1}^n L(f(\mathbf{x}_i; \theta), y_i)$$

This is typically a high-dimensional, non-convex optimization problem, especially for deep networks.

## 9. Gradient Descent Algorithms

Iterative methods based on the gradient of the loss function are the standard approach.

**9.1. Basic Gradient Descent (Full-Batch):**  
This algorithm uses the gradient computed over the entire training set at each step.

**Initialization:** Choose an initial parameter vector $\theta_0$.

**Gradient Computation:** Compute the gradient of the total loss with respect to the parameters, evaluated at the current parameters $\theta_i$:

$$\mathbf{g}_i = \nabla_\theta L(\theta_i) = \left.\frac{\partial L(\theta)}{\partial \theta}\right|_{\theta = \theta_i}$$

**Parameter Update:** Move the parameters in the direction opposite to the gradient:

$$\theta_{i+1} = \theta_i - \lambda \mathbf{g}_i$$

where $\lambda > 0$ is the learning rate, controlling the step size.

**Iteration:** Repeat gradient computation and update until a convergence criterion is met (e.g., gradient magnitude is small, loss stops decreasing, maximum iterations reached).

**9.2. Example: 1D Linear Regression Gradient Calculation:**  
**Model:** $f(x; \theta_0, \theta_1) = \theta_0 + \theta_1 x$. **Loss:** $L(\theta_0, \theta_1) = \sum_{i=1}^n ((\theta_0 + \theta_1 x_i) - y_i)^2$.  
The partial derivatives are:

$$\frac{\partial L}{\partial \theta_0} = \sum_{i=1}^n 2((\theta_0 + \theta_1 x_i) - y_i) \cdot 1$$

$$\frac{\partial L}{\partial \theta_1} = \sum_{i=1}^n 2((\theta_0 + \theta_1 x_i) - y_i) \cdot x_i$$

These gradients $\mathbf{g} = [\frac{\partial L}{\partial \theta_0}, \frac{\partial L}{\partial \theta_1}]^T$ are used in the update rule $\theta_{i+1} = \theta_i - \lambda \mathbf{g}_i$. For this specific problem, a closed-form solution (Normal Equations) exists, but gradient descent provides the template for more complex models where closed-form solutions are unavailable.

**9.3. Stochastic Gradient Descent (SGD):**  
Computing the gradient over the entire dataset (size $n$) can be computationally expensive, especially for large $n$. SGD uses an approximation of the gradient computed on a small, randomly selected subset of the data called a mini-batch, $D_B \subset D$, with $|D_B| = B \ll n$.

**Stochastic Gradient:** $\hat{\mathbf{g}}_i = \nabla_\theta L(\theta_i; D_B) = \frac{1}{B} \sum_{(\mathbf{x}, y) \in D_B} \nabla_\theta L(f(\mathbf{x}; \theta_i), y)$ (often scaled by $1/B$).

**Update Rule:** $\theta_{i+1} = \theta_i - \lambda \hat{\mathbf{g}}_i$.

**Epoch:** One full pass through the training data (typically involving $n/B$ mini-batch updates).

**Benefits:**

- **Computational Efficiency:** Faster updates as $B \ll n$.
- **Regularization Effect:** The noise in the gradient estimate ($\hat{\mathbf{g}}_i$ is a noisy estimate of $\mathbf{g}_i$) can help the optimizer escape sharp local minima or saddle points, potentially leading to solutions that generalize better.

**Learning Rate Schedule:** The learning rate $\lambda$ is often decayed over epochs (e.g., step decay, exponential decay, cosine annealing) to allow convergence as the optimization approaches a minimum.

**9.4. Momentum:**  
SGD updates can oscillate, especially in ravines of the loss landscape. Momentum adds inertia to the updates.

**Update Rule:** Introduce a velocity vector $\mathbf{v}_i$, which is an exponentially weighted moving average of past gradients:

$$\mathbf{v}_{i+1} = \beta \mathbf{v}_i + (1 - \beta) \hat{\mathbf{g}}_i \text{ (or sometimes } \mathbf{v}_{i+1} = \beta \mathbf{v}_i + \lambda \hat{\mathbf{g}}_i)$$

$$\theta_{i+1} = \theta_i - \lambda \mathbf{v}_{i+1} \text{ (or } \theta_{i+1} = \theta_i - \mathbf{v}_{i+1})$$

where $\beta \in [0, 1)$ is the momentum coefficient (e.g., 0.9).

**Intuition:** The velocity $\mathbf{v}$ accumulates gradients in consistent directions, dampening oscillations and accelerating progress along stable directions of descent.

**Nesterov Accelerated Momentum (NAG):** A modification that calculates the gradient $\hat{\mathbf{g}}_i$ after making a preliminary step in the direction of the current velocity: $\hat{\mathbf{g}}_i = \nabla_\theta L(\theta_i - \lambda \beta \mathbf{v}_i; D_B)$. Often converges faster than standard momentum.

**9.5. Adam (Adaptive Moment Estimation):**  
Adam combines momentum with adaptive learning rates for each parameter, based on estimates of the first and second moments of the gradients.

**Moment Estimates:** Maintains exponentially weighted moving averages of the gradient ($\mathbf{m}_i$, first moment) and the squared gradient ($\mathbf{v}_i$, second moment, elementwise square):

$$\mathbf{m}_{i+1} = \beta_1 \mathbf{m}_i + (1 - \beta_1) \hat{\mathbf{g}}_i$$
$$\mathbf{v}_{i+1} = \beta_2 \mathbf{v}_i + (1 - \beta_2) (\hat{\mathbf{g}}_i \odot \hat{\mathbf{g}}_i)$$

**Bias Correction:** Corrects for initialization bias: $\hat{\mathbf{m}}_{i+1} = \mathbf{m}_{i+1} / (1 - \beta_1^{i+1})$, $\hat{\mathbf{v}}_{i+1} = \mathbf{v}_{i+1} / (1 - \beta_2^{i+1})$.

**Parameter Update:** Updates parameters using the bias-corrected moment estimates:

$$\theta_{i+1} = \theta_i - \lambda \frac{\hat{\mathbf{m}}_{i+1}}{\sqrt{\hat{\mathbf{v}}_{i+1}} + \epsilon}$$

where $\lambda$ is the base learning rate, $\epsilon$ is a small constant for numerical stability, and the division/sqrt are elementwise.

**Benefit:** Automatically adapts the learning rate for each parameter. Parameters receiving large or frequent gradients have their effective learning rate reduced, while infrequent ones get larger updates. Often converges quickly with default hyperparameters ($\beta_1 \approx 0.9, \beta_2 \approx 0.999$).

**9.6. RMSProp (Root Mean Square Propagation)**
RMSProp scales learning rates based on a moving average of squared gradients, reducing the learning rate for parameters with large (noisy) gradients.

**Update Rule:** Let $\mathbf{v}_i$ be the running average of squared gradients:

$$
v_{i+1} = \beta v_i + (1 - \beta)(\hat{g}_i \odot \hat{g}_i)
$$

Update the parameters using:
$$
\theta_{i+1} = \theta_i - \lambda \frac{\hat{g}_i}{\sqrt{v_{i+1}} + \epsilon}
$$

- $\beta$ is typically set around 0.9  
- $\epsilon$ prevents division by zero

**Intuition:**  RMSProp slows down updates for parameters with consistently large gradients, allowing more stable training.

**Comparison to Adam:**  RMSProp uses only second-order moment estimates, while Adam combines both first and second.

**9.7. Normalized Gradient Descent (NGD)**
Normalized Gradient Descent removes the effect of gradient magnitude by rescaling the gradient to have unit norm.

**Update Rule:**

$$
\theta_{i+1} = \theta_i - \lambda \cdot \frac{\hat{g}_i}{\|\hat{g}_i\|}
$$

**Intuition:** This ensures every update step moves the same distance (in parameter space), regardless of how large or small the gradient is. It focuses purely on the direction of steepest descent.

**Benefit:**  Helps in stabilizing training when gradients fluctuate widely. May prevent overly aggressive updates from large gradients.

**Limitation:**  May ignore useful curvature information. Often used in conjunction with other techniques like momentum or second-order scaling (as in trust-region methods or natural gradients).

## 10. Gradient Computation: Backpropagation and Jacobians

Efficiently computing the gradient $\nabla_\theta L$ is crucial. Backpropagation is the standard algorithm, based on the chain rule and dynamic programming.

**10.1. The Challenge:** For a deep network $f(\mathbf{x}; \theta)$, the loss $L(f(\mathbf{x}; \theta), y)$ is a deeply nested composition of functions. Calculating the gradient $\nabla_\theta L$ requires differentiating through all these layers.

**10.2. Chain Rule Application:**  
The chain rule allows breaking down the derivative of a composite function. If $L$ depends on $\mathbf{y}$, $\mathbf{y}$ depends on $\mathbf{h}_L$, ..., $\mathbf{h}_1$ depends on $\theta$, then the gradient with respect to parameters in layer $l$ involves products of derivatives (Jacobians) from layer $L$ back to $l$. For example, if $\theta_l$ are parameters in layer $l$:

$$\frac{\partial L}{\partial \theta_l} = \frac{\partial L}{\partial \mathbf{h}_L} \frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}} \ldots \frac{\partial \mathbf{h}_{l+1}}{\partial \mathbf{h}_l} \frac{\partial \mathbf{h}_l}{\partial \theta_l}$$

This involves matrix/tensor products.

**10.3. Jacobians:**  
The derivative of a vector-valued function $f: \mathbb{R}^N \rightarrow \mathbb{R}^O$ with respect to its vector input is the Jacobian matrix $J_f \in \mathbb{R}^{O \times N}$:

$$[J_f(\mathbf{x})]_{ij} = \frac{\partial f_i(\mathbf{x})}{\partial x_j}$$

**Multidimensional Arguments:** If inputs/outputs are tensors (e.g., $f: \mathbb{R}^{M \times N} \rightarrow \mathbb{R}^{P \times Q}$), the Jacobian becomes a higher-order tensor containing all partial derivatives $\frac{\partial f_{pq}}{\partial x_{mn}}$. Its conceptual size is (output dimensions) $\times$ (input dimensions).

**Multiple Inputs:** For $f(\mathbf{x}, \mathbf{y})$, the Jacobian can be viewed as a tuple $(J_{f,x}, J_{f,y})$, where $J_{f,x} = \frac{\partial f}{\partial \mathbf{x}}$ and $J_{f,y} = \frac{\partial f}{\partial \mathbf{y}}$.

**10.4. Chain Rule for Jacobians:**  
If $f(\mathbf{x}) = g(h(\mathbf{x}))$, where $h: \mathbb{R}^N \rightarrow \mathbb{R}^H$ and $g: \mathbb{R}^H \rightarrow \mathbb{R}^O$, the chain rule in matrix form is:

$$J_f(\mathbf{x}) = J_g(h(\mathbf{x})) \cdot J_h(\mathbf{x})$$

The product is standard matrix multiplication ($O \times H$ matrix times $H \times N$ matrix yields $O \times N$ matrix).

**Multiple Arguments:** If $f(\mathbf{x}, \mathbf{y}) = g(h(\mathbf{x}), k(\mathbf{y}))$, the chain rule applies component-wise or requires careful handling of partial Jacobians depending on the exact structure.

**10.5. Reverse Mode Differentiation (Backpropagation):**  
Computing and storing full Jacobians ($J_l$ for each layer $l$) is infeasible due to their size ($O \times N$ can be huge). Backpropagation uses reverse mode automatic differentiation. Instead of propagating Jacobians forward, it propagates the gradient of the final scalar loss $L$ backward through the network. It relies on computing Vector-Jacobian Products (VJPs).

**Vector-Jacobian Product (VJP):** For a function $f: \mathbb{R}^N \rightarrow \mathbb{R}^O$ and a vector $\mathbf{v} \in \mathbb{R}^O$ (representing the gradient of the final loss w.r.t. the output of $f$), the VJP is:

$$\text{VJP}_f(\mathbf{x}, \mathbf{v}) = \mathbf{v}^T J_f(\mathbf{x})$$

This results in a row vector of dimension $1 \times N$ (or a tensor with the shape of the input $\mathbf{x}$), representing the gradient of the final loss w.r.t. the input $\mathbf{x}$. Crucially, the VJP can often be computed without explicitly forming the full Jacobian $J_f$.

**Scalar Loss Advantage:** Since the final loss $L$ is scalar, the initial vector $\mathbf{v}$ for the last layer is simply $\frac{\partial L}{\partial \mathbf{y}}$ (a $1 \times K$ row vector). Subsequent VJPs maintain this "vector" nature (often row vectors or tensors matching input shapes), avoiding matrix-matrix products involving large Jacobians.

**10.6. VJPs for Basic Operations:**  
Backpropagation applies VJP rules recursively. For each elementary operation $\mathbf{y} = f(\mathbf{x})$ in the network: given the gradient of the final loss w.r.t. $\mathbf{y}$ ($\bar{\mathbf{y}} = \frac{\partial L}{\partial \mathbf{y}}$), compute the gradient w.r.t. $\mathbf{x}$ ($\bar{\mathbf{x}} = \frac{\partial L}{\partial \mathbf{x}} = \bar{\mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$).

- **Vector Addition:** $f(\mathbf{a}, \mathbf{b}) = \mathbf{a} + \mathbf{b}$. $J_{f,\mathbf{a}} = I$, $J_{f,\mathbf{b}} = I$. Given  
  $\bar{f} = \frac{\partial L}{\partial f}$, the VJPs are  
  $\bar{\mathbf{a}} = \bar{f}^T I = \bar{f}^T$ and $\bar{\mathbf{b}} = \bar{f}^T I = \bar{f}^T$. The gradient is simply passed back.

- **Matrix Multiplication:** $f(A, B) = P = AB$. Let $\bar{P} = \frac{\partial L}{\partial P}$ be the incoming gradient (same shape as $P$).
  - **VJP w.r.t. $A$:** $\frac{\partial L}{\partial A} = \bar{P} B^T$. (Shape: $(M \times N) \times (N \times H) = M \times H$, matches $A$).
  - **VJP w.r.t. $B$:** $\frac{\partial L}{\partial B} = A^T \bar{P}$. (Shape: $(H \times M) \times (M \times N) = H \times N$, matches $B$).

- **Elementwise Activation:** $\mathbf{h} = \phi(\mathbf{z})$. $J_\phi$ is a diagonal matrix with $[J_\phi]_{ii} = \phi'(z_i)$. Given  
  $\bar{\mathbf{h}} = \frac{\partial L}{\partial \mathbf{h}}$, the VJP gives  
  $\bar{\mathbf{z}} = \bar{\mathbf{h}} \odot \phi'(\mathbf{z})$ (elementwise product).

**10.7. Backpropagation Algorithm Steps:**

1. **Forward Pass:** Compute and store all activations $\mathbf{h}_l$ and pre-activations $\mathbf{z}_l$ for $l = 1, \ldots, L$, given an input $\mathbf{x}$. Compute the final loss $L$.
2. **Backward Pass:**
   - **Initialize:** Compute the gradient of the loss w.r.t. the final output $\bar{\mathbf{y}} = \frac{\partial L}{\partial \mathbf{y}}$.
   - For $l = L, L-1, \ldots, 1$:
     - Compute gradient w.r.t. layer $l$'s parameters ($A_l, \mathbf{b}_l$) using the incoming gradient $\bar{\mathbf{h}}_l$ (gradient w.r.t. layer $l$'s output) and the stored values ($\mathbf{h}_{l-1}$). This uses the VJP rule for the affine transformation.
     - Compute gradient w.r.t. layer $l$'s input ($\mathbf{h}_{l-1}$) using $\bar{\mathbf{h}}_l$ and the stored pre-activation $\mathbf{z}_l$. This uses VJP rules for the activation function $\phi$ and the affine transformation. This becomes the incoming gradient $\bar{\mathbf{h}}_{l-1}$  for the previous layer.
   - **Gradient Aggregation:** Sum or average gradients computed for each sample in the mini-batch.

**10.8. Computation Model:** Backpropagation naturally maps onto the computation graph (a Directed Acyclic Graph or DAG) of the network, where nodes are operations and edges are data flow. The backward pass traverses this graph in reverse topological order.

# Part IV: Practical Considerations in Training

## 11. Initialization and Scaling

Proper initialization is crucial for successful training, preventing activations or gradients from becoming too large (exploding) or too small (vanishing).

**11.1. Importance of Input Scaling:** Input features should be normalized (e.g., to zero mean and unit variance, or scaled to $[0, 1]$ or $[-1, 1]$). This ensures that gradients are well-scaled initially and helps optimization.

**11.2. Variance Propagation (Linear Layer):** Consider a linear layer $\mathbf{p} = A \mathbf{x}$, where $\mathbf{x} \in \mathbb{R}^N$, $A \in \mathbb{R}^{M \times N}$, $\mathbf{p} \in \mathbb{R}^M$. Assume inputs $x_j$ and weights $A_{ij}$ are independent random variables with zero mean and variances $\text{Var}(x_j) = \sigma_x^2$ and $\text{Var}(A_{ij}) = \sigma_A^2$. The variance of an element $p_i = \sum_{j=1}^N A_{ij} x_j$ is:

$$\text{Var}(p_i) = \sum_{j=1}^N \text{Var}(A_{ij} x_j) = \sum_{j=1}^N \text{Var}(A_{ij}) \text{Var}(x_j) = \sum_{j=1}^N \sigma_A^2 \sigma_x^2 = N \sigma_A^2 \sigma_x^2$$

To **maintain signal variance** ($\text{Var}(p_i) \approx \sigma_x^2$), we need $N \sigma_A^2 \approx 1$, suggesting initializing weights from a distribution with variance:

$$\sigma_A^2 = \frac{1}{N} \text{ (Glorot/Xavier initialization for linear/tanh)}$$

Weights are often drawn from $\mathcal{N}(0, \sigma_A^2)$ or $\text{Uniform}(-\sqrt{3/N}, \sqrt{3/N})$.

**11.3. Adjustments for ReLU Activation:**  
ReLU activation $\phi(p) = \max(0, p)$ zeros out negative values. If $p$ is zero-mean symmetric, ReLU halves the input density. The variance of the output $h = \phi(p)$ is approximately half the variance of the input: $\text{Var}(h) \approx \frac{1}{2} \text{Var}(p)$.  
Combining this with the linear layer variance: $\text{Var}(h_i) \approx \frac{1}{2} N \sigma_A^2 \sigma_x^2$.  
To maintain variance ($\text{Var}(h_i) \approx \sigma_x^2$), we now need $\frac{1}{2} N \sigma_A^2 \approx 1$, leading to:

$$\sigma_A^2 = \frac{2}{N} \text{ (He initialization for ReLU)}$$

Weights are drawn from $\mathcal{N}(0, 2/N)$ or $\text{Uniform}(-\sqrt{6/N}, \sqrt{6/N})$. This helps keep activation variances stable across layers in deep ReLU networks. Similar analysis applies to backpropagated gradients.

**11.4. Importance:** Proper initialization prevents the exponential growth or decay of activation/gradient magnitudes through deep networks, enabling stable training.

## 12. Numerical Stability

Floating-point arithmetic has finite precision, requiring care in implementations.

**12.1. Softmax Issues:** The computation $z_k = \exp(y_k) / \sum_{k'} \exp(y_{k'})$ can suffer from overflow if some $y_k$ are large (exp(large) is huge) or underflow/loss of precision if all $y_k$ are large and negative (exp(negative) approaches zero).

**12.2. LogSoftmax:** Computing the logarithm of the softmax is often more stable and directly useful for the cross-entropy loss:

$$\log z_k = \log\left(\frac{\exp(y_k)}{\sum_{k'} \exp(y_{k'})}\right) = y_k - \log\sum_{k'} \exp(y_{k'})$$

**12.3. Stable Log-Sum-Exp Trick:** The $\log\sum\exp$ term is the source of instability. Let $y_{\max} = \max_k(y_k)$. We can rewrite:

$$\log\sum_{k'} \exp(y_{k'}) = \log\left(\exp(y_{\max}) \sum_{k'} \exp(y_{k'} - y_{\max})\right)$$

$$= y_{\max} + \log\sum_{k'} \exp(y_{k'} - y_{\max})$$

Now, the arguments to the inner exp are $(y_{k'} - y_{\max}) \leq 0$. The largest argument is 0 (where $\exp(0) = 1$), preventing overflow. If all $y_k$ are very negative, $y_{\max}$ factors out the large negative scale, potentially improving precision in the sum. This numerically stable computation is standard in implementations.

## 13. Hyperparameters and Model Capacity

Hyperparameters are settings chosen before training begins.

**13.1. Definition:** Examples include network architecture (depth, width), learning rate $\lambda$, momentum $\beta$, regularization strength $\gamma$, choice of activation function $\phi$. They are typically tuned via experimentation, often using a separate validation dataset.

**13.2. Depth:** The number of layers $L$. Deeper networks can potentially model more complex, hierarchical functions more efficiently (parameter-wise) than shallower ones.

**13.3. Width:** The number of neurons $M_l$ in layer $l$. Wider layers increase the number of parameters and the representational capacity within that layer (e.g., more basis functions in a shallow network, more hyperplanes for partitioning space).

**13.4. Capacity:** A qualitative measure of the complexity of functions a model can represent. Related to the number of parameters, but also structure (depth/width). High capacity allows fitting complex training data but risks overfitting.

**Regions Example:** As noted, the number of linear regions in a ReLU network grows with width and depth, reflecting increased capacity. A shallow network with 6 hidden units (approx. $1 \times 6 + 6 + 6 \times 1 + 1 = 19$ or 20 params depending on bias details) might create 7-9 regions, while one with 500 units ($1 \times 500 + 500 + 500 \times 1 + 1 = 1501$ params) can create a vastly larger number.

**13.5. Balancing Capacity and Overfitting:** A central challenge. The model needs sufficient capacity to capture the true underlying patterns (underfitting if too low), but not so much capacity that it memorizes noise in the training data (overfitting). Hyperparameter tuning and regularization aim to find this balance.

## 14. Regularization: Improving Generalization

Regularization techniques aim to reduce overfitting and improve the model's performance on unseen data (generalization).

**14.1. Problem:** Overfitting occurs when a model learns idiosyncrasies (noise) of the training set, leading to poor performance on new data. The model has low bias but high variance.

**14.2. Explicit Regularization:** Modifying the objective function or network structure.

**L2 Regularization (Weight Decay):** Adds a penalty term to the loss function proportional to the squared magnitude (L2 norm) of the weight parameters $\theta_W$ (typically excludes biases):

$$L_{\text{reg}}(\theta) = L(\theta) + \frac{\gamma}{2} \|\theta_W\|_2^2 = L(\theta) + \frac{\gamma}{2} \sum_j (\theta_{W,j})^2$$

The gradient of the penalty term is $\gamma \theta_W$. During gradient descent updates $\theta_W \leftarrow \theta_W - \lambda (\nabla L + \gamma \theta_W) = (1 - \lambda \gamma) \theta_W - \lambda \nabla L$. This effectively shrinks weights towards zero at each step (hence "weight decay"). It encourages smaller weights, leading to smoother functions less sensitive to input variations. Corresponds to assuming a Gaussian prior on weights in a Bayesian view.

**14.3. Implicit Regularization:** Effects arising from the training process itself.

- **Gradient Descent Dynamics:** The path taken by GD/SGD in the parameter space can implicitly favor certain types of solutions (e.g., solutions with smaller norms, especially when starting from $\theta_0 = 0$).
- **SGD Noise:** The stochasticity in SGD acts as a form of noise injection, which can prevent the optimizer from converging to overly sharp minima that might not generalize well.
- **Early Stopping:** Monitor the loss on a separate validation set during training. Stop training when the validation loss starts increasing, even if the training loss is still decreasing. This prevents the model from progressing too far into the overfitting regime.
- **Gradient Norm Penalty** One interpretation of this implicit regularization is to modify the loss function to penalize the gradient norm. This encourages the optimizer to prefer flatter regions of the loss landscape, where the gradient is small. It discourages sharp curvature in the loss function around the optimum:

$$
L_{\text{GD}}(\theta) = L(\theta) + \frac{\lambda}{4} \left\| \frac{dL}{d\theta} \right\|^2
$$


- **Gradient Variance Penalty:** When using SGD, gradients are estimated from minibatches. A second implicit regularization arises by encouraging consistency between minibatch gradients and the full-batch gradient:

$$
L_{\text{SGD}}(\theta) = L(\theta) + \frac{\lambda}{4} \left\| \frac{dL}{d\theta} \right\|^2 + \frac{\alpha}{4B} \sum_{B_t} \left\| \frac{dL}{d\theta}(B_t) - \frac{dL}{d\theta}(\mathcal{D}) \right\|^2
$$


- **Insight**: Larger learning rates amplify the noise in SGD, which increases the implicit regularization effect. This may partly explain why models trained with higher learning rates often generalize better.

**14.4. Other Techniques:**

- **Dropout:** During training, randomly set the output of each neuron to zero with some probability $p$ (independently for each neuron/batch). Scales remaining activations by $1/(1 - p)$ at training time (inverted dropout). At test time, use all neurons (no dropout). Acts as training an ensemble of many "thinned" networks, forcing redundancy and preventing co-adaptation of neurons.
- **Ensembling:** Train multiple independent models (e.g., with different initializations or data subsets) and average their predictions. Reduces variance.
- **Data Augmentation:** Increase the effective size and diversity of the training set by applying random transformations to the input data that preserve the label (e.g., image rotation/cropping/flipping, MixUp, RandAugment). Makes the model more robust to these variations.
- **Transfer Learning:** Initialize model weights from a model pre-trained on a large, related dataset. Fine-tune on the target task. Leverages knowledge learned from the large dataset. This strategy imposes a prior on the target model and acts as a regularizer, limiting overfitting on the smaller target dataset.
- **Stochastic Depth:** For very deep networks (like ResNets, used in ViT context), randomly drop entire layers (blocks) during training by replacing their function with the identity (skip connection).
- **Multi-Task Learning:** Multi-task learning simultaneously trains a single model on several related tasks. The requirement to perform well across different tasks forces the shared representations to capture more general features, thus mitigating the risk of overfitting to any single task’s noise.
- **Self-Supervised Learning:**  Self-supervised learning leverages the inherent structure of the data to create surrogate tasks (for example, predicting missing parts of an input). By pre-training a model on these tasks with massive amounts of unlabeled data, the network learns robust feature representations that can later be fine-tuned on a specific task. This external supervision acts as an effective regularizer by embedding additional domain knowledge into the learned representations.


![[reg-overview.png]]

**14.5 Architectural Regularization and Invariance**

**Normalization Techniques:** Normalization techniques are employed to stabilize training dynamics and enable the use of deeper architectures by controlling the distribution of intermediate activations. These techniques define transformations applied to neural activations at each layer of a network.

Let $\mathbf{x}^{(i)} \in \mathbb{R}^D$ denote the pre-activation vector (i.e., the output of a linear layer before nonlinearity) for the $i$-th input sample. We now describe two widely-used normalization schemes as functions applied to these activations. Normalization is typically applied before the activation function:

$$
\begin{aligned}
\mathbf{z} &= \mathbf{W} \mathbf{x} + \mathbf{b} \\
\hat{\mathbf{z}} &= \mathcal{N}(\mathbf{z}) \quad \text{(e.g., batch or layer norm)} \\
\mathbf{a} &= \phi(\hat{\mathbf{z}})
\end{aligned}
$$

- **Layer Normalization:** Layer Normalization defines a mapping:

$$
\mathcal{N}_{\text{layer}} : \mathbb{R}^D \to \mathbb{R}^D,
$$

which operates independently on each input vector $\mathbf{x} \in \mathbb{R}^D$, normalizing across the features (i.e., dimensions) of the vector. Given an input $\mathbf{x} = (x_1, \dots, x_D)$, the normalized output $\tilde{\mathbf{x}} = \mathcal{N}_{\text{layer}}(\mathbf{x})$ is computed elementwise as:

$$
\tilde{x}_j = \frac{x_j - \mu(\mathbf{x})}{\sigma(\mathbf{x})} \quad \text{for } j = 1, \dots, D,
$$

where

$$
\mu(\mathbf{x}) = \frac{1}{D} \sum_{j=1}^D x_j, \quad \sigma(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{j=1}^D (x_j - \mu(\mathbf{x}))^2 + \epsilon}.
$$

Here, $\epsilon > 0$ is a small constant added for numerical stability. Optionally, this transformation may be followed by an affine transformation:

$$
\mathcal{N}_{\text{layer}}(\mathbf{x}) = \boldsymbol{\gamma} \odot \tilde{\mathbf{x}} + \boldsymbol{\beta},
$$

where $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^D$ are learnable parameters and $\odot$ denotes elementwise multiplication.

- **Batch Normalization** Batch Normalization defines a mapping:

$$
\mathcal{N}_{\text{batch}} : \mathbb{R}^{B \times D} \to \mathbb{R}^{B \times D},
$$

which acts on a batch of $B$ input vectors $\{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(B)}\} \subset \mathbb{R}^D$, normalizing across the batch dimension for each feature coordinate $j = 1, \dots, D$. For each feature $j$, the batch-wise mean and variance are computed as:

$$
\mu_B(j) = \frac{1}{B} \sum_{i=1}^B x_j^{(i)}, \quad \sigma_B(j) = \sqrt{\frac{1}{B} \sum_{i=1}^B \left(x_j^{(i)} - \mu_B(j)\right)^2 + \epsilon}.
$$

Each activation $x_j^{(i)}$ is then normalized and scaled as:

$$
\tilde{x}_j^{(i)} = \gamma_j \cdot \frac{x_j^{(i)} - \mu_B(j)}{\sigma_B(j)} + \beta_j,
$$

where $\gamma_j, \beta_j \in \mathbb{R}$ are learnable parameters associated with each feature $j$. During inference, $\mu_B(j)$ and $\sigma_B(j)$ are replaced by running averages maintained during training to ensure deterministic outputs.


| Property              | Layer Normalization                  | Batch Normalization                         |
| --------------------- | ------------------------------------ | ------------------------------------------- |
| Operates on           | Single sample ($x \in \mathbb{R}^D$) | Mini-batch ($x^{(i)} \in \mathbb{R}^{D}$)   |
| Normalizes over       | Features ($j = 1,\dots,D$)           | Batch samples ($i = 1,\dots,B$)             |
| Suitable for          | RNNs, Transformers, online learning  | CNNs, MLPs, large-batch training            |
| Inference behavior    | Same as training                     | Requires stored running statistics          |
| Batch-size dependency | Independent of batch size            | Performance degrades with small batch sizes |
|                       |                                      |                                             |
**Invariance in Deep Learning:** Techniques for enforcing invariance ensure that the network’s output remains stable under specified transformations of the input:

- **Data Augmentation:**  By augmenting the training data with transformations (e.g., rotations, translations, scaling), the network is exposed to a broader variety of inputs, which encourages it to learn features that are invariant to these transformations. This is expensive, and does not guarantee invariance. 

- **Max Pooling and Sliding Windows:**  Architectural designs can explicitly induce invariance. For example, consider a non-invariant function $g: \mathbb{R}^{28 \times 28} \to \mathbb{R}^{10}$. An invariant function can be constructed as:
  $$
  f(X) = \max_\phi g(X * \phi),
  $$
  where $\phi$ indexes over a set of allowed transformations (such as translations). Taking the maximum (or similarly, a sum) over these transformations ensures that the output $f(X)$ remains (approximately) invariant to the variations in $X$. See that
  
$$
\begin{aligned}
f(X * \phi) &= \max_{\phi'} g((X * \phi) * \phi') \\
            &= \max_{\phi'} g(X * (\phi \phi')) \\
            &= \max_{\phi''} g(X * \phi'') \\
            &= f(X)
\end{aligned}
$$

**Convolutions, Channels, Bottlenecks, and Upsampling:** These architectural components contribute to regularization by embedding strong inductive biases into the model:

- **Bottlenecks:** Architectural bottlenecks intentionally compress the representation into a lower-dimensional space. For example, by using downsampling operations such as max or mean pooling, the network is forced to retain only the most salient features of the input. This compression acts as a regularizer by reducing overfitting.

![[downsampling.png]]
The **Information Bottleneck (IB)** objective is to find a representation $T$ of input $X$ that **maximizes the mutual information** with the target $Y$, while **minimizing the mutual information** with the input $X$. Mathematically:
$$\min_{p(t \mid x)} \; I(X; T) - \beta I(T; Y)$$
where:
- $I(X; T)$ = how much information $T$ keeps from the input $X$  → we want this to be **small** (**compression**)
- $I(T; Y)$ = how much information $T$ has about the label $Y$  → we want this to be **large** (**prediction**)
- $\beta$ controls the **trade-off** between compression and prediction

- **Upsampling Techniques:** In tasks requiring output at higher spatial resolutions (e.g., image segmentation), upsampling methods like transposed convolutions or bilinear interpolation are used. These techniques complement the bottleneck by restoring resolution while preserving the learned invariances.

![[upsampling.png]]
- **Convolutional Filters:** Convolutions enforce **weight sharing** by applying the same kernel across different spatial locations. This dramatically reduces the number of free parameters compared to fully connected layers and *naturally encodes translation invariance.*

- **Channels:** Multiple filters (channels) are applied in parallel to the input. Each channel learns to detect different features, and their combined output helps form a rich representation of the input while keeping the parameter count efficient.

*Convolutional Operations and Design Choices:*
**Stride:**
- Defines the step size of the filter during convolution.  
- A stride of $k$ means the kernel moves $k$ positions at a time.  
- **Effect:** Reduces the spatial size of the output feature map.  
- Larger stride = more aggressive downsampling.

**Kernel Size:**
- Number of input elements each output considers.  
- Controls the receptive field (i.e., how much of the input is "seen").  
- Kernel size $= 5$ means each output is influenced by 5 inputs.  
- Still uses only 5 parameters, regardless of input size.

**Dilated (Atrous) Convolutions:**
- Insert zeros between kernel weights.  
- Increase receptive field without increasing parameter count.  
- Useful for capturing broader context with fewer computations.

Collectively, these elements enforce **inductive biases** that restrict the model’s capacity to only what is necessary for the task, thus aiding generalization. **Inductive bias** is the set of assumptions a learning algorithm uses to predict outputs for unseen inputs. Inductive bias is what a learning algorithm _assumes_ about the data before seeing any examples. It's how a model “guesses” the right function out of infinitely many that could fit the data.

## 15. Lottery Ticket Hypothesis

(Zhang et al., 2018) An intriguing empirical finding.

**15.1. Concept:** A randomly initialized, dense, large network contains subnetworks ("winning tickets") that, when trained in isolation (from the same random initialization, keeping only the subnetwork weights), can achieve accuracy comparable to the original dense network trained for the same number of iterations.

**15.2. Implication:** Suggests that the effectiveness of large, overparameterized networks might stem partly from the high probability of containing a "good" subnetwork structure upon initialization. Finding these tickets efficiently (pruning) is an active research area. Underscores the importance of initialization and network structure beyond just parameter count.

# Part V: Advanced Architectures and Applications

## 16. Sequence Modeling: Recurrent Neural Networks (RNNs)
Designed for data with *sequential structure* (e.g., time series, language).

**16.1. Task:** Predict future elements based on past context.
- **Language Modeling:** Estimate $P(w_{i+1} \mid w_1, \ldots, w_i)$, where $w_i$ are tokens (words/subwords).
- **Sequence-to-Answer:** Map sequence $(x_1, \ldots, x_T)$ to a single output $y$ (e.g., sentiment classification).
- **Sequence-to-Sequence:** Map input sequence $(x_1, \ldots, x_T)$ to output sequence $(y_1, \ldots, y_{T'})$ (e.g., machine translation, tagging). $T$ can vary.

![[rnn.png]]

**16.2. Input Representation:**

- **Tokenization:** Map discrete symbols (words) to integer IDs from a vocabulary $V$.
- **One-Hot Encoding:** Represent token $w$ as a sparse vector in $\mathbb{R}^{|V|}$ with 1 at the index corresponding to $w$ and 0 elsewhere. High-dimensional and inefficient.
- **Embedding Matrix:** Learn a dense vector representation (embedding) for each token. Use an embedding matrix $A_0 \in \mathbb{R}^{D \times |V|}$ (or $|V| \times D$). The embedding for token $w_i$ (with one-hot vector $\mathbf{e}_i$) is $\mathbf{x}_i = A_0 \mathbf{e}_i$ (effectively selecting the $i$-th column/row of $A_0$). $A_0$ is learned via backpropagation. $D$ is the embedding dimension.

![[emb-matrix.png]]

**16.3. Basic RNN Dynamics:** Process sequence step-by-step, maintaining a hidden state $\mathbf{h}_t$ that summarizes past information.

**State Update:** The hidden state at time $t$ depends on the current input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$:

$$\mathbf{h}_t = \phi(A \mathbf{x}_t + C \mathbf{h}_{t-1} + \mathbf{b})$$

where $A, C, \mathbf{b}$ are learned parameters (input-to-hidden weights, hidden-to-hidden weights, hidden bias) and $\phi$ is a nonlinearity (often tanh or ReLU). $\mathbf{h}_0$ is typically initialized to $\mathbf{0}$.

**Output:** An output $\mathbf{y}_t$ can be generated at each step from the hidden state:

$$\mathbf{y}_t = \phi'(D \mathbf{h}_t + \mathbf{e})$$

where $D, \mathbf{e}$ are output parameters and $\phi'$ is an output activation. For sequence-to-answer, only the final output $\mathbf{y}_T$ might be used.

**Parameter Sharing:** Crucially, the *same parameters $(A, C, \mathbf{b}, D, \mathbf{e})$ are used at every time step $t$.* This allows RNNs to handle variable-length sequences and generalize across time.

**16.4. Residual Connections in RNNs:** Similar to feedforward networks, skip connections can improve training.

**Formula Example:** Add the previous state (potentially transformed) to the standard update:

$$\mathbf{h}_t = Q \mathbf{h}_{t-1} + \phi(A \mathbf{x}_t + C \mathbf{h}_{t-1} + \mathbf{b})$$

Helps gradient flow through time (mitigating vanishing/exploding gradients).

![[rnn-variants.png]]

**16.5. Recurrent Unit Abstraction:** The core recurrence can be abstracted as a function $R$:

$$(\mathbf{h}_t, \mathbf{y}_t) = R(\mathbf{h}_{t-1}, \mathbf{x}_t; \theta_R)$$

This encapsulates different RNN variants (Simple RNN, LSTM, GRU).

![[rnn-bidirection.png]]

**16.6. Gated Recurrent Units (GRU):** (Cho et al., 2014) Introduces gates to control information flow, aiming to capture longer-range dependencies.

- **Update Gate ($i_o$):** Decides how much of the previous state $\mathbf{h}_{t-1}$ to keep vs. the new candidate state $\tilde{\mathbf{h}}_t$. 

$$
i_o = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)
$$

- **Reset/Forger Gate ($i_f$):** Controls how much of the past state influences the candidate state.

$$
i_f = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)
$$

- **Candidate State ($\tilde{\mathbf{h}}_t$):** Computed based on current input and reset-gated previous state. 

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \mathbf{x}_t + \mathbf{U}(i_f \odot \mathbf{h}_{t-1}) + \mathbf{b})
$$

- **Final State:** Linear interpolation between previous state and candidate state, controlled by update gate.

$$
\mathbf{h}_t = (1 - i_o) \odot \mathbf{h}_{t-1} + i_o \odot \tilde{\mathbf{h}}_t
$$


![[gru.png]]
**16.7. Long Short-Term Memory (LSTM):** (Hochreiter & Schmidhuber, 1997) Another gated RNN, explicitly designed to prevent vanishing/exploding gradients and model long dependencies using a separate cell state $\mathbf{c}_t$.

- **Gates:** Three primary gates (Forget $f_t$, Input $i_t$, Output $o_t$), all computed using sigmoid $\sigma$ based on $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$.
- **Candidate Cell State ($\tilde{\mathbf{c}}_t$):** Computed using tanh based on $\mathbf{x}_t$ and $\mathbf{h}_{t-1}$.
- **Cell State Update:** Combines forgetting part of old state $\mathbf{c}_{t-1}$ and adding part of new candidate state:

$$\mathbf{c}_t = f_t \odot \mathbf{c}_{t-1} + i_t \odot \tilde{\mathbf{c}}_t$$

- **Hidden State Update:** Computed by applying the output gate to the (tanh-transformed) cell state:

$$\mathbf{h}_t = o_t \odot \tanh(\mathbf{c}_t)$$

**Intuition:** The cell state $\mathbf{c}_t$ acts as a memory conveyor belt, with gates controlling what information is removed, added, or read out to the hidden state $\mathbf{h}_t$.

![[lstm.png]]

**Equations**: 

$$\begin{aligned}
&\text{Forget gate:} \qquad && f_t = \sigma(W_f \mathbf{x}_t + U_f \mathbf{h}_{t-1} + b_f), \\
&\text{Input gate:}  \qquad && i_t = \sigma(W_i \mathbf{x}_t + U_i \mathbf{h}_{t-1} + b_i), \\
&\text{Output gate:} \qquad && o_t = \sigma(W_o \mathbf{x}_t + U_o \mathbf{h}_{t-1} + b_o), \\
&\text{Candidate cell:} \qquad && \tilde{c}_t = \tanh(W_c \mathbf{x}_t + U_c \mathbf{h}_{t-1} + b_c), \\
&\text{Cell state update:} \qquad && \mathbf{c}_t = f_t \odot \mathbf{c}_{t-1} + i_t \odot \tilde{c}_t, \\
&\text{Hidden state:} \qquad && \mathbf{h}_t = o_t \odot \tanh(\mathbf{c}_t).
\end{aligned}$$


**16.8. Highway Connections:** (Srivastava et al., 2015) Allow unimpeded information flow across layers/steps using gates. Let $H(\mathbf{x})$ be a standard transformation and $T(\mathbf{x})$ be a "transform gate" (e.g., sigmoid output). The output is:

$$\mathbf{y} = H(\mathbf{x}) \odot T(\mathbf{x}) + \mathbf{x} \odot (1 - T(\mathbf{x}))$$

If $T(\mathbf{x}) \approx 0$, input passes through; if $T(\mathbf{x}) \approx 1$, transformation is applied. Useful for very deep networks.

![[highway-connection.png]]
**Basic Cell:**
```python
def R_basic(h_prev, x, W, b):
    h = phi(W_x @ x + b_h + W_h @ h_prev)
    ...
    return h
```

**Highway Cell**
```python
def R_highway(h_prev, x, W, b):
    h_tilde = phi(W_x @ x + b_h + W_h @ h_prev)
    a = sigmoid(W_x_a @ x + b_a + W_h_a @ h_prev)
    h = gate(a, h_tilde, x)
    ...
    return h
```

## 17. Sequence Modeling: Attention and Transformers
Attention mechanisms overcome limitations of fixed-length context vectors or purely sequential RNN processing. Transformers rely entirely on attention.

**17.1. Motivation for Attention:**
- RNNs struggle with long-range dependencies (vanishing gradients, information bottleneck in single state vector).
- Sequential processing is inherently slow and not easily parallelizable over the time dimension.
- Attention allows modeling direct dependencies between distant elements in the sequence.

**17.2. Dot Product Attention (Simplified/Conceptual):**  
Given a sequence of input representations (e.g., hidden states) $\mathbf{h}_1, \ldots, \mathbf{h}_T$. To compute an output $\mathbf{y}_t$ that "attends" to all inputs:

1. **Attention Scores (Alignment):** Measure the relevance of each input $\mathbf{h}_k$ to the current position $t$ (or to a query derived from position $t$). A simple measure is the dot product:

$$e_{tk} = \mathbf{h}_t^T \mathbf{h}_k$$

(Requires $\mathbf{h}_t$ to act as a query).

2. **Softmax Normalization:** Convert scores into probabilities (attention weights) summing to 1:

$$a_{tk} = \frac{\exp(e_{tk})}{\sum_{j=1}^T \exp(e_{tj})}$$

3. **Output Computation (Context Vector):** Compute the output as a weighted sum of the input representations:

$$\mathbf{y}_t = \sum_{k=1}^T a_{tk} \mathbf{h}_k$$

The output $\mathbf{y}_t$ is a context vector dynamically weighted based on relevance scores.

**17.3. Query, Key, Value (QKV) Attention:** (Vaswani et al., 2017, "Attention Is All You Need") The standard mechanism in Transformers.

**Learned Projections:** Project the input sequence $X \in \mathbb{R}^{D \times T}$ (where columns are $\mathbf{x}_t$) into three matrices: Query $Q \in \mathbb{R}^{d_k \times T}$, Key $K \in \mathbb{R}^{d_k \times T}$, Value $V \in \mathbb{R}^{d_v \times T}$.

$$Q = W_Q X, \quad K = W_K X, \quad V = W_V X$$

where $W_Q, W_K \in \mathbb{R}^{d_k \times D}$, $W_V \in \mathbb{R}^{d_v \times D}$ are learned weight matrices. (Biases can also be added).

**Attention Scores:** Compute dot products between queries and keys:

$$\text{Scores} = Q^T K \in \mathbb{R}^{T \times T}$$

The entry $(t, k)$ represents the attention score from query $t$ to key $k$.

**Scaled Dot-Product Attention:** Scale scores before softmax for stability (prevents very large dot products when $d_k$ is large, which would lead to vanishing gradients in softmax):

$$A = \text{softmax}\left(\frac{Q^T K}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

$A$ is the attention weight matrix, $A_{tk}$ is weight from input $k$ to output $t$.

**Output Computation:** Multiply the attention weights by the Value matrix:

$$\text{Output} = V A^T \in \mathbb{R}^{d_v \times T}$$

The $t$-th column of the output is $\sum_k A_{tk} \mathbf{v}_k$.

**17.4. Multi-Head Attention:** Instead of one set of Q, K, V projections, use $h$ different sets ("heads"), computing attention in parallel:

$$\text{head}_i = \text{Attention}(W_{Q,i} X, W_{K,i} X, W_{V,i} X)$$

The outputs of the heads (each in $\mathbb{R}^{d_v \times T}$, where $d_v = D_{\text{model}} / h$) are concatenated and then linearly projected back to the original dimension $D_{\text{model}}$ with matrix $W_O$:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O^T$$

Allows the model to attend to information from different representation subspaces simultaneously.

**17.5. Self-Attention:** When Q, K, V are all derived from the same input sequence $X$ (as described above). This allows positions in the sequence to attend to other positions within the same sequence.

**17.6. Positional Encoding:** Since self-attention itself is permutation-equivariant (doesn't inherently know the order of inputs), explicit positional information must be added.

**Sinusoidal Encoding:** A common fixed (non-learned) method. Add a vector $\mathbf{p}_t \in \mathbb{R}^D$ to each input embedding $\mathbf{x}_t$: $\mathbf{x}_t' = \mathbf{x}_t + \mathbf{p}_t$. The components of $\mathbf{p}_t$ are defined using sine and cosine functions of different frequencies depending on the position $t$ and dimension $i$:

$$p_{t,2i} = \sin(t / 10000^{2i/D})$$
$$p_{t,2i+1} = \cos(t / 10000^{2i/D})$$

**Rationale:** Relative positions can be captured by linear transformations of these encodings. Learned positional embeddings are also common.

![[pos-encoding.png]]

**17.7. Transformer Architecture (Decoder/Language Model Focus):** Stacks multiple layers, each typically containing Multi-Head Self-Attention and a Position-wise Feed-Forward Network (FFN). Residual connections and Layer Normalization are crucial.

**Layer Structure:**

1. Input $X_{l-1}$.
2. **Layer Normalization:** $X^\circ = \text{LayerNorm}(X_{l-1})$.
3. **Multi-Head Self-Attention (Masked for LM):** $Z = \text{MultiHead}(X^\circ)$. Masking ensures position $t$ only attends to positions $k \leq t$.
4. **Residual Connection 1:** $R = X_{l-1} + \text{Dropout}(Z)$.
5. **Layer Normalization:** $R^\circ = \text{LayerNorm}(R)$.
6. **Position-wise FFN:** Two linear layers with a ReLU/GELU activation in between, applied independently to each position $t$. $\text{FFN}(\mathbf{r}_t^\circ) = \max(0, \mathbf{r}_t^\circ W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$. Let the output be $F$.
7. **Residual Connection 2:** $X_l = R + \text{Dropout}(F)$.

**Input:** Token Embedding + Positional Encoding.

**Output:** Final layer output $X_L$.

**17.8. Transformer Training (Language Model):**

**Objective:** Predict the next token in a sequence (autoregressive). Given $\mathbf{x}_1, \ldots, \mathbf{x}_{T-1}$, predict $w_T$.

**Output Projection:** Map final layer outputs $X_L \in \mathbb{R}^{D \times T}$ to vocabulary logits $P \in \mathbb{R}^{V \times T}$. Typically uses a linear layer: $P = W_{\text{out}} X_L$.

**Weight Tying:** Often, the output projection weights $W_{\text{out}}$ are tied to the input embedding matrix $A_0$ (i.e., $W_{\text{out}} = A_0^T$ if $A_0$ is $|V| \times D$). Reduces parameters, can improve performance.

**Loss Function:** Calculate cross-entropy loss between predicted probabilities (softmax of logits $P_{:,t}$) and the actual next token $w_{t+1}$, summed/averaged over the sequence:

$$L = \sum_{t=1}^{T-1} \text{CrossEntropy}(\text{softmax}(P_{:,t}), w_{t+1})$$

**Self-Supervised:** Trained on vast amounts of text data without explicit labels, using the text itself to provide supervision.

**17.9. Decoding/Generation Strategies:** Generating sequences from a trained LM.

- **Greedy Sampling:** Pick the most likely token at each step.
- **Random Sampling:** Sample from the predicted probability distribution $p(w_t \mid w_{<t})$. Temperature scaling ($\tau$) can control randomness: $\text{softmax}(\text{logits} / \tau)$.
- **Beam Search:** Maintain $k$ most probable partial sequences (beams) at each step, expanding each and keeping the top $k$ overall. Finds higher probability sequences than greedy.
- **RLHF (Reinforcement Learning from Human Feedback):** Fine-tune LM using RL based on human preferences for generated outputs.

**17.10. Transformer Applications:**

- **Text Classification:** Prepend a special [CLS] token. Use its final hidden state $X_{\text{CLS}}^L$ as input to a classification MLP head. Fine-tune the entire model.
- **Machine Translation (Encoder-Decoder):**
  - **Encoder:** Processes source sequence using self-attention layers.
  - **Decoder:** Processes target sequence using masked self-attention. Adds a Cross-Attention layer in each block, where Queries $Q$ come from the decoder's state, and Keys $K$ and Values $V$ come from the encoder's final output. This allows the decoder to attend to relevant parts of the source sentence while generating the translation.

![[cross-attention.png]]

## 18. Vision Transformers (ViT)
(Dosovitskiy et al., 2020) Adapting Transformers to image data.

**18.1. Concept:**
- Divide image into a grid of non-overlapping patches (e.g., 16x16 pixels).
- Linearly embed each patch into a vector (e.g., flatten patch, multiply by weight matrix).
- Add positional embeddings (usually learned) to patch embeddings.
- Optionally prepend a learnable [CLS] token embedding.
- Feed this sequence of vectors (patch embeddings + [CLS] token) into a standard Transformer encoder.
- Use the output embedding of the [CLS] token for image classification (fed into an MLP head).

![[viT.png]]

**18.2. Integration with CNNs:** Hybrid approaches exist. CNNs might extract initial features (e.g., ResNet stem), and a Transformer processes the resulting feature maps (treated as a sequence). Attention mechanisms can also be added within CNN architectures.

**18.3. Training without Large Pretraining:** ViTs typically require large datasets (like ImageNet-21k or JFT-300M) for pretraining to perform well, unlike CNNs which have stronger inductive biases (locality, translation equivariance). If large pretraining is unavailable:

- **Regularization:** Heavy use of Weight Decay, Stochastic Depth, Dropout is crucial.
- **Data Augmentation:** Advanced techniques like MixUp (interpolating images and labels) and RandAugment (applying random sequences of augmentations) are vital for good performance when training from scratch on smaller datasets like ImageNet-1k.

**18.4. Distillation:** Improve ViT performance by training it to match the output logits (or feature representations) of a pre-trained, powerful teacher model (often a CNN like ResNet or EfficientNet). Uses a distillation loss term alongside the standard cross-entropy loss.

## 19. Generative Models

Learning the underlying probability distribution $p(\mathbf{x})$ of the data.

**19.1. Goal:** Learn a model $p_\theta(\mathbf{x})$ that approximates the true data distribution $p_{\text{data}}(\mathbf{x})$. Allows generating new samples $\mathbf{x} \sim p_\theta(\mathbf{x})$. Often optimized by maximizing the log-likelihood $\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_\theta(\mathbf{x})]$ on the training data.

**19.2. Variational Autoencoders (VAEs):** (Kingma & Welling, 2013; Rezende et al., 2014) Latent variable models that use neural networks for both generation (decoder) and inference (encoder).

**Latent Variable Model:** Assume data $\mathbf{x}$ is generated from a latent variable $\mathbf{z}$ via a process:
1. Sample $\mathbf{z} \sim p(\mathbf{z})$ (prior, typically $\mathcal{N}(0, I)$).
2. Sample $\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})$ (likelihood/decoder).

The marginal likelihood is $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$. This integral is usually intractable.

![[vaes.png]]

**Decoder:** A neural network $f_{\text{dec}}(\mathbf{z}; \theta)$ that outputs parameters of the distribution $p_\theta(\mathbf{x} \mid \mathbf{z})$. For continuous data, often Gaussian: $p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{x} \mid \mu_\theta(\mathbf{z}), \Sigma_\theta(\mathbf{z}))$.

**Encoder (Amortized Inference):** The true posterior $p_\theta(\mathbf{z} \mid \mathbf{x}) = p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}) / p_\theta(\mathbf{x})$ is also intractable. Introduce an approximate posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ (encoder), typically modeled as a Gaussian $\mathcal{N}(\mathbf{z} \mid \mu_\phi(\mathbf{x}), \Sigma_\phi(\mathbf{x}))$, where a network $f_{\text{enc}}(\mathbf{x}; \phi)$ outputs $\mu_\phi$ and $\Sigma_\phi$.

**Training Objective (Maximize ELBo):** Maximize the Evidence Lower Bound (ELBo) on the marginal log-likelihood $\log p_\theta(\mathbf{x})$:

$$\mathcal{L}(\mathbf{x}; \theta, \phi) = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z})) \leq \log p_\theta(\mathbf{x})$$

- **Reconstruction Term:** $\mathbb{E}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$ encourages the decoder to reconstruct $\mathbf{x}$ from latent codes $\mathbf{z}$ sampled from the encoder's approximation. Evaluated using Monte Carlo sampling and the reparameterization trick (sample $\epsilon \sim \mathcal{N}(0, I)$, then $\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon$ where $\Sigma = \text{diag}(\sigma^2)$).
- **KL Divergence Term:** $D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$ acts as a regularizer, pushing the approximate posterior $q_\phi$ towards the prior $p(\mathbf{z})$. Often analytically computable if both are Gaussian.

**Generation:** Sample $\mathbf{z} \sim p(\mathbf{z})$, then sample $\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})$ (or just use $\mu_\theta(\mathbf{z})$).

**19.3. Generative Adversarial Networks (GANs):** (Goodfellow et al., 2014) Implicit generative models based on a game between two networks.

- **Generator ($G$):** Maps noise $\mathbf{z}$ (from prior $p(\mathbf{z})$) to data space, $G(\mathbf{z}; \theta_G)$, trying to produce realistic samples.
- **Discriminator ($D$):** Tries to distinguish real data $\mathbf{x} \sim p_{\text{data}}$ from fake data $G(\mathbf{z})$. Outputs probability $D(\mathbf{x}; \theta_D)$ that $\mathbf{x}$ is real.

**Objective (Minimax Game):**

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

$D$ tries to maximize the objective (correct classification), $G$ tries to minimize it (fool $D$).

**Variants:** DCGAN (uses CNNs), Progressive GANs (grow resolution), Conditional GANs (generate based on labels $c$: $G(\mathbf{z}, c)$, $D(\mathbf{x}, c)$), WGAN, StyleGAN, etc.

![[dcgans.png]]

**19.4. Diffusion Models:** (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., Score-Based Models) State-of-the-art generative models.

**Forward Process (Diffusion):** Gradually add Gaussian noise to data $\mathbf{x}_0$ over $T$ steps, producing noisy versions $\mathbf{x}_1, \ldots, \mathbf{x}_T$. $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t \mid \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t I)$, where $\beta_t$ is a noise schedule. $\mathbf{x}_T$ approaches pure noise $\mathcal{N}(0, I)$.

**Reverse Process (Denoising):** Learn a model $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to reverse this process. Parameterized by a neural network (often U-Net based) $f_\theta(\mathbf{x}_t, t)$ that typically predicts the noise added at step $t$, or $\mathbf{x}_0$.

**Training:** Train $f_\theta$ to predict the noise $\epsilon_t$ added to get $\mathbf{x}_t$ from $\mathbf{x}_0$ (using properties of the forward process): $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$, where $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$. Objective often minimizes MSE: $\mathbb{E}_{t, \mathbf{x}_0, \epsilon}[\|\epsilon - f_\theta(\mathbf{x}_t, t)\|^2]$.

**Generation:** Start with noise $\mathbf{x}_T \sim \mathcal{N}(0, I)$, iteratively sample $\mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ for $t = T, \ldots, 1$ using the learned model $f_\theta$.

**Architecture:** U-Net is common for $f_\theta$ due to its ability to handle multi-scale features, crucial for image generation. Often incorporates time embedding $t$ and self-attention mechanisms.

![[unet-diff.png]]

# Part VI: Summary and Synthesis

## 20. Key Takeaways and Recurring Themes

**20.1. Core Idea:** Machine learning, particularly deep learning, focuses on learning a parameterized function $f(\mathbf{x}; \theta)$ by minimizing a loss function $L$ that measures discrepancy between predictions $f(\mathbf{x}_i; \theta)$ and observations $y_i$ over a dataset $D$.

**20.2. Function Construction:** The choice of $f$ evolves from simple linear models, to linear combinations of basis functions, to shallow and deep neural networks, representing increasingly complex and adaptive function classes.

**20.3. Importance of Nonlinearity:** Nonlinear activation functions $\phi$ are essential for the expressive power of multi-layer networks, allowing them to approximate complex functions beyond simple linear mappings.

**20.4. Training:** Gradient-based optimization (GD, SGD, Adam) is the workhorse. Backpropagation, leveraging the chain rule via efficient Vector-Jacobian Products (VJPs), is the fundamental algorithm for computing gradients in deep networks.

**20.5. Practicalities:** Success relies heavily on careful initialization (e.g., He init for ReLU), input/activation scaling, numerical stability considerations (e.g., log-sum-exp trick), and regularization techniques (L2, dropout, early stopping, data augmentation) to prevent overfitting and ensure stable training.

**20.6. Architectural Evolution:** Progression from simple MLPs to architectures specialized for different data types: CNNs for grid data (images), RNNs (LSTM/GRU) for sequences, and Attention/Transformers providing parallelizable and powerful sequence modeling and beyond (ViT).

**20.7. Inductive Biases:** Architectures often incorporate prior assumptions about the data: CNNs assume locality and translation equivariance; RNNs assume sequential dependence; Attention focuses on pairwise interactions. ViTs initially had weaker inductive biases, requiring more data or specific regularization.

**20.8. Trade-offs:** Key decisions involve balancing model capacity (expressiveness) against the risk of overfitting, and choosing between computational paradigms (sequential RNNs vs. parallel Transformers).

**20.9. Applications:** These principles and techniques underpin successes in diverse areas including regression, classification, natural language processing (translation, generation), computer vision (classification, generation), and scientific discovery (e.g., AlphaFold).