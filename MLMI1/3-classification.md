# Classification

Classification $\rightarrow$ predicting a **discrete output** $y^\ast$ for a given input $x^\ast$, based on a training set of input-output pairs $\{(x_n, y_n)\}_{n=1}^N$. 

**Classification Goal**: To find a function $f: \mathbb{R}^D \rightarrow \mathbb{R}^{K}$ that maps inputs to probabilities over $K$ classes, allowing us to make future predictions by assigning $x^\star$ to one of the $K$ classes.

---

# Binary Logistic Classification (K=2)

Logistic regression models the probability that a given input $x_n$ belongs to class $y_n = 1$ (assuming binary classes $y_n \in \{0, 1\}$).

- **Linear Combination**:
  $$a_n = w^\top x_n = \sum_{d=1}^D w_d x_{n,d}$$
  - $w$: Weight vector.
  - $a_n$: Linear activation for input $x_n$.

- **Sigmoid Function**:
  $$p(y_n = 1 \mid x_n, w) = \sigma(a_n) = \frac{1}{1 + \exp(-a_n)}$$
  - Maps the linear activation $a_n$ to a probability between 0 and 1.

![Linear Neuron](linear_neuron.gif)

![Single Neuron](single_neuron.gif)

### Decision Boundary

- The decision boundary is defined where $p(y_n = 1 \mid x_n, w) = 0.5$, which corresponds to $a_n = 0$.
- The orientation of the decision boundary is determined by $w$.

### Maximum Likelihood Estimation for Logistic Regression

To find the optimal weights $w$, we use maximum likelihood estimation.

#### Likelihood Function

Given the training data $\{(x_n, y_n)\}_{n=1}^N$, the likelihood is:
$$L(w) = \prod_{n=1}^N p(y_n \mid x_n, w)$$

For binary outputs:
$$p(y_n \mid x_n, w) = \sigma(a_n)^{y_n} [1 - \sigma(a_n)]^{1 - y_n}$$

Thus, the likelihood becomes:
$$L(w) = \prod_{n=1}^N \sigma(a_n)^{y_n} [1 - \sigma(a_n)]^{1 - y_n}$$

#### Log-Likelihood Function

Taking the logarithm simplifies calculations:
$$\ell(w) = \sum_{n=1}^N \left[ y_n \log \sigma(a_n) + (1 - y_n) \log (1 - \sigma(a_n)) \right]$$

#### Gradient of the Log-Likelihood

To maximize $\ell(w)$, compute its gradient:
$$\nabla_w \ell(w) = \sum_{n=1}^N (y_n - \sigma(a_n)) x_n$$

The gradient is a sum over the training examples, weighted by the difference between the actual label $y_n$ and the predicted probability $\sigma(a_n)$.

#### Optimization via Gradient Ascent

Update the weights iteratively:
$$w_{t+1} = w_t + \eta \nabla_w \ell(w_t)$$

- $\eta$: Learning rate.
- Continue until convergence.

![Single Neuron](class_1d_animation.gif)
### Convergence and Uniqueness

- The log-likelihood function is concave, ensuring a unique global maximum.
- The Hessian matrix is negative definite:
  $$\nabla_w^2 \ell(w) = - \sum_{n=1}^N \sigma(a_n) [1 - \sigma(a_n)] x_n x_n^\top$$

### Decision Boundaries and Thresholds

#### Predicting Class Labels

- Default rule: Predict $y_n = 1$ if $p(y_n = 1 \mid x_n, w) \ge 0.5$.
- Decision boundary is where $a_n = 0$.

#### Adjusting for Misclassification Costs

- In some cases, misclassification costs are asymmetric.
- Adjust the threshold to minimize expected cost:
  $$\text{Threshold} = \frac{L(0, 0) - L(0, 1)}{[L(0, 0) - L(0, 1)] + [L(1, 1) - L(1, 0)]}$$

- $L(y, \hat{y})$: Loss function representing the cost of predicting $\hat{y}$ when the true label is $y$.

---

# Multi-class Softmax Classification (K>2)

We extend the binary logistic regression model to handle multiple classes by introducing the **softmax classification model**. In this model, each data point consists of an input vector $\mathbf{x}_n$ and an output label $y_n$, where $y_n \in \{1, 2, \dots, K\}$ indicates the class of the $n$-th data point.

## The Softmax Classification Model

The model comprises two stages:

1. **Computing Activations**:

   For each class $k$, compute the activation:
   $$
   a_{n,k} = \mathbf{w}_k^\top \mathbf{x}_n
   $$

   where $\mathbf{w}_k$ is the weight vector associated with class $k$.

2. **Softmax Function**:

   The activations are passed through the softmax function to obtain the probability that data point $\mathbf{x}_n$ belongs to class $k$:
   $$
   p(y_n = k \mid \mathbf{x}_n, \{\mathbf{w}_k\}_{k=1}^K) = \frac{\exp(a_{n,k})}{\sum_{k'=1}^K \exp(a_{n,k'})} = \frac{\exp(\mathbf{w}_k^\top \mathbf{x}_n)}{\sum_{k'=1}^K \exp(\mathbf{w}_{k'}^\top \mathbf{x}_n)}
   $$

By construction, the probabilities are normalized:

$$
\sum_{k=1}^K p(y_n = k \mid \mathbf{x}_n, \{\mathbf{w}_k\}_{k=1}^K) = 1
$$

Thus, the softmax function parameterizes a categorical distribution over the output classes.

![[Pasted image 20241107180455.png]]
![[Pasted image 20241107180525.png]]
## Fitting Using Maximum Likelihood Estimation

To estimate the weight vectors $\{\mathbf{w}_k\}$, we use maximum likelihood estimation (MLE).

### Likelihood Function

We first represent the output labels using one-hot encoding. For each data point $n$, the label $y_n$ is encoded as a vector $\mathbf{y}_n$ of length $K$:

$$
y_{n,k} =
\begin{cases}
1 & \text{if } y_n = k \\
0 & \text{otherwise}
\end{cases}
$$

The likelihood of the parameters given the data is:

$$
p(\{ y_n \}_{n=1}^N \mid \{ \mathbf{x}_n \}_{n=1}^N, \{ \mathbf{w}_k \}_{k=1}^K) = \prod_{n=1}^N \prod_{k=1}^K s_{n,k}^{y_{n,k}}
$$

where:

$$
s_{n,k} = p(y_n = k \mid \mathbf{x}_n, \{\mathbf{w}_k\}_{k=1}^K)
$$

### Log-Likelihood Function

Taking the logarithm of the likelihood simplifies the product into a sum:

$$
\mathcal{L}(\{ \mathbf{w}_k \}_{k=1}^K) = \sum_{n=1}^N \sum_{k=1}^K y_{n,k} \log s_{n,k}
$$

### Gradient of the Log-Likelihood

To maximize the log-likelihood, we compute its gradient with respect to each weight vector $\mathbf{w}_j$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_j} = \sum_{n=1}^N (y_{n,j} - s_{n,j}) \mathbf{x}_n
$$

### Optimization via Gradient Ascent

We can use gradient ascent to iteratively update the weights:

$$
\mathbf{w}_j^{(t+1)} = \mathbf{w}_j^{(t)} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}_j}
$$

where $\eta$ is the learning rate.


---

# Non-linear Classification

Linear classification models are limited to linear decision boundaries, which may not be suitable for complex datasets. To handle non-linear decision boundaries, we introduce **non-linear basis functions**.

## Non-linear Classification through Basis Functions

We enhance the model by transforming the input features into a higher-dimensional space using non-linear basis functions.

### Basis Function Expansion

Define a set of basis functions $\boldsymbol{\Phi}(\mathbf{x}_n) = [\phi_1(\mathbf{x}_n), \phi_2(\mathbf{x}_n), \dots, \phi_D(\mathbf{x}_n)]^\top$.

Compute the activation:

$$
a_n = \mathbf{w}^\top \boldsymbol{\Phi}(\mathbf{x}_n)
$$

The probability is then:

$$
p(y_n = 1 \mid \mathbf{x}_n, \mathbf{w}) = \sigma(a_n) = \frac{1}{1 + \exp(-a_n)}
$$

### Radial Basis Functions (RBFs)

A common choice for basis functions is radial basis functions:

$$
\phi_d(\mathbf{x}) = \exp\left( -\frac{\|\mathbf{x} - \boldsymbol{\mu}_d\|^2}{2l^2} \right)
$$

- $\boldsymbol{\mu}_d$ is the center of the $d$-th basis function.
- $l$ is the length-scale parameter controlling the width.

### Overfitting in Non-linear Classification

Using too many basis functions or a small length-scale can lead to overfitting:

- The model becomes overly complex, fitting the noise in the training data.
- Poor generalization to unseen data.

### Visualizing Predictions

By plotting the probability contours or decision boundaries, we can observe how the non-linear model captures complex patterns in the data.


---
## Mitigating Overfitting

- **Regularization**: Introduce penalty terms in the loss function to discourage large weights.
- **Cross-Validation**: Use validation sets to monitor performance and select model parameters.
- **Early Stopping**: Halt training when performance on validation data starts to degrade.

