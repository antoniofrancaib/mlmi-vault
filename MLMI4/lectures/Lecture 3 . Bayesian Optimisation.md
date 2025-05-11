# --FIRST HALF-- (and last bit of last lecture)
Lecture by Jose Miguel Hernandez Lobato

# 1. Introduction and Setting

When confronted with a black-box objective function $f(\mathbf{x})$ whose internal mechanics are unknown or very expensive to evaluate, Bayesian Optimization (BO) provides a systematic strategy for finding the global minimum (or maximum, with trivial sign-flips) efficiently. The classic workflow:

1. Maintain a probabilistic surrogate model over the unknown function $f$.
2. Derive an acquisition function that balances exploration (searching where uncertainty is high) and exploitation (searching where the model predicts good objective values).
3. Select a candidate $\mathbf{x}^*$ by maximizing or minimizing the acquisition function.
4. Evaluate the black-box at $\mathbf{x}^*$ (incurring the expensive cost).
5. Update the surrogate model with the newly observed data.
6. Repeat until resources are exhausted or the optimum is found.

The overarching idea is to minimize the number of costly evaluations of the black-box function by leveraging a carefully chosen model (often a Gaussian Process) and an intelligent, uncertainty-aware strategy (the acquisition function).

# 2. Gaussian Process Model for BO

## 2.1 Gaussian Process Basics

The Gaussian Process (GP) is the most common surrogate model in BO, owing to:

- Analytic tractability of its predictive distributions.
- Flexible means of capturing correlation structure in the data (through a kernel or covariance function).
- Closed-form expressions for many key quantities, including marginal likelihoods and posterior predictive means and covariances.

A Gaussian Process defines a prior over functions:

$$
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), C(\mathbf{x}, \mathbf{x}'; \boldsymbol{\theta})),
$$

where $m(\mathbf{x})$ is often taken as zero (for notational convenience) and $C(\cdot, \cdot; \boldsymbol{\theta})$ is the covariance function (or kernel) parameterized by hyper-parameters $\boldsymbol{\theta}$.

### Joint Distribution Over Training and Test Points

Given:

- Training inputs $\{\mathbf{x}_n\}_{n=1}^N$ with observed outputs $\{y_n\}_{n=1}^N$.
- Test inputs $\{\mathbf{x}_m^*\}_{m=1}^M$ whose outputs $\{y_m^*\}_{m=1}^M$ we want to predict.

The joint distribution of the (unknown) test outputs $\mathbf{y}^*$ and the (observed) training outputs $\mathbf{y}$ under a GP is:

$$
p(\mathbf{y}^*, \mathbf{y}) = \mathcal{N}\left(
\begin{bmatrix}
\mathbf{0} \\
\mathbf{0}
\end{bmatrix},
\begin{bmatrix}
\boldsymbol{\kappa}_{\boldsymbol{\theta}} & \mathbf{k}_{\boldsymbol{\theta}}^\top \\
\mathbf{k}_{\boldsymbol{\theta}} & \mathbf{K}_{\boldsymbol{\theta}}
\end{bmatrix}
\right),
$$

where:

- $\mathbf{K}_{\boldsymbol{\theta}}$ is the $N \times N$ covariance matrix of the training inputs.
- $\mathbf{k}_{\boldsymbol{\theta}}$ is the $N \times M$ cross-covariance matrix between training points and test points.
- $\boldsymbol{\kappa}_{\boldsymbol{\theta}}$ is the $M \times M$ covariance matrix for the test points themselves.

Each element is computed by evaluating the covariance function $C(\mathbf{x}_i, \mathbf{x}_j; \boldsymbol{\theta})$.

## 2.2 GP Predictive Distribution

Conditioning on observed data $\{\mathbf{x}_n, y_n\}$, the posterior predictive distribution for the unknown test outputs $\mathbf{y}^*$ is itself Gaussian:

$$
\mathbf{y}^* \mid \mathbf{y} \sim \mathcal{N}(\mathbf{m}, \boldsymbol{\Sigma}),
$$

with

$$
\mathbf{m} = \mathbf{k}_{\boldsymbol{\theta}}^\top \mathbf{K}_{\boldsymbol{\theta}}^{-1} \mathbf{y}, \quad \boldsymbol{\Sigma} = \boldsymbol{\kappa}_{\boldsymbol{\theta}} - \mathbf{k}_{\boldsymbol{\theta}}^\top \mathbf{K}_{\boldsymbol{\theta}}^{-1} \mathbf{k}_{\boldsymbol{\theta}}.
$$

These closed-form expressions enable rapid, exact Bayesian posterior updates after each new observation.

## 2.3 Hyper-Parameter Learning

The hyper-parameters $\boldsymbol{\theta}$ of the covariance function can be chosen in different ways. A common approach is maximum marginal likelihood, where we maximize

$$
\log p(\mathbf{y} \mid \boldsymbol{\theta}) = -\frac{N}{2} \log(2\pi) - \frac{1}{2} \log|\mathbf{K}_{\boldsymbol{\theta}}| - \frac{1}{2} \mathbf{y}^\top \mathbf{K}_{\boldsymbol{\theta}}^{-1} \mathbf{y}.
$$

Alternatively, one can place a prior over $\boldsymbol{\theta}$ and perform MCMC to sample from $p(\boldsymbol{\theta} \mid \mathbf{y})$, then form a predictive acquisition by averaging over these samples. This is sometimes preferable because maximizing the marginal likelihood alone can lead to overconfident predictions.

# 3. Examples of Covariance Functions

Covariance functions encode our assumptions about smoothness, periodicity, or other structure in the unknown function $f$. Common examples include:

- **Squared Exponential (RBF) Kernel**: Encourages very smooth functions.
- **Matérn Family**: Offers tunable smoothness, controlling the differentiability of sampled functions.
- **Others** (not explicitly detailed in the provided slides, but commonly known): Periodic, Rational Quadratic, etc.

The choice of kernel has a strong impact on performance and must reflect prior beliefs about how $f$ behaves.

# 4. Alternative Models to GPs

Despite their usefulness, Gaussian Processes have some drawbacks:

- **Computational Scalability**: Exact GP inference is $\mathcal{O}(N^3)$ in the number of observations $N$. This quickly becomes prohibitive for large datasets.
- **Strong Dependence on Kernel Choice**: Performance can degrade significantly if the chosen covariance function is a poor fit.
- **Kernel Form Not Learned from Scratch**: Standard GP regression requires you to assume a kernel structure rather than learning it automatically from data.

## 4.1 Sparse Gaussian Processes

A popular remedy for the $\mathcal{O}(N^3)$ cost is to use sparse GPs with a set of $M$ inducing inputs (where $M \ll N$). This yields computational savings by approximating the full GP posterior with a lower-rank representation, reducing cost to $\mathcal{O}(NM^2)$. The intuition is that you maintain a smaller, representative set of "virtual" points that approximate the full covariance structure.

## 4.2 Bayesian Neural Networks (BNNs)

Bayesian neural networks can, at least in theory:

- Scale better with data size.
- Learn flexible function representations (addressing kernel engineering).
- Potentially handle nonstationary behaviors more naturally.

However, they are:

- Intractable under exact Bayesian inference; approximate methods (e.g., variational inference, Hamiltonian Monte Carlo) must be used.
- Often more complicated to implement reliably compared to GPs.

A common approximate strategy is to sample weights $\{\mathbf{W}_k\}_{k=1}^K$ from the posterior $p(\mathbf{W} \mid \mathcal{D})$ and then average the predictions:

$$
p(y \mid \mathbf{x}, \mathcal{D}) \approx \frac{1}{K} \sum_{k=1}^K p(y \mid \mathbf{x}, \mathbf{W}_k).
$$

Although more flexible than GPs, these methods are typically more complex and can require significant expertise.

# 5. Acquisition Functions

Bayesian Optimization can be posed as a multi-period planning problem solved by dynamic programming:

$$
\alpha_N(\mathbf{x}_1) = \int \left[ \prod_{n=1}^N p(y_n \mid \mathbf{x}_n, \mathcal{D} \cup \{\mathbf{x}_m, y_m\}_{m=1}^{n-1}) \, p(\mathbf{x}_n \mid \mathcal{D} \cup \{\mathbf{x}_m, y_m\}_{m=1}^{n-1}) \right] U[y_N \mid \mathbf{x}_N, \mathcal{D} \cup \{\mathbf{x}_n, y_n\}_{n=1}^{N-1}] \, d\mathbf{x}_2 \cdots d\mathbf{x}_N \, dy_1 \cdots dy_N,
$$

subject to

$$
p(\mathbf{x}_n \mid \mathcal{D} \cup \{\mathbf{x}_m, y_m\}_{m=1}^{n-1}) = \delta(\mathbf{x}_n - \arg\max_{\mathbf{x}} \alpha_{N-n+1}(\mathbf{x})).
$$

Exact dynamic programming is intractable for typical problems, because of the nested integrals and nested $\arg\max$. Thus, we generally employ a myopic (single-step) approach. We approximate the acquisition function by considering only the immediate utility of an evaluation:

$$
\alpha_1(\mathbf{x}) = \int U[y \mid \mathbf{x}, \mathcal{D}] \, p(y \mid \mathbf{x}, \mathcal{D}) \, dy.
$$

Below are some of the most common acquisition functions, assuming a Gaussian predictive $p(y \mid \mathbf{x}, \mathcal{D}) = \mathcal{N}(y \mid \mu(\mathbf{x}), \sigma^2(\mathbf{x}))$. Define:

- $\eta = \mu(\mathbf{x}_{\text{rec}})$, where $\mathbf{x}_{\text{rec}} = \arg\min_{\mathbf{x}} \mu(\mathbf{x})$ (the current best predicted value).
- $\gamma(\mathbf{x}) = \frac{\eta - \mu(\mathbf{x})}{\sigma(\mathbf{x})}$.

Let $\Phi(\cdot)$ and $\phi(\cdot)$ denote the standard Gaussian CDF and PDF, respectively.

## 5.1 Probability of Improvement (PI)

- **Utility**: $U(y \mid \mathbf{x}, \mathcal{D}) = \mathbb{1}(y < \eta)$.
- **Acquisition**:
  $$
  \alpha_{\text{PI}}(\mathbf{x}) = \Phi(\gamma(\mathbf{x})).
  $$

PI measures how likely the new evaluation will produce an objective value better (lower, in a minimization problem) than $\eta$. It can be too conservative, however, because it only measures probability, not magnitude of improvement.

## 5.2 Expected Improvement (EI)

- **Utility**: $U(y \mid \mathbf{x}, \mathcal{D}) = \max(0, \eta - y)$.
- **Acquisition**:
  $$
  \alpha_{\text{EI}}(\mathbf{x}) = \sigma(\mathbf{x}) \left[ \gamma(\mathbf{x}) \Phi(\gamma(\mathbf{x})) + \phi(\gamma(\mathbf{x})) \right].
  $$

EI is a very popular choice since it accounts for both the probability and the expected magnitude of improvement over $\eta$. Often yields balanced exploration and exploitation.

## 5.3 Lower Confidence Bound (LCB)

- **Acquisition**:
  $$
  \alpha_{\text{LCB}}(\mathbf{x}) = -\left[ \mu(\mathbf{x}) - \kappa \sigma^2(\mathbf{x}) \right],
  $$
  where $\kappa \geq 0$.

LCB encourages exploration by penalizing large posterior uncertainties ($\sigma^2(\mathbf{x})$) and exploitation by favoring low mean. Tuning $\kappa$ adjusts the trade-off between these two.

## 5.4 Entropy Search (ES)

- **Utility**:
  $$
  U[y \mid \mathbf{x}, \mathcal{D}] = H[p(\mathbf{x}^* \mid \mathcal{D})] - H[p(\mathbf{x}^* \mid \mathcal{D} \cup \{\mathbf{x}, y\})],
  $$
  where $H(\cdot)$ denotes entropy and $\mathbf{x}^*$ is the global minimizer.

ES aims to reduce uncertainty in the location of the minimizer $\mathbf{x}^*$. It often performs well in practice but can be computationally expensive, as it involves approximating how an observation at $\mathbf{x}$ will refine the posterior over $\mathbf{x}^*$.

# 6. Thompson Sampling (A Non-Standard Acquisition)

Instead of explicitly constructing an acquisition function, Thompson Sampling samples from the posterior over possible functions $f(\cdot)$. One draws a random function sample $f'$ from $p(f \mid \mathcal{D})$ and picks

$$
\mathbf{x} = \arg\min_{\mathbf{x}'} f'(\mathbf{x}').
$$

This approach:

- **Exploits**: On average, a sample from $p(f \mid \mathcal{D})$ reflects the current best guess of the objective.
- **Explores**: Because each sample is random, different draws create different possible minima, driving exploration into uncertain regions.
- Is conceptually simple and often performs well with easy sampling mechanisms (e.g., approximate function samples from GPs or from BNN weight posteriors).

# 7. Performance and Hyper-Parameter Selection

In empirical comparisons (e.g., on synthetic tasks sampled from GP priors), Entropy Search can sometimes outperform other acquisition strategies but may be too costly in high dimensions. Expected Improvement (EI) remains the most widely used, balancing efficiency and effectiveness.

When learning GP hyper-parameters, maximizing $\log p(\mathbf{y} \mid \boldsymbol{\theta})$ can produce overconfident predictive distributions. An alternative is MCMC sampling for $\boldsymbol{\theta}$. In practice, we can average the acquisition function across samples:

$$
\alpha(\mathbf{x}) = \int \alpha(\mathbf{x} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \mathbf{y}) \, d\boldsymbol{\theta} \approx \frac{1}{K} \sum_{k=1}^K \alpha(\mathbf{x} \mid \boldsymbol{\theta}^{(k)}),
$$

with $\boldsymbol{\theta}^{(k)} \sim p(\boldsymbol{\theta} \mid \mathbf{y})$.

# 8. Cost-Sensitive Optimization

Many optimization problems have variable evaluation costs. For example, training a larger neural network (with more hidden units) often takes longer. In such situations, the standard BO pipeline should prioritize "cheaper" evaluations if they are equally informative.

## 8.1 Utility per Unit Cost

We can modify common acquisition functions (like EI) by dividing by the cost:

$$
\alpha(\mathbf{x}) = \frac{\text{EI}(\mathbf{x})}{\text{cost}(\mathbf{x})}.
$$

When the cost function is unknown, one can model (say, the log-cost) with another surrogate to ensure positivity. In practice, this encourages trying configurations that yield a good improvement per unit time or per unit budget.

## 8.2 Example

In hyper-parameter tuning for deep neural networks on large datasets (e.g., CIFAR), one might see a dramatic difference in evaluation times depending on architecture size. Adopting a cost-aware approach (such as "EI per second of training time") can accelerate progress toward a good solution.

# 9. Software Tools

Several open-source packages facilitate Bayesian Optimization workflows:

- **Spearmint (Python)**:
  - Easy configuration via JSON files.
  - GP hyper-parameter sampling using Slice Sampling (MCMC).
  - Supports nonstationary kernels (beta warping).
  - Can distribute computations across clusters with SLURM.
  - Uses a database to store inputs and outputs.

Other popular packages include:

- SMAC (Java)
- Hyperopt (Python)
- BayesOpt (C++)
- PyBO (Python)
- MOE (Python/C++)

Each offers different features, e.g., specific kernels, multi-threading, parallel scheduling, or specialized handling of constraints.

# 10. Exercises

1. **Derive PI and EI for Gaussian and Bayesian Neural Network Models**
   - Show the full derivation of Probability of Improvement (PI) and Expected Improvement (EI) from their utility definitions. In the BNN case, assume the network posterior predictive is Gaussian with known variance given weights $\mathbf{W}$. Write how you would integrate out the posterior over $\mathbf{W}$ (or $\boldsymbol{\theta}$ in a GP).

2. **Why Does $U[y] = y$ Fail in the Myopic Case?**
   - Recall that using $U[y \mid \mathbf{x}, \mathcal{D}] = y$ would be "optimal" if you could do exact dynamic programming over a horizon $N$. Explain precisely why it fails when only a myopic (one-step) approximation is used, and why it does not fail in the full non-myopic case.

3. **Comparing $\epsilon$-Greedy to Bayesian Optimization**
   - The $\epsilon$-greedy strategy picks the current best $\mathbf{x}$ from a point estimate of $f$ with probability $1-\epsilon$, and picks $\mathbf{x}$ uniformly at random with probability $\epsilon$. Discuss the advantages and disadvantages of this approach compared to typical BO acquisition functions in terms of exploration, exploitation, and sample efficiency.

4. **Efficient Thompson Sampling for GPs**
   - Outline how to implement Thompson Sampling in a GP efficiently. Consider how to draw approximate function samples from the GP posterior, especially if the input domain is large or continuous. Provide some practical hints on discretization or low-rank approximations.

# Conclusion

These notes offer an advanced perspective on Bayesian Optimization, covering Gaussian Processes, sparse/alternative models, a range of acquisition functions, hyper-parameter learning, and cost-sensitive extensions. A firm grasp of these topics prepares you to address real-world, computationally intensive black-box optimization tasks and to tailor solutions that best fit the practical constraints (time/budget) and domain assumptions (smoothness, stationarity, etc.) you encounter.