 Lecture by Mark Girolami

*topics: intractable integrals, Monte Carlo estimation, Markov chains, ergodicity, detailed balance, Metropolis-Hastings algorithm, Gibbs sampler, transition kernels, data augmentation, binary probit regression, convergence, burn-in, proposal tuning. 

# LECTURE-5
# 1. Bayesian Probabilistic Inference and Probability Inversion

## 1.1 The Bayesian Paradigm
We have observed data $\mathbf{x} \in \mathbf{X}$ and wish to infer unknown parameters (which may include latent variables) $\mathbf{\theta} \in \mathbf{\Theta}$. In Bayesian inference, we begin with a prior distribution $\pi_0(\mathbf{\theta})$ that encodes our beliefs (or assumptions) about $\mathbf{\theta}$ before observing $\mathbf{x}$. The likelihood function $p(\mathbf{x} \mid \mathbf{\theta})$ quantifies how probable the observed data $\mathbf{x}$ is, given a specific parameter value $\mathbf{\theta}$. By Bayes’ Rule, the posterior distribution for $\mathbf{\theta}$ given data $\mathbf{x}$ is

$$
\pi(\mathbf{\theta} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathbf{\theta}) \pi_0(\mathbf{\theta})}{\int_{\mathbf{\Theta}} p(\mathbf{x} \mid \mathbf{\theta}) \pi_0(\mathbf{\theta}) d\mathbf{\theta}}.
$$

The denominator is just the normalizing constant (or marginal likelihood), often intractable to compute.

## 1.2 The Challenge of Intractable Integrals
Many tasks of interest in Bayesian statistics require integrating against this posterior. For example, the posterior mean of some function $f(\mathbf{\theta})$ is

$$
E_{\mathbf{\theta} \mid \mathbf{x}}[f(\mathbf{\theta})] = \int_{\mathbf{\Theta}} f(\mathbf{\theta}) \pi(\mathbf{\theta} \mid \mathbf{x}) d\mathbf{\theta}.
$$

Typically, such integrals are not available in closed form unless $\pi_0(\mathbf{\theta})$ and $p(\mathbf{x} \mid \mathbf{\theta})$ are conjugate, a **rare and restrictive condition**. Consequently, direct numerical methods often fail because the integrals can be of very high dimension or otherwise **intractable**.

## 1.3 Monte Carlo Estimation
Monte Carlo methods bypass closed-form integration by simulating random draws $\mathbf{\theta}^{(1)}, \dots, \mathbf{\theta}^{(N)}$ from the target density $\pi(\mathbf{\theta} \mid \mathbf{x})$. We form the empirical average:

$$
\hat{I}_N = \frac{1}{N} \sum_{n=1}^N f(\mathbf{\theta}^{(n)}),
$$

which is a consistent estimator of $\int f(\mathbf{\theta}) \pi(\mathbf{\theta} \mid \mathbf{x}) d\mathbf{\theta}$. If $\mathbf{\theta}^{(n)}$ were *independent and identically distributed* (i.i.d.) from $\pi(\mathbf{\theta} \mid \mathbf{x})$, then by the **Law of Large Numbers**,

$$
\hat{I}_N \to E_{\mathbf{\theta} \mid \mathbf{x}}[f(\mathbf{\theta})] \quad \text{as } N \to \infty.
$$

We can also assume that the variance of $f(\theta)$ under the target distribution is finite, i.e., $\sigma^2 = \operatorname{Var}_{\theta \mid x} [f(\theta)] < \infty.$ Then, the **Central Limit Theorem (CLT)** implies that  

$$\sqrt N \left( \hat{I}_N - \mathbb{E}_{\theta \mid x} [f(\theta)] \right) \xrightarrow{d} N(0, \sigma_f^2).$$ 
This result shows that the error in the Monte Carlo estimator decreases at the rate $O(N^{-1/2})$. In other words, the typical magnitude of the estimation error (often called the Monte Carlo standard error) is approximately $\frac{\sigma_f}{\sqrt{N}}.$

However, generating i.i.d. samples from complicated posteriors is itself usually ***infeasible***. This sets the stage for Markov chain methods.

# 2. Markov Chains: Key Concepts

## 2.1 Markov Property
A (time-homogeneous - transition probabilities do not change over time) Markov chain $\{\mathbf{\theta}^{(i)}\}_{i=1}^\infty$ on a state space $\mathbf{\Theta}$ is a sequence of random variables satisfying

$$
p(\mathbf{\theta}^{(i)} \mid \mathbf{\theta}^{(i-1)}, \mathbf{\theta}^{(i-2)}, \dots) = T(\mathbf{\theta}^{(i)} \mid \mathbf{\theta}^{(i-1)}),
$$

where $T$ is the Markov transition kernel (or operator). Intuitively, the next state depends only on the immediately preceding state.

## 2.2 Invariant Distributions and Detailed Balance
A distribution $\pi(\mathbf{\theta})$ is called **invariant** (or stationary) for a transition $T$ if

$$
\pi(\mathbf{\theta}) = \int T(\mathbf{y} \to \mathbf{\theta}) \pi(\mathbf{y}) d\mathbf{y}
$$

or in discrete notation, $\pi(\mathbf{\theta}_i) = \sum_j \pi(\mathbf{\theta}_j) T(\mathbf{\theta}_j \to \mathbf{\theta}_i)$. If the chain is **irreducible** (can move between all regions of the space with nonzero probability) and **aperiodic** (no inescapable cycles), it will converge from any initial distribution to its invariant distribution $\pi$.

These conditions guarantee that all eigenvalues of the transition matrix $\mathbf{T}$, except for the dominant eigenvalue $\lambda_1=1$, satisfy $|\lambda_i|<1$ for $i\geq 2$. Since any initial distribution $\boldsymbol{\mu}_0$ can be expressed as a linear combination of the eigenvectors of $\mathbf{T}$, we have:

$$
\boldsymbol{\mu}_0 \mathbf{T}^n = \boldsymbol{\pi} + \sum_{i=2}^d c_i \lambda_i^n \mathbf{v}_i,
$$

where $\boldsymbol{\pi}$ is the invariant distribution, $\mathbf{v}_i$ are eigenvectors of $\mathbf{T}$, and $c_i$ are coefficients depending on $\boldsymbol{\mu}_0$. As $n \to \infty$, the terms $\lambda_i^n \to 0$, so $\boldsymbol{\mu}_0 \mathbf{T}^n \to \boldsymbol{\pi}$. This explains why *irreducibility and aperiodicity* ensure convergence to a unique invariant distribution, independently of the initial state.

### Aperiodicity and Irreducibility:
**The transition operator as an integral:**  Define for any density $f$:

$$(Tf)(\theta) = \int T(y \rightarrow \theta)\, f(y)\, dy.$$

In the discrete case this was  
$$(Tf)_i = \sum_j T_{j \rightarrow i} f_j.$$

**Eigen‐functions and spectral expansion:** We look for functions $v_i(\theta)$ and scalars $\lambda_i$ such that

$$\int T(y \rightarrow \theta)\, v_i(y)\, dy = \lambda_i\, v_i(\theta).$$

Under appropriate compactness (e.g. a reversible chain on a compact space), you get a countable spectrum  
$$\lambda_1 = 1 > |\lambda_2| \geq |\lambda_3| \geq \cdots \rightarrow 0,$$  
with eigen‐functions $\{v_i\}$ forming a basis (in an $L^2$ sense).

**Expanding your initial density:** Assume you can write

$$\mu_0(\theta) = \sum_{i=1}^\infty c_i\, v_i(\theta),$$  
where $c_i = \int v_i(\theta)\, \mu_0(\theta)\, d\theta$   (if the $v_i$ are orthonormal under the right inner product).

**Acting $T^n$ and taking the limit:** Applying $T^n$ times gives

$$\mu_n(\theta) = (T^n \mu_0)(\theta) = \sum_{i=1}^\infty c_i\, \lambda_i^n\, v_i(\theta).$$

Since $\lambda_1 = 1$ and all $|\lambda_{i \geq 2}| < 1$, as $n \rightarrow \infty$:

$$
\mu_n(\theta) = c_1\, v_1(\theta) + \sum_{i=2}^\infty c_i\, \lambda_i^n\, v_i(\theta) \longrightarrow c_1\, v_1(\theta) = \pi(\theta),
$$

showing convergence to the invariant density $\pi = v_1$.

### Detailed Balance
A sufficient (though not necessary) condition for $\pi$ to be invariant is reversibility, also known as satisfying detailed balance:

$$
\pi(\mathbf{x}) T(\mathbf{x} \to \mathbf{y}) = \pi(\mathbf{y}) T(\mathbf{y} \to \mathbf{x}),
$$

for all $\mathbf{x}, \mathbf{y}$. This condition ensures $\pi$ is stationary, simplifying the design of Markov transitions that preserve $\pi$.

### Short Proof:

Assume detailed balance holds: $\pi(x)T(x \to y) = \pi(y)T(y \to x), \quad \forall x,y.$ 

Integrate both sides over $x$:

$$\int \pi(x)T(x \to y) \,dx = \pi(y) \int T(y \to x) \,dx.$$  

Since $\int T(y \to x) \,dx = 1$ (by definition of a probability kernel), we get:

$$\int \pi(x)T(x \to y) \,dx = \pi(y).$$  
Thus, $\pi$ is invariant under $T$. 

### A Discrete Example:

Consider the following $3 \times 3$ transition matrix:

$$T = \begin{bmatrix} 0.5 & 0.25 & 0.25 \\ 0.25 & 0.5 & 0.25 \\ 0.25 & 0.25 & 0.5 \end{bmatrix}.$$  
**Detailed Balance:** Since $T$ is symmetric, for any states $i$ and $j$ we have: $\pi(i)T(i \to j) = \pi(j)T(j \to i)$ when $\pi$ is the uniform distribution: $\pi = \left( \frac{1}{3}, \frac{1}{3}, \frac{1}{3} \right).$ This means $T$ satisfies detailed balance.

**Stationarity:** Detailed balance implies that $\pi$ is invariant under $T$.

**Long-Term Behavior ($T^{\infty}$):** As we raise $T$ to a large power, the effect of the starting state is "forgotten". Specifically:
$$T^{\infty} = \lim_{n \to \infty} T^n = \begin{bmatrix} \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \\ \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \\ \frac{1}{3} & \frac{1}{3} & \frac{1}{3} \end{bmatrix}.$$  
Every row becomes the stationary distribution $\pi$. This means that no matter where you start, after many transitions the probability of being in any state converges to $\frac{1}{3}$.


## 2.3 Using Markov Chains for Monte Carlo
Instead of needing i.i.d. draws from $\pi$, we construct a Markov chain with $\pi$ as its invariant distribution, then run the chain for many steps. By the ergodic theorem for Markov chains, under suitable conditions (e.g., irreducibility, aperiodicity), the time average of any function $f$ along the chain converges to the target mean:

$$
\frac{1}{N} \sum_{i=1}^N f(\mathbf{\theta}^{(i)}) \to \int f(\mathbf{\theta}) \pi(\mathbf{\theta}) d\mathbf{\theta}, \quad \text{as } N \to \infty.
$$

This is exactly what we want for Bayesian inference: approximate integrals $\int f(\mathbf{\theta}) \pi(\mathbf{\theta} \mid \mathbf{x}) d\mathbf{\theta}$ by simulating a Markov chain that has the posterior as invariant distribution.

**Note on the Ergodic Theorem:**

The **Ergodic Theorem** is like the **Law of Large Numbers (LLN)** — and in spirit, also the **CLT** — **but for dependent samples**, specifically those generated by a **Markov chain**.

Given random variables $X_0, X_1, \dots, X_n$, governed by the transition kernel $\mathbf{T}$, and they are ergodic (i.e., irreducible + aperiodic + positive recurrent). Then:
$$
\frac{1}{n} \sum_{k=0}^{n-1} f(X_k) \xrightarrow{\text{a.s.}} \mathbb{E}_{\boldsymbol{\pi}}[f]
$$

even though the $X_k$'s are not independent!

Roughly speaking, the ergodic theorem implies that if a Markov chain is ergodic (i.e., it is irreducible, aperiodic, and positive recurrent -returns to all states in **finite expected time**), then it "forgets" its initial state. This means that, in the long run, the proportion of time the chain spends in any region of the state space converges to the probability mass that $\pi$ assigns to that region.  As a consequence, even though the samples are dependent, the time average of $f$ along the chain converges to the expected value of $f$ under $\pi$.

# 3. The Metropolis–Hastings Algorithm

## 3.1 Motivation and Construction
Suppose we can propose a “candidate” move $\mathbf{y}$ from a simpler density $q(\mathbf{y} \mid \mathbf{x})$ (called the proposal), but $q$ itself is not equal to the target $\pi$. We adjust the transitions to satisfy detailed balance with respect to $\pi$. Specifically:
- We first propose $\mathbf{y} \sim q(\cdot \mid \mathbf{x})$.
- We then accept or reject this proposal with probability $\alpha(\mathbf{x}, \mathbf{y})$.

The transition kernel is thus

$$
P(\mathbf{x} \to d\mathbf{y}) = q(\mathbf{x} \to \mathbf{y}) \alpha(\mathbf{x}, \mathbf{y}) d\mathbf{y} + \left[1 - \int q(\mathbf{x} \to \mathbf{z}) \alpha(\mathbf{x}, \mathbf{z}) d\mathbf{z}\right] \delta_{\mathbf{x}}(d\mathbf{y}),
$$

where $\delta_{\mathbf{x}}$ is the Dirac measure (the probability of “staying put”).

## 3.2 Acceptance Probability and Detailed Balance

For transitions where the state does not change (i.e., the "self-transition"), detailed balance is satisfied trivially. So, while the formal detailed balance condition is written as

$$\pi(x) P(x \to dy) = \pi(y) P(y \to dx),$$

we just need to ensure reversibility for the non-trivial case, i.e. $\mathbf{x} \neq \mathbf{y}$:

$$
\pi(\mathbf{x}) q(\mathbf{x}, \mathbf{y}) \alpha(\mathbf{x}, \mathbf{y}) = \pi(\mathbf{y}) q(\mathbf{y}, \mathbf{x}) \alpha(\mathbf{y}, \mathbf{x}).
$$

A standard choice is

$$
\alpha(\mathbf{x}, \mathbf{y}) = \min\left\{1, \frac{\pi(\mathbf{y}) q(\mathbf{y}, \mathbf{x})}{\pi(\mathbf{x}) q(\mathbf{x}, \mathbf{y})}\right\}.
$$

If $\pi(\mathbf{y}) q(\mathbf{y}, \mathbf{x}) \geq \pi(\mathbf{x}) q(\mathbf{x}, \mathbf{y})$, the ratio is $\geq 1$, so $\alpha(\mathbf{x}, \mathbf{y}) = 1$ and the proposal is always accepted. Otherwise, it is accepted with probability equal to that ratio.

## 3.3 Algorithmic Steps
Putting it all together:

**Metropolis–Hastings Algorithm**

1. Start from some initial state $\mathbf{x}^{(1)}$.
2. At iteration $j$:
   - Propose $\mathbf{y} \sim q(\mathbf{y} \mid \mathbf{x}^{(j)})$.
   - Draw $u \sim \text{Uniform}(0,1)$.
   - If $u \leq \alpha(\mathbf{x}^{(j)}, \mathbf{y}) = \min\left\{1, \frac{\pi(\mathbf{y}) q(\mathbf{y}, \mathbf{x}^{(j)})}{\pi(\mathbf{x}^{(j)}) q(\mathbf{x}^{(j)}, \mathbf{y})}\right\}$, accept and set $\mathbf{x}^{(j+1)} = \mathbf{y}$.
   - Else reject and set $\mathbf{x}^{(j+1)} = \mathbf{x}^{(j)}$.
3. Collect samples $\{\mathbf{x}^{(j)}\}$.

## 3.4 Special Case: Symmetric Proposals
If $q(\mathbf{x}, \mathbf{y}) = q(\mathbf{y}, \mathbf{x})$, e.g. a Gaussian centered at the current point, then

$$
\alpha(\mathbf{x}, \mathbf{y}) = \min\left\{1, \frac{\pi(\mathbf{y})}{\pi(\mathbf{x})}\right\}.
$$

This simpler acceptance ratio is typical of the Metropolis (as opposed to Metropolis–Hastings) algorithm.

## 3.5 Practical Observations
- The choice of $q$ affects mixing: if the proposal is too narrow, the chain moves slowly through the space. If the proposal is too wide, many moves are rejected.
- For multi-modal targets, a simple local proposal may get “stuck” in one mode. More sophisticated proposals or additional MCMC tricks may be required.
- Note in this whole set up, we generally require the target density $\pi(x)$ to be known up to a normalizing constant. In other words:

$$\pi(x) = \frac{\tilde{\pi}(x)}{Z},$$
- If we truly know nothing about the target distribution—that is, if we have no way of evaluating it even up to a constant—then we cannot directly implement the standard Metropolis–Hastings algorithm.
- However, in practical problems, especially in Bayesian inference, we usually know the target distribution up to a constant. For example, the posterior density is often known as

$$\pi(x \mid y) \propto L(y \mid x) p(x),$$

where the likelihood $L(y \mid x)$ and the prior $p(x)$ are known functions. Great!!

# V. MCMC and Monte Carlo Estimation

## 1. The Ideal: i.i.d. Samples for Monte Carlo

### Why i.i.d.?
- **Law of Large Numbers & CLT**: With independent samples, the empirical average converges to the true expectation, and the variance of the estimator decreases at the rate $O(N^{-1/2})$, where $N$ is the number of samples.

### Benefits:
- Straightforward uncertainty quantification (e.g., via Monte Carlo standard errors).
- Clear theoretical guarantees on convergence.

## 2. MCMC Samples: Not i.i.d. but Still Useful

### Dependence Structure:
- MCMC generates samples via a Markov chain whose transition kernel is designed so that its stationary (or invariant) distribution is the target $\pi$.
- Because of the Markov property, successive samples are generally correlated.

### Ergodicity:
- If the chain is ergodic (i.e., irreducible, aperiodic, and positive recurrent), the ergodic theorem ensures that time averages converge to the expected value under $\pi$:
  $$
  \frac{1}{N} \sum_{i=1}^{N} f(\theta^{(i)}) \to \int f(\theta) \pi(\theta) d\theta \quad \text{as } N \to \infty.
  $$

### Effective Sample Size (ESS):
- Because of serial correlations, the effective number of independent samples is lower than $N$.
- ESS estimation helps assess the “information content” of your chain.

# --LECTURE 6--
# 4. Product and Mixture Transition Kernels

## 4.1 Product Transition Kernels
Often, $\mathbf{\theta}$ is high-dimensional, say $\mathbf{\theta} = (\mathbf{x}_1, \mathbf{x}_2)$ or more generally $(\mathbf{x}_1, \dots, \mathbf{x}_d)$. One strategy is to update the components in “blocks” (or subsets) in sequence. Suppose $P_1$ updates $\mathbf{x}_1$ conditional on $\mathbf{x}_2$, and $P_2$ updates $\mathbf{x}_2$ conditional on $\mathbf{x}_1$. If both $P_1$ and $P_2$ leave the overall target $\pi$ invariant when the other coordinate is held fixed, then their product $P_1 P_2$ also leaves $\pi$ invariant. In symbols:

$$
(P_1 P_2)(\mathbf{x}_1, \mathbf{x}_2 \to d\mathbf{y}_1, d\mathbf{y}_2) = P_1(\mathbf{x}_1 \to d\mathbf{y}_1 \mid \mathbf{x}_2) P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1).
$$

One can prove that this product kernel has $\pi$ as its invariant measure by verifying an integral identity mirroring detailed balance.

### Short Proof:

The setting is as follows: 

- State vector:  $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2) \in \mathbb{R}^d.$  
- Target distribution factors as:  
$$\pi(\mathbf{x}_1, \mathbf{x}_2) = \pi_2(\mathbf{x}_2) \pi_{1|2}(\mathbf{x}_1 \mid \mathbf{x}_2) = \pi_1(\mathbf{x}_1) \pi_{2|1}(\mathbf{x}_2 \mid \mathbf{x}_1).$$

- Block-update kernels:  
$$P_1(\mathbf{x}_1 \to d\mathbf{y}_1 \mid \mathbf{x}_2), \quad \text{leaving } \pi_{1|2}(\cdot \mid \mathbf{x}_2) \text{ invariant.}$$
$$P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1), \quad \text{leaving } \pi_{2|1}(\cdot \mid \mathbf{y}_1) \text{ invariant.}$$  
- ***Goal***: Show that the product has $\pi(\mathbf{x}_1, \mathbf{x}_2)$ as an invariant measure. 
$$P_1 P_2((\mathbf{x}_1, \mathbf{x}_2), d(\mathbf{y}_1, \mathbf{y}_2)) = P_1(\mathbf{x}_1 \to d\mathbf{y}_1 \mid \mathbf{x}_2) P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1)$$ 
Observe that
$$
\begin{aligned}
&\int \int P_1(\mathbf{x}_1 \to d\mathbf{y}_1 \mid \mathbf{x}_2) P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1) \pi(\mathbf{x}_1, \mathbf{x}_2) d\mathbf{x}_1 d\mathbf{x}_2 \\ &= \int P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1) \left( \int P_1(\mathbf{x}_1 \to d\mathbf{y}_1 \mid \mathbf{x}_2) \pi_{1|2}(\mathbf{x}_1 \mid \mathbf{x}_2) d\mathbf{x}_1 \right) \pi_2(\mathbf{x}_2) d\mathbf{x}_2 \\
&= \int P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1) \pi^*_{1|2}(d\mathbf{y}_1 \mid \mathbf{x}_2) \pi_2(\mathbf{x}_2) d\mathbf{x}_2 \\
&= \int P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1) \frac{\pi_{2|1}(\mathbf{x}_2 \mid \mathbf{y}_1) \pi^*_1(d\mathbf{y}_1)}{\pi_2(\mathbf{x}_2)} \pi_2(\mathbf{x}_2) d\mathbf{x}_2 \\
&= \pi^*_1(d\mathbf{y}_1) \int P_2(\mathbf{x}_2 \to d\mathbf{y}_2 \mid \mathbf{y}_1) \pi_{2|1}(\mathbf{x}_2 \mid \mathbf{y}_1) d\mathbf{x}_2 \\
&= \pi^*_1(d\mathbf{y}_1) \pi^*_{2|1}(d\mathbf{y}_2 \mid \mathbf{y}_1) \\
&= \pi^*(d\mathbf{y}_1, d\mathbf{y}_2).
\end{aligned}
$$

Here $\pi_{1|2}^*$, $\pi_{2|1}^*$, and $\pi^*$ denote the updated or "pushed-forward" measures (the exact notation can vary, but they all match the same respective conditional or full target distribution).  

This completes the argument that $\pi$ is invariant under the product $P_1 P_2$. 

## 4.2 Mixture Transition Kernels
Another useful construction: if $P_1$ and $P_2$ both preserve $\pi$, then any convex combination (a “mixture,” i.e., a weighted average with non-negative weights summing to 1) of them also preserves $\pi$. Concretely, for $0 \leq \gamma \leq 1$,

$$
P = \gamma P_1 + (1 - \gamma) P_2
$$

also has $\pi$ as an invariant distribution. This approach can help the chain explore complicated targets by occasionally switching between different updates (multiple modes).

# 5. The Gibbs Sampler

## 5.1 Exact Conditional Updates
Gibbs sampling is a special case of Metropolis–Hastings in which proposed moves are always accepted (acceptance probability = 1). Set up: let $\mathbf{\theta} = (\mathbf{x}_1, \dots, \mathbf{x}_d)$. Suppose we know how to sample exactly from each conditional distribution $\pi(\mathbf{x}_i \mid \text{all other } \mathbf{x}_j)$. The basic Gibbs Sampler cycle updates each coordinate (or block) in turn from the exact conditional distribution, for instance:

$$
\mathbf{x}_1^{(j+1)} \sim \pi(\mathbf{x}_1 \mid \mathbf{x}_2^{(j)}, \dots, \mathbf{x}_d^{(j)}), \quad
\mathbf{x}_2^{(j+1)} \sim \pi(\mathbf{x}_2 \mid \mathbf{x}_1^{(j+1)}, \mathbf{x}_3^{(j)}, \dots), \quad
\dots
$$

until all coordinates have been updated.

## 5.2 Why Gibbs Sampler Always Accepts
Each coordinate update is a Metropolis–Hastings step with proposal = “draw from the exact conditional.” In that scenario, the acceptance ratio

$$
\frac{\pi(\mathbf{y}) q(\mathbf{y} \rightarrow \mathbf{x})}{\pi(\mathbf{x}) q(\mathbf{x} \rightarrow \mathbf{y})}
$$

becomes exactly 1, so the move is always accepted.

$$
\begin{align}
\underbrace{\frac{\pi(\mathbf{y}) \, q(\mathbf{y} \to \mathbf{x})}{\pi(\mathbf{x}) \, q(\mathbf{x} \to \mathbf{y})}}_{\text{M-H ratio}} 
& \longrightarrow 
\underbrace{\frac{\pi(\mathbf{y₁}, \mathbf{y₂}) \, q([\mathbf{y₁}, \mathbf{y₂}] \to [\mathbf{x₁}, \mathbf{x₂}])}{\pi(\mathbf{x₁}, \mathbf{x₂}) \, q([\mathbf{x₁}, \mathbf{x₂}] \to [\mathbf{y₁}, \mathbf{y₂}])}}_{\substack{\text{Expand } \mathbf{x} = (\mathbf{x₁}, \mathbf{x₂}), \\ \mathbf{y} = (\mathbf{y₁}, \mathbf{y₂}); \text{ Gibbs proposal}}} 
\\[10pt] % Adds vertical space between equations
& \longrightarrow 
\frac{\pi(\mathbf{y}_1, \mathbf{y}_2)\, \pi(\mathbf{x}_1 \mid \mathbf{y}_2)\, \delta_{\mathbf{y}_2}(\mathbf{x}_2)}{\pi(\mathbf{x}_1, \mathbf{x}_2)\, \pi(\mathbf{y}_1 \mid \mathbf{x}_2)\, \delta_{\mathbf{x}_2}(\mathbf{y}_2)}.
\\[10pt] % Adds vertical space between equations
& \longrightarrow 
\underbrace{\frac{\pi(\mathbf{y₁} \mid \mathbf{x₂}) \pi(\mathbf{x₂}) \times \pi(\mathbf{x₁} \mid \mathbf{x₂})}{\pi(\mathbf{x₁} \mid \mathbf{x₂}) \pi(\mathbf{x₂}) \times \pi(\mathbf{y₁} \mid \mathbf{x₂})}}_{\substack{\text{Rewrite via Bayes'/conditional identities;} \\ \text{all factors match}}} 
= 1
\end{align}
$$

***Remark***:  In a single‐block Gibbs update of $(\mathbf{x}_1, \mathbf{x}_2)$, we only change $\mathbf{x}_1$, drawing $\mathbf{y}_1 \sim \pi(\mathbf{y}_1 \mid \mathbf{x}_2)$ while keeping $\mathbf{x}_2$ fixed. Thus the new state becomes $(\mathbf{y}_1, \mathbf{x}_2)$, so $\mathbf{y}_2$ is literally the same as the old $\mathbf{x}_2$. Consequently, we identify $\pi(\mathbf{y}_1 \mid \mathbf{y}_2)$ with $\pi(\mathbf{y}_1 \mid \mathbf{x}_2)$. Furthermore, when we look at the reverse transition $([\mathbf{y}_1, \mathbf{y}_2] \to [\mathbf{x}_1, \mathbf{x}_2])$, it’s really an update of the same first coordinate but going from $\mathbf{y}_1$ back to $\mathbf{x}_1$. Since $\mathbf{x}_2$ remains unchanged in that move, the corresponding proposal density is $\pi(\mathbf{x}_1 \mid \mathbf{x}_2)$. This is why in the Metropolis–Hastings ratio both “forward” and “reverse” proposal kernels appear as conditionals $\pi(\cdot \mid \mathbf{x}_2)$.

**Rationale:**  In a Gibbs update on coordinate 1, we draw  
$\mathbf{y}_1 \sim \pi(\mathbf{y}_1 \mid \mathbf{x}_2)$  
and we leave coordinate 2 unchanged.  
That “leave it be” is encoded by a Dirac $\delta_{\mathbf{x}_2}(\mathbf{y}_2)$.

Thus

$$
q((\mathbf{x}_1, \mathbf{x}_2) \rightarrow (\mathbf{y}_1, \mathbf{y}_2)) = \pi(\mathbf{y}_1 \mid \mathbf{x}_2)\, \delta_{\mathbf{x}_2}(\mathbf{y}_2),
$$

$$
q((\mathbf{y}_1, \mathbf{y}_2) \rightarrow (\mathbf{x}_1, \mathbf{x}_2)) = \pi(\mathbf{x}_1 \mid \mathbf{y}_2)\, \delta_{\mathbf{y}_2}(\mathbf{x}_2).
$$

Substitute into $r$:

$$
r = \frac{\pi(\mathbf{y}_1, \mathbf{y}_2)\, \pi(\mathbf{x}_1 \mid \mathbf{y}_2)\, \delta_{\mathbf{y}_2}(\mathbf{x}_2)}{\pi(\mathbf{x}_1, \mathbf{x}_2)\, \pi(\mathbf{y}_1 \mid \mathbf{x}_2)\, \delta_{\mathbf{x}_2}(\mathbf{y}_2)}.
$$


**Enforce the Dirac deltas:** Both $\delta_{\mathbf{y}_2}(\mathbf{x}_2)$ and $\delta_{\mathbf{x}_2}(\mathbf{y}_2)$ vanish unless $\mathbf{y}_2 = \mathbf{x}_2$.  
On that event, we set $\mathbf{y}_2 := \mathbf{x}_2$ everywhere, and the deltas drop out:

$$
r = \frac{\pi(\mathbf{y}_1, \mathbf{x}_2)\, \pi(\mathbf{x}_1 \mid \mathbf{x}_2)}{\pi(\mathbf{x}_1, \mathbf{x}_2)\, \pi(\mathbf{y}_1 \mid \mathbf{x}_2)}.
$$


## 5.3 Metropolis-within-Gibbs
Sometimes not all conditionals are easy to sample directly. In that case, we combine a Gibbs update for the easy coordinates with a Metropolis–Hastings update for the difficult ones. This is known as **Metropolis-within-Gibbs**. The overall kernel can still be viewed as a product of individual transitions, each preserving the same target $\pi$.

# 6. Data Augmentation

## 6.1 Idea of Augmented Variables
Suppose $\pi(\mathbf{\theta})$ is difficult to sample directly. We might introduce extra (latent) variables $\mathbf{\phi}$ to form an augmented distribution $\pi(\mathbf{\theta}, \mathbf{\phi})$. By design,

$$
\int \pi(\mathbf{\theta}, \mathbf{\phi}) d\mathbf{\phi} = \pi(\mathbf{\theta}).
$$

If we can set up a scheme in which $\pi(\mathbf{\theta} \mid \mathbf{\phi})$ and $\pi(\mathbf{\phi} \mid \mathbf{\theta})$ are easier to sample from, then we can run a Gibbs sampler on $(\mathbf{\theta}, \mathbf{\phi})$. The marginal draws of $\mathbf{\theta}$ will still have the original $\pi(\mathbf{\theta})$ distribution.

## 6.2 Binary Probit Regression Example
We have observed binary responses $t_i \in \{0,1\}$ with predictor vectors $\mathbf{x}_i$. The probit model says:

$$
p(t_i = 1 \mid \mathbf{\beta}, \mathbf{x}_i) = \Phi(\mathbf{\beta}^\top \mathbf{x}_i),
$$

where $\Phi$ is the standard normal CDF. Direct sampling from the posterior $\pi(\mathbf{\beta} \mid \mathbf{t}, \mathbf{X})$ is nontrivial. Instead:
- Augment with latent variables $y_i$, such that

$$
y_i = \mathbf{\beta}^\top \mathbf{x}_i + \varepsilon_i, \quad \varepsilon_i \sim N(0,1),
$$

and impose $t_i = 1 \iff y_i > 0$, and otherwise. Then

$$
\begin{aligned}
 p(t_i, y_i \mid \beta, x_i) &= p(t_i \mid y_i) p(y_i \mid \beta, x_i)
 \\[10pt]
 &=\left[ \delta(t_i = 1) \delta(y_i > 0) + \delta(t_i = 0) \delta(y_i \leq 0) \right] \times N(y_i \mid \beta^\top x_i, 1).
\end{aligned}
$$

and the joint likelihood becomes 

$$
 p(\mathbf{t}, \mathbf{y} \mid \beta, \mathbf{X}) = \prod_{i=1}^{n} \left\{ \left[ \delta(t_i = 1) \delta(y_i > 0) + \delta(t_i = 0) \delta(y_i \leq 0) \right] \times N(y_i \mid \beta^\top x_i, 1) \right\}.
$$

Full conditionals become:
- $\mathbf{\beta} \mid \{y_i\}, \{t_i\}$ is Gaussian (since the product of normals is normal).
- Each $y_i \mid \mathbf{\beta}, t_i$ is a truncated normal: if $t_i = 1$, then $y_i > 0$; if $t_i = 0$, then $y_i \leq 0$.

If we take an (improper) uniform prior on $\boldsymbol{\beta}$, then the posterior is

$$
p(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{X}) \propto \exp\left\{ -\frac{1}{2} (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) \right\}.
$$

Completing the square gives

$$
\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{X} \sim \mathcal{N}\left( (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y},\, (\mathbf{X}^\top \mathbf{X})^{-1} \right).
$$

We can alternate these two updates in a Gibbs fashion. Marginally, the $\mathbf{\beta}$ draws match the posterior $\pi(\mathbf{\beta} \mid \mathbf{t}, \mathbf{X})$.

# 7. Convergence and Practical Considerations
- **Non-i.i.d. Samples**: Markov chain outputs are typically correlated. Nevertheless, the ergodic theorem guarantees asymptotically correct averages.
- **Burn-In**: While not explicitly discussed in the provided text, one often discards early “burn-in” iterations before the chain is deemed to have “converged” to $\pi$.
- **Choosing Proposals $q$**: In Metropolis–Hastings, a poorly tuned $q$ can lead to inefficient exploration (too many rejections or very slow movement).
- **High Dimensionality**: Blocking (product kernels) and data augmentation can help break down complex targets and keep acceptance rates reasonable.
- **Multi-Modality**: Single-site updates (or local proposals) may struggle to move between widely separated modes. Strategies like mixture kernels or advanced sampling schemes can mitigate this.

