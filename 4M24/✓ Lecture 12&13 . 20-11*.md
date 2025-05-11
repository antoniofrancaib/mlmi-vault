Lecture by Mark Girolami

*topics:

# LECTURE-12

# 1. **Introduction and Overview**  
Langevin Monte Carlo methods serve as a bridge between continuous stochastic processes and computational sampling schemes in high-dimensional statistics and Bayesian inference. These notes focus on:

- Langevin Stochastic Differential Equations (SDEs) and their connection with target probability distributions.  
- Two main sampling algorithms derived from the Langevin SDE:  
  - Metropolis-adjusted Langevin Algorithm (MALA)  
  - Unadjusted Langevin Algorithm (ULA)  
- Special emphasis is placed on the role of log-concavity (and its stronger variant, $m$–strongly log–concavity) for deriving convergence rates and guaranteeing exponential contraction in probability metric spaces.  
- Applications to modern Bayesian machine learning are underscored by showing how these techniques help sample from complex posterior distributions.

# 2. **The Langevin SDE: Formulation and Stationarity**  
At the heart of the method is the Langevin SDE given by  

$$d\mathbf{X}_t = -\nabla U(\mathbf{X}_t)\,dt + \sqrt{2}\,d\mathbf{B}_t,$$  

where:  
- $U:\mathbb{R}^d \to \mathbb{R}$ is a potential function, often chosen as the negative logarithm of the target density (i.e., $U(\mathbf{x}) = -\log \pi(\mathbf{x})$).  
- $(\mathbf{B}_t)_{t \geq 0}$ is a standard Brownian motion.  

**Stationarity Insight:**  
By setting  

$$U(\mathbf{x}) = -\log \pi(\mathbf{x}),$$  
we obtain  

$$\nabla U(\mathbf{x}) = -\nabla \log \pi(\mathbf{x}),$$  

which allows us to rewrite the SDE as:  

$$d\mathbf{X}_t = \nabla \log \pi(\mathbf{X}_t)\,dt + \sqrt{2}\,d\mathbf{B}_t.$$  
In this formulation, the stationary (long-time) distribution of the process is the target distribution $\pi(\mathbf{x})$, up to a normalization constant. In other words, as $t \to \infty$, the distribution of $\mathbf{X}_t$ converges to $\pi$.

# 3. **The Fokker–Planck Equation and Stationary Measures**  
The evolution of the probability density $\rho_t$ of $\mathbf{X}_t$ is governed by the Fokker–Planck equation:  

$$\frac{\partial \rho_t}{\partial t} = -\nabla \cdot (\rho_t \nabla \log \pi) + \Delta \rho_t,$$  

where $\Delta$ denotes the Laplacian operator.  

**Stationary Solution:**  Setting $\frac{\partial \rho_t}{\partial t} = 0$ leads to a balance condition where the incoming and outgoing "flows" of probability match. Direct substitution shows that  

$$\rho_\infty = \pi.$$  
Thus, by simulating the Langevin SDE, one eventually obtains samples from $\pi$, making this approach a natural candidate for Markov chain Monte Carlo (MCMC) sampling.

Note: Finding a fixed point of the Fokker–Planck equation is exactly the same as finding an invariant distribution of the stochastic process $X_t$​. The **Fokker–Planck equation** describes the evolution over time of the **probability density** $\rho_t(x)$, where:

$$\rho_t(x) = \text{density of } X_t \text{ at location } x$$

So it’s a **dynamical system**, not for points in space, but for **distributions**

Derivation:  We plug $\rho_t = \pi$ into the right-hand side of the Fokker–Planck equation:

$$
\frac{\partial \pi}{\partial t} = -\nabla \cdot (\pi \nabla \log \pi) + \Delta \pi
$$

Recall:

$$
\nabla \log \pi = \frac{\nabla \pi}{\pi} \quad \Rightarrow \quad \pi \nabla \log \pi = \nabla \pi
$$

So:

$$
\nabla \cdot (\pi \nabla \log \pi) = \nabla \cdot (\nabla \pi) = \Delta \pi
$$

Substituting back into the equation:

$$
\frac{\partial \pi}{\partial t} = -\Delta \pi + \Delta \pi = 0
$$

**This shows:** $\pi$ is a **stationary (equilibrium)** solution of the Fokker–Planck equation. The **Fokker–Planck equation** describes how a probability density evolves over time under the dynamics of an SDE. When you simulate the Langevin dynamics:

$$
dX_t = \nabla \log \pi(X_t) \, dt + \sqrt{2} \, db_t,
$$

you are drawing paths $X_t$ whose **density $\rho_t$ evolves toward $\pi$**.

After enough time, the distribution of $X_t$ converges to $\pi$. This is **why Langevin dynamics is used in sampling**:  It **converges to the target distribution** $\pi$.

# 4. **Convergence Analysis in Log–Concave Settings**  
## 4.1 **Log–Concavity and $m$–Strongly Log–Concave Distributions**  

**Log–Concavity:**  A probability measure with density $\pi$ is said to be log–concave if, for any $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ and $\lambda \in (0,1)$,  

$$\log \pi(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \geq \lambda \log \pi(\mathbf{x}) + (1-\lambda) \log \pi(\mathbf{y}).$$  
An equivalent multiplicative form is:  

$$\pi(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \geq \pi(\mathbf{x})^\lambda \pi(\mathbf{y})^{1-\lambda}.$$  
This property implies that the density has a "single hump" with no flat regions.  

**$m$–Strong Log–Concavity:**   A stronger condition is defined by:  

$$\log \pi(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \geq \lambda \log \pi(\mathbf{x}) + (1-\lambda) \log \pi(\mathbf{y}) + \frac{m\lambda(1-\lambda)}{2} \|\mathbf{x} - \mathbf{y}\|^2.$$  
This ensures the existence of a unique mode (maximum) and provides uniform curvature (similar to a quadratic form as seen in Gaussian densities).  

**$m$-strong log-concavity** strengthens that idea: the graph lies **strictly below the chord by at least a quadratic amount**. The added term ensures a certain amount of "bending downward" — i.e., **uniform curvature**.

Why It Matters

- **Uniqueness of the mode**: Strong log-concavity ensures that $\pi$ has a **unique global maximum** (no plateaus or multiple peaks).
    
- **Exponential concentration**: Samples from $\pi$ concentrate around the mode, and tail probabilities decay like Gaussians.
    
- **Convex optimization**: Strong log-concavity implies that $-\log \pi$ is **strongly convex**, so gradient-based methods have good convergence properties.

## 4.2 **Exponential Convergence in the Wasserstein Metric**  
For an $m$–strongly log–concave target distribution, one can show that the convergence to the stationary distribution is exponentially fast. In particular, when measured in a suitable metric (e.g., the Wasserstein-2 distance $W_2$), one has an inequality of the form: 

$$
W_2(\pi_t, \pi) \leq e^{-mt} \left\{ \|x - x^\star\| + \left( \frac{d}{m} \right)^{1/2} \right\},
$$

for $\pi_0 = \delta_x$ (the starting point) and $x^\star = \arg\max_{x \in \mathbb{R}^d} \log \pi(x)$. The additional term reflects the inherent geometry and curvature of the problem. The key takeaway is that strong log–concavity guarantees rapid "forgetting" of the initial condition.

# 5. **Discretisation: The Euler Scheme**  
Because the continuous-time Langevin SDE cannot be simulated exactly, discretisation schemes are required. The simplest approach is the first–order Euler discretisation:  

$$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1},$$  
where:  
- $\gamma > 0$ is the step–size.  
- $(\mathbf{W}_k)$ is a sequence of independent standard normal random variables.  

While straightforward, this discretisation introduces bias relative to the continuous process. The bias can be corrected via a Metropolis step.

# 6. **Metropolis–Adjusted Langevin Algorithm (MALA)**  
To correct for the bias introduced by discretisation, the Metropolis–adjusted Langevin Algorithm is used.  

## 6.1 **The Metropolis Correction Step**  
After generating a proposal using the Euler scheme:  

$$\mathbf{X}_{k+1}^{\text{prop}} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1},$$  
one applies a Metropolis–Hastings acceptance step.  

## 6.2 **Detailed Balance and Acceptance Probability**  
The acceptance probability is computed as:  

$$\alpha = \min\left\{1, \frac{\pi(\mathbf{X}_{k+1}^{\text{prop}}) q(\mathbf{X}_k \mid \mathbf{X}_{k+1}^{\text{prop}})}{\pi(\mathbf{X}_k) q(\mathbf{X}_{k+1}^{\text{prop}} \mid \mathbf{X}_k)}\right\},$$  
where the proposal density is given by:  

$$q(\mathbf{x}' \mid \mathbf{x}) \propto \exp\left(-\frac{1}{4\gamma} \|\mathbf{x}' - \mathbf{x} - \gamma \nabla \log \pi(\mathbf{x})\|^2\right).$$  
This ratio ensures that the Markov chain satisfies detailed balance with respect to the target distribution $\pi$. Consequently, the chain has $\pi$ as its invariant measure.  

## 6.3 **Practical Considerations and High–Dimensional Issues**  
**Step–Size Tuning:** The acceptance probability is sensitive to $\gamma$, and theoretical studies suggest tuning the step–size so that $\alpha \approx 0.574$.  

**High Dimensions:**  In high–dimensional settings, the acceptance probability may deteriorate (i.e., become exponentially small in the dimension), leading to poor mixing.

# 7. **Unadjusted Langevin Algorithm (ULA)**  
An alternative is to simply run the discretised Langevin dynamics without the Metropolis adjustment:  

$$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1}.$$ 
## 7.1 **Bias and Efficiency Trade–Offs**  
**Inherent Bias:**  The resulting chain converges to a stationary measure $\pi_\gamma$ that differs from the intended target $\pi$. The bias introduced can be quantified and is non–asymptotic in nature.  

**High–Dimensional Performance:**  Avoiding the accept/reject step often leads to improved performance in high dimensions, making ULA attractive despite its bias.  

## 7.2 **Convergence Guarantees**  
For $m$–strongly log–concave targets with gradients that are $L$–Lipschitz, convergence of the ULA can be characterized by the following inequality (using the Wasserstein-2 distance):  

$$W_2(\pi_k, \pi) \leq (1 - m\gamma)^k W_2(\pi_0, \pi) + 1.65 \frac{L}{m} (\gamma d)^{1/2}.$$  
**Key points:**  
- The first term shows an exponential decay of the error if the initial condition is far from $\pi$.  
- The second term quantifies the asymptotic bias, which is of order $O(\gamma^{1/2})$ and can be made arbitrarily small with a sufficiently small step–size.

# 8. **Special Case: Gaussian Targets and the Ornstein–Uhlenbeck Process**  
When the target distribution is Gaussian,  

$$\pi(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}),$$  
the Langevin SDE becomes an Ornstein–Uhlenbeck (OU) process:  

$$d\mathbf{X}_t = -\boldsymbol{\Sigma}^{-1} (\mathbf{X}_t - \boldsymbol{\mu})\,dt + \sqrt{2}\,d\mathbf{B}_t.$$  
## 8.1 **ULA for Gaussian Targets**  
The discretised update is:  

$$\mathbf{X}_{k+1} = \mathbf{X}_k - \gamma \boldsymbol{\Sigma}^{-1} (\mathbf{X}_k - \boldsymbol{\mu}) + \sqrt{2\gamma} \mathbf{W}_{k+1}.$$  
Under appropriate conditions, the stationary (or invariant) measure for this discretisation is given by:  

$$\pi_\gamma = \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}\left(\mathbf{I} - \frac{\gamma}{2} \boldsymbol{\Sigma}^{-1}\right)^{-1}\right).$$  
This example illustrates how the discretisation modifies the covariance structure of the target Gaussian. It also offers a tractable setting for analyzing the bias introduced by the ULA.

# 9. **Implications and Applications in Bayesian Machine Learning**  
Langevin-based methods, including both MALA and ULA, have become popular in Bayesian machine learning for several reasons:  
- They allow sampling from high–dimensional posterior distributions with complex geometries.  
- Convergence properties based on strong log–concavity lead to rigorous performance guarantees.  
- Even though MALA provides exact invariance (via detailed balance), ULA's bias can be controlled by reducing the step–size, making it a viable option in high–dimensional contexts where the Metropolis step may be inefficient.  

Applications range from uncertainty quantification in neural network weights to Bayesian inference in complex probabilistic models.

# 10. **Summary and Conclusions**  
- Langevin SDEs offer a principled way to sample from target distributions $\pi$ by simulating dynamics that have $\pi$ as their stationary measure.  
- By setting $U(\mathbf{x}) = -\log \pi(\mathbf{x})$, the drift term becomes $\nabla \log \pi(\mathbf{x})$, leading to the SDE:  

$$d\mathbf{X}_t = \nabla \log \pi(\mathbf{X}_t)\,dt + \sqrt{2}\,d\mathbf{B}_t.$$  
- The Fokker–Planck equation confirms $\pi$ as the stationary distribution.  
- Restricting attention to log–concave (and $m$–strongly log–concave) distributions allows one to derive exponential convergence rates in metrics like $W_2$.  
- MALA corrects for discretisation bias via a Metropolis–Hastings accept–reject step, ensuring detailed balance; however, tuning and high–dimensional challenges must be managed.  
- ULA forgoes the acceptance step, leading to an invariant measure $\pi_\gamma$ that approximates $\pi$ with a bias of order $O(\gamma^{1/2})$, but it offers improved computational efficiency.  
- In the special case of Gaussian targets, these ideas are concretely illustrated through the Ornstein–Uhlenbeck process and its corresponding invariant measure under discretisation.  

# LECTURE-13
# 1. Introduction to Langevin MCMC in Bayesian Inference
In Bayesian inference, we aim to sample from the posterior distribution of parameters given data. Formally, if we denote our parameter vector by $\mathbf{x}$ and have a prior $p(\mathbf{x})$, along with likelihoods $L(y_i\mid\mathbf{x})$ for data points $y_i$ (with $i=1,\ldots,n$), the target distribution is

$$\pi(\mathbf{x}) \propto p(\mathbf{x}) \prod_{i=1}^n L(y_i\mid\mathbf{x}).$$

The Langevin MCMC framework leverages ideas from continuous-time stochastic processes (namely Langevin dynamics) to design Markov chain Monte Carlo (MCMC) algorithms. Its key innovation is the use of gradient information, meaning that proposals for the Markov chain are made in directions that are informed by the local geometry of the target density.

# 2. The Basic Langevin Dynamics Framework
## 2.1 The Continuous-Time Intuition
Langevin dynamics describe a continuous stochastic process where the evolution of a state $\mathbf{x}(t)$ is driven both by a drift component (proportional to the gradient of the log-density) and a diffusion (noise) term. The intuition behind this is that if you "flow" along the gradient of $\log\pi(\mathbf{x})$ (which points in the direction of increasing probability), and add appropriately scaled noise, then the resulting process has the target distribution $\pi(\mathbf{x})$ as its stationary distribution. In discrete time, this gives rise to update rules that mimic this behavior.

## 2.2 Discrete-Time Schemes: ULA and MALA
**Unadjusted Langevin Algorithm (ULA):**
The discrete update in ULA is given by

$$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{Z}_k,$$

where $\gamma$ is the stepsize and $\mathbf{Z}_k$ is a normally distributed random vector.

**Intuition:** The term $\gamma \nabla \log \pi(\mathbf{X}_k)$ deterministically moves the state in the direction of higher probability, while the noise term $\sqrt{2\gamma} \mathbf{Z}_k$ prevents the chain from becoming trapped in local modes.

**Metropolis-Adjusted Langevin Algorithm (MALA):**
MALA enhances ULA by introducing a Metropolis acceptance step to correct for the discretization error, ensuring detailed balance. In MALA, one first proposes a move using the Langevin update and then accepts or rejects the move with probability

$$\alpha = \min\left(1, \frac{\pi(\mathbf{X}_{k+1}) q(\mathbf{X}_k\mid\mathbf{X}_{k+1})}{\pi(\mathbf{X}_k) q(\mathbf{X}_{k+1}\mid\mathbf{X}_k)}\right),$$

where $q(\cdot\mid\cdot)$ denotes the proposal density.

**Key Point:** While the use of acceptance probabilities improves accuracy, it typically requires an additional evaluation of the likelihood terms for all data points.

# 3. Challenges of Scaling Langevin MCMC in Modern Bayesian Machine Learning
In modern applications, particularly in large-scale Bayesian machine learning and Bayesian neural networks, the number of data points $n$ can be enormous. This introduces two major computational challenges:

## 3.1 Gradient Computation
**Full Gradient Cost:**
The gradient of the log posterior is given by

$$\nabla \log \pi(\mathbf{x}) = \nabla \log p(\mathbf{x}) + \sum_{i=1}^n \nabla \log L(y_i\mid\mathbf{x}).$$

Computing the sum over $n$ data points results in an $O(n)$ computational complexity at every iteration—a prohibitive cost when $n$ is large.

## 3.2 Acceptance Probability in MALA
**Costly Acceptance Step:** In MALA, calculating the acceptance probability requires evaluating the unnormalized density $\pi(\mathbf{x})$, which again involves summing over all $n$ likelihood contributions. Even though the evaluation need not be normalized, it still scales as $O(n)$ per iteration.

**Implication:** Even if the acceptance probability can sometimes be approximated more efficiently through other tricks, the need to compute the full gradient remains the primary bottleneck.

# 4. Subsampling and Unbiased Gradient Estimation
## 4.1 Rationale Behind Subsampling
The primary observation here is that if one could compute an unbiased estimator of the gradient without summing over all $n$ data points, the computational cost per iteration could be dramatically reduced. This motivates the idea of stochastic gradients via data subsampling.

## 4.2 The Mini-Batch Gradient Estimator
At iteration $k$, rather than summing over the entire data set, we select a random subset (or mini-batch) $I_k \subset \{1,\ldots,n\}$ and compute:

$$\nabla \widetilde{\log \pi}(\mathbf{X}_k) = \nabla \log p(\mathbf{X}_k) + \frac{n}{|I_k|} \sum_{i \in I_k} \nabla \log L(y_i\mid\mathbf{X}_k).$$

**Unbiasedness:** Since each data point is equally likely to be included in the mini-batch, the expectation of the estimated gradient equals the true gradient:

$$\mathbb{E}[\nabla \widetilde{\log \pi}(\mathbf{X}_k)] = \nabla \log \pi(\mathbf{X}_k).$$

**Intuitive Explanation:** You can think of this as "scaling up" the contribution from a small, randomly chosen subset of the data to approximate the full sum. The factor $\frac{n}{|I_k|}$ ensures that the mini-batch gradient is an unbiased estimate of the full gradient.

## 4.3 Considerations and Trade-Offs
**Variance Control:**
While unbiased, the estimator will have higher variance compared to the full gradient. This variance must be controlled (e.g., by choosing a sufficiently large mini-batch or using variance reduction techniques) to ensure that the chain converges reliably.

**Iteration Cost:** The cost per iteration becomes $O(|I_k|)$ instead of $O(n)$, which is a significant reduction when $|I_k| \ll n$.

# 5. Stochastic Gradient Langevin Dynamics (SGLD)
## 5.1 Formulation of SGLD
With the unbiased mini-batch gradient estimator in place, we can modify the ULA update to get the Stochastic Gradient Langevin Dynamics (SGLD) scheme:

$$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla \widetilde{\log \pi}(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1},$$

where:
- $\gamma$ is the step size,
- $\nabla \widetilde{\log \pi}(\mathbf{X}_k)$ is the unbiased (mini-batch) gradient estimator,
- $\mathbf{W}_{k+1}$ is a standard Gaussian noise term.

## 5.2 Why SGLD Works
**Unbiased Gradient:**
The key property that makes SGLD viable is that the mini-batch gradient is an unbiased estimator of the true gradient. This ensures that, in expectation, the SGLD update has similar drift properties to those in ULA.

**Reduced Computational Burden:**
Each SGLD iteration only requires processing a mini-batch, making it scalable for large datasets. This efficiency is especially critical in training Bayesian neural networks or other large-scale probabilistic models.

**Convergence Guarantees:**
Under standard assumptions (for example, when the target $\pi$ is $m$-strongly log-concave), SGLD exhibits convergence guarantees akin to ULA. Specifically, bounds in Wasserstein distance $W_2$ of the form

$$W_2(\pi_k, \pi) \leq O(e^{-\gamma m k} + \gamma^{1/2})$$

have been derived. This shows that as long as the variance of the gradient estimates is controlled, SGLD can converge to the true target distribution.

**Connection to Bayesian Neural Networks:**
The ability to use automatic differentiation in modern frameworks allows one to handle complex likelihoods automatically. This is particularly advantageous for Bayesian neural networks, where the likelihood and corresponding gradients are highly nontrivial.

## 5.3 Variance Concerns
**Importance of Variance Control:**
While the mini-batch estimator is unbiased, its variance can affect the quality of the samples generated. High variance may lead to slow mixing or even divergence if not managed properly.

**Potential Solutions:**
Techniques such as increasing the mini-batch size, using control variates, or averaging over multiple mini-batch estimates may be employed to mitigate the variance issue. The exact approach will depend on the specific application and the structure of the likelihood function.

# 6. Practical Considerations and Extensions
## 6.1 Scalability and Algorithmic Efficiency
**Cost per Iteration:**
With SGLD, each iteration's cost scales with the size of the mini-batch rather than the full dataset. This makes the algorithm particularly suitable for datasets encountered in modern Bayesian machine learning.

**Metropolis-Free Approach:**
By design, SGLD avoids the costly acceptance-rejection step found in MALA. Although this introduces a discretization bias, the resulting algorithm remains practical for many applications where computational efficiency is paramount.

## 6.2 Generality of the Stochastic Gradient Approach
**Broad Applicability:**
The technique of substituting full gradients with unbiased mini-batch estimates is not limited to Langevin-based MCMC methods. It serves as a general strategy in other gradient-based MCMC algorithms, allowing them to scale to large datasets.

**Automatic Differentiation:**
The use of automatic differentiation simplifies the implementation of complex models and likelihoods, further promoting the use of these stochastic gradient schemes in contemporary research and applications.

# 7. Summary and Insights
**Langevin MCMC Fundamentals:**
The Langevin approach leverages gradient information to guide the sampling process, moving the state in regions of high probability and adding noise to ensure exploration.

**Challenges with Large $n$:**
In modern applications, particularly those involving large datasets or complex models like Bayesian neural networks, the need for $O(n)$ computations per iteration in evaluating gradients and acceptance ratios becomes a major barrier.

**Subsampling with Mini-Batches:**
By using mini-batch gradients, we obtain an unbiased estimator of the full gradient, effectively reducing the per-iteration computational cost from $O(n)$ to $O(|I_k|)$. The unbiasedness is critical to ensure that the long-term behavior of the chain is not adversely affected.

**Stochastic Gradient Langevin Dynamics (SGLD):**
SGLD adapts the ULA update by incorporating the mini-batch gradient, providing a practical and scalable method for Bayesian inference. Despite the trade-off of introducing additional variance, the overall convergence properties remain similar under appropriate conditions.

**Key Trade-Off – Variance vs. Efficiency:**
The success of these methods hinges on managing the variance introduced by subsampling. In practice, this often requires balancing mini-batch size and step size to achieve both computational efficiency and accurate posterior approximations.

