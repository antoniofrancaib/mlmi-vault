
## Part 1: Foundations and Extension to Infinite Dimensions

### 1. Revisiting the Classical Metropolis-Hastings (MH) Algorithm (Finite Dimensions)

**1.1. Objective:** 
Our fundamental goal is to generate samples $\{ \mathbf{x}^{(j)} \}_{j=1}^N$ from a target probability distribution $\pi$, which is defined on a state space $\mathcal{X} \subseteq \mathbb{R}^d$. We often only know $\pi$ up to a normalization constant, i.e., $\pi(\mathbf{x}) = \tilde{\pi}(\mathbf{x}) / Z$, where $\tilde{\pi}(\mathbf{x})$ can be evaluated but the normalization constant $Z = \int_{\mathcal{X}} \tilde{\pi}(\mathbf{x}) d\mathbf{x}$ is unknown or intractable.

**1.2. Mechanism:** 
The MH algorithm constructs a Markov chain $(\mathbf{X}^{(j)})_{j \geq 0}$ on $\mathcal{X}$ whose transitions are designed such that its unique stationary (or invariant) distribution is the target distribution $\pi$. If the chain is constructed correctly (i.e., it is irreducible and aperiodic), the Ergodic Theorem ensures that for suitable functions $f$:

$$\lim_{N \to \infty} \frac{1}{N} \sum_{j=1}^N f(\mathbf{X}^{(j)}) = \int_{\mathcal{X}} f(\mathbf{x}) \pi(\mathbf{x}) d\mathbf{x} = \mathbb{E}_\pi[f(\mathbf{X})] \quad (\text{almost surely}).$$

**1.3. Algorithm Steps:** 
Given the current state $\mathbf{x}^{(j)}$, the next state $\mathbf{x}^{(j+1)}$ is generated as follows:

- **1.3.1. Propose:** Draw a candidate state $\mathbf{z}$ from a *proposal distribution* $q(\cdot | \mathbf{x}^{(j)})$. This distribution may depend on the current state $\mathbf{x}^{(j)}$. We denote its density function by $q(\mathbf{z} | \mathbf{x}^{(j)})$.

- **1.3.2. Accept/Reject:** Compute the *acceptance probability*:

$$\alpha(\mathbf{x}^{(j)}, \mathbf{z}) = \min\left\{1, \frac{\pi(\mathbf{z}) q(\mathbf{x}^{(j)} | \mathbf{z})}{\pi(\mathbf{x}^{(j)}) q(\mathbf{z} | \mathbf{x}^{(j)})}\right\}$$

Note that this ratio only requires the *unnormalized* density $\tilde{\pi}$, as the normalization constant $Z$ cancels out: 

$$\frac{\pi(\mathbf{z}) q(\mathbf{x}^{(j)} | \mathbf{z})}{\pi(\mathbf{x}^{(j)}) q(\mathbf{z} | \mathbf{x}^{(j)})} = \frac{(\tilde{\pi}(\mathbf{z})/Z) q(\mathbf{x}^{(j)} | \mathbf{z})}{(\tilde{\pi}(\mathbf{x}^{(j)})/Z) q(\mathbf{z} | \mathbf{x}^{(j)})} = \frac{\tilde{\pi}(\mathbf{z}) q(\mathbf{x}^{(j)} | \mathbf{z})}{\tilde{\pi}(\mathbf{x}^{(j)}) q(\mathbf{z} | \mathbf{x}^{(j)})}$$


- **1.3.3. Update:** Draw a uniform random variable $u \sim \text{Uniform}[0,1]$.
	* If $u \leq \alpha(\mathbf{x}^{(j)}, \mathbf{z})$, accept the proposal: $\mathbf{x}^{(j+1)} = \mathbf{z}$.
	* Otherwise, reject the proposal: $\mathbf{x}^{(j+1)} = \mathbf{x}^{(j)}$.

The core requirement ensuring $\pi$ is the stationary distribution is the *detailed balance condition*:

$$\pi(\mathbf{x}) p(\mathbf{y}|\mathbf{x}) = \pi(\mathbf{y}) p(\mathbf{x}|\mathbf{y})$$

where $p(\mathbf{y}|\mathbf{x})$ is the transition kernel density of the Markov chain. The MH acceptance probability $\alpha$ is constructed precisely to satisfy this condition.

**1.4. Underlying Assumption:** The standard formulation implicitly assumes that the target distribution $\pi$ and the proposal distribution $q$ possess density functions with respect to the $d$-dimensional *Lebesgue measure* $\lambda_d$ on $\mathbb{R}^d$. *This assumption breaks down fundamentally in infinite-dimensional spaces.*

### 2. Metropolis-Hastings in Infinite-Dimensional Hilbert Spaces

**2.1. Motivation:** 
Many modern statistical problems, especially in Bayesian inverse problems (e.g., inferring a function causing an observed signal) or functional data analysis, are naturally formulated on *infinite-dimensional spaces*, typically separable Hilbert spaces $(\mathbf{H}, \langle \cdot, \cdot \rangle)$.

**2.2. Fundamental Challenge:** 
There is no *analogue of the Lebesgue measure on an infinite-dimensional Hilbert space*. A measure that is translation-invariant (like Lebesgue) would necessarily assign infinite measure to bounded sets or zero measure to all sets. This means we cannot define probability distributions via densities w.r.t. such a measure. Probability measures must instead be defined directly or via densities (Radon-Nikodym derivatives) with respect to a suitable *reference measure*.

**2.3. Challenge: Naïve Proposals and Divergence:**

* **2.3.1. Naïve Proposal:** A natural first thought is to generalize the standard Gaussian random walk proposal. Let $\mathbf{I}$ be the identity operator on $\mathbf{H}$. Consider $q(\cdot | \mathbf{x}^{(j)}) = \mathcal{N}(\mathbf{x}^{(j)}, \sigma^2 \mathbf{I})$. This corresponds to proposing $z_n = x_n^{(j)} + \sigma \xi_n$ where $\xi_n \sim \mathcal{N}(0,1)$ are i.i.d. across an orthonormal basis $\{e_n\}_{n=1}^\infty$.

* **2.3.2. Problem: Divergence of Norms:** Let $\mathbf{z} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$. Its squared norm is $\|\mathbf{z}\|^2 = \sum_{n=1}^\infty z_n^2$, where $z_n = \langle \mathbf{z}, e_n \rangle \sim \mathcal{N}(0, \sigma^2)$. The expected squared norm is:

$$\mathbb{E}[\|\mathbf{z}\|^2] = \mathbb{E}\left[\sum_{n=1}^\infty z_n^2\right] = \sum_{n=1}^\infty \mathbb{E}[z_n^2] = \sum_{n=1}^\infty \sigma^2 = \infty.$$

By Kolmogorov's zero-one law or direct calculation, $\|\mathbf{z}\|^2 = \infty$ almost surely.

* **2.3.3. Consequence:** Samples drawn from $\mathcal{N}(0, \sigma^2 \mathbf{I})$ do not lie within the Hilbert space $\mathbf{H}$ (which requires finite norm) almost surely. The proposal distribution is not supported on $\mathbf{H}$.

* **2.3.4. Resolution: Trace Class Covariance:** To define a Gaussian measure supported on $\mathbf{H}$, its covariance operator $\mathbf{C}$ must be *trace class*. $\mathbf{C}$ is a linear, self-adjoint, positive semi-definite operator. Let $\{\lambda_n\}_{n=1}^\infty$ be its eigenvalues corresponding to an orthonormal basis of eigenvectors $\{e_n\}_{n=1}^\infty$. $\mathbf{C}$ is trace class if its trace is finite:

$$\text{Tr}(\mathbf{C}) = \sum_{n=1}^\infty \langle \mathbf{C} e_n, e_n \rangle = \sum_{n=1}^\infty \lambda_n < \infty.$$

If $\mathbf{z} \sim \mathcal{N}(0, \mathbf{C})$, then $\mathbf{z} = \sum_{n=1}^\infty z_n e_n$ where $z_n \sim \mathcal{N}(0, \lambda_n)$ are independent. The expected squared norm is $\mathbb{E}[\|\mathbf{z}\|^2] = \sum \lambda_n = \text{Tr}(\mathbf{C}) < \infty$. This finiteness ensures that $\|\mathbf{z}\|^2 < \infty$ almost surely, so $\mathbf{z} \in \mathbf{H}$ a.s.

* **2.3.5. Modified Proposal:** We must use proposals based on trace-class covariances, e.g., $q(\cdot | \mathbf{x}^{(j)}) = \mathcal{N}(\mathbf{x}^{(j)}, \mathbf{C})$.

**2.4. Reformulating the Target and Acceptance Probability:**

* **2.4.1. Bayesian Formulation:** In Bayesian settings on $\mathbf{H}$, the prior distribution $\mu_0$ is often chosen as a Gaussian measure, $\mu_0 = \mathcal{N}(0, \mathbf{C}_0)$, where $\mathbf{C}_0$ is trace class. Given data $\mathbf{y}$, the posterior measure $\mu_y$ is defined via Bayes' theorem in terms of Radon-Nikodym derivatives relative to the prior:

$$\frac{d\mu_y}{d\mu_0}(\mathbf{x}) = \frac{1}{Z} L(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z} \exp(-\Phi(\mathbf{x}))$$

where $L(\mathbf{y} \mid \mathbf{x})$ is the likelihood function, often expressed via a potential or negative log-likelihood $\Phi(\mathbf{x})$. $Z = \int_{\mathbf{H}} \exp(-\Phi(\mathbf{x})) d\mu_0(\mathbf{x})$ is the evidence.

* **2.4.2. Measure-Theoretic Acceptance Ratio:** The classical MH ratio involving densities w.r.t. Lebesgue measure is replaced by a ratio of Radon-Nikodym derivatives w.r.t. a suitable reference measure. Let the target be $\mu = \mu_y$ and the proposal transition kernel be $Q(\mathbf{x}, d\mathbf{z})$, defining $q(d\mathbf{z} | \mathbf{x})$. The MH acceptance probability must ensure detailed balance for measures: 

$$\mu(d\mathbf{x}) Q(\mathbf{x}, d\mathbf{z}) = \mu(d\mathbf{z}) Q(\mathbf{z}, d\mathbf{x})$$

The acceptance probability is formally:

$$\alpha(\mathbf{x}, \mathbf{z}) = \min\left\{1, \frac{d(\mu Q)}{d(\mu \otimes Q)}(\mathbf{z}, \mathbf{x}) \right\}$$

where $\mu \otimes Q$ is the measure defined by $(\mu \otimes Q)(A \times B) = \int_A Q(\mathbf{x}, B) \mu(d\mathbf{x})$. If we choose the Gaussian prior $\mu_0$ as the reference measure and assume a symmetric proposal kernel $Q$ ($Q(\mathbf{x}, d\mathbf{z}) = Q(\mathbf{z}, d\mathbf{x})$), the ratio involves the Radon-Nikodym derivative of the target w.r.t. the prior:

$$\frac{d\mu_y}{d\mu_0}(\mathbf{z}) / \frac{d\mu_y}{d\mu_0}(\mathbf{x}) = \frac{\exp(-\Phi(\mathbf{z}))}{\exp(-\Phi(\mathbf{x}))} = \exp(\Phi(\mathbf{x}) - \Phi(\mathbf{z}))$$

*Caution:* This simplified form holds only under specific proposal choices (see below).

* **2.4.3. Log-Posterior "Energy" Functional:** We can formally define an "energy" associated with the posterior measure $\mu_y$, relative to the (non-existent) Lebesgue measure, by combining the negative log-likelihood and the negative log-prior density (itself formally defined relative to Lebesgue):

$$J(\mathbf{v}) := -\log L(\mathbf{y} | \mathbf{v}) - \log \pi_0(\mathbf{v}) = \Phi(\mathbf{v}) + \frac{1}{2} \langle \mathbf{v}, \mathbf{C}_0^{-1} \mathbf{v} \rangle + \text{const.}$$

Here, $\pi_0$ is the formal density of $\mathcal{N}(0, \mathbf{C}_0)$, and $\langle \mathbf{v}, \mathbf{C}_0^{-1} \mathbf{v} \rangle = \|\mathbf{C}_0^{-1/2} \mathbf{v}\|^2$. This term represents the prior energy.

* **2.4.4. General Acceptance Probability Form (Formal):** If we were using a generic proposal $q$ and could use densities relative to some base measure, the acceptance ratio would compare the "posterior densities" at the proposed and current points, adjusted for proposal asymmetry:

$$\frac{\pi_y(\mathbf{z}) q(\mathbf{x} | \mathbf{z})}{\pi_y(\mathbf{x}) q(\mathbf{z} | \mathbf{x})} \propto \frac{\exp(-J(\mathbf{z})) q(\mathbf{x} | \mathbf{z})}{\exp(-J(\mathbf{x})) q(\mathbf{z} | \mathbf{x})}$$

Leading to $\alpha(\mathbf{x}, \mathbf{z}) = \min\left\{1, \exp(J(\mathbf{x}) - J(\mathbf{z})) \frac{q(\mathbf{x} | \mathbf{z})}{q(\mathbf{z} | \mathbf{x})} \right\}$.

**2.5. Challenge: Divergence of the Prior Energy Term:**

* **2.5.1. Problem:** Consider the prior energy term $\|\mathbf{C}_0^{-1/2} \mathbf{v}\|^2$. Let $\mathbf{v} \sim \mu_0 = \mathcal{N}(0, \mathbf{C}_0)$. Then $\mathbf{v} = \sum v_n e_n$ with $v_n \sim \mathcal{N}(0, \lambda_n)$. The term $\mathbf{C}_0^{-1/2}\mathbf{v}$ has components $\langle \mathbf{C}_0^{-1/2}\mathbf{v}, e_n \rangle = v_n / \sqrt{\lambda_n} \sim \mathcal{N}(0, 1)$. The squared norm is:

$$\|\mathbf{C}_0^{-1/2} \mathbf{v}\|^2 = \sum_{n=1}^\infty \left(\frac{v_n}{\sqrt{\lambda_n}}\right)^2$$

* **2.5.2. Derivation of Divergence:** The expectation is:

$$\mathbb{E}_{\mu_0} [\|\mathbf{C}_0^{-1/2} \mathbf{v}\|^2] = \sum_{n=1}^\infty \mathbb{E}\left[\left(\frac{v_n}{\sqrt{\lambda_n}}\right)^2\right] = \sum_{n=1}^\infty \mathbb{E}[(\mathcal{N}(0,1))^2] = \sum_{n=1}^\infty 1 = \infty.$$

Similar to the norm divergence, this prior energy term diverges almost surely for samples $\mathbf{v}$ drawn from the prior $\mu_0$.

* **2.5.3. Consequence:** If both the current state $\mathbf{x}$ and the proposed state $\mathbf{z}$ are typical samples related to $\mu_0$ (as is often the case, e.g., in a random walk proposal $\mathbf{z} = \mathbf{x} + \xi$ with $\xi \sim \mathcal{N}(0, \mathbf{C})$), then both $J(\mathbf{x})$ and $J(\mathbf{z})$ will involve adding $\Phi(\cdot)$ (assumed finite) to an infinite term. The difference $J(\mathbf{x}) - J(\mathbf{z})$ becomes ill-defined ($\infty - \infty$), rendering the acceptance probability unusable for such proposals.

**2.6. Resolution via Stochastic Differential Equations (SDEs):** 
A successful approach avoids this issue by constructing a proposal mechanism that inherently preserves the prior measure $\mu_0$. This is often achieved by discretizing an SDE whose invariant measure is $\mu_0$.

* **2.6.1. Underlying SDE:** Consider the infinite-dimensional Ornstein-Uhlenbeck process:

$$d\mathbf{U}_s = -\mathbf{U}_s \, ds + \sqrt{2\mathbf{C}_0} \, d\mathbf{B}_s$$

where $\mathbf{B}_s$ is a cylindrical Brownian motion appropriate for $\mathbf{H}$. This SDE has $\mu_0 = \mathcal{N}(0, \mathbf{C}_0)$ as its unique invariant measure.

* **2.6.2. Discretization: The Theta Method:** Applying the $\theta$-method with step size $\delta$ to discretize this SDE from time $s$ (state $\mathbf{u} = \mathbf{U}_s$) to time $s+\delta$ (state $\mathbf{v} \approx \mathbf{U}_{s+\delta}$) yields:

$$\mathbf{v} - \mathbf{u} = \delta \left( -(1-\theta)\mathbf{u} - \theta \mathbf{v} \right) + \sqrt{2\delta \mathbf{C}_0} \, \mathbf{\xi}_0$$

where $\mathbf{\xi}_0 \sim \mathcal{N}(0, \mathbf{I})$ (formally, requires care in infinite dim). Rearranging gives:

$$(\mathbf{I} + \delta \theta) \mathbf{v} = (\mathbf{I} - \delta(1-\theta)) \mathbf{u} + \sqrt{2\delta \mathbf{C}_0} \, \mathbf{\xi}_0$$
$$\mathbf{v} = (\mathbf{I} + \delta \theta)^{-1} \left[ (\mathbf{I} - \delta(1-\theta)) \mathbf{u} + \sqrt{2\delta \mathbf{C}_0} \, \mathbf{\xi}_0 \right]$$

* **2.6.3. Optimal Choice $\theta = 1/2$:** A crucial insight (related to preserving Gaussian measures under linear transformations and ensuring reversibility) is that choosing $\theta = 1/2$ (the Crank-Nicolson scheme) leads to a discretization that *exactly* preserves the prior measure $\mu_0$. If $\mathbf{u} \sim \mathcal{N}(0, \mathbf{C}_0)$, then the resulting $\mathbf{v}$ also follows $\mathcal{N}(0, \mathbf{C}_0)$. This choice is essential for well-definedness in infinite dimensions.

* **2.6.4. Resulting Proposal: Preconditioned Crank-Nicolson (pCN):** With $\theta = 1/2$ and some algebraic manipulation (often introducing a parameter $\beta$ related to $\delta$, e.g., $\beta^2$ related to step size), the proposal takes the form:

$$\mathbf{v} = \sqrt{1 - \beta^2} \, \mathbf{u} + \beta \, \mathbf{\xi}, \quad \text{where } \mathbf{\xi} \sim \mathcal{N}(0, \mathbf{C}_0)$$

Here, $\beta \in (0, 1)$ controls the step size. This proposal mixes the current state $\mathbf{u}$ with a fresh draw from the prior.

**2.7. The Well-Defined pCN Metropolis-Hastings Algorithm:**

* **2.7.1. Key Property:** The pCN proposal $Q(\mathbf{u}, d\mathbf{v})$ leaves the prior measure $\mu_0 = \mathcal{N}(0, \mathbf{C}_0)$ invariant. That is, $\int_{\mathbf{H}} Q(\mathbf{u}, A) d\mu_0(\mathbf{u}) = \mu_0(A)$ for any measurable set $A$. Furthermore, the proposal is reversible w.r.t. $\mu_0$.

* **2.7.2. Simplified Acceptance Probability:** When using the pCN proposal to sample from the posterior $\mu_y$ (defined by $d\mu_y/d\mu_0 \propto \exp(-\Phi(\mathbf{x}))$), the MH acceptance probability simplifies dramatically. The general ratio involves terms related to the proposal and the target density relative to the *invariant measure of the proposal*. Since the pCN proposal's invariant measure is $\mu_0$, the ratio becomes:

$$\frac{ (d\mu_y/d\mu_0)(\mathbf{z}) }{ (d\mu_y/d\mu_0)(\mathbf{u}) } = \frac{\exp(-\Phi(\mathbf{z}))}{\exp(-\Phi(\mathbf{u}))} = \exp(\Phi(\mathbf{u}) - \Phi(\mathbf{z}))$$

Thus, the acceptance probability is:

$$\alpha(\mathbf{u}, \mathbf{v}) = \min\{1, \exp(\Phi(\mathbf{u}) - \Phi(\mathbf{v}))\}$$

Crucially, this expression only involves the (negative) log-likelihood term $\Phi$, which is assumed to be well-defined and finite. The problematic prior energy terms $\|\mathbf{C}_0^{-1/2} \cdot \|^2$ have vanished from the ratio because the proposal was specifically designed to preserve the measure $\mu_0$ whose formal density involves this term.

* **2.7.3. Justification: Equivalence of Measures:** The rigorous justification relies on the concept of absolute continuity of measures. For the MH ratio to be well-defined, the measures $\mu_y(d\mathbf{x}) Q(\mathbf{x}, d\mathbf{z})$ and $\mu_y(d\mathbf{z}) Q(\mathbf{z}, d\mathbf{x})$ must be mutually absolutely continuous. Using the pCN proposal ensures that the proposal kernel $Q$ is absolutely continuous w.r.t. $\mu_0$. The absolute continuity of the full measures then depends on properties of $\Phi$.

* **2.7.4. Variance Ratio Condition:** The requirement that the discretization preserves the Gaussian measure structure (essential for avoiding the infinite energy terms) is deeply connected to ensuring the measures induced by the proposal remain equivalent. Analyzing the variance of individual components $v_i$ vs $u_i$ under the $\theta$-scheme leads to a condition on the sum of squared relative changes in standard deviations:

$$\sum_{i=1}^\infty \left(\frac{\text{std}(v_i)}{\text{std}(u_i)} - 1\right)^2 < \infty$$

This condition, related to the Kakutani dichotomy for infinite product measures, is satisfied only when $\theta = 1/2$. This mathematically necessitates the Crank-Nicolson scheme for well-definedness in this context.

**2.8. Summary:** The pCN-MH algorithm provides a principled and robust way to perform MCMC sampling on infinite-dimensional Hilbert spaces. It overcomes the limitations of classical MH by:
1.  Working with measures relative to a Gaussian prior $\mu_0$.
2.  Employing a carefully constructed pCN proposal (derived from SDE discretization with $\theta=1/2$) that preserves $\mu_0$.
3.  Resulting in a simplified and well-defined acceptance probability involving only the log-likelihood potential $\Phi$.
This makes the algorithm robust to the dimension of the problem, avoiding the curse of dimensionality inherent in naïve generalizations.

## Part 2: Langevin Dynamics MCMC Methods

### 3. Continuous-Time Langevin Dynamics

**3.1. Core Concept:** 
Langevin MCMC methods leverage gradient information from the target distribution $\pi$ to propose moves more intelligently, potentially leading to faster exploration compared to random walk proposals. The foundation is a continuous-time stochastic process whose dynamics are related to $\pi$.

**3.2. The Langevin SDE:** 
Consider a particle moving in a potential landscape $U(\mathbf{x})$ subject to friction and random thermal fluctuations. Its position $\mathbf{X}_t \in \mathbb{R}^d$ evolves according to the Langevin Stochastic Differential Equation (SDE):

$$d\mathbf{X}_t = -\nabla U(\mathbf{X}_t)\,dt + \sqrt{2}\,d\mathbf{B}_t$$

where $(\mathbf{B}_t)_{t \geq 0}$ is a $d$-dimensional standard Brownian motion. The term $-\nabla U(\mathbf{X}_t)$ is the drift force, pulling the particle towards lower potential, and $\sqrt{2}\,d\mathbf{B}_t$ represents isotropic random noise.

* **Potential Function:** In the context of sampling from $\pi$, we define the potential as the negative logarithm of the target density (up to an additive constant):

$$U(\mathbf{x}) = -\log \pi(\mathbf{x})$$

* **Alternative Form:** Substituting $U(\mathbf{x})$, we have $\nabla U(\mathbf{x}) = -\nabla \log \pi(\mathbf{x}) = -\frac{\nabla \pi(\mathbf{x})}{\pi(\mathbf{x})}$. The SDE becomes:

$$d\mathbf{X}_t = \nabla \log \pi(\mathbf{X}_t)\,dt + \sqrt{2}\,d\mathbf{B}_t$$

The drift term $\nabla \log \pi(\mathbf{X}_t)$ points in the direction of increasing log-probability density.

* **3.3. Stationarity:** Under suitable regularity conditions on $U$ (or $\pi$), typically ensuring sufficient confinement (e.g., $\int e^{-U(\mathbf{x})} d\mathbf{x} < \infty$ and $\lim_{\|\mathbf{x}\|\to\infty} U(\mathbf{x}) = \infty$), the stochastic process $(\mathbf{X}_t)$ defined by the Langevin SDE is ergodic and possesses a unique stationary distribution. This stationary distribution is precisely the target distribution $\pi(\mathbf{x}) \propto e^{-U(\mathbf{x})}$. Simulating this SDE for a long time yields samples approximately distributed according to $\pi$.

* **3.4. Fokker-Planck Equation:** The evolution of the probability density $\rho(t, \mathbf{x})$ (or $\rho_t(\mathbf{x})$) of the process $\mathbf{X}_t$ is described by the Fokker-Planck equation (also known as the forward Kolmogorov equation for this process):

$$\frac{\partial \rho_t}{\partial t} = -\nabla \cdot \mathbf{J}_t$$

where $\mathbf{J}_t$ is the probability flux. For the Langevin SDE $d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t) dt + \sigma d\mathbf{B}_t$ with drift $\mathbf{b}(\mathbf{x}) = \nabla \log \pi(\mathbf{x})$ and constant diffusion $\sigma = \sqrt{2}\mathbf{I}$, the Fokker-Planck equation is:

$$\frac{\partial \rho_t}{\partial t} = -\nabla \cdot (\mathbf{b}(\mathbf{x}) \rho_t) + \frac{1}{2} \sum_{i,j=1}^d \frac{\partial^2}{\partial x_i \partial x_j} ((\sigma \sigma^T)_{ij} \rho_t)$$
$$\frac{\partial \rho_t}{\partial t} = -\nabla \cdot (\rho_t \nabla \log \pi) + \Delta \rho_t$$

where $\Delta = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2}$ is the Laplacian operator.

**Verification of Stationarity:** A distribution $\pi$ is stationary if $\frac{\partial \pi}{\partial t} = 0$. Let's substitute $\rho_t = \pi$ into the right-hand side:

$$-\nabla \cdot (\pi \nabla \log \pi) + \Delta \pi$$

Using the identity $\nabla \log \pi = \frac{\nabla \pi}{\pi}$, we have $\pi \nabla \log \pi = \nabla \pi$. So the expression becomes:

$$-\nabla \cdot (\nabla \pi) + \Delta \pi = -\Delta \pi + \Delta \pi = 0$$

This confirms that $\pi(\mathbf{x}) \propto e^{-U(\mathbf{x})}$ is indeed the stationary solution of the Fokker-Planck equation, and thus the stationary distribution of the Langevin SDE.

### 4. Convergence Properties (Log-Concave Settings)

The rate at which the distribution of $\mathbf{X}_t$, denoted $\pi_t$, converges to the stationary distribution $\pi$ is crucial for understanding the efficiency of Langevin-based methods. Strong assumptions on $\pi$ yield strong convergence guarantees.

 **4.1. Log-Concavity:** A probability density $\pi$ on $\mathbb{R}^d$ is *log-concave* if its logarithm, $\log \pi(\mathbf{x})$, is a concave function. That is, for any $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ and $\lambda \in (0,1)$:

$$\log \pi(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \geq \lambda \log \pi(\mathbf{x}) + (1-\lambda) \log \pi(\mathbf{y})$$

Equivalently, the potential $U(\mathbf{x}) = -\log \pi(\mathbf{x})$ is a *convex* function. Log-concave distributions include Gaussian, Exponential, Uniform on convex sets, Laplace, etc. They are unimodal and have desirable geometric properties.

**4.2. $m$-Strong Log-Concavity:** A density $\pi$ is *$m$-strongly log-concave* for some $m > 0$ if $\log \pi(\mathbf{x})$ is strongly concave with parameter $m$. That is, for any $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ and $\lambda \in (0,1)$:

$$\log \pi(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \geq \lambda \log \pi(\mathbf{x}) + (1-\lambda) \log \pi(\mathbf{y}) + \frac{m\lambda(1-\lambda)}{2} \|\mathbf{x} - \mathbf{y}\|^2$$

This is equivalent to the potential $U(\mathbf{x}) = -\log \pi(\mathbf{x})$ being *$m$-strongly convex*. That is, the function $U(\mathbf{x}) - \frac{m}{2}\|\mathbf{x}\|^2$ is convex. Equivalently, the Hessian matrix $\nabla^2 U(\mathbf{x})$ satisfies $\nabla^2 U(\mathbf{x}) \succeq m \mathbf{I}$ (in the sense of positive semi-definite matrices) for all $\mathbf{x}$. Gaussian distributions $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ are $m$-strongly log-concave with $m = \lambda_{\min}(\boldsymbol{\Sigma}^{-1})$. Strong log-concavity implies a unique mode and ensures that the potential $U$ curves upwards quadratically in all directions.

**4.3. Exponential Convergence in Wasserstein-2 ($W_2$) Metric:** The *Wasserstein-2 distance* between two probability measures $\mu$ and $\nu$ on $\mathbb{R}^d$ is defined as:

$$W_2(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|\mathbf{x} - \mathbf{y}\|^2 d\gamma(\mathbf{x}, \mathbf{y}) \right)^{1/2}$$

where $\Gamma(\mu, \nu)$ is the set of all joint probability measures on $\mathbb{R}^d \times \mathbb{R}^d$ with marginals $\mu$ and $\nu$. It measures the "cost" of transporting mass from $\mu$ to $\nu$.

* **Result:** If the target distribution $\pi$ is $m$-strongly log-concave, the distribution $\pi_t$ of the Langevin SDE $d\mathbf{X}_t = \nabla \log \pi(\mathbf{X}_t) dt + \sqrt{2} d\mathbf{B}_t$ converges exponentially fast to $\pi$ in the $W_2$ metric:

$$W_2(\pi_t, \pi) \leq e^{-mt} W_2(\pi_0, \pi)$$

where $\pi_0$ is the distribution of the initial state $\mathbf{X}_0$. This result, stemming from the Bakry-Émery theory or optimal transport approaches (like Jordan-Kinderlehrer-Otto gradient flow interpretation), shows that the distance to stationarity decays exponentially with rate $m$. The $m$-strong convexity of $U$ drives this rapid convergence. (Note: The formula in the lecture notes involving $\mathbf{x}^\star$ and $(d/m)^{1/2}$ applies to a specific initial condition $\pi_0 = \delta_\mathbf{x}$ and relates $W_2$ to initial distance and dimension-dependent variance).

### 5. Discretized Langevin Algorithms

Since SDEs cannot be simulated exactly, we use numerical integration schemes.

**5.1. Euler-Maruyama Discretization:** The simplest first-order scheme applied to $d\mathbf{X}_t = \nabla \log \pi(\mathbf{X}_t) dt + \sqrt{2} d\mathbf{B}_t$ with a step-size $\gamma > 0$ yields the update rule:

$$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1}$$

where $\mathbf{X}_k \approx \mathbf{X}_{k\gamma}$ and $\mathbf{W}_{k+1} \sim \mathcal{N}(0, \mathbf{I})$ are independent standard Gaussian random vectors representing the increment $\Delta \mathbf{B}_k = \mathbf{B}_{(k+1)\gamma} - \mathbf{B}_{k\gamma} \sim \mathcal{N}(0, \gamma \mathbf{I})$.

* **Issue:** This discretization introduces an error of order $O(\gamma)$ relative to the continuous process. The stationary distribution of this discrete-time Markov chain is generally *not* $\pi$.

**5.2. Metropolis-Adjusted Langevin Algorithm (MALA)**

* **5.2.1. Goal:** To correct the discretization bias of the Euler scheme and design a Markov chain whose stationary distribution is exactly $\pi$.

* **5.2.2. Method:** Use the Euler step as a proposal mechanism within a Metropolis-Hastings framework.
	1.  *Propose:* $\mathbf{X}_{k+1}^{\text{prop}} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1}$.
	2.  *Accept/Reject:* Calculate the acceptance probability $\alpha$ and accept/reject as in standard MH.

* **5.2.3. Proposal Density:** The proposal $\mathbf{X}_{k+1}^{\text{prop}}$ is drawn from a Gaussian distribution centered at the deterministic part of the Euler step. The conditional density of proposing $\mathbf{x}'$ given $\mathbf{x}$ is:

$$q(\mathbf{x}' | \mathbf{x}) = (4\pi\gamma)^{-d/2} \exp\left(-\frac{1}{4\gamma} \|\mathbf{x}' - (\mathbf{x} + \gamma \nabla \log \pi(\mathbf{x}))\|^2\right)$$

* **5.2.4. Acceptance Probability:** Substituting $\pi$ and this $q$ into the MH ratio gives:

$$\alpha(\mathbf{X}_k, \mathbf{X}_{k+1}^{\text{prop}}) = \min\left\{1, \frac{\pi(\mathbf{X}_{k+1}^{\text{prop}}) q(\mathbf{X}_k \mid \mathbf{X}_{k+1}^{\text{prop}})}{\pi(\mathbf{X}_k) q(\mathbf{X}_{k+1}^{\text{prop}} \mid \mathbf{X}_k)}\right\}$$

Note that the proposal $q$ is *not* symmetric, so the full ratio must be computed.

* **5.2.5. Property:** By construction, the MALA algorithm satisfies the detailed balance condition with respect to $\pi$. If the chain is irreducible and aperiodic, its unique stationary distribution is $\pi$.

* **5.2.6. Practical Issues:**
	* *Step-size Tuning:* The choice of $\gamma$ is crucial. Too large $\gamma$ leads to low acceptance rates ($\alpha \to 0$); too small $\gamma$ leads to high acceptance but slow exploration (adjacent states are highly correlated). Optimal scaling theory suggests tuning $\gamma$ such that the average acceptance rate is around 0.574 for high dimensions $d$.
	* *High Dimensions:* The acceptance rate can degrade significantly as $d$ increases, potentially requiring $\gamma \propto d^{-1/3}$ for constant acceptance, limiting the effective step size.

**5.3. Unadjusted Langevin Algorithm (ULA)**

* **5.3.1. Method:** Simply use the Euler-Maruyama discretization directly without any Metropolis-Hastings correction:
$$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla \log \pi(\mathbf{X}_k) + \sqrt{2\gamma} \mathbf{W}_{k+1}$$

* **5.3.2. Properties:**
	* *Bias:* The ULA chain does *not* have $\pi$ as its stationary distribution. It converges to a perturbed stationary measure $\pi_\gamma$, which approximates $\pi$ for small $\gamma$. The bias typically scales as $O(\gamma)$, meaning $W_2(\pi_\gamma, \pi) = O(\sqrt{\gamma d})$ under suitable conditions.
	* *Efficiency:* Each step is computationally cheaper than MALA (no acceptance probability calculation needed). It avoids the potentially low acceptance rates of MALA in high dimensions, often leading to faster exploration despite the bias.

* **5.3.3. Convergence Guarantee (under $m$-strong log-concavity, $L$-Lipschitz gradient $\nabla \log \pi$):** The convergence of the distribution $\pi_k$ of $\mathbf{X}_k$ to the *true target* $\pi$ can be bounded. A typical result for $W_2$ distance:

$$W_2(\pi_k, \pi) \leq (1 - c_1 m\gamma)^k W_2(\pi_0, \pi) + c_2 \frac{L}{m} \sqrt{\gamma d}$$

(Constants $c_1, c_2$ depend on specific assumptions, the $1.65$ factor in notes is one possibility). This shows two terms:
1.  An exponential decay of the initial error (with rate related to $m\gamma$).
2.  An asymptotic bias term that depends on the step size $\gamma$, dimension $d$, strong convexity $m$, and Lipschitz constant $L$. The bias can be made small by choosing small $\gamma$, but this slows down the exponential decay.

### 6. Special Case: Gaussian Targets

Analyzing Langevin methods for Gaussian targets provides concrete insights.

**6.1. Target:** $\pi(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$. The potential is $U(\mathbf{x}) = \frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) + \text{const}$.

**6.2. Continuous SDE:** The gradient is $\nabla U(\mathbf{x}) = \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$. The Langevin SDE becomes the *Ornstein-Uhlenbeck (OU) process*:

$$d\mathbf{X}_t = -\boldsymbol{\Sigma}^{-1} (\mathbf{X}_t - \boldsymbol{\mu})\,dt + \sqrt{2}\,d\mathbf{B}_t$$

This is a linear SDE whose stationary distribution is exactly $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.

**6.3. ULA for Gaussian Target:**
* **Update:** The ULA update rule is linear:

$$\mathbf{X}_{k+1} = \mathbf{X}_k - \gamma \boldsymbol{\Sigma}^{-1} (\mathbf{X}_k - \boldsymbol{\mu}) + \sqrt{2\gamma} \mathbf{W}_{k+1}$$

$$\mathbf{X}_{k+1} = (\mathbf{I} - \gamma \boldsymbol{\Sigma}^{-1})\mathbf{X}_k + \gamma \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} + \sqrt{2\gamma} \mathbf{W}_{k+1}$$

* **Invariant Measure $\pi_\gamma$**: This is a discrete-time linear Gaussian process (an AR(1) process). Its stationary distribution $\pi_\gamma$ is also Gaussian. We can find its mean and covariance. Let $\mathbf{X}_\infty \sim \pi_\gamma$. Taking expectation in the update yields $\mathbb{E}[\mathbf{X}_\infty] = (\mathbf{I} - \gamma \boldsymbol{\Sigma}^{-1})\mathbb{E}[\mathbf{X}_\infty] + \gamma \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$, which solves to $\mathbb{E}[\mathbf{X}_\infty] = \boldsymbol{\mu}$. The covariance $\boldsymbol{\Sigma}_\gamma = \text{Cov}(\mathbf{X}_\infty)$ satisfies the discrete Lyapunov equation:

$$\boldsymbol{\Sigma}_\gamma = (\mathbf{I} - \gamma \boldsymbol{\Sigma}^{-1}) \boldsymbol{\Sigma}_\gamma (\mathbf{I} - \gamma \boldsymbol{\Sigma}^{-1})^T + 2\gamma \mathbf{I}$$

Solving this yields (after some algebra, related to continuous/discrete time Lyapunov equations):

$$\pi_\gamma = \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}_\gamma \right) \quad \text{where} \quad \boldsymbol{\Sigma}_\gamma = \boldsymbol{\Sigma} \left(\mathbf{I} + (\mathbf{I} - \gamma \boldsymbol{\Sigma}^{-1})^{-1}\right)^{-1} \approx \boldsymbol{\Sigma} \left( \mathbf{I} + \frac{\gamma}{2} \boldsymbol{\Sigma}^{-1} \right)$$

Alternatively, the form from the notes $\boldsymbol{\Sigma}_\gamma = \boldsymbol{\Sigma}\left(\mathbf{I} - \frac{\gamma}{2} \boldsymbol{\Sigma}^{-1}\right)^{-1}$ (assuming $\mathbf{I} - \gamma \boldsymbol{\Sigma}^{-1}$ is symmetric, which requires $\boldsymbol{\Sigma}$ to be diagonal or $\gamma$ small) also shows that $\boldsymbol{\Sigma}_\gamma \approx \boldsymbol{\Sigma}( \mathbf{I} + \frac{\gamma}{2} \boldsymbol{\Sigma}^{-1} )$ for small $\gamma$.

The key point is that $\boldsymbol{\Sigma}_\gamma \neq \boldsymbol{\Sigma}$, explicitly demonstrating the bias introduced by the ULA discretization. The bias is $O(\gamma)$.



## Part 3: Scaling Langevin MCMC for Large Datasets

### 7. Langevin MCMC in Bayesian Inference

* **7.1. Context:** Consider Bayesian inference where the posterior distribution is the target:
    $\pi(\mathbf{x} | \mathbf{y}) \propto p(\mathbf{x}) L(\mathbf{y} | \mathbf{x})$
    where $\mathbf{x} \in \mathbb{R}^d$ are parameters, $p(\mathbf{x})$ is the prior, and $\mathbf{y} = \{y_1, \dots, y_n\}$ is the dataset. If data points are conditionally independent given $\mathbf{x}$, the likelihood is $L(\mathbf{y} | \mathbf{x}) = \prod_{i=1}^n L(y_i | \mathbf{x})$. The log-posterior is:
    $$\log \pi(\mathbf{x} | \mathbf{y}) = \log p(\mathbf{x}) + \sum_{i=1}^n \log L(y_i | \mathbf{x}) + \text{const.}$$

* **7.2. Computational Challenges for Large $n$:**
    * **7.2.1. Full Gradient Cost:** Langevin methods require the gradient of the log-posterior:
        $$\nabla \log \pi(\mathbf{x} | \mathbf{y}) = \nabla \log p(\mathbf{x}) + \sum_{i=1}^n \nabla \log L(y_i | \mathbf{x})$$
        Computing this gradient requires summing contributions from all $n$ data points. The computational cost per iteration is therefore $O(nd_i)$, where $d_i$ is the cost per data point, typically dominated by $O(n)$ when $n$ is large. This is prohibitive for massive datasets (e.g., $n > 10^6$).
    * **7.2.2. MALA Acceptance Cost:** The MALA acceptance probability $\alpha$ requires evaluating $\pi(\mathbf{x}^{\text{prop}})$ and $\pi(\mathbf{x})$, which in turn requires computing the sum $\sum_{i=1}^n \log L(y_i | \mathbf{x})$ at both points. This also incurs an $O(n)$ cost per iteration.

### 8. Stochastic Gradient Estimation via Subsampling

* **8.1. Rationale:** The $O(n)$ cost stems from the sum over the entire dataset. If we could approximate this sum cheaply, we could reduce the per-iteration cost. Stochastic Gradient Descent (SGD) in optimization uses gradients estimated from small subsets (mini-batches) of data. We adapt this idea for MCMC.

* **8.2. Mini-Batch Gradient Estimator:** At iteration $k$, select a mini-batch $I_k$ of size $m = |I_k| \ll n$, typically by sampling $m$ indices uniformly *with replacement* from $\{1, \dots, n\}$. An unbiased estimator for the sum term in the gradient is constructed by scaling the mini-batch sum:
    $$\sum_{i=1}^n \nabla \log L(y_i | \mathbf{X}_k) \approx \frac{n}{m} \sum_{i \in I_k} \nabla \log L(y_i | \mathbf{X}_k)$$
    The full gradient estimator is then:
    $$\nabla \widetilde{\log \pi}(\mathbf{X}_k) := \nabla \log p(\mathbf{X}_k) + \frac{n}{m} \sum_{i \in I_k} \nabla \log L(y_i | \mathbf{X}_k)$$

* **8.3. Key Property: Unbiasedness:** This estimator is unbiased for the true gradient, assuming uniform sampling of indices in $I_k$. Let $i^*$ be a random index drawn uniformly from $\{1, \dots, n\}$. Then $\mathbb{E}_{i^*}[\nabla \log L(y_{i^*} | \mathbf{x})] = \frac{1}{n} \sum_{j=1}^n \nabla \log L(y_j | \mathbf{x})$. If $I_k = \{i_1, \dots, i_m\}$ where each $i_j$ is drawn independently and uniformly:
    $$
    \begin{align*} \mathbb{E}_{I_k}\left[ \nabla \widetilde{\log \pi}(\mathbf{x}) \right] &= \mathbb{E}_{I_k}\left[ \nabla \log p(\mathbf{x}) + \frac{n}{m} \sum_{j=1}^m \nabla \log L(y_{i_j} | \mathbf{x}) \right] \\ &= \nabla \log p(\mathbf{x}) + \frac{n}{m} \sum_{j=1}^m \mathbb{E}_{i_j}[\nabla \log L(y_{i_j} | \mathbf{x})] \\ &= \nabla \log p(\mathbf{x}) + \frac{n}{m} \sum_{j=1}^m \left( \frac{1}{n} \sum_{l=1}^n \nabla \log L(y_l | \mathbf{x}) \right) \\ &= \nabla \log p(\mathbf{x}) + \frac{n}{m} \cdot m \cdot \left( \frac{1}{n} \sum_{l=1}^n \nabla \log L(y_l | \mathbf{x}) \right) \\ &= \nabla \log p(\mathbf{x}) + \sum_{l=1}^n \nabla \log L(y_l | \mathbf{x}) \\ &= \nabla \log \pi(\mathbf{x}) \end{align*}
    $$
    The scaling factor $n/m$ is crucial for unbiasedness. (Note: Sampling *without* replacement also yields an unbiased estimator and often lower variance, but analysis is slightly different).

* **8.4. Trade-off:**
    * *Cost:* The cost per iteration is reduced to $O(m d_i)$, significantly less than $O(n d_i)$ when $m \ll n$.
    * *Variance:* The stochastic gradient estimator $\nabla \widetilde{\log \pi}$ has higher variance than the true gradient $\nabla \log \pi$. This introduces additional noise into the algorithm. $\text{Var}(\nabla \widetilde{\log \pi}) \propto n/m$ (under certain conditions).

### 9. Stochastic Gradient Langevin Dynamics (SGLD)

* **9.1. Algorithm:** SGLD modifies the ULA update by replacing the true gradient with the unbiased mini-batch estimator:
    $$\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma_k \nabla \widetilde{\log \pi}(\mathbf{X}_k) + \sqrt{2\gamma_k} \mathbf{W}_{k+1}$$
    Note the use of a potentially decreasing step-size sequence $\gamma_k$.

* **9.2. Properties:**
    * *Scalability:* Highly scalable to large datasets due to $O(m)$ cost per iteration.
    * *Metropolis-Free:* It avoids the $O(n)$ cost of the MALA acceptance step. This means it inherits a discretization bias similar to ULA.
    * *Convergence:* Under appropriate conditions on $\pi$ (e.g., strong log-concavity) and the step sizes ($\sum \gamma_k = \infty, \sum \gamma_k^2 < \infty$), SGLD converges to the true posterior $\pi$. The stochastic gradient noise effectively acts like additional diffusion, but its unbiasedness ensures the drift is correct on average. Convergence rates often resemble ULA but may have additional terms related to the gradient variance. The decreasing step size $\gamma_k \to 0$ is typically required to eliminate the asymptotic bias, unlike fixed-$\gamma$ ULA which converges to $\pi_\gamma$. Bounds often look like $W_2(\pi_k, \pi) \leq O((1+C\gamma_k)^{-k} + \text{Bias}(\gamma_k) + \text{VarianceTerm})$.

* **9.3. Importance of Variance Control:** The variance introduced by the mini-batch gradients can significantly impact performance. If the variance is too high, the chain may mix slowly or require very small step sizes. Techniques to reduce variance (e.g., control variates like SVRG-Langevin, larger mini-batch size $m$) can improve performance but may increase cost per iteration.

* **9.4. Applications:** SGLD and its variants are very popular for large-scale Bayesian inference, particularly for training Bayesian neural networks where $n$ (number of data points) and $d$ (number of parameters/weights) can both be very large. It integrates naturally with automatic differentiation frameworks for computing $\nabla \log L(y_i | \mathbf{x})$.

---

## Part 4: Addressing Multimodality with Parallel Tempering

### 10. The Challenge of Multimodal Distributions

* **10.1. Problem:** Many target distributions $\pi(\mathbf{x})$ of interest are *multimodal*, meaning they possess multiple distinct regions of high probability (modes) separated by regions of low probability (valleys or energy barriers).
* **10.2. Consequence:** Standard MCMC algorithms (MH, MALA, ULA, SGLD), which rely on local moves (random walks or gradient-based steps), can become *trapped* within the basin of attraction of a single mode. The probability of proposing a move that jumps across a low-probability valley to another mode can be vanishingly small, especially in high dimensions.
* **10.3. Implication:** The chain fails to explore the entire support of $\pi$ adequately, leading to biased estimates and poor characterization of uncertainty. This is problematic in:
    * *Machine Learning:* Posterior distributions over model parameters (e.g., mixture models, some neural networks) can be multimodal.
    * *Physics/Chemistry:* Energy landscapes of molecules (e.g., protein folding) have many local minima (modes).
    * *Optimization:* Finding global optima in non-convex landscapes requires escaping local optima.

### 11. Tempered Distributions

* **11.1. Idea:** To overcome energy barriers, Parallel Tempering (PT) uses a ladder of auxiliary distributions that interpolate between an easy-to-sample distribution and the target distribution $\pi$. These intermediate distributions are "flattened" versions of $\pi$, making it easier to move between modes.

* **11.2. Potential Function:** Define $U(\mathbf{x}) = -\log \pi(\mathbf{x})$.

* **11.3. Tempered Density Definition:** Introduce an *inverse temperature* parameter $\beta \in (0, 1]$. Define the $\beta$-tempered distribution $\pi_\beta$ as:
    $$\pi_\beta(\mathbf{x}) \propto [\pi(\mathbf{x})]^\beta = \exp(-\beta \log \pi(\mathbf{x})) = \exp(-\beta U(\mathbf{x}))$$
    The normalization constant depends on $\beta$: $Z_\beta = \int \exp(-\beta U(\mathbf{x})) d\mathbf{x}$.

* **11.4. Properties:**
    * **Target:** When $\beta=1$, $\pi_1(\mathbf{x}) = \pi(\mathbf{x})$ is the original target distribution.
    * **High Temperature (Low $\beta$):** As $\beta \to 0$, the distribution $\pi_\beta$ becomes flatter. The potential landscape $\beta U(\mathbf{x})$ has lower barriers compared to $U(\mathbf{x})$. A chain sampling from $\pi_\beta$ with small $\beta$ can more easily traverse the state space and jump between regions that correspond to modes of $\pi$. In the limit $\beta \to 0$, $\pi_\beta$ approaches a uniform distribution over the accessible state space (if compact) or might become improper.
    * **Low Temperature (High $\beta$, sometimes used):** If $\beta > 1$, the distribution becomes more sharply peaked around the modes of $\pi$.

### 12. Parallel Tempering (Replica Exchange MCMC) Algorithm

* **12.1. Setup:** Run $K$ Markov chains (replicas) in parallel, indexed $i=1, \dots, K$.
* **12.2. Temperature Ladder:** Assign an inverse temperature $\beta_i$ to each chain $i$, such that they form a ladder:
    $1 = \beta_1 > \beta_2 > \dots > \beta_K > 0$
    Chain $i$ is designed to sample from the tempered distribution $\pi_{\beta_i}$.
* **12.3. Algorithm Steps (per iteration):**
    * **12.3.1. Local Moves:** Each chain $i$ evolves independently for one or more steps using a standard MCMC algorithm (e.g., MH, MALA) whose stationary distribution is its target $\pi_{\beta_i}$. Let the state of chain $i$ be $\mathbf{x}_i$.
    * **12.3.2. Replica Exchange Moves:** Periodically (e.g., every few local moves, or randomly), attempt to swap the *entire states* between selected pairs of chains. A common strategy is to attempt swaps only between adjacent chains in the temperature ladder, i.e., between chain $i$ and chain $i+1$.

* **12.4. Swap Acceptance Probability:** Consider attempting to swap the states between chain $i$ (with temperature $\beta_i$, state $\mathbf{x}_i$) and chain $j$ (with temperature $\beta_j$, state $\mathbf{x}_j$). The proposed move is to change the state vector $(\dots, \mathbf{x}_i, \dots, \mathbf{x}_j, \dots)$ to $(\dots, \mathbf{x}_j, \dots, \mathbf{x}_i, \dots)$. This swap is accepted with probability $\alpha_{\text{swap}}(i \leftrightarrow j)$ determined by the Metropolis-Hastings criterion applied to the *extended state* $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_K)$ on the product space $\mathcal{X}^K$. The target distribution on this extended space is the product of the individual tempered distributions:
    $$\Pi(\mathbf{X}) = \prod_{k=1}^K \pi_{\beta_k}(\mathbf{x}_k) \propto \prod_{k=1}^K \exp(-\beta_k U(\mathbf{x}_k))$$
    The swap proposal is symmetric (swapping $i,j$ then back yields the original state). The MH acceptance probability is therefore:
    $$
    \begin{align*} \alpha_{\text{swap}}(i \leftrightarrow j) &= \min\left\{1, \frac{\Pi(\dots, \mathbf{x}_j, \dots, \mathbf{x}_i, \dots)}{\Pi(\dots, \mathbf{x}_i, \dots, \mathbf{x}_j, \dots)}\right\} \\ &= \min\left\{1, \frac{ \pi_{\beta_i}(\mathbf{x}_j) \pi_{\beta_j}(\mathbf{x}_i) \prod_{k \neq i,j} \pi_{\beta_k}(\mathbf{x}_k) }{ \pi_{\beta_i}(\mathbf{x}_i) \pi_{\beta_j}(\mathbf{x}_j) \prod_{k \neq i,j} \pi_{\beta_k}(\mathbf{x}_k) }\right\} \\ &= \min\left\{1, \frac{\pi_{\beta_i}(\mathbf{x}_j) \pi_{\beta_j}(\mathbf{x}_i)}{\pi_{\beta_i}(\mathbf{x}_i) \pi_{\beta_j}(\mathbf{x}_j)}\right\} \end{align*}
    $$
    Substituting $\pi_\beta(\mathbf{x}) \propto \exp(-\beta U(\mathbf{x}))$, the ratio becomes:
    $$
    \begin{align*} \frac{\exp(-\beta_i U(\mathbf{x}_j)) \exp(-\beta_j U(\mathbf{x}_i))}{\exp(-\beta_i U(\mathbf{x}_i)) \exp(-\beta_j U(\mathbf{x}_j))} &= \exp\left( -\beta_i U(\mathbf{x}_j) - \beta_j U(\mathbf{x}_i) + \beta_i U(\mathbf{x}_i) + \beta_j U(\mathbf{x}_j) \right) \\ &= \exp\left( (\beta_i - \beta_j) U(\mathbf{x}_i) - (\beta_i - \beta_j) U(\mathbf{x}_j) \right) \\ &= \exp\left[ (\beta_i - \beta_j) (U(\mathbf{x}_i) - U(\mathbf{x}_j)) \right] \end{align*}
    $$
    So, the final acceptance probability is:
    $$\alpha_{\text{swap}}(i \leftrightarrow j) = \min\left\{1, \exp\left[ (\beta_i - \beta_j) (U(\mathbf{x}_i) - U(\mathbf{x}_j)) \right]\right\}$$

* **12.5. Justification:** The overall algorithm defines a Markov chain on the extended state space $\mathcal{X}^K$. The combination of local moves (which leave each $\pi_{\beta_i}$ invariant marginally) and swap moves (which satisfy detailed balance with respect to the product measure $\Pi$) ensures that the stationary distribution of the extended chain is $\Pi(\mathbf{X})$. Consequently, the marginal distribution of the state $\mathbf{x}_i$ for chain $i$ converges to the desired tempered distribution $\pi_{\beta_i}$. In particular, the chain $i=1$ (with $\beta_1=1$) provides samples from the original target distribution $\pi$.

### 13. Practical Considerations for Parallel Tempering

* **13.1. Temperature Schedule:** The choice of temperatures $\{ \beta_i \}_{i=1}^K$ is critical.
    * *Spacing:* Temperatures must be close enough so that adjacent distributions $\pi_{\beta_i}$ and $\pi_{\beta_{i+1}}$ have sufficient overlap. If they are too dissimilar, the term $|U(\mathbf{x}_i) - U(\mathbf{x}_{i+1})|$ in the swap probability will likely be large, leading to very low acceptance rates ($\alpha_{\text{swap}} \approx 0$). This prevents information flow between chains. A common strategy is geometric spacing: $\beta_{i+1} = \beta_i / c$ for some constant $c > 1$.
    * *Range:* The lowest temperature $\beta_K$ must be small enough to allow exploration over the largest energy barriers in $U(\mathbf{x})$.
    * *Adaptation:* Adaptive methods exist to adjust the temperatures during the run to achieve target swap acceptance rates (e.g., 20-40%) between adjacent chains.
* **13.2. Swap Strategy:**
    * *Pairs:* Swapping only adjacent temperatures $(\beta_i, \beta_{i+1})$ is standard, as non-adjacent swaps usually have negligible acceptance probability.
    * *Frequency:* Swaps should be attempted frequently enough to allow states to effectively diffuse up and down the temperature ladder, but not so frequently that it excessively disrupts local exploration within each chain.
* **13.3. Computational Aspects:**
    * *Parallelism:* The local moves for all $K$ chains can be performed entirely in parallel. Swap steps require minimal communication (exchanging states and energy values $U(\mathbf{x}_i)$). PT is highly amenable to parallel computing.
    * *Convergence Diagnosis:* Assessing convergence is more complex than for standard MCMC. One needs to check convergence within each chain *and* ensure sufficient mixing across the temperature ladder (i.e., states should travel from $\beta_K$ to $\beta_1$ and back multiple times).
* **13.4. Benefit:** By allowing states discovered by high-temperature chains (low $\beta$) to be swapped down the ladder, eventually reaching the target chain ($\beta_1=1$), PT enables the sampler to escape local modes and explore the full state space much more effectively than standard MCMC methods when dealing with multimodal distributions. The target chain $\mathbf{x}_1$ benefits from the global exploration facilitated by the auxiliary chains.
