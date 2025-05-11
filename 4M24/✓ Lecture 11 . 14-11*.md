Lecture by Mark Girolami

*topics:

# LECTURE-11

## Overview
MCMC methods, and in particular the Metropolis–Hastings (MH) algorithm, are central to computational statistics and machine learning. In many applications—especially in Bayesian inverse problems and functional data analysis—it is natural to work in an infinite-dimensional setting. These infinite-dimensional problems are naturally formulated in a Hilbert space. However, the classical finite-dimensional MH algorithm does not directly carry over because many underlying assumptions (such as the existence of a Lebesgue measure) break down.

The lecture explores:

- **Revisiting the classical MH algorithm**: To set the stage.
- **Defining an MH algorithm in an infinite-dimensional space**: Overcoming the nonexistence of standard Lebesgue measure.
- **Ensuring correctness in Hilbert space**: Guaranteeing that proposals and acceptance probabilities are well defined.
- **Developing a Markov chain sampler that is robust to the curse of dimensionality**: So that performance does not degrade as the dimension grows.

### Revisiting the Metropolis–Hastings Algorithm

In its simplest form, the MH algorithm is used to simulate from a target distribution $\pi(\cdot)$ by constructing a Markov chain whose stationary distribution is $\pi$. The basic algorithm is:

For $j=1$ to $N$:

1. **Propose**: Draw $\mathbf{z}$ from a proposal distribution $q(\mathbf{x}^{(j)}, \cdot)$.
2. **Accept/Reject Step**: Draw $u$ from the Uniform[0,1] distribution and calculate the acceptance probability
   $$\alpha(\mathbf{x}^{(j)}, \mathbf{z}) = \min\left\{\frac{\pi(\mathbf{z}) q(\mathbf{z}, \mathbf{x}^{(j)})}{\pi(\mathbf{x}^{(j)}) q(\mathbf{x}^{(j)}, \mathbf{z})}, 1\right\}.$$

3. **Update**:
   - If $u \leq \alpha(\mathbf{x}^{(j)}, \mathbf{z})$, set $\mathbf{x}^{(j+1)} = \mathbf{z}$.
   - Otherwise, set $\mathbf{x}^{(j+1)} = \mathbf{x}^{(j)}$.

Return the sample $\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)}\}$.

In finite dimensions, the target density is normally defined with respect to the Lebesgue measure. However, once we step into an infinite-dimensional Hilbert space, the Lebesgue measure is no longer defined, which necessitates a rethink of how densities and transition probabilities are formulated.

### Challenges of Infinite-Dimensional Spaces

#### Unbounded Norms and Divergence
Consider a Hilbert space $\mathbf{H}$ and suppose we try to define the proposal as

$$q(\mathbf{x}^{(j)}, \cdot) = \mathcal{N}(\mathbf{x}^{(j)}, \mathbf{I}),$$

where $\mathbf{I}$ is the identity operator. In an infinite-dimensional space, if one were to sample each coordinate independently with $\mathcal{N}(0,1)$, the norm of $\mathbf{x}^{(j)}$ will almost surely be unbounded. In short:

**Divergence of Norms**: Random variables $x_n \sim \mathcal{N}(0,1)$ tend to diverge since the sum of infinitely many independent variances is infinite.

**Derivation:** Let’s say you're working in the Hilbert space $\ell^2$, the space of square-summable sequences:

$$
\ell^2 = \left\{ x = (x_1, x_2, \dots) \in \mathbb{R}^\infty \ \middle| \ \sum_{n=1}^\infty x_n^2 < \infty \right\}.
$$

Now suppose we define a **standard Gaussian proposal**:

$$
q(\cdot) = \mathcal{N}(0, I),
$$

where $I$ is the identity operator on $\ell^2$. That is, we're trying to define a Gaussian measure whose coordinates are i.i.d.:

$$
x_n \sim \mathcal{N}(0, 1) \quad \text{independently for } n = 1, 2, \dots
$$
The squared Hilbert norm is:
$$
\|x\|^2 = \sum_{n=1}^\infty x_n^2.
$$

Since each $x_n \sim \mathcal{N}(0,1)$, we know $\mathbb{E}[x_n^2] = 1,$ and the variables are independent. So by linearity of expectation:

$$
\mathbb{E}[\|x\|^2] = \sum_{n=1}^\infty \mathbb{E}[x_n^2] = \sum_{n=1}^\infty 1 = \infty.
$$

That is, the **expected squared norm is infinite**. This Means the Norm Diverges Almost Surely. To be in $\ell^2$, a sample $x$ must satisfy $\sum x_n^2 < \infty$. So, almost every sample from  does not live in  — meaning it has infinite norm.

### Resolution with a Trace Class Covariance Operator

To make the proposal well defined:

**Covariance Modification**: Replace the identity with a trace class covariance operator $\mathbf{C}$ to obtain

$$q(\mathbf{x}^{(j)}, \cdot) = \mathcal{N}(\mathbf{x}^{(j)}, \mathbf{C}).$$

This ensures that random draws will have finite norms (since the trace class property guarantees that the sum of eigenvalues is finite) and that convergence properties are maintained.

## Redefining the Proposal Mechanism in Hilbert Space

The traditional acceptance probability

$$\alpha(\mathbf{x}^{(j)}, \mathbf{z}) = \min\left\{\frac{\pi(\mathbf{z}) q(\mathbf{z}, \mathbf{x}^{(j)})}{\pi(\mathbf{x}^{(j)}) q(\mathbf{x}^{(j)}, \mathbf{z})}, 1\right\}$$

relies on the target density $\pi(\mathbf{z})$. In the infinite-dimensional setting, the target is not defined with respect to Lebesgue measure but rather via a Radon–Nikodym derivative with respect to a Gaussian measure.

### Bayes' Formula in Infinite Dimensions

If $\mu_y$ is the posterior measure and $\mu_0$ is the Gaussian prior $\mathcal{N}(0, \mathbf{C})$, then Bayes’ formula is reformulated as:

$$\frac{d\mu_y}{d\mu_0}(\mathbf{x}) \propto L(y \mid \mathbf{x}),$$

where $L(y \mid \mathbf{x})$ is the likelihood. In our context, one often expresses the likelihood in the form

$$L(y \mid \mathbf{x}) = \exp(-\Phi(\mathbf{x})).$$

Thus, the acceptance probability involves quantities related to the change of measure and must be redefined accordingly.

## Acceptance Probability: From Finite to Infinite Dimensions

### Redefinition Using Measure-Theoretic Concepts

To adapt the acceptance probability, the term $\pi(\mathbf{z})q(\mathbf{z}, \mathbf{x}^{(j)})$ is replaced by the measure $\mu_y(d\mathbf{z})q(d\mathbf{z}, \mathbf{x}^{(j)})$. In particular, one shows that for the choice

$$\mathbf{v} = \mathbf{u} + \beta \mathbf{\xi} \quad \text{with} \quad \mathbf{\xi} \sim \mathcal{N}(0, \mathbf{C}),$$

if the likelihood is given by

$$L(y \mid \mathbf{v}) = \exp(-\Phi(\mathbf{v})),$$

then the acceptance probability takes the form:

$$\alpha(\mathbf{u}, \mathbf{v}) = \min\{\exp(J(\mathbf{v}) - J(\mathbf{u})), 1\}$$

with

$$J(\mathbf{v}) = \log L(y \mid \mathbf{v}) - \frac{1}{2} \langle \mathbf{v}, \mathbf{C}^{-1} \mathbf{v} \rangle = -\Phi(\mathbf{v}) - \frac{1}{2} \|\mathbf{C}^{-1/2} \mathbf{v}\|^2.$$

## MH acceptance in measure form.

Let $Q(\mathbf{u}, d\mathbf{v})$ be your proposal kernel. The general MH rule is

$$
\alpha(\mathbf{u}, \mathbf{v}) = 1 \wedge \frac{\mu^y(d\mathbf{v}) \; Q(\mathbf{v}, d\mathbf{u})}{\mu^y(d\mathbf{u}) \; Q(\mathbf{u}, d\mathbf{v})}
= 1 \wedge \frac{\frac{d\mu^y}{d\mu_0}(\mathbf{v}) \; \mu_0(d\mathbf{v}) \; Q(\mathbf{v}, d\mathbf{u})}{\frac{d\mu^y}{d\mu_0}(\mathbf{u}) \; \mu_0(d\mathbf{u}) \; Q(\mathbf{u}, d\mathbf{v})}.
$$

**Derivation:** We’ll build up the theory from scratch and derive the infinite-dimensional Metropolis–Hastings acceptance probability expression:

$$
\alpha(u, v) = \min\left(1, \exp\left(J(v) - J(u)\right)\right),
$$

where:
$$
J(v) = -\Phi(v) - \frac{1}{2} \| \mathbf{C}^{-1/2} v \|^2.
$$

*Step 0: The Goal*:  We want to sample from a posterior distribution $\mu_y$ over some space $\mathbf{H}$ (possibly infinite-dimensional, e.g., a function space). This posterior is defined as:

$$
\mu_y(dv) \propto L(y \mid v) \cdot \mu_0(dv),
$$

where:

- $L(y \mid v) = \exp(-\Phi(v))$ is the likelihood,
- $\mu_0$ is a Gaussian prior, $\mathcal{N}(0, \mathbf{C})$, over $\mathbf{H}$,
- $\mu_y$ is the posterior measure we wish to sample from.

This is Bayes' Rule in measure-theoretic form:

$$
\frac{d\mu_y}{d\mu_0}(v) = \frac{1}{Z} \exp(-\Phi(v)),
$$

with normalization constant

$$
Z = \int \exp(-\Phi(v)) \, d\mu_0(v).
$$

*Step 1: Proposal Distribution:* We define a proposal mechanism:

$$
v = u + \beta \, \xi, \quad \xi \sim \mathcal{N}(0, \mathbf{C})
$$

This means:

- $u$ is the current state (sample from previous iteration),
- $\xi$ is Gaussian noise with the same covariance as the prior,
- $\beta$ is a scaling parameter.

This is a symmetric random walk: the probability of proposing $v$ from $u$ is the same as proposing $u$ from $v$, because both just differ by symmetric Gaussian noise.

*Step 2: Metropolis–Hastings Acceptance Probability:* In general, for target measure $\mu$ and proposal kernel $q$, the Metropolis–Hastings acceptance probability is:

$$
\alpha(u, v) = \min\left(1, \frac{d\mu}{dq}(v) \cdot \frac{dq}{d\mu}(u)\right)
$$
In continuous settings, we replace densities (which may not exist) with **Radon–Nikodym derivatives**, so the Metropolis–Hastings acceptance probability becomes:

$$
\alpha(u, v) = \min\left(1,\ 
\frac{\displaystyle \frac{d\mu}{d\lambda}(v) \cdot \frac{dq(v, \cdot)}{d\lambda}(u)}{\displaystyle \frac{d\mu}{d\lambda}(u) \cdot \frac{dq(u, \cdot)}{d\lambda}(v)}
\right)
$$

for some **reference measure** $\lambda$ on $\mathbf{H} \times \mathbf{H}$. This is valid for any $\lambda$ **dominating** $\mu_y$, $q(u, \cdot)$, and $q(v, \cdot)$. Now we choose $\lambda = \mu_0$, and use the **chain rule of Radon–Nikodym derivatives**:

$$
\alpha(u, v) = \min\left(1,\ 
\frac{\displaystyle \frac{d\mu_y}{d\mu_0}(v) \cdot \frac{dq(v, \cdot)}{dq(u, \cdot)}(u)}{\displaystyle \frac{d\mu_y}{d\mu_0}(u)}
\right)
$$

The last term — the **ratio of proposal kernels** — reflects the **asymmetry** of the proposal. If $q$ is symmetric (as it is here), this simplifies to:

$$
\alpha(u, v) = \min\left(1, \frac{d\mu_y}{d\mu_0}(v) \cdot \frac{d\mu_0}{d\mu_y}(u)\right)
$$

This form compares the likelihood of moving forward versus backward under the two measures $\mu_y$ and $\mu_0$.


*Step 3: Compute the Radon–Nikodym Derivatives:* We know from Bayes’ rule:

$$
\frac{d\mu_y}{d\mu_0}(v) = \frac{1}{Z} \exp(-\Phi(v))
$$

and therefore, the reciprocal:

$$
\frac{d\mu_0}{d\mu_y}(u) = Z \cdot \exp(\Phi(u))
$$

So the acceptance ratio becomes:

$$
\frac{d\mu_y}{d\mu_0}(v) \cdot \frac{d\mu_0}{d\mu_y}(u)
= \left( \frac{1}{Z} \exp(-\Phi(v)) \right) \cdot \left( Z \cdot \exp(\Phi(u)) \right)
= \exp(\Phi(u) - \Phi(v))
$$

But this is not yet enough — it ignores the prior’s true density.

We want to describe the posterior relative to a base measure like Lebesgue, so we include the prior’s density explicitly.

*Step 4: Density of the Prior $\mathcal{N}(0, \mathbf{C})$:* Let’s compute the (formal) log-density of the Gaussian prior $\mu_0$ at point $v$.  In finite dimensions, the density of $\mathcal{N}(0, \mathbf{C})$ is:

$$
\mu_0(v) \propto \exp\left( -\frac{1}{2} \langle v, \mathbf{C}^{-1} v \rangle \right)
= \exp\left( -\frac{1}{2} \| \mathbf{C}^{-1/2} v \|^2 \right)
$$

In infinite dimensions, we cannot use Lebesgue measure as a reference, but we can use another Gaussian measure as the base. So if we define the “energy” functional:

$$
J(v) := \log L(y \mid v) + \log \mu_0(v)
$$

then:

$$
\log L(y \mid v) = -\Phi(v), \quad \log \mu_0(v) = -\frac{1}{2} \| \mathbf{C}^{-1/2} v \|^2,
$$

so:

$$
J(v) = -\Phi(v) - \frac{1}{2} \| \mathbf{C}^{-1/2} v \|^2
$$

This combines:

- the **negative log-likelihood** (data term),
- and the **negative log-prior** (regularization term).

*Final Expression:* The Metropolis–Hastings acceptance probability becomes:

$$
\alpha(u, v) = \min\left(1, \exp(J(v) - J(u))\right)
$$

This generalizes the classical MH rule to **infinite-dimensional Hilbert spaces**, where:

- We use **Radon–Nikodym derivatives** instead of densities w.r.t. Lebesgue,
- And compute **log-densities** using the likelihood and Gaussian prior energy.

*Interpretation:*

- $\Phi(v)$: the **data misfit** — how well $v$ explains observations.
- $\| \mathbf{C}^{-1/2} v \|^2$: how likely $v$ is under the **prior**.
- $J(v)$: total "log-likelihood + log-prior" — a **score** to compare states.

**Acceptance** compares whether the new point $v$ is more probable than the old one $u$.

### Problem with the Naïve Proposal

Even with the trace class operator $\mathbf{C}$, using the additive proposal $\mathbf{v} = \mathbf{u} + \beta \mathbf{\xi}$ leads to difficulties:

**Divergence of the Ratio**:
The $i$th coordinate of $\mathbf{C}^{-1/2} \mathbf{u}$ is distributed as $\mathcal{N}(0,1)$, and the variance of the norm,

$$\mathbb{E}\|\mathbf{C}^{-1/2} \mathbf{u}\|^2 = \sum_{i=1}^\infty \mathbb{E}\left(\frac{u_i}{\sqrt{\lambda_i}}\right)^2,$$

diverges since there are infinitely many terms. Hence, the ratio inherent in the acceptance probability is not well defined.

Derivation: This Quantity Diverges. To understand why $\| \mathbf{C}^{-1/2} v \|^2$ diverges almost surely, let's break it down carefully.

*Step 1*: What is $\mathbf{C}^{-1/2} v$? Assume:

- $\mathbf{C}$ is a **trace-class covariance operator** on a Hilbert space $\mathbf{H}$,
- It has eigenfunctions $\{ e_i \}_{i=1}^\infty$ with eigenvalues $\lambda_i > 0$,

Then for any $u \in \mathbf{H}$, we can write:

$$
u = \sum_{i=1}^\infty u_i e_i \quad \Rightarrow \quad \mathbf{C}^{-1/2} u = \sum_{i=1}^\infty \frac{u_i}{\sqrt{\lambda_i}} e_i
$$
*Step 2*: Compute the Expected Norm

$$
\| \mathbf{C}^{-1/2} u \|^2 = \sum_{i=1}^\infty \left( \frac{u_i}{\sqrt{\lambda_i}} \right)^2
$$

Take expectation under $\mu_0 = \mathcal{N}(0, \mathbf{C})$.  Since $u_i \sim \mathcal{N}(0, \lambda_i)$, then:

$$
\mathbb{E}[u_i^2] = \lambda_i \quad \Rightarrow \quad \mathbb{E}\left[ \left( \frac{u_i}{\sqrt{\lambda_i}} \right)^2 \right] = 1
$$

So the total expected norm becomes:

$$
\mathbb{E} \| \mathbf{C}^{-1/2} u \|^2 = \sum_{i=1}^\infty 1 = \infty
$$

This means the norm **diverges with probability 1**: almost every sample from $\mu_0$ has **infinite prior energy**:
$$
\| \mathbf{C}^{-1/2} u \|^2 = \infty
$$

*Why This Breaks Metropolis–Hastings*. Recall: the acceptance probability depends on the difference of log-densities:

$$
J(v) = -\Phi(v) - \frac{1}{2} \underbrace{\| \mathbf{C}^{-1/2} v \|^2}_{\infty \text{ a.s.}}
$$

If both $u$ and $v \sim \mu_0$, then:

$$
\| \mathbf{C}^{-1/2} u \|^2 = \infty, \quad \| \mathbf{C}^{-1/2} v \|^2 = \infty
$$

So their difference:

$$
J(v) - J(u) = \text{undefined } (\infty - \infty)
$$

This means the **acceptance ratio is ill-defined**, so **Metropolis–Hastings breaks**.

## Derivation via Stochastic Differential Equations and Discretization

A key insight into constructing a well-defined MH algorithm in infinite dimensions comes from considering an underlying stochastic differential equation (SDE) whose invariant measure is the Gaussian prior.

### The Underlying SDE

Consider the SDE:

$$d\mathbf{u} = -\mathbf{u} \, ds + \sqrt{2\mathbf{C}} \, d\mathbf{b},$$

where $\mathbf{b}$ is a standard Brownian motion. This SDE is chosen because:

**Invariant Measure**:
Its invariant measure is $\mu_0 = \mathcal{N}(0, \mathbf{C})$.

### Discretisation Scheme: The Theta Method

To develop a practical algorithm, one discretizes the SDE using the Theta scheme:

$$\mathbf{v} - \mathbf{u} = -\delta((1 - \theta)\mathbf{u} + \theta \mathbf{v}) + \sqrt{2\delta} \, \mathbf{C} \, \mathbf{\xi}_0,$$

where:

- $\mathbf{u}$ is the current state,
- $\mathbf{v}$ the proposed state,
- $\delta$ is the time-step,
- $\theta$ is a discretisation parameter,
- $\mathbf{\xi}_0$ is a standard normal random variable.

Rearranging yields:

$$\mathbf{v} = (\mathbf{I} + \delta \theta)^{-1} \left[\mathbf{u} - \delta(1 - \theta)\mathbf{u} + \sqrt{2\delta} \, \mathbf{C} \, \mathbf{\xi}_0\right].$$

### Choosing the Optimal $\theta$

By setting $\theta = \frac{1}{2}$, one achieves the following:

**Preservation of the Prior**: The proposal is modified such that if $\mathbf{u} \sim \mathcal{N}(0, \mathbf{C})$ then the proposed $\mathbf{v}$ also satisfies $\mathbf{v} \sim \mathcal{N}(0, \mathbf{C})$. This is crucial because preserving the Gaussian prior avoids issues with diverging norms.

**Acceptance Probability Simplification**: With this choice, the resulting acceptance probability simplifies to

$$\alpha(\mathbf{u}, \mathbf{v}) = \min\{\exp(\Phi(\mathbf{u}) - \Phi(\mathbf{v})), 1\},$$

provided $\Phi(\cdot)$ is well defined.

### Expressing the Proposal in a Compact Form

After further algebra (which involves introducing a scaling parameter $\beta$ related to the time-step $\delta$), the final proposal can be written in the form:

$$\mathbf{v} = \sqrt{1 - \beta^2} \, \mathbf{u} + \beta \, \mathbf{\xi}, \quad \mathbf{\xi} \sim \mathcal{N}(0, \mathbf{C}).$$

This form is sometimes referred to as a **preconditioned Crank–Nicolson proposal** and has the crucial property of preserving the Gaussian prior $\mathcal{N}(0, \mathbf{C})$.

## Ensuring Equivalence of Measures

The formal justification of this method rests on showing that the two measures

$$\mu_0(d\mathbf{v}) q(\mathbf{v}, d\mathbf{u}) \quad \text{and} \quad \mu_0(d\mathbf{u}) q(\mathbf{u}, d\mathbf{v})$$

are equivalent when the proposal is defined using the above mechanism.

### The Variance Ratio Condition

**Pointwise Analysis**:
By analyzing each coordinate of the proposal, one finds for the $i$th component:

$$v_i = \frac{1 - \delta(1 - \theta)}{I + \delta \theta} u_i + \sqrt{\frac{2\delta \lambda_i^2}{(I + \delta \theta)^2}} g_i,$$

where $g_i$ is standard Normal.

**Ratio of Variances**:
One requires that the series

$$\sum_{i=1}^\infty \left(\frac{\sigma(v_i)}{\sigma(u_i)} - 1\right)^2$$

is finite. In other words, the fluctuations in the variances between the proposal $\mathbf{v}$ and the current state $\mathbf{u}$ should be controlled so that the infinite product of Gaussian measures remains absolutely continuous with respect to one another.

**Imposition on $\theta$**:
It turns out that the necessary condition is satisfied only when $\theta = \frac{1}{2}$. This is not an arbitrary choice; it is a consequence of matching the variances so that the algorithm is well defined in infinite dimensions.

## Summary and Key Insights

Transitioning from Finite to Infinite Dimensions:

- In a standard finite-dimensional setting, the MH algorithm is defined with densities with respect to the Lebesgue measure. In an infinite-dimensional Hilbert space, one must redefine the target measure using a Radon–Nikodym derivative with respect to a Gaussian prior.
- A naïve application of the finite-dimensional approach leads to divergence issues because the sum of variances (or norms) becomes infinite.

**Proposal Modification**:

- Instead of the additive proposal $\mathbf{v} = \mathbf{u} + \beta \mathbf{\xi}$ which does not preserve the structure of the infinite-dimensional space, a new form

  $$\mathbf{v} = \sqrt{1 - \beta^2} \, \mathbf{u} + \beta \, \mathbf{\xi}$$

  is introduced.
- This proposal is derived via the discretisation of an underlying SDE that leaves the prior invariant. Its careful construction preserves $\mu_0 = \mathcal{N}(0, \mathbf{C})$.

**Correct Acceptance Probability**:

- With the new proposal, the acceptance probability simplifies to

  $$\alpha(\mathbf{u}, \mathbf{v}) = \min\{\exp(\Phi(\mathbf{u}) - \Phi(\mathbf{v})), 1\},$$

  which remains well defined in the infinite-dimensional setting.
- This relies on showing that the measures required in the acceptance ratio are equivalent; a fact guaranteed by the specific choice $\theta = \frac{1}{2}$.

**Practical Implications**:

- The construction ensures that performance does not degrade as the dimensionality increases. This is crucial for applications in fields such as Bayesian inverse problems, where the dimensionality can effectively be infinite.
- The derivation, anchored in both measure theory and stochastic differential equations, highlights the interplay between continuous-time processes and their discrete-time approximations in MCMC.

## Concluding Remarks

The development of MCMC methods in infinite-dimensional spaces is not a trivial extension of finite-dimensional algorithms. Instead, it requires:

1. **Rigorous Measure-Theoretic Reformulation**:
   Adapting Bayes’ rule and the acceptance probability to account for the nonexistence of a Lebesgue measure in infinite dimensions.
2. **Carefully Designed Proposals**:
   Constructing proposals that preserve the underlying Gaussian measure—often through careful discretisation of SDEs—and ensuring the mathematical equivalence of measures for the acceptance ratio.
3. **Optimal Parameter Choice**:
   Demonstrating that, for the correct behavior, the discretisation parameter (e.g., $\theta = \frac{1}{2}$) is not merely a numerical convenience but a mathematical necessity.



