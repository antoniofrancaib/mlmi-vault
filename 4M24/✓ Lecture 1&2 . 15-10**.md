Lecture by Mark Girolami

*topics: deterministic integration methods, monte carlo integration, statistical expectations, central limit theorem and convergence, monte carlo error, variance reduction techniques (importance sampling and control variates)*

# LECTURE-1
# 1. Introduction and Historical Context

**Origin of the Name "Monte Carlo"**

Monte Carlo is the name of a district in Monaco, famous for its casino. Gambling involves random processes, so this was an apt code-name suggested by Nicholas Metropolis for research at the Los Alamos National Laboratory. The original impetus was to use randomization to estimate nuclear diffusion processes during the development of nuclear weapons.

---

# 2. Evaluating Integrals Numerically

## 2.1 Deterministic Approaches

Consider a one-dimensional Riemann integral:

$$\int_0^1 \exp(-x^3) \, dx.$$

Sometimes, no closed-form antiderivative exists, so we resort to numerical methods such as:

**Midpoint Rule:**

$$\int_a^b f(x) \, dx \approx (b-a) f\left(\frac{a+b}{2}\right).$$

**Trapezoidal Rule:**

$$\int_a^b f(x) \, dx \approx (b-a) \frac{f(a) + f(b)}{2}.$$

![[midpoint-vs-trapezoidal.png]]

Higher-Order Quadrature (e.g., Simpson’s rule, Gaussian quadrature).

Deterministic rules become increasingly sophisticated to achieve higher accuracy. However:

- Defining *accuracy and controlling error* in these rules can be intricate.
- Scaling to ***multiple dimensions** is nontrivial*. Deterministic rules in dimensions greater than one suffer from a *curse of dimensionality*, where the number of grid points grows exponentially.

## 2.2 Stochastic (Monte Carlo) Approach

Monte Carlo sidesteps some of the dimensionality issues. To see how:

**Area-Fraction Interpretation (basic illustration)**

Suppose we want $\int_a^b f(x) \, dx$ and know $f(x) \leq M$ for $x \in [a, b]$. Then the area under $f$ is the fraction of a bounding rectangle $[(b-a) \times M]$.

1. Pick $N$ random points uniformly inside that rectangle.
2. Let $\text{acc}$ be the count of points that fall under the graph of $f$.
3. The fraction $\frac{\text{acc}}{N}$ approximates the fraction of the rectangle's area under $f$.

Hence,

$$\int_a^b f(x) \, dx \approx (b-a) \times M \times \frac{\text{acc}}{N}.$$

This is conceptually simple but can be highly inefficient if ***$M$ is large*** or **$f$ is sharply peaked**, leading to many rejections.

![[rejection-sampling.png]]

**Uniform-Sampling Interpretation**

A more direct route avoids explicitly sampling the “vertical” direction. Recall that if $X \sim \text{Uniform}[a, b],$ then

$$E[f(X)] = \frac{1}{b-a} \int_a^b f(x) \, dx.$$

Rearranging yields

$$\int_a^b f(x) \, dx = (b-a) E[f(X)].$$

So by drawing $N$ i.i.d. samples $x_1, \dots, x_N$ from $\text{Uniform}[a, b],$

$$\int_a^b f(x) \, dx \approx (b-a) \times \frac{1}{N} \sum_{n=1}^N f(x_n).$$

This principle generalizes directly to any dimension $d$. Crucially, the statistical error depends only on $N$ and the variance of $f(X);$ it ***does not explode with dimension*** (although practical considerations can still be complicated in very large dimensions).

---

# 3. Evaluating Statistical Expectations

## 3.1 Relationship to Integrals

Often, one needs to compute an expectation

$$E_{X \sim p}[f(X)] = \int f(x) p(x) \, dx,$$

where $p(x)$ is a probability density on the domain of interest. The connection to integrals is immediate.

As an example, if we want $\int_a^b f(x) \, dx$ and $p(x)$ is the uniform density on $[a, b],$ then

$$\int_a^b f(x) \, dx = (b-a) E_p[f(X)].$$

Hence, *numerical integration is often a special case of computing a statistical expectation.*

## 3.2 Monte Carlo Estimation of an Expectation

Vanilla Monte Carlo (MC) estimate:

$$E_{X \sim p}[f(X)] \approx \frac{1}{N} \sum_{n=1}^N f(x_n),$$

where $x_1, \dots, x_N$ are i.i.d. samples from $p.$

This estimator is **unbiased**, meaning its expected value equals the true expectation:

$$E\left[\frac{1}{N} \sum_{n=1}^N f(x_n)\right] = E_{X \sim p}[f(X)].$$

How it converges (and how quickly) is addressed by the Law of Large Numbers and the Central Limit Theorem, as discussed next.

---

# 4. Central Limit Theorem (CLT) and Convergence

## 4.1 Statement of the Central Limit Theorem for Monte Carlo

When we approximate $E_{X \sim p}[f(X)]$ by

$$\frac{1}{N} \sum_{n=1}^N f(x_n),$$

the Central Limit Theorem (CLT) gives a precise asymptotic distribution for the estimator. Specifically:

Let $f_1, f_2, \dots, f_N$ denote the i.i.d. random variables $f(x_n)$ with $E[f_i] = \mu$ and $\text{Var}(f_i) = \sigma_f^2.$

Define

$$\bar{f}_N = \frac{1}{N} \sum_{n=1}^N f_n.$$

Then the CLT states:

$$\sqrt{N} (\bar{f}_N - \mu) \xrightarrow{d} N(0, \sigma_f^2),$$

equivalently

$$\bar{f}_N \sim N\left(\mu, \frac{\sigma_f^2}{N}\right) \text{ for large } N.$$

In words:
- The distribution of the Monte Carlo estimator $\bar{f}_N$ is approximately normal for sufficiently large $N.$
- The standard deviation of the estimator decays like $\frac{1}{\sqrt{N}}.$

### Remark:
We can make the same argument using Moment-Generating Functions (MGF)

<new-argument-goes-here>

</new-argument-goes-here>

> **Refresher:** A ***moment-generating function (MGF)*** of a random variable $X$ is defined as  $$
 M_X(t) = \mathbb{E}[e^{tX}].
 $$
> It’s called the “moment-generating” function because, when you expand $e^{tX}$ as a power series, the coefficients involve the moments $\mathbb{E}[X^n]$. In other words,  $$
 M_X(t) = 1 + t \mathbb{E}[X] + \frac{t^2}{2!} \mathbb{E}[X^2] + \cdots,
 $$
> so knowing the MGF (for all $t$) allows you to recover all the moments of $X$. 


Again, consider a sequence of i.i.d. random variables $f_1, f_2, \dots, f_N$, each with mean $\mathbb{E}[f_i] = \mu$ and variance $\sigma_f^2$. Define their sum as

$$
S_N = \sum_{n=1}^N f_n \quad \text{and the sample mean} \quad \bar{f}_N = \frac{1}{N} S_N.
$$

he MGF of a single $f_i$ is $M_f(t) = \mathbb{E}[e^{t f_i}]$. Since the $f_i$ are i.i.d., the MGF of the sum $S_N$ is

$$
M_{S_N}(t) = \mathbb{E}[e^{t S_N}] = (M_f(t))^N.
$$

Next, note that $f_i$ has mean $\mu$ and variance $\sigma_f^2$. A Taylor expansion of $M_f(t)$ near $t=0$ then looks like

$$
M_f(t) = 1 + t \mu + \frac{t^2}{2} \sigma_f^2 + O(t^3).
$$

To examine the distribution of the centered and scaled sample mean, define

$$
Z_N = \frac{1}{\sqrt{N} \sigma_f} \sum_{n=1}^N (f_n - \mu) = \frac{S_N - N \mu}{\sqrt{N} \sigma_f}.
$$

Its MGF is

$$
M_{Z_N}(t) = \mathbb{E}[e^{t Z_N}] = \mathbb{E}\left[e^{\frac{t}{\sqrt{N} \sigma_f}(S_N - N \mu)}\right].
$$

We can rewrite this in terms of $S_N$ itself:

$$
M_{Z_N}(t) = e^{-\frac{t N \mu}{\sqrt{N} \sigma_f}} \, \mathbb{E}\left[e^{\frac{t}{\sqrt{N} \sigma_f} S_N}\right] = \exp\left(-\frac{t \sqrt{N} \mu}{\sigma_f}\right) \, M_{S_N}\left(\frac{t}{\sqrt{N} \sigma_f}\right).
$$

Since $M_{S_N}(u) = (M_f(u))^N$, we substitute $u = \frac{t}{\sqrt{N} \sigma_f}$ and use the Taylor expansion of $M_f(u)$. For large $N$, this expansion leads to

$$
M_f\left(\frac{t}{\sqrt{N} \sigma_f}\right) = 1 + \frac{t \mu}{\sqrt{N} \sigma_f} + \frac{t^2}{2N} \sigma_f^2 + O\left(\frac{1}{N^{3/2}}\right).
$$

Raising this to the $N$th power and combining the exponential factor $\exp(-t \sqrt{N} \mu / \sigma_f)$ ultimately yields, in the limit $N \to \infty$, the MGF of a standard normal distribution $\mathcal{N}(0, 1)$ for $Z_N$. By the uniqueness of MGFs, we conclude

$$
Z_N = \frac{S_N - N \mu}{\sqrt{N} \sigma_f} \xrightarrow{d} \mathcal{N}(0, 1).
$$

Equivalently, this statement translates to the sample mean $\bar{f}_N$ being approximately normal for large $N$ with mean $\mu$ and variance $\sigma_f^2 / N$. In other words,

$$
\frac{\sqrt{N}}{\sigma_f}(\bar{f}_N - \mu) \xrightarrow{d} \mathcal{N}(0, 1).
$$
which implies the more direct “CLT-like” result:
$$
\bar{f}_N = \frac{1}{N} \sum_{n=1}^N f_n \sim \mathcal{N}\left(\mu, \frac{\sigma_f^2}{N} \right) \quad \text{for large } N,
$$

## 4.2 Rate of Convergence

The error in the Monte Carlo estimate, as measured by standard deviation, decreases on the order of $1/\sqrt{N}.$ 

This rate is dimension-independent in principle (though in high dimensions, designing effective Monte Carlo estimators can still be challenging). The variance constant $\sigma_f^2$ is a property of $f$ and the distribution $p.$ Reducing that variance is often the crucial step in improving Monte Carlo performance.

## 4.3 Example: Estimating a Small-Tail Probability

To see how variance can dominate, consider:

$$I = P(T_\text{life} \geq 25),$$

where $T_\text{life} \sim \text{Exponential}(1).$ Hence, $p(x) = e^{-x}$ for $x \geq 0.$ 

One can attempt a vanilla Monte Carlo approach:

$$\frac{1}{N} \sum_{n=1}^N 1\{x_n \geq 25\}, \quad x_n \sim e^{-x}.$$

But the region $x \geq 25$ has extremely small probability under this distribution ($e^{-25}$ is tiny), so with moderate $N,$ often no samples fall above 25, leading to a naive estimate of $0.$ The variance and computational cost for a good approximation become extremely large.

Such problems motivate the ***Importance Sampling technique***, where one simulates from a more suitable distribution that places more mass on the relevant tails.

---

# LECTURE-2 
# 5. Monte Carlo in Practice

## 5.1 Basic Assumptions

We assume ***one can simulate*** (at least approximately) from the desired distribution $p(x).$ In many standard cases (e.g., Gaussians, Gamma, Beta distributions), efficient routines are readily available. Later in more advanced contexts (e.g., Bayesian posterior distributions that lack a closed form), we relax the requirement for exact sampling and rely on algorithms like Markov Chain Monte Carlo.

## 5.2 Monte Carlo Error

From the CLT analysis:

$$\bar{f}_N = \frac{1}{N} \sum_{n=1}^N f(x_n) \sim N\left(E[f(X)], \frac{\sigma_f^2}{N}\right),$$

with $\sigma_f^2 = E[f^2(X)] - E[f(X)]^2.$ Thus, the variance of the estimate is

$$\text{Var}(\bar{f}_N) = \frac{\sigma_f^2}{N}.$$

This leads to the “error bars” on the estimate shrinking at the rate $1/\sqrt{N}.$ However, $\sigma_f^2$ can be large if $f$ is high-variance under $p.$ That challenge is precisely what ***variance reduction strategies*** address.

e.g. consider evaluating $\mathbb{E}\{x^{10}\}$ where expectation is w.r.t. the uniform distribution on the unit line. Most of the samples would be ***uninformative***, inflating the variance of the estimator.

> **Note:** Even though the variance $\sigma_f^2 = \frac{2541}{100}$ for $f(x) = x^{10}$ might appear small in absolute terms, it is relatively large compared to the expected value $\mathbb{E}[x^{10}] = \frac{1}{11}$.  
> This means that most samples yield extremely small values, with only a few high values near $x = 1$ contributing substantially.  
> As a result, the estimator’s relative error is high, making the Monte Carlo estimation challenging without variance reduction techniques.

---

# 6. Variance Reduction Techniques

Two widely used variance reduction methods in the Monte Carlo toolkit are ***Importance Sampling*** and ***Control Variates***. Both exploit certain insights to reduce $\sigma_f^2,$ making the estimator converge faster in practice (though the theoretical $1/\sqrt{N}$ rate remains the same, the constant factor is reduced).

## 6.1 Importance Sampling (IS)

**Motivation**

Returning to the example of estimating $P(T_\text{life} \geq 25)$ with $T_\text{life} \sim e^{-x},$ the main issue was that the original distribution $p(x) = e^{-x}$ rarely produces samples in $\{x \geq 25\}.$

**Idea**: Replace the original sampling distribution $p$ by another one $q$ (the importance density) that gives more samples in critical regions. Then adjust each sample’s contribution so we still compute the same expectation.

**Derivation**
Any expectation $E_p[f(X)]$ can be rewritten as:

$$\int f(x) p(x) \, dx = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx = E_q\left[f(X) \frac{p(X)}{q(X)}\right],$$

provided $q(x) > 0$ wherever $p(x) > 0.$

Hence, the Importance Sampling estimator is:

$$\frac{1}{N} \sum_{n=1}^N f(x_n) \frac{p(x_n)}{q(x_n)}, \quad x_n \sim q.$$

This estimate remains unbiased.

**Variance Analysis and Optimal Proposal**

Let $\tilde{f}(x) = f(x) \frac{p(x)}{q(x)}.$ Then

$$\text{Var}_q(\tilde{f}(X)) = E_q[\tilde{f}(X)^2] - \left(E_q[\tilde{f}(X)]\right)^2.$$

The difference in errors between $\tilde{f}(x)$ and the original MC estimate is:

$$
\text{Var}_p(f(X)) - \text{Var}_q(\tilde{f}(X)) = \mathbb{E}_p\{f^2(X)\} - \mathbb{E}_q\{\overline{f}^2(X)\}
$$

$$
= \int f^2(x)p(x)dx - \int \frac{f(x)^2p(x)^2}{q(x)}dx
$$

We want to *maximize this difference* because this directly measures _how much better_ the new estimator (under $q$) is compared to the old one (under $p$). If that difference is _positive and large_, it means you have cut down a _lot_ of variance (thus gaining a more efficient estimator).

The biggest decrease in error is achieved when (Lagrange multiplier + Differentiate and set to zero):

$$
q^*(x) = \frac{|f(x)|p(x)}{\int |f(x)|p(x)dx}
$$

This happens to be when $\text{Var}_q(\tilde{f}(X)) = 0$, which is great as **we want to reduce the variance of the importance‐sampling estimator** as much as possible. 

> **Derivation:**  
> We start by noting that the improvement in variance by switching from the original sampling density $p(x)$ to an importance sampling density $q(x)$ is given by the difference $$\Delta = \mathrm{Var}_p(f(X)) - \mathrm{Var}_q(\tilde{f}(X)) = \int f^2(x) p(x) \, dx - \int \frac{f(x)^2 p(x)^2}{q(x)} \, dx.$$
> Since the first term does not depend on $q(x)$, maximizing $\Delta$ is equivalent to minimizing the second term $$\mathcal{J}(q) = \int \frac{f(x)^2 p(x)^2}{q(x)} \, dx,$$
> subject to the normalization constraint $\int q(x) \, dx = 1$. To solve this constrained optimization problem, we introduce a Lagrange multiplier $\lambda$ and form the Lagrangian
> $$\mathcal{L}[q] = \int \frac{f(x)^2 p(x)^2}{q(x)} \, dx + \lambda \left( \int q(x) \, dx - 1 \right).$$
> We now seek the density $q(x)$ that minimizes $L[q]$. To do so, we differentiate the integrand with respect to $q(x)$ (treating $x$ as fixed) and set the derivative to zero. Including the derivative of the constraint term, we have  $$
 -\frac{f(x)^2 p(x)^2}{q(x)^2} + \lambda = 0.
$$
> Solving for $q(x)$ yields $$
 q(x)^2 = \frac{f(x)^2 p(x)^2}{\lambda},
 $$
> or, taking the positive square root (since $q(x)$ must be nonnegative), $$
 q^*(x) = \frac{|f(x)| p(x)}{\sqrt{\lambda}}.
 $$
> The multiplier $\lambda$ is then determined by enforcing the normalization condition. Substituting back, the optimal importance sampling density becomes
> $$
 q^*(x) = \frac{|f(x)| p(x)}{\int |f(x)| p(x) \, dx}.
 $$
Intuitively, for good variance properties, $q$ should be large where $f(x) p(x)$ is large, so that $\frac{p(x)}{q(x)}$ remains near-constant.

In that ideal case, the variance can be driven to zero. In practice, exact matching might be difficult or impossible, but the closer $q$ is to $|f(x)| p(x),$ the smaller the variance.

**Intuition:** To efficiently estimate an integral, we want to allocate more sampling effort to the regions where the function $f(x)$ has a significant contribution. In areas where $f(x)$ is small or negligible, the sampling density should also be low to avoid wasting resources. Conversely, in regions where $f(x)$ is large, the density should be higher to ensure that sampled function evaluations meaningfully impact the estimate. 

This principle is captured by the optimal importance sampling density, given by $q(x) \propto |f(x)|p(x)$, which ensures that samples are drawn in proportion to the function's magnitude, weighted by the original density.

**Illustrative Example**: Estimating $P(T_\text{life} \geq 25)$ for $T_\text{life} \sim e^{-x}.$

- We can pick a proposal $q$ (e.g., a normal distribution centered near 25, or an exponential distribution with a larger mean) to get more frequent large samples.
- We then weight each sample by $\frac{p(x_n)}{q(x_n)}.$

**Key Insight**: If the tail region is crucial for your integral or expectation, choose an importance distribution $q$ that pays extra attention to that tail, and correct for it with weights.


## 6.2 Control Variates

**Motivation**
Suppose $f$ is expensive or has high variance, but we can identify a related function $g$ whose integral $\int g(x) p(x) \, dx$ is known exactly. If $f$ and $g$ are highly correlated, we can exploit that to reduce variance.

**Single Control Variate Formulation**
We know $E[g(X)]$ exactly. Write

$$E[f(X)] = E[f(X) - \alpha g(X)] + \alpha E[g(X)],$$

for any scalar $\alpha.$

A Monte Carlo estimator with a “control variate” becomes:

$$\hat{I}^{} = \frac{1}{N} \sum_{n=1}^N (f(x_n) - \alpha g(x_n)) + \alpha E[g(X)].$$

Because $\frac{1}{N} \sum_{n=1}^N g(x_n)$ is canceled by subtracting $\alpha \frac{1}{N} \sum_{n=1}^N g(x_n)$ and then adding $\alpha E[g(X)],$ the estimator remains unbiased.

**Optimal Choice of $\alpha$**
The single-sample variance of $f(X) - \alpha g(X)$ is:

$$\text{Var}(f(X) - \alpha g(X)) = \text{Var}[f(X)] + \alpha^2 \text{Var}[g(X)] - 2 \alpha \text{Cov}(f(X), g(X)).$$

Differentiating and setting the result to zero yields:

$$\alpha_\text{opt} = \frac{\text{Cov}[f(X), g(X)]}{\text{Var}[g(X)]}.$$

Plugging this back in shows the resulting variance is strictly reduced, provided $f$ and $g$ have nonzero covariance. In particular,

$$\text{Var}(f(X) - \alpha_{\text{opt}} g(X)) = \text{Var}[f(X)] - \frac{(\text{Cov}[f(X), g(X)])^2}{\text{Var}[g(X)]} < \text{Var}[f(X)].$$


**Multiple Control Variates**
Suppose we have $M$ different functions $g_1, \dots, g_M$ whose expectations are known. We define:

$$\hat{I} = \frac{1}{N} \sum_{n=1}^N \left(f(x_n) + c^T [g(x_n) - E[g(X)]]\right),$$

with $g(x) = (g_1(x), \dots, g_M(x))^T.$

An analogous variance-minimizing solution for the coefficient vector $c$ is:

$$c_\text{opt} = -C^{-1} b,$$

where $C = \text{Cov}(g(X))$ and $b = \text{Cov}(f(X), g(X)).$

**Key Insight**: Control variates exploit known integrals of functions correlated with $f.$ By subtracting off a piece that has low variance but known average, the overall variance of the estimator is lowered.


 **Derivation:**  
 We have $$\hat{I} = \frac{1}{N} \sum_{n=1}^N \left[ f(x_n) + \mathbf{c}^T (g(x_n) - E[g(X)]) \right].$$Since the sample mean scales variance by $1/N$, the key is

$$
\begin{aligned}
\operatorname{Var}[\hat{I}] &= \frac{1}{N} \operatorname{Var}\left[f(X) + \mathbf{c}^T (g(X) - E[g(X)])\right] \\
&= \frac{1}{N}\operatorname{Var}[f(X)] + 2 \operatorname{Cov}(f(X), \mathbf{c}^T (g(X) - \mu_g)) + \operatorname{Var}[\mathbf{c}^T (g(X) - \mu_g)].
\end{aligned}
$$

Then

- $\operatorname{Var}[f(X)]$ is a constant (w.r.t.\ $\mathbf{c}$),
- $\operatorname{Cov}(f, \mathbf{c}^T (g - \mu_g)) = \mathbf{c}^T \operatorname{Cov}(g, f) = \mathbf{c}^T \mathbf{b}$,
- $\operatorname{Var}(\mathbf{c}^T (g - \mu_g)) = \mathbf{c}^T \operatorname{Cov}(g, g) \mathbf{c} = \mathbf{c}^T \mathbf{C} \mathbf{c}$.

where
- $\mathbf{b} = \operatorname{Cov}(g(X), f(X)) \quad (\text{an } M\text{-vector}),$
- $\mathbf{C} = \operatorname{Cov}(g(X), g(X)) \quad (M \times M \text{ matrix}).$

Putting it together,

$$
\operatorname{Var}[\hat{I}] = \frac{1}{N} \left( \underbrace{\operatorname{Var}[f(X)]}_{\text{const}} + 2 \mathbf{c}^T \mathbf{b} + \mathbf{c}^T \mathbf{C} \mathbf{c} \right).
$$

Since $\operatorname{Var}[f(X)]$ doesn’t depend on $\mathbf{c}$, minimizing $\operatorname{Var}[Y]$ is equivalent to minimizing the quadratic form

$$
Q(\mathbf{c}) = 2 \mathbf{c}^T \mathbf{b} + \mathbf{c}^T \mathbf{C} \mathbf{c}.
$$

Take the gradient of $Q(\mathbf{c})$ w.r.t.\ $\mathbf{c}$. Using standard vector‐calculus rules,

$$
\nabla_{\mathbf{c}} Q(\mathbf{c}) = 2 \mathbf{b} + 2 \mathbf{C} \mathbf{c} = 0 \quad \Rightarrow \quad \mathbf{C} \mathbf{c} = -\mathbf{b}.
$$

Assuming $\mathbf{C}$ is invertible, the unique minimizer is

$$
\mathbf{c}_{\text{opt}} = -\mathbf{C}^{-1} \mathbf{b}.
$$

**Interpretation**  
You’re fitting a linear combination of the known‐mean functions $g_i$ to be negatively correlated with the residual $f(X) - E[f]$.

The matrix $\mathbf{C} = \operatorname{Var}(g)$ “whitens” the $\mathbf{g}$-space so that you weight each control variate in proportion to how much (and how independently) it co‐moves with $f$.


