Lecture by Mark Girolami

*topics: the riemann integral; convergence of functions; the lebesgue integral; measure theory foundations; probability as a measure-theoretic construction; constructing the lebesgue integral; the radon–nikodym theorem

# LECTURE-3
# 1. Motivation and Overview

Although the Riemann integral is familiar from elementary calculus, certain limitations become quickly apparent when dealing with:

- Complex or high-dimensional domains (beyond $\mathbf{R}^d$).
- Function spaces (e.g., spaces of continuous functions).
- Swapping limits and integrals (a recurring operation in ML and Statistics for taking expectations of sequences of approximating functions).

Henri Lebesgue’s construction of the Lebesgue integral addresses these limitations elegantly by:

- Partitioning the range of the function, rather than the domain.
- Grounding integration in a measure-theoretic framework that handles very general spaces, well beyond $\mathbf{R}^d$.
- Allowing powerful convergence theorems (Monotone and Dominated Convergence Theorems) that let us interchange limits and integrals under weaker assumptions than those required by the Riemann theory.

# 2. The Riemann Integral

## 2.1 Construction of the Riemann Integral

Recall the classical definition of the Riemann integral of a (bounded) function $f$ on a closed interval $[a,b] \subset \mathbf{R}$. We partition the interval $[a,b]$ into subintervals,

$$
a = x_1 < x_2 < \cdots < x_{n+1} = b,
$$

and define:

$$
M_k = \sup_{x \in [x_k, x_{k+1}]} f(x), \quad m_k = \inf_{x \in [x_k, x_{k+1}]} f(x).
$$

We then form the upper and lower Riemann sums:

$$
S_P(f) = \sum_{k=1}^n M_k (x_{k+1} - x_k), \quad s_P(f) = \sum_{k=1}^n m_k (x_{k+1} - x_k),
$$

where $P$ denotes the chosen partition. As we refine the partition (i.e., let $\max(x_{k+1} - x_k) \to 0$), these sums squeeze closer together. We set:

$$
S(f) = \lim_{\text{all partitions } P} \inf S_P(f), \quad s(f) = \lim_{\text{all partitions } P} \sup s_P(f).
$$
![[riemann-sum.png]]

If $S(f) = s(f) = A$, then we say $f$ is Riemann integrable and define

$$
\int_a^b f(x) \, dx = A.
$$

![[riemann-equiv.png]]
This classical viewpoint partitions the domain $[a,b]$ into intervals and assumes $f$ is “nice enough” (in particular, continuous almost everywhere). It works well in standard calculus on $\mathbf{R}$, but has limitations when:

- The underlying domain is higher-dimensional ($\mathbf{R}^d$) or more abstract (e.g., space of continuous functions).
- We want to exchange integrals and limits without strong conditions like uniform convergence.

## 2.2 Shortcomings in Machine Learning Contexts

- **Partitioning Arbitrary Domains**: In many ML applications, the space on which we integrate may be very large-dimensional or even a function space $X$. It is unnatural (or impossible) to replicate “slicing into subintervals” in such spaces.
- **Exchanging Limits and Integrals**: Many ML or statistical procedures involve taking limits inside integrals (e.g., parameter estimation, posterior updates, or function approximation). The Riemann integral requires strict forms of convergence (often uniform) to justify interchanging $\lim$ and $\int$. But pointwise convergence in ML practice is more common, and, as we’ll see, the Riemann integral does not always handle this gracefully.

# 3. Convergence of Functions

To understand where the Riemann integral fails when handling limits of functions, we recall two common modes of convergence.

## 3.1 Pointwise Convergence

A sequence of real-valued functions $\{f_n\}$ on domain $D$ converges pointwise to $f$ if for every $x \in D$, $\lim_{n \to \infty} f_n(x) = f(x)$. 

Formally, $\forall x \in D$ and $\forall \varepsilon > 0$, there exists $N(\varepsilon, x)$ such that for all $n > N$, $|f_n(x) - f(x)| < \varepsilon$. The key point is that the speed of convergence can depend on $x$.

A classic example is $f_n(x) = x^n$ on $[0,1]$. Pointwise, $f_n(x)$ converges to

$$
f(x) = \begin{cases}
0, & 0 \leq x < 1, \\
1, & x = 1.
\end{cases}
$$

This limiting function is discontinuous, even though each $f_n$ is continuous. Hence pointwise convergence need not preserve continuity.

On the other side, the function $g_n(x ) = x/n, x ∈ [0, 1]$ converges uniformly to g (x ) = 0
![[unif-conv.png]]
## 3.2 Uniform Convergence

A stronger requirement is uniform convergence: $\{f_n\}$ converges uniformly to $f$ on $D$ if the largest discrepancy $\sup_{x \in D} |f_n(x) - f(x)| \to 0$. Formally,

$$
\forall \varepsilon > 0, \exists N(\varepsilon) \text{ independent of } x, \text{ such that } n > N(\varepsilon) \implies \sup_{x \in D} |f_n(x) - f(x)| < \varepsilon.
$$

Uniform convergence is quite restrictive but does allow exchanging limits and Riemann integrals:

$$
\lim_{n \to \infty} \int_D f_n(x) \, dx = \int_D \lim_{n \to \infty} f_n(x) \, dx,
$$

whereas mere pointwise convergence can fail to justify $\lim \leftrightarrow \int$ swaps. A memorable example is

$$
f_n(x) = 2n^2 x e^{-n^2 x^2} \text{ on } [0,1].
$$
![[lim-int-swap.png]]
One can show

$$
\lim_{n \to \infty} \int_0^1 f_n(x) \, dx = \lim_{n \to \infty}  1 - e^{-n^2} = 1,
$$

while

$$
\int_0^1 \lim_{n \to \infty} f_n(x) \, dx = \int_0^1 0 \, dx = 0.
$$

The Riemann integral cannot handle this limit interchange gracefully, but the Lebesgue integral can, via more lenient convergence theorems.

# 4. The Lebesgue Integral

## 4.1 Partitioning the Range Instead of the Domain

Henri Lebesgue’s key insight was to ***partition the range of $f$*** (rather than the domain) and measure the size of the corresponding preimages. Conceptually:

- If $f$ takes values in $[f_{\min}, f_{\max}]$, subdivide this range into small intervals $[\alpha_k, \alpha_{k+1})$.
- Each interval in the range maps back to some set in the domain, e.g.,
  $$
  X_k = \{x : f_{\min} + (k-1)\delta \leq f(x) < f_{\min} + k\delta\},
  $$
  with $\delta \to 0$. The ‘size’ of $X_k$ is given by a measure $\mu$.
- Summation $\sum \alpha_k \mu(X_k)$ approximates the integral, and taking finer partitions of the range yields the Lebesgue integral.
- If the Lebesgue sum is convergent as $n \to \infty$ such that $|f_k - f_{k+1}| \to 0,$ the limiting value is the Lebesgue Integral of the function $f(x)$ over the set $X$:

$$
\int_X f d\mu = \lim_{\max |f_k - f_{k-1}| \to 0} \sum_{k=1}^n f_k \mu(X_k).
$$

This construction relies on having a consistent notion of “measure” $\mu$ for the set $X_k$. Because we are no longer forced to partition an (often complicated) domain into intervals or rectangles, Lebesgue’s approach generalizes easily to arbitrary measurable sets—including infinite-dimensional function spaces.

![[lebesgue-size.png]]

## 4.2 Relaxed Continuity Requirements and Powerful Convergence Theorems

A function can have “many” discontinuities (as long as these discontinuities form a set of measure zero in a suitable sense), yet still be Lebesgue integrable. Moreover, limit theorems in the Lebesgue framework (like Monotone Convergence and Dominated Convergence) allow the interchange of limits and integrals under conditions of pointwise convergence plus a suitable dominating bound or monotonicity. This is a substantial improvement over the uniform-convergence requirement in Riemann theory.

**Relationship to Riemann Integrable Functions**

Every Riemann integrable function (that is bounded and continuous almost everywhere) is also Lebesgue integrable, and both integrals coincide. However, there are (pathological) *cases of Lebesgue integrable functions that are **not** Riemann integrable*.

# LECTURE-4 

# 5. Measure Theory Foundations

## 5.1 Sigma-Algebras

To formalize “size” in general spaces, we must specify which subsets are “measurable.” A $\sigma$-algebra $\Sigma$ on a set $X$ is a collection of subsets with three properties:

1. $X \in \Sigma$ and $\emptyset \in \Sigma$.
2. If $A \in \Sigma$, then its complement $A^c \in \Sigma$.
3. If $A_n \in \Sigma$ for $n = 1, 2, \dots$, then $\bigcup_{n=1}^\infty A_n \in \Sigma$.

These requirements ensure closure under countable complement, union, and intersection (via De Morgan’s laws). A pair $(X, \Sigma)$ is a **measurable space**; any $A \in \Sigma$ is a **measurable set**.

## 5.2 Measures

A measure $\mu$ on $(X, \Sigma)$ is a function $\mu: \Sigma \to [0, \infty]$ that satisfies:

1. $\mu(\emptyset) = 0$.
2. (Countable Additivity) For disjoint measurable sets $A_1, A_2, \dots$,
   $$
   \mu\left(\bigcup_{k=1}^\infty A_k\right) = \sum_{k=1}^\infty \mu(A_k).
   $$

The triple $(X, \Sigma, \mu)$ is then a **measure space**. We say $\mu$ is finite if $\mu(X) < \infty$ and is a probability measure if $\mu(X) = 1$.

## 5.3 Lebesgue Measure on $\mathbf{R}^d$

The classical “length” on $\mathbf{R}$, “area” on $\mathbf{R}^2$, or “volume” in $\mathbf{R}^d$ is captured by the Lebesgue measure. Defining it rigorously requires a delicate construction. Key properties:

- For intervals $[a, b] \subset \mathbf{R}$, $\mu([a, b]) = b - a$.
- For higher dimensions, $\mu([a_1, b_1] \times \cdots \times [a_d, b_d]) = \prod_{j=1}^d (b_j - a_j)$.
- All countable sets (e.g., rationals) have measure zero, i.e. $\mu(\mathbb{Q}) = 0$
- Many complicated (non-measurable) subsets of $\mathbf{R}^d$ exist (Banach–Tarski paradox). We only assign “length/volume” to the subsets in the Lebesgue $\sigma$-algebra, which is strictly smaller than the full power set.

## 5.4 Measurable Functions

A function between measurable spaces is the bridge that allows us to connect abstract measure spaces to more familiar settings, such as the real numbers. Suppose we have two measurable spaces, $(X, \Sigma)$ and $(Y, T)$. A function  

$$
f: X \to Y
$$

is called **measurable** if the preimage of every measurable set in $Y$ is a measurable set in $X$; that is,  

$$
\forall B \in T, \quad f^{-1}(B) \in \Sigma.
$$

# 6. Probability as a Measure-Theoretic Construction

## 6.1 Probability Spaces

A probability space $(\Omega, \mathcal{F}, P)$ is just a *measure space where $P(\Omega) = 1$*. We interpret:

- $\Omega$: the set of all possible outcomes of a random experiment.
- $\mathcal{F}$: a $\sigma$-algebra of events, subsets of $\Omega$ for which we can assign probabilities.
- $P$: a probability measure, with $P(\Omega) = 1$.

For finite discrete outcomes (e.g., rolling a die), it suffices to define $P$ on each singleton. For continuous outcomes (like a real-valued variable), we rely on Borel $\sigma$-algebras and possibly define $P$ via a probability density function with respect to Lebesgue measure.

## 6.2 Borel $\sigma$-Algebra on a General Topological Space

Let $(X,\tau)$ be a topological space, where $X$ is a set and $\tau$ is a collection of subsets of $X$ (the open sets) that defines the topology on $X$. The Borel sigma-algebra on $X$, denoted by $B(X)$, is defined as the **smallest sigma-algebra** that contains all the open sets in $\tau$. Formally, we define

$$B(X) = \sigma(\tau) = \bigcap \{ \Sigma \subseteq P(X) : \Sigma \text{ is a sigma-algebra and } \tau \subseteq \Sigma \}.$$

## 6.3 Random Variables

A random variable $X: \Omega \to \mathbf{R}$ is a measurable function from $(\Omega, \mathcal{F})$ to $(\mathbf{R}, \mathcal{B}(\mathbf{R}))$, where $\mathcal{B}(\mathbf{R})$ is the Borel $\sigma$-algebra on $\mathbf{R}$. “Measurable” means pre-images of measurable sets must be events in $\mathcal{F}$. The advantage is we can define new measures on $\mathbf{R}$ by “pushing forward” along $X$:

$$
P_X(B) = P(\{\omega : X(\omega) \in B\}).
$$

$P_X$ is called the law (or distribution) of $X$. For example, we define an expectation by the Lebesgue integral of $X$ with respect to $P$:

$$
E[X] = \int_\Omega X(\omega) P(d\omega).
$$

In many typical scenarios, $E[X]$ reduces to $\int x p(x) \, dx$ where $p$ is the usual probability density function (when it exists).

*Proof:* 
Recall the definition of the expectation of a random variable $X$ on the sample space $\Omega$ as $E[X] = \int_{\Omega} X(\omega) P(d\omega).$

Since $X$ is measurable, we can transfer the measure $P$ from $\Omega$ to $\mathbb{R}$ by defining the pushforward measure $P_X$ (also known as the law or distribution of $X$) via

$$P_X(B) = P(\{\omega \in \Omega : X(\omega) \in B\})$$

for every Borel set $B$. With this change of variable, the expectation can be re-expressed as

$$E[X] = \int_{\mathbb{R}} x P_X(dx).$$
Note in this step we made use of the **Law of the Unconscious Statistician.** 

In many common scenarios, the measure $P_X$ is absolutely continuous with respect to Lebesgue measure, so that (by the **Radon–Nikodym theorem**) there exists a probability density function $p(x)$ satisfying $P_X(dx) = p(x)dx.$

Substituting this into the integral gives the familiar form

$$E[X] = \int_{\mathbb{R}} x p(x) dx.$$

# 7. Constructing the Lebesgue Integral

The Lebesgue integral $\int_X f(x) \mu(dx)$ extends the idea of “area under the curve” by focusing on how large the sets of points are that map into certain values of $f$. The formal steps:

1. **Simple Functions**: A “simple function” $s(x)$ takes finitely many distinct real values on measurable sets. Its integral is a direct sum:
   $$
   s(x) = \sum_{i=1}^N c_i 1_{E_i}(x), \quad \int s \, d\mu = \sum_{i=1}^N c_i \mu(E_i).
   $$
2. **Nonnegative Measurable Functions**: Approximate $f$ from above by simple functions. If $\{s_n\}$ are simple and $s_n(x) \geq f(x)$, then the Lebesgue integral is defined as
   $$
   \int f \, d\mu = \inf_{s_n \geq f} \int s_n \, d\mu.
   $$
3. **General (Signed) Functions**: Decompose $f$ into positive and negative parts $f = f^+ - f^-$. Integrate each part separately, if finite.

Key ***results*** ensuring the power of Lebesgue integrals are:

## Monotone Convergence Theorem (MCT)

Let $(X,M,\mu)$ be a measure space, and let $\{f_n\}_{n=1}^{\infty}$ be a sequence of nonnegative measurable functions on $X$ such that

$$f_1(x) \leq f_2(x) \leq f_3(x) \leq \dots \quad \text{for all } x \in X.$$

Define

$$f(x) = \lim_{n \to \infty} f_n(x) \quad \text{for all } x \in X.$$

Then $f$ is measurable and

$$\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X \left( \lim_{n \to \infty} f_n \right) d\mu = \int_X f \, d\mu.$$

In other words, the integral of the limit function equals the limit of the integrals when the sequence increases monotonically.

## Dominated Convergence Theorem (DCT)

Let $(X,M,\mu)$ be a measure space, and let $\{f_n\}_{n=1}^{\infty}$ be a sequence of measurable functions that converge pointwise almost everywhere to a measurable function $f$; that is,

$$\lim_{n \to \infty} f_n(x) = f(x) \quad \text{for almost every } x \in X.$$

Assume there exists an integrable function $g: X \to [0,\infty)$ (i.e., $\int_X g \, d\mu < \infty$) such that

$$|f_n(x)| \leq g(x) \quad \text{for all } n \text{ and almost every } x \in X.$$

Then $f$ is integrable and

$$\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X \left( \lim_{n \to \infty} f_n \right) d\mu = \int_X f \, d\mu.$$

That is, if a sequence of functions converges pointwise and is dominated in absolute value by some integrable function, then the limit of the integrals equals the integral of the limit.


# 8. Radon–Nikodym Theorem and Density Functions

Let $(X,M)$ be a measurable space and let $\mu$ and $\nu$ be measures on this space.

## Absolute Continuity

We say that $\nu$ is **absolutely continuous** with respect to $\mu$ (written $\nu \ll \mu$) if for every measurable set $A \in M$,

$$\mu(A) = 0 \implies \nu(A) = 0.$$

In other words, $\nu$ does not assign positive measure to any set that $\mu$ considers negligible.

## Singular Measures

We say that $\nu$ is **singular** with respect to $\mu$ (written $\nu \perp \mu$) if there exists a measurable set $E \in M$ such that

$$\mu(E) = 0 \quad \text{and} \quad \nu(X \setminus E) = 0.$$

Equivalently, there exists a partition of $X$ into two disjoint measurable sets $E$ and $F$ (with $X = E \cup F$) such that $\mu(E) = 0$ and $\nu(F) = 0$. This means that the measures $\mu$ and $\nu$ "live" on disjoint parts of the space.

## Lebesgue Decomposition Theorem

Let $(X,M)$ be a measurable space and let $\mu$ and $\nu$ be two $\sigma$-finite measures on $(X,M)$. Then there exist unique measures $\nu_a$ and $\nu_s$ on $(X,M)$ such that

$$\nu = \nu_a + \nu_s,$$

where:

- $\nu_a$ is absolutely continuous with respect to $\mu$ (written $\nu_a \ll \mu$); that is, for every $A \in M$ with $\mu(A) = 0$, we have $\nu_a(A) = 0$.
- $\nu_s$ is singular with respect to $\mu$ (written $\nu_s \perp \mu$); that is, there exists a measurable set $N \in M$ such that $\mu(N) = 0$ and $\nu_s(X \setminus N) = 0$.

This theorem states that any $\sigma$-finite measure $\nu$ can be uniquely decomposed into a part that “follows” $\mu$ (the absolutely continuous part) and a part that is concentrated on a $\mu$-null set (the singular part).

## Radon–Nikodym Theorem and Derivative

Let $(X,M)$ be a measurable space and let $\mu$ and $\nu$ be $\sigma$-finite measures on $(X,M)$ with $\nu \ll \mu$ (that is, $\nu$ is absolutely continuous with respect to $\mu$). Then there exists a $\mu$-integrable function $f: X \to [0,\infty)$ such that for every measurable set $A \in M$,

$$\nu(A) = \int_A f \, d\mu.$$

The function $f$ is called the **Radon–Nikodym derivative** of $\nu$ with respect to $\mu$ and is denoted by

$$f = \frac{d\nu}{d\mu}, \quad \nu(dx) = f(x) \mu(dx).$$

Moreover, the function $f$ is unique in the sense that if $g$ is another function satisfying

$$\nu(A) = \int_A g \, d\mu \quad \text{for all } A \in M,$$

then $f = g$ $\mu$-almost everywhere. Intuitively, $f$ acts like a “density” that converts measure $\mu$ into measure $\nu$.

### A Very Common Special Case: Probability Densities

Let $\mu$ denote the Lebesgue measure (often written as $\lambda$) on $\mathbb{R}^d$. Suppose $\nu$ is a probability measure on $\mathbb{R}^d$ that is absolutely continuous with respect to $\mu$ (i.e., $\nu \ll \mu$). Then, by the **Radon–Nikodym theorem**, there exists a nonnegative measurable function $p: \mathbb{R}^d \to [0,\infty)$ such that for every Lebesgue measurable set $A \subset \mathbb{R}^d$,

$$\nu(A) = \int_A p(x) d\mu(x) = \int_A p(x) dx,$$

with the normalization condition

$$\int_{\mathbb{R}^d} p(x) dx = 1.$$

The function $p$ is called the **probability density function (pdf)** of $\nu$ with respect to the Lebesgue measure. For example, the standard **Gaussian measure** on $\mathbb{R}$ is given by

$$d\nu(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} dx,$$

so that its **Radon–Nikodym derivative** is the **Gaussian density**

$$p(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}.$$

This density is nonnegative and integrates to one, confirming that the **Gaussian measure** is indeed a probability measure absolutely continuous with respect to $dx$.

### Changing Measures in Integrals

Assume that $\nu$ is a measure on a measurable space $(X,M)$ and that $\nu$ is absolutely continuous with respect to another measure $\mu$; that is, there exists a nonnegative measurable function $f$ (the **Radon–Nikodym derivative**) such that for every $A \in M$,

$$\nu(A) = \int_A f \, d\mu.$$

Then, for any measurable function $g: X \to \mathbb{R}$ (or $\mathbb{C}$) for which the integrals are defined, the following **change-of-measure formula** holds:

$$\int_X g(x) d\nu(x) = \int_X g(x) f(x) d\mu(x).$$

This result is fundamental because it allows one to compute **integrals with respect to $\nu$** by instead integrating with respect to $\mu$, provided that the density $f$ is known. Such a **change of measure** is crucial in various applications, including **sampling algorithms** and techniques like **importance sampling**, where one often transforms an integral from one measure to another via an appropriate density ratio.
