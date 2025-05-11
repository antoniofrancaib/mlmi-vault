Lecture by Mark Girolami

*topics: Sobolev spaces, classical smooth function spaces $C^k(\Omega)$, Lebesgue spaces $L^p(\Omega)$, weak derivatives, Hilbert Sobolev spaces $H^k(\Omega)$, convergence rates in $L^2$, curse of dimensionality, function approximation in high dimensions, failure of Lebesgue measure in infinite dimensions, Gaussian measures, infinite product measures, weighted Hilbert spaces $\ell_a^2$, normalization of Gaussian densities, Gaussian measures on function spaces, trace-class operators, full covariance operators, Cameron–Martin space, image of $\mathbf{C}^{1/2}$, equivalence and singularity of Gaussian measures, quasi-invariance, Radon–Nikodym derivatives in infinite dimensions, Bayes’ rule in Hilbert space, Gaussian process priors, kernel methods in infinite dimensions, and Bayesian inference on function spaces.*

# LECTURE-9

# 1. Introduction and Motivation
In many areas—from the numerical solution of partial differential equations (PDEs) to machine learning function estimation—it is critical to measure and control the smoothness of functions. Classical spaces such as $C^k(\Omega)$ (functions having continuous derivatives up to order $k$) do not always lead to complete spaces under the usual norms, nor do they adequately capture the "borderline" cases encountered in practical applications.

Sobolev spaces address these issues by combining $L^p$ norms of a function and its (weak) derivatives into a single framework. When $p=2$, these spaces become Hilbert spaces, which are particularly amenable to both theoretical analysis and numerical computation. In machine learning, such spaces are useful for studying generalisation performance by relating smoothness to function complexity.

# 2. Sobolev Spaces: Definitions and Properties
## 2.1. Background: From $C^k(\Omega)$ to $L^p(\Omega)$
**Classical smooth functions:**  
We start with the space $C^k(\Omega)$ of functions that possess continuous derivatives up to order $k$ on a domain $\Omega \subset \mathbf{R}$. In our simple one-dimensional setting, $\Omega$ is typically an open interval of the real line.

**The classical norm:** For $f \in C^k(\Omega)$, a norm that reflects smoothness is given by:

$$
\|f\|_{W^{k}_p} = \left( \sum_{s=0}^k \int_\Omega |D^s f(x)|^p \, dx \right)^{1/p},
$$

where $D^s f$ denotes the $s$th (classical) derivative of $f$. This norm is a natural way to "penalize" irregular behavior, since it aggregates information about the function and all derivatives up to order $k$.

## 2.2. Definition of a Sobolev Space $W^{k}_p(\Omega)$

Let $\Omega \subset \mathbb{R}^n$ be an open set, $k \in \mathbb{N}$, and $1 \leq p \leq \infty$. The Sobolev space $W^{k}_p(\Omega)$ (also denoted $H^k(\Omega)$ when $p=2$) is the set of all functions $f \in L^p(\Omega)$ whose weak derivatives $D^{\alpha}f$ of all multiorders $|\alpha| \leq k$ exist in the weak sense and also belong to $L^p(\Omega)$.

That is,

$$
W^{k}_p(\Omega) = \{ f \in L^p(\Omega) : D^{\alpha}f \in L^p(\Omega),\ \forall \alpha \text{ with } |\alpha| \leq k \},
$$

where $\alpha \in \mathbb{N}_0^n$ is a multi-index and $D^{\alpha}f$ denotes the weak derivative of order $\alpha$. The associated Sobolev norm is:

$$
\|f\|_{W^{k}_p(\Omega)} = \left( \sum_{|\alpha| \leq k} \|D^{\alpha}f\|_{L^p(\Omega)}^p \right)^{1/p}, \quad \text{for } 1 \leq p < \infty,
$$

or

$$
\|f\|_{W^{k}_{\infty}(\Omega)} = \max_{|\alpha| \leq k} \|D^{\alpha}f\|_{L^{\infty}(\Omega)}.
$$


### Interpretation 

$W^{k}_p(\Omega)$ is the completion of $C^k(\Omega)$ under the $W^{k}_p$-norm. These spaces generalize classical smooth functions by relaxing the requirement that derivatives be classically defined — weak derivatives suffice.

## 2.3. Completion via Weak Derivatives: The General Sobolev Space $W^{k,p}(\Omega)$
**Limitations of $C^k(\Omega)$:**  The norm defined on $C^k(\Omega)$ is not complete. In other words, the limit of a convergent sequence of smooth functions (with respect to this norm) might not be smooth in the classical sense.

To remedy this, one enlarges the space by considering functions in the Lebesgue space $L^p(\Omega)$. The enlarged space includes functions whose classical derivatives may not exist everywhere but whose weak derivatives do.

**Introducing Lebesgue spaces:** Let $\Omega \subset \mathbb{R}^n$ be an open set and $1 \leq p < \infty$. The Lebesgue space $L^p(\Omega)$ is defined as:

$$
L^p(\Omega) = \{ f : \Omega \to \mathbb{R} \ \text{measurable} \mid \int_{\Omega} |f(x)|^p \, dx < \infty \}.
$$

It is a Banach space with norm:

$$
\|f\|_{L^p(\Omega)} = \left( \int_{\Omega} |f(x)|^p \, dx \right)^{1/p}.
$$

When $p=2$, $L^2(\Omega)$ becomes a Hilbert space with inner product:

$$
\langle f, g \rangle_{L^2(\Omega)} = \int_{\Omega} f(x) g(x) \, dx.
$$

**Weak Derivatives** Let $f \in L^p(\Omega)$. A function $g \in L^p(\Omega)$ is called the weak derivative of $f$ (denoted $D^{\alpha}f = g$, for a multi-index $\alpha$) if:

$$
\int_{\Omega} f(x) D^{\alpha} \varphi(x) \, dx = (-1)^{|\alpha|} \int_{\Omega} g(x) \varphi(x) \, dx
$$

for all $\varphi \in C_c^{\infty}(\Omega)$, where $C_c^{\infty}(\Omega)$ is the space of infinitely differentiable functions with compact support in $\Omega$, and $D^{\alpha} \varphi$ denotes the classical derivative of $\varphi$.

This definition allows us to consider "derivatives" of functions that are not differentiable in the classical sense — as long as they satisfy this identity under integration.

**Special Case: $p = 2$ — A Hilbert Space**

When $p=2$, the Sobolev space $W^{k,2}(\Omega)$ becomes a Hilbert space, typically denoted $H^k(\Omega)$, with inner product:

$$
\langle f, g \rangle_{W^{k}_2(\Omega)} = \sum_{|\alpha| \leq k} \langle D^{\alpha} f, D^{\alpha} g \rangle_{L^2(\Omega)}.
$$

This inner product incorporates all weak derivatives up to order $k$, and the resulting structure enables the use of powerful geometric and functional-analytic tools — especially important in PDE theory, variational methods, and learning theory.

# 3. Weak Derivatives: Intuition and Formal Definition
## 3.1. Motivation for Weak Differentiation
For many functions encountered in applications, classical derivatives may fail to exist at certain points (often on a set of measure zero). The idea of a weak derivative allows one to extend the notion of differentiation to such functions by "transferring" the differentiation onto test functions. In effect, the weak derivative is defined not by pointwise limits but by satisfying an integration-by-parts formula.

## 3.2. Integration by Parts and the Weak Derivative
**Basic integration-by-parts argument:**  Suppose $f$ and $g$ are sufficiently smooth functions defined on the interval $[0,1]$ with boundary values $g(0)=g(1)=0$. Then,

$$
\int_0^1 f(x) g'(x) \, dx = -\int_0^1 f'(x) g(x) \, dx,
$$

assuming that $f$ has a classical derivative $f'$. This classical integration-by-parts formula is the starting point for defining the weak derivative.

**Definition using test functions:**  A function $h$ is called the weak derivative of $f$ on $[0,1]$ if for every test function $g$ (smooth function with $g(0)=g(1)=0$) it holds that:

$$
\int_0^1 f(x) g'(x) \, dx = -\int_0^1 h(x) g(x) \, dx.
$$

Notice that the requirement on $h$ is made in an integrated sense. This definition effectively "transfers" the differentiability requirement from $f$ to the smooth function $g$.

## 3.3. An Illustrative Example
Consider the function $f: [0,2] \to \mathbf{R}$ defined by:

$$
f(x) = \begin{cases} 
x, & x \in [0,1), \\
1, & x \in [1,2].
\end{cases}
$$

**Classical differentiation problem:**  At $x=1$, there is a discontinuity in the derivative (a jump) which means $f$ does not have a classical derivative at that point.

**Computing the weak derivative:**  Choose any differentiable test function $\varphi$ with $\varphi(0)=\varphi(2)=0$. Splitting the domain, one computes:

$$
-\int_0^2 f(x) \varphi'(x) \, dx = \left[ -\int_0^1 x \, \varphi'(x) \, dx \right] - \int_1^2 1 \cdot \varphi'(x) \, dx.
$$

Integration by parts separately on each interval and handling the boundary terms (which vanish due to the conditions on $\varphi$) yields an expression that identifies a weak derivative $g(x)$ defined by:

$$
g(x) = \begin{cases} 
1, & x \in [0,1), \\ 
0, & x \in [1,2]. 
\end{cases}
$$

Here, although $g$ is discontinuous, it suffices as the weak derivative of $f$ because the integration identity holds for all valid test functions. This approach generalises to more dimensions and higher-order derivatives, albeit with more elaborate notation.

**Derivation:** Consider the function $f$ defined on the interval $[0, 2]$ as defined above. For any differentiable test function $\varphi : [0, 2] \to \mathbb{R}$, with $\varphi(0) = \varphi(2) = 0$, then:

$$
- \int_0^1 x \varphi'(x) \, dx = -x \varphi(x) \big|_0^1 + \int_0^1 \varphi(x) \, dx
= -\varphi(1) + \int_0^1 \varphi(x) \, dx
$$

Furthermore,

$$
- \int_1^2 \varphi'(x) \, dx = -\varphi(2) + \varphi(1) = \varphi(1)
$$

Finally,

$$
- \int_0^2 f(x) \varphi'(x) \, dx = \int_0^1 \varphi(x) \, dx = \int_0^2 g(x) \varphi(x) \, dx
$$

where $g$ is

$$
g(x) =
\begin{cases}
1 & \text{for } x \in [0, 1) \\
0 & \text{for } x \in [1, 2]
\end{cases}
$$

The weak derivative we intuitively wanted is a non-continuous function defined on sets up to measure zero.

# 4. Function Approximation by Polynomial (Fourier) Expansion
## 4.1. The Motivation in Machine Learning
One of the fundamental tasks in machine learning is the estimation of functions from data. A key question is: How fast can an approximation of a function converge to the true function as we increase the model complexity? Here, polynomial approximations (or their trigonometric analogs) offer an illustrative and mathematically tractable example to understand the interplay between function smoothness and the convergence rate.

## 4.2. Function Representation via Fourier Series
**Setting:**  Consider the space:

$$
C^2[-\pi,\pi] = \{ f: [-\pi,\pi] \to \mathbf{R} \mid f \text{ is continuous and } f \in L^2[-\pi,\pi] \}.
$$

Functions in this space can be represented by their Fourier series:

$$
f(x) = \sum_{k=0}^\infty c_k \exp(ikx), \text{ with } c_k \propto \int_{-\pi}^\pi f(x) \exp(-ikx) dx.
$$

From the previous lecture, we know that the $L^2$ norm is:

$$
\|f\|_{L^2}^2 = \sum_{k=0}^{\infty} c_k^2
$$

Note this identity (without the $2\pi$) is just a version **up to constant scaling**, assuming normalized coefficients.

**Link with Sobolev spaces:**  The Sobolev space $W^{s}_2$ on this interval can be understood in terms of the Fourier coefficients. In these spaces, the norm is given by:

$$
\|f\|_{W_2^s}^2 = \|D^s f\|_{L^2}^2 = \sum_{k=1}^{\infty} k^{2s} c_k^2
$$

which shows that for $f$ to be in $W^{s}_2$, the Fourier coefficients $c_k$ must decay at a rate that increases with $s$.

## 4.3. Approximation via Trigonometric Polynomials
**Approximating space:**  Let $H_n$ denote the space of trigonometric polynomials of degree $n$:

$$
p(x) = \sum_{k=0}^n a_k \exp(ikx).
$$

**Optimal approximation:**  For a function $f(x) = \sum_{k=0}^\infty c_k \exp(ikx)$, the best approximation (in the $L^2$ sense) by a polynomial of degree $n$ is given by:

$$
f_n(x) = \sum_{k=0}^n c_k \exp(ikx).
$$

**Error analysis:** The approximation error in the $L^2$ norm is:

$$
\epsilon_n[f] = \|f - f_n\|_{L^2} = \left( \sum_{k=n+1}^\infty c_k^2 \right)^{1/2}.
$$

By introducing the weighting $k^{2s}$ from the Sobolev norm and noticing that

$$
c_k^2 = c_k^2 \cdot k^{2s} \cdot k^{-2s},
$$

one obtains a bound:

$$
\epsilon_n[f]^2 = \sum_{k=n+1}^\infty c_k^2 \cdot k^{2s} \cdot k^{-2s}< \frac{1}{n^{2s}} \sum_{k=n+1}^\infty k^{2s} c_k^2 \leq \frac{1}{n^{2s}} \sum_{k=1}^\infty k^{2s} c_k^2  = \frac{1}{n^{2s}} \|f\|_{W^{s}_2}^2.
$$

**Intuitive significance:**  As $s$ (the measure of smoothness) increases, the decay in Fourier coefficients accelerates, and the same degree $n$ yields a lower error. In other words, smoother functions (with higher $s$) are approximated more efficiently.

## 4.4. Extension to Higher Dimensions
**Multi-dimensional setting:**  For functions defined on a $d$-dimensional domain (e.g., for $d=2$):

$$
p(x) = \sum_{k,m=0}^n a_{k,m} \exp(i(kx + my)),
$$

a similar error bound can be established:

$$
\epsilon_n[f] < \frac{1}{(2n)^{2s/d}} \|f\|_{W^{s}_2}.
$$

**Curse of dimensionality:**  
The exponent $2s/d$ in the error bound reveals that as the dimension $d$ increases, convergence becomes slower. This is a clear manifestation of the "curse of dimensionality"—even if the function is smooth, the sheer increase in dimensionality can severely hamper the rate at which the approximation error decreases.

# 5. Implications in Machine Learning and Computational Statistics
**Function learning:**  
Many machine learning methods aim to learn an unknown function from data. The above analysis provides a framework for understanding how the smoothness of the true function (measured in an appropriate Sobolev norm) impacts the rate of convergence of the learned model.

**Practical applications:**  
- **Deep Learning and Neural Networks:** Although deep neural networks are far more complex than simple polynomial models, similar ideas about smoothness and convergence rates are underlying factors in the generalisation performance of these models.  
- **Gaussian Processes and Kernel Methods:** In these contexts, the chosen kernel can often be related to a Sobolev space, and the prior smoothness assumptions encoded in the kernel influence prediction error rates.

**Model complexity versus data:**  
The analysis illustrates that increasing the number of parameters (or the degree $n$) improves the approximation error at a rate governed by the function's smoothness. However, in high-dimensional settings, the curse of dimensionality implies that a much larger number of parameters might be necessary to achieve a comparable error rate—underscoring the importance of incorporating regularisation or kernel methods that effectively manage complexity.

# 6. Conclusion and Further Directions
The lecture has covered:  
- **Sobolev Spaces:** Defined as completions of classical function spaces by including functions whose weak derivatives exist in $L^p(\Omega)$. For $p=2$, these spaces form a Hilbert space with an inner product that encapsulates function smoothness.  
- **Weak Derivatives:** Introduced through the integration-by-parts framework, allowing differentiation to be defined in a "weak" sense. This extension is crucial for handling functions with discontinuities or irregularities.  
- **Function Approximation:** Using polynomial (or Fourier) expansions to study how error bounds depend on the smoothness of the function and the dimensionality of the domain. Smooth functions allow for faster convergence, while higher dimensions penalize the rate of approximation improvement.  
- **Machine Learning Connections:** Highlighting that these theoretical insights underpin many practical methods in computational statistics and machine learning, including deep networks and Gaussian processes.

# LECTURE-10

# 1. Lebesgue Measure and Its Limitations in Infinite Dimensions  
## 1.1. Recap of Lebesgue Measure in Finite Dimensions  

**Definition (σ-Algebra)**: Let $X$ be any non-empty set. A collection $\mathcal{F} \subseteq \mathcal{P}(X)$ is called a **σ-algebra** (or **σ-field**) over $X$ if:

- $X \in \mathcal{F}$,
- $A \in \mathcal{F} \Rightarrow X \setminus A \in \mathcal{F}$,
- If $\{A_n\}_{n \in \mathbb{N}} \subseteq \mathcal{F}$, then $\bigcup_{n=1}^\infty A_n \in \mathcal{F}$.

A pair $(X, \mathcal{F})$ is called a **measurable space**.

**Definition (Topology):** Let $X$ be a non-empty set. A **topology** $\tau$ on $X$ is a collection of subsets of $X$ satisfying:

- $\emptyset \in \tau$ and $X \in \tau$,
- If $\{U_i\}_{i \in I} \subseteq \tau$ is any collection (finite or infinite), then the union $\bigcup_{i \in I} U_i \in \tau$,
- If $U_1, \dots, U_n \in \tau$ (a finite collection), then the intersection $\bigcap_{i=1}^n U_i \in \tau$.

The pair $(X, \tau)$ is called a **topological space**, and the sets in $\tau$ are called **open sets**.

**Definition (Borel σ-Algebra):** Let $(X, \tau)$ be a topological space. The **Borel σ-algebra** on $X$, denoted $\mathcal{B}(X)$, is the smallest σ-algebra that contains the topology $\tau$; that is,

$$
\mathcal{B}(X) = \sigma(\tau),
$$

where $\sigma(\tau)$ denotes the **σ-algebra generated by** $\tau$ — the smallest collection of subsets of $X$ that:

- contains all open sets in $\tau$,
- and is closed under:
  - complementation,
  - countable unions,
  - and hence countable intersections.

The measurable sets in $\mathcal{B}(X)$ are called **Borel sets**.

**Interpretation and Comparison: Topology vs. σ-Algebra**

- A **topology** on $X$ encodes the structure of **closeness, continuity, and convergence**. Open sets serve as the “building blocks” for defining **limits**, **continuous functions**, and **compactness**. Topology is thus concerned with **geometry and analysis** without requiring a notion of distance.
    
- A **σ-algebra**, in contrast, encodes the structure of **measurability**. It tells us **which subsets of $X$ can be assigned a measure** (e.g., length, probability, volume). It is tailored for **integration theory** and **probability theory**, focusing on operations like countable unions and complements to ensure consistency of measure.
    
- While topologies and σ-algebras both consist of collections of subsets, their goals differ:

| **Concept**            | **Topology**                            | **σ-Algebra**                     |
| ---------------------- | --------------------------------------- | --------------------------------- |
| **Structure type**     | Geometric / analytic                    | Measure-theoretic                 |
| **Basic sets**         | Open sets                               | Measurable sets                   |
| **Closure operations** | Finite intersections + arbitrary unions | Countable unions + complements    |
| **Purpose**            | Define continuity, limits               | Define measurability, integration |
| **Smallest generated** | Basis → Topology                        | Generator → σ-Algebra             |
|                        |                                         |                                   |
- **Borel σ-algebra** bridges both worlds: it is the smallest σ-algebra generated by a topology. It captures all **measurable events** that arise from open sets and their countable operations.

**Definition (Measure)**: Let $(X, \mathcal{F})$ be a measurable space. A function $\mu : \mathcal{F} \to [0, \infty]$ is called a **measure** if:

- $\mu(\emptyset) = 0$,
- (**Countable Additivity**) For any countable collection $\{A_n\}_{n=1}^\infty$ of disjoint sets in $\mathcal{F}$:

$$
\mu\left( \bigcup_{n=1}^\infty A_n \right) = \sum_{n=1}^\infty \mu(A_n)
$$

The triple $(X, \mathcal{F}, \mu)$ is called a **measure space**.

**Definition (Outer Measure)**: An **outer measure** is a function $\mu^* : \mathcal{P}(X) \to [0, \infty]$ satisfying:

- $\mu^*(\emptyset) = 0$,
- (**Monotonicity**) If $A \subseteq B$, then $\mu^*(A) \leq \mu^*(B)$,
- (**Countable Subadditivity**) For any sequence $\{A_n\} \subseteq \mathcal{P}(X)$:

$$
\mu^*\left( \bigcup_{n=1}^\infty A_n \right) \leq \sum_{n=1}^\infty \mu^*(A_n)
$$


**Definition (Carathéodory Measurable Set)**: A set $A \subset X$ is **Carathéodory-measurable** with respect to $\mu^*$ if for all $E \subset X$,

$$
\mu^*(E) = \mu^*(E \cap A) + \mu^*(E \setminus A)
$$

The collection of such sets $\mathcal{L} \subset \mathcal{P}(X)$ forms a **σ-algebra**.

**Definition (Measure Induced by Outer Measure)**: The restriction $\mu := \mu^*|_{\mathcal{L}}$ is a **measure** on the measurable space $(X, \mathcal{L})$.

Let $E \subseteq \mathbb{R}^n$ be any set. The **Lebesgue outer measure** $\mu^*(E)$ is defined as:

$$
\mu^*(E) := \inf \left\{ \sum_{i=1}^\infty \text{vol}(R_i) \ \middle|\ E \subseteq \bigcup_{i=1}^\infty R_i,\ R_i \in \mathcal{R} \right\},
$$

where:

- $\mathcal{R}$ is the collection of all **open rectangles** (products of open intervals) in $\mathbb{R}^n$,
- $\text{vol}(R_i)$ is the **volume** of the rectangle $R_i$ (i.e., the product of its side lengths),
- The **infimum** is taken over all **countable covers** $\{R_i\}$ of $E$ using such rectangles.


**Lebesgue Measure on $\mathbb{R}^n$**: To recover the **Lebesgue measure** on $\mathbb{R}^n$:

- Define the outer measure $\mu^*$ by covering sets using countable unions of rectangles, and taking the **infimum of their volumes**.
- Use **Carathéodory's method** to obtain the σ-algebra $\mathcal{L}$ of Lebesgue-measurable sets.
- The resulting measure $\mu$ is the **Lebesgue measure** on $(\mathbb{R}^n, \mathcal{L})$.


**Definition and Properties:**  In one dimension, the Lebesgue measure $\mu$ on an interval $\mathbf{I}=[a,b]$ is given by:  

$$\mu(\mathbf{I})=b−a$$  
Fundamental properties include:  

- **Finiteness:** For bounded intervals, $\mu(\mathbf{I})$ is finite.  
- **Monotonicity:** If $\mathbf{A}\subset\mathbf{B}$, then $\mu(\mathbf{A})\leq\mu(\mathbf{B})$.  
- **Translation Invariance:** For any $x_0\in\mathbb{R}$, shifting the set does not change its measure: $\mu(\mathbf{A}+x_0)=\mu(\mathbf{A})$.  

**Extension to $D$-Dimensions:**  In $\mathbb{R}^D$, the measure for a rectangle $[a_1,b_1]\times\cdots\times[a_D,b_D]$ is the product of the intervals’ lengths:  

$$\mu(\mathbf{I})=\prod_{d=1}^D(b_d−a_d)$$  

## 1.2. The Infinite-Dimensional Problem  
**Infinite-Dimensional Hilbert Space Setup:**  Consider a separable Hilbert space $\mathbf{H}$ with a countable orthonormal basis $\{\mathbf{e}_i\}_{i=1}^\infty$. By definition, every element $\mathbf{x}\in\mathbf{H}$ can be represented as  
$$\mathbf{x}=\sum_{i=1}^\infty x_i\mathbf{e}_i$$  
with $\|\mathbf{x}\|^2=\sum_{i=1}^\infty|x_i|^2<\infty$.  


**Why Lebesgue Measure Cannot Extend to $\mathbf{H}$:**  

- **Distance Between Basis Elements:**  Any two distinct basis elements $\mathbf{e}_i$ and $\mathbf{e}_j$ satisfy  
  $$\|\mathbf{e}_i−\mathbf{e}_j\|^2=\|\mathbf{e}_i\|^2+\|\mathbf{e}_j\|^2=2.$$  

- **Covering the Space with Balls:**  Consider the ball centered at the origin with radius 2, $\mathbf{B}(0,2)$, in $\mathbf{H}$. Place non-overlapping balls of radius $1/2$ centered at each basis element:  
  $$\{\mathbf{B}(\mathbf{e}_i,\frac{1}{2})\}_{i=1}^\infty.$$  ![[balls-hilbert.png]]
Then 
$$
B(e_i, \tfrac{1}{2}) \cap B(e_j, \tfrac{1}{2}) = \emptyset \quad \text{and} \quad \bigcup_{i \in \mathbb{N}} B(e_i, \tfrac{1}{2}) \subset B(0, 2)
$$

Now if Lebesgue measure in infinite dimensions is translation invariant, then

$$
\mu(B(e_i, \tfrac{1}{2})) = \mu(B(e_j, \tfrac{1}{2}))
$$

for all $i, j \in \mathbb{N}$.

  Since these balls are disjoint and each one would have the same Lebesgue measure (due to translation invariance), the monotonicity and countable additivity properties:  
  
  $$\sum_{i=1}^\infty\mu(\mathbf{B}(\mathbf{e}_i,\frac{1}{2}))\leq\mu(\mathbf{B}(0,2)).$$  
- **Contradiction via Countable Additivity:**  The left-hand side is an infinite sum of equal positive constants, and hence would diverge to infinity. However, $\mu(\mathbf{B}(0,2))$ is finite in the classical (finite-dimensional) context. This contradiction shows that there is **no translation-invariant, σ-finite Lebesgue measure in infinite dimensions.**  
**Implications for Probability Densities:**   In probability theory, densities are typically defined with respect to Lebesgue measure. When no such reference measure exists in infinite-dimensional spaces, it becomes a serious problem for defining probability densities on function spaces—a central issue in many modern applications.  

# 2. Measures in Infinite Dimensions and the Need for Gaussian Measures  
## 2.1. Infinite-Dimensional Applications  
Many areas in machine learning and computational statistics naturally operate in infinite-dimensional settings:  

- **Function Approximation:** Kernel methods operate in feature spaces that are often modeled as infinite dimensional.  
- **Stochastic Processes:** Signal processing, Gaussian processes, and Dirichlet processes rely on measures over infinite-dimensional spaces.  
- **Inverse Problems:** Many engineering and scientific problems involve functional spaces, making the definition of proper measures critical.  

## 2.2. Replacement by Gaussian Measures  

**Gaussian Measure as a Reference:**  Since Lebesgue measure fails in infinite dimensions, a common and effective substitute is the Gaussian measure. Gaussian measures have properties that are well-behaved in infinite-dimensional contexts.  

**Constructing Gaussian Measures:**  

- In $\mathbb{R}$, the standard Gaussian is given by:  
  $$g(\mathbf{B})=\frac{1}{\sqrt{2\pi}}\int_{\mathbf{B}}\exp\left(−\frac{x^2}{2}\right)\,dx.$$  
- In the infinite-dimensional setting, one can form the infinite product measure:  
  $$\mu=\bigotimes_{n=1}^\infty\mu_n\quad\text{with each }\mu_n=g,$$  
  where the product measure is well defined on a suitably restricted sequence space (e.g., $\ell^2$ with certain weightings).  

**Function Space Consideration – The $\ell_a^2$ Space:**  Define a weighted sequence space:  

$$\ell_a^2=\left\{\mathbf{x}\in\mathbb{R}^\infty:\sum_{k=1}^\infty a_kx_k^2<\infty\right\},$$  
where **coordinates are scaled by given weights** $a_k$. The **weights** $a_k$ in the definition of the **weighted Hilbert space** $\ell_a^2$ are part of the construction of that space. This space has an inner product  
$$\langle\mathbf{x},\mathbf{y}\rangle=\sum_{k=1}^\infty a_kx_ky_k.$$  
A function $f_\lambda(\mathbf{x})$ is introduced:  

$$f_\lambda(\mathbf{x})=\exp\left(−\frac{\lambda}{2}\sum_{k=1}^\infty a_kx_k^2\right),$$  
which is well defined on $\ell_a^2$ provided the series converges (for instance, if $\sum_{k=1}^\infty a_k<\infty$).  

**Normalization and Finite Measure:**  Through this construction, if the infinite product  

$$\prod_{k=1}^\infty(1+\lambda a_k)$$  
converges, then  

$$\int_{\mathbb{R}^\infty}f_\lambda(\mathbf{x})\,d\mu(\mathbf{x})=\frac{1}{\prod_{k=1}^\infty(1+\lambda a_k)}$$  
is finite, demonstrating that the Gaussian measure properly ‘lives’ on this infinite-dimensional space and assigns a total probability of one.  

**Derivation:** For each coordinate $x_k$, the integral over $\mathbb{R}$ with respect to the measure $\mu_k$  
(which is such that $x_k^2$ is exponentially distributed with rate $1/2$) is:

$$
\int_{\mathbb{R}} \exp\left(- \frac{\lambda a_k}{2}  x_k^2\right) \, d\mu_k(x_k).
$$

Using the **Laplace transform property** of the exponential distribution, we know that if $x_k^2$ is exponentially distributed with rate $1/2$, then:

$$
\int_{\mathbb{R}} \exp(-c x_k^2) \, d\mu_k(x_k) = \frac{1}{1 + 2c}.
$$

Substituting $c = \frac{\lambda a_k}{2}$, we get:

$$
\int_{\mathbb{R}} \exp\left(-\frac{\lambda a_k}{2} x_k^2\right) \, d\mu_k(x_k) = \frac{1}{1 + \lambda a_k}.
$$

Since the measure $\mu$ is a **product measure**, the entire integral factors into the product of these one-dimensional integrals:

$$
\int_{\mathbb{R}^\infty} f_\lambda(x) \, d\mu(x) 
= \prod_{k=1}^\infty \int_{\mathbb{R}} \exp\left(-\frac{\lambda a_k}{2}x_k^2\right) \, d\mu_k(x_k) 
= \prod_{k=1}^\infty \frac{1}{1 + \lambda a_k}.
$$

Thus, the integral is the **reciprocal of the infinite product**:
$$
\frac{1}{\prod_{k=1}^\infty (1 + \lambda a_k)}.
$$

There are a number of things we can see here:

- We have replaced the Lebesgue measure with the product Gaussian measure in $d\mu$
- If the infinite product converges then $\int_{\mathbb{R}^\infty} f_\lambda \, d\mu$ is finite and positive
- If $\sum_{k=1}^\infty a_k < \infty$ then the product converges
- Note that $\lambda \to 0$ then $f_\lambda(x) = 1$ if $x \in \ell_{a}^2$
- Also as $\lambda \to 0$ then

$$
\int_{\mathbb{R}^\infty} f_\lambda \, d\mu 
= \int_{\mathbb{R}^\infty} \mathbb{I}_{x \in \ell_{2,a}} \, d\mu 
= \int_{\ell_{2,a}} d\mu 
= \mu(\ell_{2,a}) = 1
$$

- This is awesome — we have a probability measure which gives full measure to the whole sequence space — and as it is isomorphic to $L^2$ to the corresponding function space

| **Infinite-dim’l** | **Finite-$n$ analogue** |
|--------------------|--------------------------|
| $\ell_a^2 = \left\{ \mathbf{x} : \sum a_k x_k^2 < \infty \right\}$ | $\mathbb{R}^n$, equipped with the weighted inner-product $\mathbf{x}^\top A \mathbf{y}$ where $A = \mathrm{diag}(a_1, \dots, a_n)$ |
| Inner-product $\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{k=1}^\infty a_k x_k y_k$ | $\langle \mathbf{x}, \mathbf{y} \rangle_A = \mathbf{x}^\top A \mathbf{y}$ |
| $\displaystyle f_\lambda(\mathbf{x}) = \exp\left(-\frac{\lambda}{2} \sum a_k x_k^2\right)$ | Unnormalized Gaussian density $\displaystyle f_\lambda(\mathbf{x}) = \exp\left(-\frac{\lambda}{2} \mathbf{x}^\top A \mathbf{x}\right)$ on $\mathbb{R}^n$ |
| Product-Gaussian measure $\mu = \bigotimes g$ on coordinates | Lebesgue measure $d\mathbf{x}$ on $\mathbb{R}^n$ (or a single $n$-variate Gaussian if you like, but typically one uses plain $d\mathbf{x}$) |
| $\displaystyle \prod_{k=1}^\infty (1 + \lambda a_k)$ | Determinant $\det(I + \lambda A) = \prod_{k=1}^n (1 + \lambda a_k)$ |
| $\displaystyle \int_{\mathbb{R}^\infty} f_\lambda\, d\mu = \prod_{k=1}^\infty (1 + \lambda a_k)^{-1/2}$ | $\displaystyle \int_{\mathbb{R}^n} \exp\left(-\frac{\lambda}{2} \mathbf{x}^\top A \mathbf{x} \right)\, d\mathbf{x} = (2\pi)^{n/2} \det(\lambda A)^{-1/2} = (2\pi)^{n/2} \lambda^{-n/2} \prod_{k=1}^n a_k^{-1/2}$ |

# 3. Full Covariance Operators in Infinite Dimensions  
## 3.1. Beyond Diagonal Covariance  
**Diagonal Case Recap:**  Up to now, Gaussian measures in infinite-dimensional spaces have been discussed with diagonal covariance operators; that is, the covariance operator $\mathbf{S}$ acting on a sequence $\mathbf{x}$ is given by  

$$\mathbf{Sx}=(a_nx_n)_{n=1}^\infty.$$  
**General Covariance Operators:**  In many applications, it is necessary to consider full (non-diagonal) covariance operators. A general covariance operator $\mathbf{C}$ on a Hilbert space $\mathbf{H}$ must be trace class (or nuclear), meaning:  

$$\text{trace}(\mathbf{C})=\sum_{n=1}^\infty\langle\mathbf{e}_n,\mathbf{Ce}_n\rangle<\infty,$$  
where $\{\mathbf{e}_n\}$ is any orthonormal basis of $\mathbf{H}$.  

##  3.2. Connection to Covariance Functions  

**Covariance Inner Product:**  For two zero-mean functions $\mathbf{f},\mathbf{g}\in\mathbf{H}$ defined on a domain $\mathbf{D}$, the covariance between $\mathbf{f}$ and $\mathbf{g}$ with respect to a random function $\mathbf{u}$ is given by:  

$$\langle\mathbf{f},\mathbf{Cg}\rangle=\mathbb{E}[\langle\mathbf{f},\mathbf{u}\rangle\langle\mathbf{u},\mathbf{g}\rangle].$$  
**Integral Representation:**  Writing this in integral form:  

$$
\begin{aligned}
\langle \mathbf{f}, \mathbf{C} \mathbf{g} \rangle 
&= \mathbb{E}(\mathbf{f}, \mathbf{u})(\mathbf{u}, \mathbf{g}) \quad \text{with } \mathbf{u} \in H \\
&= \mathbb{E} \int_D \int_D \mathbf{f}(x) \left( \mathbf{u}(x) \mathbf{u}(y) \right) \mathbf{g}(y) \, dy \, dx \\
&= \mathbb{E} \int_D \mathbf{f}(x) \left( \int_D \mathbf{u}(x) \mathbf{u}(y) \mathbf{g}(y) \, dy \right) dx \\
&= \int_D \mathbf{f}(x) \left( \int_D \mathbb{E} \left\{ \mathbf{u}(x) \mathbf{u}(y) \right\} \mathbf{g}(y) \, dy \right) dx \\
&= \int_D \mathbf{f}(x) \left( \int_D c(x, y) \mathbf{g}(y) \, dy \right) dx
\end{aligned}
$$


So $\mathbf{C} e_n(x) = \int_D c(x, y) e_n(y) \, dy$. One can identify $c(x,y)$ as the covariance function, which is precisely the function that defines a Gaussian Process (a central tool in probabilistic machine learning).  

**Practical Importance:**  Defining full covariance operators is crucial for modeling dependencies across dimensions in function spaces, enabling the use of Gaussian process priors and other advanced techniques in machine learning and inverse problem settings.  

Notation!

**The Hilbert space $\mathcal{H}$**
You pick some real separable Hilbert space $\mathcal{H}$. Two of the most common choices are:

- $\mathcal{H} = \ell_a^2$, the weighted sequence space with inner product  
  $$
  \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{k=1}^\infty a_k u_k v_k
  $$

- $\mathcal{H} = L^2(D)$, a space of square‐integrable functions on some domain $D$, with  
  $$
  \langle u, v \rangle = \int_D u(x) v(x)\, dx
  $$

**What are $\mathbf{u}$ and $\mathbf{v}$?**
They’re arbitrary (deterministic) elements of that same $\mathcal{H}$.

- In the sequence-space case, they’re infinite sequences  
  $\mathbf{u} = (u_1, u_2, \dots)$, $\mathbf{v} = (v_1, v_2, \dots)$

- In the function-space case, they’re functions  
  $u = u(x)$, $v = v(x)$

**What is $\mathbf{x}$?**
Here $\mathbf{x}$ is a **random** element of $\mathcal{H}$, drawn from your Gaussian measure $\mu$.

- In the sequence-space picture: $\mathbf{x} = (x_1, x_2, \dots)$ is a random sequence.

- In the function-space picture: $x(\cdot)$ is a random function in $L^2$.

So $\langle \mathbf{x}, \mathbf{u} \rangle$ means “inner product of the sample $\mathbf{x}$ with the fixed vector $\mathbf{u}$.”

- If both live in $\ell_a^2$, that’s  
  $$
  \sum_k a_k x_k u_k
  $$

- If both live in $L^2(D)$, that’s  
  $$
  \int_D x(t) u(t)\, dt
  $$


**Covariance operator:**
By definition:

$$
\langle \mathbf{u}, C \mathbf{v} \rangle = \mathbb{E}[\langle \mathbf{X}, \mathbf{u} \rangle \langle \mathbf{X}, \mathbf{v} \rangle] = \int_{\mathcal{H}} \langle \mathbf{x}, \mathbf{u} \rangle \langle \mathbf{x}, \mathbf{v} \rangle\, d\mu(\mathbf{x})
$$

This integral makes perfect sense because for each fixed $\mathbf{u}, \mathbf{v}$, the scalar function  
$$
\mathbf{x} \mapsto \langle \mathbf{x}, \mathbf{u} \rangle \langle \mathbf{x}, \mathbf{v} \rangle
$$  
is an ordinary real‐valued function on the sample space, which we integrate against the probability measure $\mu$.

# 4. Gaussian Measures in Hilbert Space  
## 4.1. Definition and Properties  

**Gaussian Measure in $\mathbf{H}$:**  A Gaussian measure in a Hilbert space, such as the $\ell^2$ space, is defined with mean zero and a covariance operator $\mathbf{C}$ (possibly full, not merely diagonal). One typically denotes this as:  

$$\mathcal{N}(0,\mathbf{C}).$$  
**Reference Measure:**  In the absence of a Lebesgue measure in infinite dimensions, the Gaussian measure serves as the canonical reference measure.  

**Quasi Invariance:**  Unlike the finite-dimensional Lebesgue measure, the Gaussian measure is **not fully translation invariant**; it is only **quasi invariant**. This subtlety has important implications in establishing Radon–Nikodym derivatives between shifted measures.  

## 4.2. Radon–Nikodym Derivative in $\mathbb{R}$ and Extension   

**Finite-Dimensional Case:**  For two Gaussian measures on $\mathbb{R}$ such as $\mu=\mathcal{N}(0,\sigma^2)$ and its shifted version $\mu_m=\mathcal{N}(m,\sigma^2)$, the Radon–Nikodym derivative is given by:  

$$\frac{d\mu_m}{d\mu}(x)=\exp\left(−\frac{m^2}{2\sigma^2}+\frac{mx}{\sigma^2}\right).$$  
Both $\mu$ and $\mu_m$ have positive densities with respect to **Lebesgue measure**. It follows that for $\sigma \ne 0$, then for any $A \in \mathbb{R}$,

$$
\mu(A) = 0 \quad \text{if and only if} \quad \mu_m(A) = 0 \quad \text{if and only if} \quad \text{Lebesgue}(A) = 0
$$

The measures are *absolutely continuous* (or equivalent) with respect to each other.

**Implication for Equivalence:**  Both measures assign positive densities with respect to the Lebesgue measure, so they are **mutually absolutely continuous**; that is, they are equivalent in the sense that no set has zero measure under one and positive under the other.  

# 5. Equivalence and Singularity of Gaussian Measures in Infinite Dimensions  
##  5.1. Dichotomy in Infinite Dimensions  

**Equivalence vs. Singularity:**  In an infinite-dimensional Hilbert space, two Gaussian measures $\mathcal{N}(0,\mathbf{C})$ and $\mathcal{N}(\mathbf{v},\mathbf{C})$ will either be equivalent (mutually absolutely continuous) or singular (completely non-overlapping support).  

- **Absolute Continuity:** Two measures are absolutely continuous if they assign zero measure to the same sets.  
- **Singularity:** They are singular if there exists a set such that one measure assigns it full measure while the other assigns it zero.  

## 5.2. The Cameron–Martin Space  

**Characterizing Shifts:**  The set of translations for which the shifted Gaussian measure $\mathcal{N}(\mathbf{v},\mathbf{C})$ remains equivalent to $\mathcal{N}(0,\mathbf{C})$ is exactly the image of the Hilbert space under the operator $\mathbf{C}^{1/2}$.  

This subspace is known as the Cameron–Martin space (or equivalently, the Reproducing Kernel Hilbert Space, RKHS). It plays a central role in the theory of Gaussian measures by delineating the “directions” in which the measure is robust under translation.  

The set of valid shifts is exactly the image of $\mathbf{H}$ under $\mathbf{C}^{1/2}$, i.e.:

$$
\mathbf{H}_{\text{CM}} = \text{Im}(\mathbf{C}^{1/2}) = \{\mathbf{C}^{1/2}(x) \mid x \in \mathbf{H} \}.
$$

You can think of $\mathbf{C}^{1/2}$ as a **smoothing operator**. It "pulls" the standard Hilbert space $\mathbf{H}$ into a smaller, nicer, smoother space.

This image is a **subspace where things are regular enough** that the Gaussian can be shifted without becoming singular.

Since $\mathbf{C}$ is a linear operator, $\mathbf{C}(x)$ is just applying the operator to the vector/function $x \in \mathbf{H}$:

- Think of $x$ as an input (e.g., a sequence or function)
- $\mathbf{C}(x)$ is the output, transformed through the structure encoded in the operator

If $\mathbf{C}$ is diagonalizable (as it is for trace-class operators), and has:
- eigenvectors $\{e_k\}$,
- eigenvalues $\lambda_k$,

then any $x \in \mathbf{H}$ can be written as:

$$
x = \sum_{k=1}^{\infty} x_k e_k,
$$

and the action of $\mathbf{C}$ is:

$$
\mathbf{C}(x) = \sum_{k=1}^{\infty} \lambda_k x_k e_k.
$$

So $\mathbf{C}(x)$ is a **smoothed version of** $x$, where each component is scaled by the corresponding eigenvalue $\lambda_k$.

Similarly:

$$
\mathbf{C}^{1/2}(x) = \sum_{k=1}^{\infty} \sqrt{\lambda_k} x_k e_k.
$$

This is **less “strongly smoothed”** than $\mathbf{C}(x)$ — eigenvalues are scaled by $\sqrt{\lambda_k}$ instead of $\lambda_k$.


##  5.3. Implications in Machine Learning  

**Computational Methods:** In many algorithms, especially those related to Bayesian inference and stochastic processes, deciding whether two measures are equivalent or singular affects convergence properties, regularization strategies, and the feasibility of certain computational techniques.  

For instance, when designing algorithms that involve changing measures (by taking Radon–Nikodym derivatives), it is crucial to operate within the Cameron–Martin space.  

#  6. Bayes Rule in Hilbert Space  
## 6.1. Bayesian Inference in Infinite Dimensions  

**Classical Bayes’ Theorem:**  In finite dimensions, Bayes’ theorem is written as:  

$$P(\mathbf{u}\mid\mathbf{y})=\frac{P(\mathbf{y}\mid\mathbf{u})P(\mathbf{u})}{P(\mathbf{y})},$$  
where $P(\mathbf{u})$ is the prior, $P(\mathbf{y}\mid\mathbf{u})$ is the likelihood, and $P(\mathbf{y})$ is a normalizing constant.  

**Difficulties in Infinite Dimensions:**  In an infinite-dimensional setting, the concept of “density” with respect to a non-existent Lebesgue measure does not make sense. Instead, one relies on the Gaussian measure (or another reference measure) defined on the Hilbert space.  

## 6.2. Role of the Radon–Nikodym Derivative  
**Posterior Measure via Derivative:**  The relationship between a posterior measure $\mu_y$ and a prior measure $\mu_0$ is captured via the Radon–Nikodym derivative:  

$$\frac{d\mu_y}{d\mu_0}(\mathbf{u})\propto P(\mathbf{y}\mid\mathbf{u}).$$  
Here, the likelihood $P(\mathbf{y}\mid\mathbf{u})$ does not appear as a density with respect to the (absent) Lebesgue measure; rather, it serves as the proportionality factor between measures defined on the function space.  

**Gaussian Process Priors:**  In practice, the prior measure $\mu_0$ is chosen to be a Gaussian measure $\mathcal{N}(\mathbf{m},\mathbf{C})$ on the space of functions (e.g., a Gaussian process prior). This framework is now standard in modern machine learning, particularly in Bayesian nonparametrics and Gaussian process regression.  

# 7. Summary and Practical Implications  

- **Absence of Lebesgue Measure:**  The classical Lebesgue measure does not extend to infinite-dimensional Hilbert or Banach spaces because of the failure of translation invariance in the presence of countable additivity. This necessitates alternative measures.  

- **Adoption of Gaussian Measures:**  Gaussian measures, defined either through infinite products of one-dimensional Gaussians or via covariance operators, provide a well-defined probability measure on infinite-dimensional function spaces.  

- **Full Covariance Operators and Trace-Class Requirement:**  For a Gaussian measure to be well-defined in a Hilbert space, the associated full covariance operator must be trace class.  

- **Equivalence and the Cameron–Martin Space:** In infinite dimensions, Gaussian measures can be either equivalent or singular depending on the shift. The Cameron–Martin space exactly characterizes the directions in which one can translate without changing the null sets.  

- **Bayesian Inference in Function Spaces:**  By employing the Radon–Nikodym derivative with respect to a Gaussian reference measure, one obtains a rigorous formulation of Bayes’ rule in infinite dimensions. This is central to the theoretical underpinning of Gaussian process priors and many inverse problem approaches in machine learning.  






