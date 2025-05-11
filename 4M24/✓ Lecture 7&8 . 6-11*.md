Lecture by Mark Girolami

*topics: vector spaces; cauchy sequences; completeness; banach spaces; hilbert spaces; pre-hilbert spaces; orthogonality; orthonormal sets; parseval’s identity; bessel’s inequality; $\ell^2$ as a hilbert space; $L^2$ as a hilbert space; isomorphism of $\ell^2$ and $L^2$; reproducing kernel hilbert spaces (rkhs); reproducing kernels; representer theorem; gram matrix; regularization in rkhs; kernel methods in machine learning*

# LECTURE-7
# 1. Vector Spaces

## 1.1 Definition and Axioms

A vector space $\mathbf{V}$ over a field $\mathbf{F}$ (typically $\mathbf{R}$ or $\mathbf{C}$) is a collection of objects called vectors, together with two operations:

- **Vector Addition**: A rule $(\mathbf{x}, \mathbf{y}) \mapsto \mathbf{x} + \mathbf{y}$ taking two vectors in $\mathbf{V}$ to a third vector in $\mathbf{V}$.
- **Scalar Multiplication**: A rule $(\alpha, \mathbf{x}) \mapsto \alpha \mathbf{x}$ taking a scalar $\alpha \in \mathbf{F}$ and a vector $\mathbf{x} \in \mathbf{V}$ to another vector in $\mathbf{V}$.

These operations must satisfy the following **axioms** for all $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathbf{V}$ and all $\alpha, \beta \in \mathbf{F}$:

1. **Commutativity of Addition**:
   $$\mathbf{x} + \mathbf{y} = \mathbf{y} + \mathbf{x} \quad \text{and} \quad \mathbf{x} + \mathbf{y} \in \mathbf{V}.$$

2. **Associativity of Addition**:
   $$\mathbf{x} + (\mathbf{y} + \mathbf{z}) = (\mathbf{x} + \mathbf{y}) + \mathbf{z} \quad \text{and} \quad (\mathbf{x} + \mathbf{y}) + \mathbf{z} \in \mathbf{V}.$$

3. **Existence of Additive Identity**:
   There is a special vector $\mathbf{0} \in \mathbf{V}$ such that
   $$\mathbf{x} + \mathbf{0} = \mathbf{x} \quad \text{for all} \quad \mathbf{x} \in \mathbf{V}.$$

4. **Existence of Additive Inverse**:
   For every $\mathbf{x} \in \mathbf{V}$, there is a vector $-\mathbf{x} \in \mathbf{V}$ such that
   $$\mathbf{x} + (-\mathbf{x}) = \mathbf{0}.$$

5. **Closure Under Scalar Multiplication**:
   For every scalar $\alpha \in \mathbf{F}$ and $\mathbf{x} \in \mathbf{V}$,
   $$\alpha \mathbf{x} \in \mathbf{V}.$$

6. **Distributivity of Scalar Multiplication**:
   $$\alpha (\mathbf{x} + \mathbf{y}) = \alpha \mathbf{x} + \alpha \mathbf{y}.$$

7. **Associativity of Scalar Multiplication**:
   $$\alpha (\beta \mathbf{x}) = (\alpha \beta) \mathbf{x}.$$

8. **Scalar Identity**:
   $$1 \cdot \mathbf{x} = \mathbf{x}, \quad 0 \cdot \mathbf{x} = \mathbf{0}.$$

From these axioms, one sees that the set $\mathbf{V}$, together with vector addition, forms an *Abelian group*, and scalar multiplication is a compatible operation tying scalars from $\mathbf{F}$ to vectors in $\mathbf{V}$.

**Examples of vector spaces include**:

- $\mathbf{R}^n$, the usual $n$-dimensional space over the real numbers.
- $\mathbf{C}^n$, the usual $n$-dimensional space over the complex numbers.
- Spaces of functions on an interval $[a, b] \subset \mathbf{R}$, e.g., continuous functions $C[a, b]$, integrable functions $L^p[a, b]$, etc.

# 2. Inner Product Spaces

An inner product on a real (or complex) vector space $\mathbf{V}$ is a map

$$(\mathbf{x}, \mathbf{y}) \mapsto \langle \mathbf{x}, \mathbf{y} \rangle$$

that assigns to each pair of vectors $\mathbf{x}, \mathbf{y}$ a scalar (real or complex) satisfying:

1. **Symmetry (or Hermitian Symmetry for complex fields)**:
   $$\langle \mathbf{x}, \mathbf{y} \rangle = \overline{\langle \mathbf{y}, \mathbf{x} \rangle}.$$
   For real vector spaces, this is simply $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$.

2. **Linearity in the First Argument (or Second, depending on convention)**:
   $$\langle \alpha \mathbf{x} + \beta \mathbf{y}, \mathbf{z} \rangle = \alpha \langle \mathbf{x}, \mathbf{z} \rangle + \beta \langle \mathbf{y}, \mathbf{z} \rangle,$$
   for all scalars $\alpha, \beta$. 

	**Remark**: In the real case, the inner product is bilinear — linear in both arguments. In the complex case, we adopt the Hermitian inner product, where the inner product is linear in one argument, conjugate linear in the other, depending on the convention. 
	$$
	\langle \mathbf{x}, \alpha \mathbf{y} + \beta \mathbf{z} \rangle = \overline{\alpha} \langle \mathbf{x}, \mathbf{y} \rangle + \overline{\beta} \langle \mathbf{x}, \mathbf{z} \rangle
	$$
	
	This ensures **positive definiteness** still makes sense $\langle \mathbf{x}, \mathbf{x} \rangle \in \mathbb{R}_{\geq 0}$

3. **Positive Definiteness**:
   $$\langle \mathbf{x}, \mathbf{x} \rangle \geq 0 \quad \text{for all} \quad \mathbf{x}, \quad \text{with equality if and only if} \quad \mathbf{x} = \mathbf{0}.$$

A vector space endowed with such an inner product is called an **inner product space**.

**Induced Norm**: Once you have an inner product, you obtain a norm via

$$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}.$$

This norm gives the length of $\mathbf{x}$. When a norm arises directly from an inner product, we say the norm is induced by that inner product $\langle \cdot, \cdot \rangle$.

## 2.1 Examples

- Let $\mathbf{a}, \mathbf{b} \in \mathbf{C}^n$. Then the inner product is defined by:
$$
\langle \mathbf{a}, \mathbf{b} \rangle = \sum_{i=1}^n \overline{a_i} b_i
$$
The reason we take the complex conjugate $\overline{a_i}$​ in the inner product comes down to one crucial requirement of inner products: **positive definiteness**.

- Let $\mathcal{F}$ be a space of real-valued functions defined on a closed interval $[a, b] \subset \mathbb{R}$. We define an inner product on $\mathcal{F}$ by:

$$
\langle f, g \rangle = \int_a^b f(x) g(x) \, dx
$$ 
for all $f, g \in \mathcal{F}$. One checks that this satisfies the symmetry, linearity, and positivity requirements of an inner product (on the appropriate class of functions).

## 2.2 Fundamental Inequalities in Inner Product Spaces

Given an inner product space $\mathbf{V}$, the inner product and induced norm obey several fundamental geometric inequalities. These inequalities are essential for the geometry of vector spaces, including the concepts of angles, projections, and distances.

### **Cauchy–Schwarz Inequality**
For all $\mathbf{x}, \mathbf{y} \in \mathbf{V}$,
$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|.
$$
**Equality holds** if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.

**Cauchy–Schwarz as a Projection Inequality**: The Cauchy–Schwarz inequality also implies:
$$
\|\mathbf{y}\| \geq \frac{|\langle \mathbf{x}, \mathbf{y} \rangle|}{\|\mathbf{x}\|}.
$$

This expresses that the length of the projection of $\mathbf{y}$ onto $\mathbf{x}$ — given by $\frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\|}$ — is never greater than $\|\mathbf{y}\|$. It captures how much of $\mathbf{y}$ "lies in the direction" of $\mathbf{x}$.

**Interpretation**:  
This bounds the absolute value of the inner product — thought of as the projection of $\mathbf{y}$ onto $\mathbf{x}$ — by the product of the lengths.  It also ensures that the cosine of the angle between two vectors lies in $[-1, 1]$.

### **Triangle Inequality**
For all $\mathbf{x}, \mathbf{y} \in \mathbf{V}$,
$$
\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.
$$

**Proof sketch**:  
Apply Cauchy–Schwarz to expand:
$$
\|\mathbf{x} + \mathbf{y}\|^2 = \langle \mathbf{x}, \mathbf{x} \rangle + 2\langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle
\leq \|\mathbf{x}\|^2 + 2\|\mathbf{x}\|\|\mathbf{y}\| + \|\mathbf{y}\|^2 = (\|\mathbf{x}\| + \|\mathbf{y}\|)^2.
$$

### **Parallelogram Law**
For all $\mathbf{x}, \mathbf{y} \in \mathbf{V}$,
$$
\|\mathbf{x} + \mathbf{y}\|^2 + \|\mathbf{x} - \mathbf{y}\|^2 = 2(\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2).
$$

Here’s a quick inner‐product proof. Then

$$
\|\mathbf{x} + \mathbf{y}\|^2 = \langle \mathbf{x} + \mathbf{y},\, \mathbf{x} + \mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{x} \rangle + 2\langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle,
$$

$$
\|\mathbf{x} - \mathbf{y}\|^2 = \langle \mathbf{x} - \mathbf{y},\, \mathbf{x} - \mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{x} \rangle - 2\langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle.
$$

Adding them cancels the cross-terms:

$$
\|\mathbf{x} + \mathbf{y}\|^2 + \|\mathbf{x} - \mathbf{y}\|^2 = (\langle \mathbf{x}, \mathbf{x} \rangle + 2\langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle) + (\langle \mathbf{x}, \mathbf{x} \rangle - 2\langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{y}, \mathbf{y} \rangle) = 2\langle \mathbf{x}, \mathbf{x} \rangle + 2\langle \mathbf{y}, \mathbf{y} \rangle,
$$

which is exactly

$$
2(\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2).
$$

This is the **Parallelogram Law**.

**Interpretation**:  
This identity reflects the geometric structure of inner product spaces and distinguishes them from general normed spaces (i.e., Banach spaces). It's used to characterize norms induced by inner products.

# 3. Normed Vector Spaces and Cauchy Sequences

## 3.1 Normed Spaces

A norm on a vector space $\mathbf{V}$ is a function $\|\cdot\|: \mathbf{V} \to \mathbf{R}_{\geq 0}$ satisfying:

1. **Positive Scalability**:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. **Triangle Inequality**:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. **Definiteness**:
   $$\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}.$$

When $\mathbf{V}$ is equipped with such a norm $\|\cdot\|$, we say $\mathbf{V}$ is a **normed vector space**.

Every norm $\|\cdot\|$ yields a distance $\rho(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$, making $(\mathbf{V}, \rho)$ into a metric space. Conversely, not every metric arises from a norm, but many spaces in analysis do come from norms.

![[normed-spaces.png]]
 ## **3.2 Metric Vector Spaces**

A **metric vector space** is a vector space $\mathbf{V}$ equipped with a **distance function** (also called a **metric**)  
$$
\rho: \mathbf{V} \times \mathbf{V} \to \mathbb{R}_{\geq 0}
$$
that satisfies the following axioms for all $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathbf{V}$:

1. **Non-negativity and identity of indiscernibles**:
   $$
   \rho(\mathbf{x}, \mathbf{y}) \geq 0 \quad \text{and} \quad \rho(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}.
   $$

2. **Symmetry**:
   $$
   \rho(\mathbf{x}, \mathbf{y}) = \rho(\mathbf{y}, \mathbf{x}).
   $$

3. **Triangle inequality**:
   $$
   \rho(\mathbf{x}, \mathbf{y}) \leq \rho(\mathbf{x}, \mathbf{z}) + \rho(\mathbf{z}, \mathbf{y}) \quad \text{for all } \mathbf{z} \in \mathbf{V}.
   $$

Any vector space $\mathbf{V}$ equipped with such a distance function becomes a **metric vector space**.

**💡 Example: $\mathbb{R}$ as a Metric Vector Space**

Let $\rho(x, y) = |x - y|$ for $x, y \in \mathbb{R}$. Then:

- $\rho(x, y) = 0 \iff x = y$
- $\rho(x, y) = |x - y| = |y - x| = \rho(y, x)$
- For any $z \in \mathbb{R}$:
  $$
  |x - y| = |x - z + z - y| \leq |x - z| + |z - y|
  $$

So $(\mathbb{R}, \rho)$ is a **metric vector space**.

**🧠 Connection with Normed Spaces**

Every **normed vector space** $(\mathbf{V}, \|\cdot\|)$ automatically defines a **metric** via:
$$
\rho(\mathbf{x}, \mathbf{y}) := \|\mathbf{x} - \mathbf{y}\|.
$$

So **every normed vector space is also a metric vector space**, but not every metric space arises from a norm. This metric structure allows us to talk about **convergence**, **Cauchy sequences**, and **completeness**, which are foundational for defining Banach and Hilbert spaces.

## 3.3 Cauchy Sequences

A sequence $\{\mathbf{x}_n\}$ in a normed space $\mathbf{V}$ is called a **Cauchy sequence** if for every $\varepsilon > 0$, there exists $N$ such that for all $m, n > N$,

$$\|\mathbf{x}_m - \mathbf{x}_n\| < \varepsilon.$$

In simpler terms, a Cauchy sequence is one in which elements get arbitrarily close to each other as the sequence progresses.

A sequence $\{\mathbf{x}_n\}$ converges in $\mathbf{V}$ if there exists $\mathbf{x} \in \mathbf{V}$ such that $\|\mathbf{x}_n - \mathbf{x}\| \to 0$.

## 3.4 Completeness

A **complete normed space** is one in which every Cauchy sequence converges to a limit within that space. If $(\mathbf{V}, \|\cdot\|)$ is complete, we call $\mathbf{V}$ a **Banach space**.

All finite-dimensional normed spaces are complete, hence any $\mathbf{R}^n$ or $\mathbf{C}^n$ with a standard norm is a Banach space. In infinite dimensions, completeness becomes a significant property that is not automatic.

![[complete-spaces.png]]

# 4. Banach Spaces

A **Banach space** is thus defined as a complete normed vector space. Important examples include:

- **$\ell^p$ spaces**: Sequences $\mathbf{x} = (x_1, x_2, \dots)$ such that $\sum_{i=1}^\infty |x_i|^p < \infty$. Equipped with the norm
  $$\|\mathbf{x}\|_{\ell^p} = \left( \sum_{i=1}^\infty |x_i|^p \right)^{1/p}, \quad p \geq 1.$$
  Each $\ell^p$ space is complete, making it a Banach space.

- **$L^p([a, b])$ spaces**: Functions $f$ (strictly, equivalence classes of functions) on $[a, b]$ such that  $\int_a^b |f(x)|^p \, dx < \infty.$ Equipped with
  $$\|\mathbf{f}\|_{L^p} = \left( \int_a^b |f(x)|^p \, dx \right)^{1/p},$$
  each $L^p$ is also a Banach space.

# LECTURE-8
# 5. Hilbert Spaces

## 5.1 Definition

A **Hilbert space** is a complete inner product space. Thus it is both:

- A normed space (with norm induced by the inner product).
- A Banach space (it is complete).

Hence, Hilbert spaces have a rich structure: the distance, angle, projections, orthogonality, and other geometric concepts all exist in an infinite-dimensional analog of Euclidean geometry.

## 5.2 From Pre-Hilbert to Hilbert

An inner product space that is not complete (i.e., where some Cauchy sequences fail to converge to elements within the space) is called a **pre-Hilbert space**. One can often “complete” it by carefully adjoining limit points of Cauchy sequences, thus creating a Hilbert space. For instance:

- The space of continuous functions on $[a, b]$ with the $\|\cdot\|_{L^2}$ norm is not complete, because a limit of continuous functions in the $L^2$-sense might be discontinuous (or differ on sets of measure zero). The completion is precisely $L^2([a, b])$.

## 5.3 Orthogonality, Orthonormal Sets, and Completeness

The inner product enables the notion of **orthogonality**: two vectors $\mathbf{x}, \mathbf{y} \in \mathbf{V}$ are **orthogonal** if
$$
\langle \mathbf{x}, \mathbf{y} \rangle = 0.
$$

A set $\{\mathbf{e}_i\} \subset \mathbf{V}$ is called **orthonormal** if each element has unit norm and is mutually orthogonal:
$$
\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij}.
$$

**Finite Dimensions**: In finite-dimensional inner product spaces:
- Every orthonormal set is **linearly independent**.
- If it contains $n$ vectors in an $n$-dimensional space, it is a **basis**.
- Every vector $\mathbf{x} \in \mathbf{V}$ can be expressed as:
  $$
  \mathbf{x} = \sum_{i=1}^n \alpha_i \mathbf{e}_i, \quad \text{where } \alpha_i = \langle \mathbf{x}, \mathbf{e}_i \rangle.
  $$

This defines a **coordinate system** for the space.

**Example**: The standard orthonormal basis of $\mathbb{R}^3$:
$$
\mathbf{e}_1 = (1,0,0), \quad \mathbf{e}_2 = (0,1,0), \quad \mathbf{e}_3 = (0,0,1).
$$


**Infinite Dimensions and the Notion of Completeness**: In infinite-dimensional spaces, the situation becomes more subtle.

An orthonormal set $\{\mathbf{e}_i\}_{i=1}^\infty$ may **not** be a complete basis. That is, some vectors $\mathbf{f} \in \mathbf{V}$ may **not** be expressible as
$$
\mathbf{f} = \sum_{i=1}^\infty \langle \mathbf{f}, \mathbf{e}_i \rangle \mathbf{e}_i.
$$

### ⚠️ Example: Notch Function

Let:
$$
f(x) =
\begin{cases}
1 & \text{if } x \in [0,1] \setminus [0.4, 0.6], \\
0 & \text{if } x \in [0.4, 0.6].
\end{cases}
$$

This function is **not continuous**, but it is square integrable — so $f \in L^2([0,1])$, but $f \notin C([0,1])$.

You can approximate $f$ using a Fourier series:
$$
f(x) \approx \sum_{i=1}^\infty \langle f, \mathbf{e}_i \rangle \mathbf{e}_i
$$
where $\{\mathbf{e}_i\} \subset C([0,1])$. However, the limit function lies **outside** of $C([0,1])$, even though the series converges in the $L^2$-norm. **Conclusion**: $C([0,1])$ is **not complete** under the $L^2$-norm — it has “holes.”

### Cauchy Criterion and Completeness

To formally assess whether an inner product space is complete, we use the **Cauchy criterion**:
> A space $\mathbf{V}$ is complete if **every Cauchy sequence** in $\mathbf{V}$ converges to an element in $\mathbf{V}$.

That is, if
$$
\forall \varepsilon > 0, \exists N \text{ such that } \|\mathbf{x}_m - \mathbf{x}_n\| < \varepsilon \text{ for all } m,n > N
$$
then there exists $\mathbf{x} \in \mathbf{V}$ such that $\mathbf{x}_n \to \mathbf{x}$ in norm.

### 📐 Bessel’s Inequality and Parseval’s Identity

These results help quantify the relationship between an element $\mathbf{f}$ and an orthonormal set $\{\mathbf{e}_i\}$:

- **Bessel’s Inequality** (always holds):
  $$
  \sum_{i=1}^\infty |\langle \mathbf{f}, \mathbf{e}_i \rangle|^2 \leq \|\mathbf{f}\|^2.
  $$
  Equality may **not** hold unless the orthonormal set is **complete**.

- **Parseval’s Identity** (characterizes completeness):
  $$
  \|\mathbf{f}\|^2 = \sum_{i=1}^\infty |\langle \mathbf{f}, \mathbf{e}_i \rangle|^2
  \quad \Longleftrightarrow \quad \{\mathbf{e}_i\} \text{ is a complete orthonormal system (basis)}.
  $$
**Short Derivation of Bessel’s Inequality**: Let $\{\mathbf{e}_i\}$ be an infinite orthonormal set in a Hilbert space $\mathbf{V}$, and let $\mathbf{f} \in \mathbf{V}$ be any element. Define the **Fourier coefficients**:

$$
c_i := \langle \mathbf{f}, \mathbf{e}_i \rangle,
$$

and consider the $n$-term partial sum (orthogonal projection onto the span of $\{\mathbf{e}_1, \dots, \mathbf{e}_n\}$):

$$
\mathbf{f}_n := \sum_{i=1}^n c_i \mathbf{e}_i.
$$

By **orthonormality**, we immediately obtain:

$$
\|\mathbf{f}_n\|^2 = \left\langle \sum_{i=1}^n c_i \mathbf{e}_i,\ \sum_{j=1}^n c_j \mathbf{e}_j \right\rangle = \sum_{i=1}^n |c_i|^2.
$$

Since $\mathbf{f}_n$ is the **orthogonal projection** of $\mathbf{f}$ onto a subspace, we have:

$$
\|\mathbf{f}_n\| \leq \|\mathbf{f}\|,
$$

so squaring gives:

$$
\sum_{i=1}^n |\langle \mathbf{f}, \mathbf{e}_i \rangle|^2 = \|\mathbf{f}_n\|^2 \leq \|\mathbf{f}\|^2.
$$

This is **Bessel’s Inequality**. Taking the limit as $n \to \infty$:

$$
\sum_{i=1}^\infty |\langle \mathbf{f}, \mathbf{e}_i \rangle|^2 \leq \|\mathbf{f}\|^2.
$$

This shows that the **Fourier series expansion**:

$$
\sum_{i=1}^\infty \langle \mathbf{f}, \mathbf{e}_i \rangle \mathbf{e}_i
$$

is a **Cauchy sequence** in $\mathbf{V}$ (its norm is bounded), and thus **converges in the Hilbert space**. The critical question is whether it converges to $\mathbf{f}$. If it does, then $\{\mathbf{e}_i\}$ is a **complete orthonormal system**, or **Hilbert basis**.

# 6. $\ell^2$ and $L^2$ as Hilbert Spaces

## 6.1 $\ell^2$

Let $\mathbf{x} = (x_1, x_2, \dots)$ be an infinite sequence of real or complex numbers. We say $\mathbf{x} \in \ell^2$ if

$$
\sum_{i=1}^\infty |x_i|^2 < \infty.
$$

This space consists of all square-summable sequences. We define the inner product on $\ell^2$ by:

$$
\langle \mathbf{x}, \mathbf{y} \rangle := \sum_{i=1}^\infty \overline{x_i} y_i,
$$

where $\overline{x_i}$ denotes the complex conjugate of $x_i$ (if the field is $\mathbb{C}$). The induced norm is:

$$
\|\mathbf{x}\|_{\ell^2} := \sqrt{ \langle \mathbf{x}, \mathbf{x} \rangle } = \left( \sum_{i=1}^\infty |x_i|^2 \right)^{1/2}.
$$

This inner product satisfies:
- Linearity in the first argument,
- Conjugate symmetry: $\langle \mathbf{x}, \mathbf{y} \rangle = \overline{ \langle \mathbf{y}, \mathbf{x} \rangle }$,
- Positive definiteness: $\langle \mathbf{x}, \mathbf{x} \rangle = 0 \iff \mathbf{x} = 0$.

Hence, $\ell^2$ is an **inner product space**.

**Completeness of $\ell^2$**: Let $\{ \mathbf{x}^{(n)} \} \subset \ell^2$ be a **Cauchy sequence** in the norm:

$$
\| \mathbf{x}^{(n)} - \mathbf{x}^{(m)} \|_{\ell^2} \to 0 \quad \text{as } n,m \to \infty.
$$

Then for each index $i \in \mathbb{N}$, the scalar sequence $\{ x_i^{(n)} \}$ is Cauchy in $\mathbb{R}$ or $\mathbb{C}$, and thus converges:
$$
x_i^{(n)} \to x_i.
$$

Define the limit sequence $\mathbf{x} = (x_1, x_2, \dots)$. Using Fatou’s Lemma: $\sum_{i=1}^\infty |x_i|^2 \leq \liminf_{n \to \infty} \sum_{i=1}^\infty |x_i^{(n)}|^2 < \infty,$ so $\mathbf{x} \in \ell^2$. Moreover,

$$
\| \mathbf{x}^{(n)} - \mathbf{x} \|_{\ell^2}^2 = \sum_{i=1}^\infty |x_i^{(n)} - x_i|^2 \to 0.
$$

Hence $\mathbf{x}^{(n)} \to \mathbf{x}$ in norm. Conclude the space $\ell^2$ is an inner product space and is **complete** with respect to the norm induced by its inner product.

$$
\boxed{\ell^2 \text{ is a Hilbert space.}}
$$

## 6.2 $L^2([a, b])$

The space $L^2([a, b])$ consists of (equivalence classes of) functions $f: [a, b] \to \mathbb{R}$ or $\mathbb{C}$ such that:

$$
\int_a^b |f(x)|^2 \, dx < \infty.
$$

Two functions are considered **equal** in $L^2$ if they differ only on a set of measure zero. We define the inner product:

$$
\langle f, g \rangle := \int_a^b f(x) \overline{g(x)} \, dx,
$$

with induced norm:

$$
\|f\|_{L^2} := \sqrt{ \langle f, f \rangle } = \left( \int_a^b |f(x)|^2 dx \right)^{1/2}.
$$

This satisfies:
- **Linearity** in the first argument,
- **Conjugate symmetry**: $\langle f, g \rangle = \overline{ \langle g, f \rangle }$,
- **Positive definiteness**: $\langle f, f \rangle = 0 \iff f = 0$ almost everywhere.

So $L^2([a,b])$ is an **inner product space**.

**Completeness of $L^2([a, b])$**: Let $\{f_n\} \subset L^2([a, b])$ be a **Cauchy sequence** in the $L^2$-norm:
$$
\|f_n - f_m\|_{L^2}^2 = \int_a^b |f_n(x) - f_m(x)|^2 \, dx \to 0 \quad \text{as } n,m \to \infty.
$$

By the **Cauchy property**, for each $\varepsilon > 0$, there exists $N$ such that for all $n, m > N$,
$$
\int_a^b |f_n(x) - f_m(x)|^2 \, dx < \varepsilon.
$$

This implies that $\{f_n(x)\}$ is Cauchy in the metric space $L^2([a,b])$. From the completeness of the Lebesgue integral, there exists a function $f \in L^2([a, b])$ such that:

$$
\int_a^b |f_n(x) - f(x)|^2 \, dx \to 0 \quad \text{as } n \to \infty.
$$

In other words, $f_n \to f$ in the $L^2$-norm. Hence, **every Cauchy sequence in $L^2$ converges** in norm to a function within $L^2$.

### Interpretation of $L^2$ Convergence

We say:
$$
f_n \to f \quad \text{in } L^2 \text{ sense if } \|f_n - f\|_{L^2} \to 0,
$$
i.e.,
$$
\int_a^b |f_n(x) - f(x)|^2 \, dx \to 0.
$$

This implies that $f_n(x) \to f(x)$ **in measure**, and up to a subsequence, **almost everywhere**.

### Conclude the space $L^2([a, b])$ is an inner product space and **complete** in the induced norm.

$$
\boxed{L^2([a, b]) \text{ is a Hilbert space.}}
$$

## 6.3 Isomorphism of $\ell^2$ and $L^2$

Let $f(x) \in L^2$ be square-integrable. We can represent $f$ as a **Fourier series**:

$$
f(x) = \sum_{k=1}^\infty c_k \phi_k(x),
$$

where $\{\phi_k\}$ is an **orthonormal basis** of $L^2$ and $c_k = \langle f, \phi_k \rangle$ are the **Fourier coefficients**.

From **Bessel’s inequality**:

$$
\sum_{k=1}^\infty |c_k|^2 \leq \|f\|_{L^2}^2,
$$

so the sequence $\{c_k\}$ belongs to $\ell^2$. Therefore, every function $f \in L^2$ corresponds to a sequence $c \in \ell^2$ via the mapping $c_k = \langle f, \phi_k \rangle$. This establishes an isomorphism between $L^2$ and $\ell^2$ — a one-to-one correspondence that preserves the inner product structure. We may interpret $\ell^2$ as a coordinate system for $L^2$.

*Key points:*
- **Bessel’s inequality** requires only an **orthonormal system** $\{e_n\}$.
- **Parseval’s identity** holds **iff** that orthonormal system is **complete** (i.e.\ an orthonormal basis).

# 7. Reproducing Kernel Hilbert Spaces (RKHS)

**Preliminaries:** A **linear functional** on $\mathcal{H}$ is a map $L:\mathcal{H}\to\mathbb{R}$ satisfying

1. $L(af+bg)=aL(f)+bL(g)$ for all $f,g\in\mathcal{H}$ and scalars $a,b$.  
2. $L$ is continuous (equivalently, bounded: there exists $C<\infty$ with $|L(f)|\le C\|f\|$ for all $f$).

By the **Riesz Representation Theorem**, every continuous linear functional $L$ on a Hilbert space $\mathcal{H}$ can be represented as

$$
L(f)=\langle f, h\rangle_{\mathcal{H}},
$$

for a unique element $h\in\mathcal{H}$.

## 7.1 Definition

A Hilbert space $\mathcal{H}$ of functions $f:X\to\mathbb{R}$ is called a **Reproducing Kernel Hilbert Space** if, for every point $x\in X$, the evaluation functional

$$
L_x: \mathcal{H}\to\mathbb{R},\qquad L_x(f)=f(x),
$$

is continuous (bounded).

By the Riesz Representation Theorem, there exists a unique element $K_x\in\mathcal{H}$ such that for all $f\in\mathcal{H}$,

$$
f(x) = L_x(f) = \langle f, K_x\rangle_{\mathcal{H}}.
$$

We define the **reproducing kernel**

$$
K:X\times X\to\mathbb{R},\qquad K(x,y) := K_y(x) = \langle K_x, K_y\rangle_{\mathcal{H}}.
$$

Properties of $K$:

1. **Symmetry**: $K(x,y)=K(y,x)$.
2. **Positive Definiteness**: For any finite set $\{x_1,\dots,x_n\}\subset X$ and any real coefficients $c_1,\dots,c_n$,

   $$
   \sum_{i=1}^n\sum_{j=1}^n c_ic_jK(x_i,x_j) \;=\; \bigl\langle\sum_i c_iK_{x_i},\sum_j c_jK_{x_j}\bigr\rangle_{\mathcal{H}} \;\ge\;0.
   $$

Conversely, a function $K:X\times X\to\mathbb{R}$ that is symmetric and positive definite uniquely determines an RKHS in which it is the reproducing kernel (the **Moore–Aronszajn theorem**).

## 7.2 Construction of an RKHS from a Kernel

Given a symmetric positive-definite kernel $K$, define the vector space

$$
\mathcal{H}_0 = \mathrm{span}\{K(\cdot,x):x\in X\},
$$

and equip it with the inner product determined by

$$
\left\langle \sum_{i=1}^n\alpha_iK(\cdot,x_i),\;\sum_{j=1}^m\beta_jK(\cdot,y_j)\right\rangle_{\mathcal{H}_0}
= \sum_{i=1}^n\sum_{j=1}^m \alpha_i\beta_j\,K(x_i,y_j).
$$

Completing this pre-Hilbert space under the induced norm yields a Hilbert space $\mathcal{H}$, in which one verifies that the reproducing property holds and $K$ is indeed its reproducing kernel.

## 7.3 Examples of RKHS Kernels

1. **Linear Kernel** on $\mathbb{R}^d$:  
   $$K(x,y)=x^\top y,\qquad \mathcal{H}=\{f(x)=w^\top x: w\in\mathbb{R}^d\},\;\|f\|=\|w\|_2.$$  
2. **Gaussian (RBF) Kernel** on $\mathbb{R}^d$:  
   $$K(x,y)=\exp\bigl(-\|x-y\|^2/(2\sigma^2)\bigr).$$  
   The associated RKHS consists of functions with rapidly decaying Fourier spectra.  

3. **Polynomial Kernel**:  
   $$K(x,y)=(\gamma\,x^\top y + c)^p,\quad p\in\mathbb{N}.$$  

Each kernel imposes a different geometry (norm) on function space, controlling smoothness and complexity.


## 7.4 The Reproducing Property

For any $f\in\mathcal{H}$ and $x\in X$, since $K_x\in\mathcal{H}$ by construction,

$$
f(x) = \langle f, K_x \rangle_{\mathcal{H}},
$$

and in particular, taking $f=K_y$ gives

$$
K_y(x) = K(x,y) = \langle K_y, K_x \rangle_{\mathcal{H}},
$$

which shows symmetry and ties pointwise evaluation directly to the inner product.

## 7.5 Representer Theorem

**Setup:** Given data $\{(x_n,y_n)\}_{n=1}^N\subset X\times\mathbb{R}$, consider the regularized risk minimization problem

$$
\min_{f\in\mathcal{H}} \;J(f):=\frac1N\sum_{n=1}^N\bigl(f(x_n)-y_n\bigr)^2 + \lambda\|f\|_{\mathcal{H}}^2,
$$

where $\lambda>0$ is a regularization parameter.

**Theorem (Representer):** Any minimizer $f^\star$ of $J$ admits a representation of the form


$$
f^\star(x)
= \sum_{n=1}^N \alpha_n\,K(x,x_n),
$$

for some coefficients $\alpha=(\alpha_1,\dots,\alpha_N)^\top\in\mathbb{R}^N$.

**Proof Sketch:**  
1. Decompose an arbitrary $f\in\mathcal{H}$ into its projection onto the span of $\{K(\cdot,x_n)\}_{n=1}^N$ and its orthogonal complement:  
   $$f = f_\parallel + f_\perp,\quad f_\parallel\in\mathrm{span}\{K(\cdot,x_n)\},\quad f_\perp\perp \mathrm{span}\{K(\cdot,x_n)\}.$$  
2. For each $n$, $f_\perp(x_n)=\langle f_\perp, K(\cdot,x_n)\rangle=0$, so the empirical loss depends only on $f_\parallel$.  
3. Meanwhile, $\|f\|^2 = \|f_\parallel\|^2 + \|f_\perp\|^2 \ge \|f_\parallel\|^2$.  
4. Hence replacing $f$ by $f_\parallel$ does not increase the objective, so an optimizer must lie in the finite-dimensional span.

**Solving for the Coefficients:**

Writing $f^\star(x)=\sum_{n=1}^N\alpha_nK(x,x_n)$, define the **Gram matrix** $K\in\mathbb{R}^{N\times N}$ by $K_{ij}=K(x_i,x_j)$.  Let $y=(y_1,\dots,y_N)^\top$.  Then one shows that the $\alpha$ minimizing $J$ satisfies the linear system

$$
\bigl(K + N\lambda I\bigr)\alpha = y.
$$

Thus implementation reduces to solving an $N\times N$ system.

**Interpretation of the RKHS Norm**

The squared norm $\|f\|_{\mathcal{H}}^2$ serves as a measure of complexity.  In the linear-kernel case it coincides with the squared Euclidean norm of parameters.  For general kernels, it penalizes functions that oscillate or vary too sharply under the geometry induced by $K$.  Regularization by this norm ensures well-posedness and prevents overfitting.

# 8. Concluding Remarks and Applications

**Machine Learning and Statistical Modeling**:

- Banach and Hilbert spaces underlie many methods involving high-dimensional or infinite-dimensional objects.
- Hilbert spaces ($L^2$, $\ell^2$, etc.) let us use geometry-like concepts (angles, projections) in function spaces.
- RKHS frameworks are central in kernel-based learning methods (e.g., Support Vector Machines, Gaussian Process regression, kernel ridge regression), where a cleverly chosen kernel $K$ encodes prior assumptions about function smoothness or similarity.

**Further Insights**:

- In finite dimensions, completeness is always guaranteed. The subtleties arise in infinite-dimensional settings, where we can encounter incomplete inner product spaces (pre-Hilbert spaces).
- Completing such spaces typically involves extending them to include all limit points of Cauchy sequences. This