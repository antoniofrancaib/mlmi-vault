## Introduction

This report provides a comprehensive overview of key concepts from functional analysis, approximation theory, and infinite-dimensional probability, drawing upon the material presented in Lectures 7 through 10 by Mark Girolami. The aim is to establish a rigorous mathematical foundation necessary for understanding advanced techniques in machine learning, statistical inference, and related fields where functions themselves are the primary objects of study. We begin by exploring the fundamental algebraic and topological structures of vector spaces, progressing through normed, Banach, inner product, and Hilbert spaces. Subsequently, we delve into specialized function spaces crucial for machine learning, namely Reproducing Kernel Hilbert Spaces (RKHS) and Sobolev spaces, examining their properties and applications in function approximation. Finally, we address the complexities of defining measures in infinite-dimensional spaces, focusing on Gaussian measures and their role in formulating Bayesian inference for functional data, such as in Gaussian Processes. This structured approach illuminates the theoretical underpinnings essential for tackling modern data analysis challenges.

## 1. Foundations: From Vector Spaces to Banach Spaces

The journey into functional analysis begins with the fundamental algebraic structure of vector spaces, upon which layers of topological and geometric structure are built. Understanding these foundational concepts is crucial for appreciating the frameworks used in advanced machine learning and statistics.

### 1.1. Vector Spaces: The Algebraic Bedrock

A vector space, formally defined as a set $V$ over a field $\mathbb{F}$ (typically the real numbers $\mathbb{R}$ or complex numbers $\mathbb{C}$), is equipped with two operations: vector addition ($+$) and scalar multiplication ($\cdot$). These operations must satisfy a set of axioms ensuring a consistent algebraic structure.

Vector addition must form an Abelian group, meaning it satisfies:
- Commutativity: $\mathbf{x}+\mathbf{y}=\mathbf{y}+\mathbf{x}$ for all $\mathbf{x},\mathbf{y}\in V$.
- Associativity: $\mathbf{x}+(\mathbf{y}+\mathbf{z})=(\mathbf{x}+\mathbf{y})+\mathbf{z}$ for all $\mathbf{x},\mathbf{y},\mathbf{z}\in V$.
- Additive Identity: There exists a unique zero vector $\mathbf{0}\in V$ such that $\mathbf{x}+\mathbf{0}=\mathbf{x}$ for all $\mathbf{x}\in V$.
- Additive Inverse: For every $\mathbf{x}\in V$, there exists a unique vector $-\mathbf{x}\in V$ such that $\mathbf{x}+(-\mathbf{x})=\mathbf{0}$.

Scalar multiplication must satisfy:
- Closure: For any scalar $\alpha\in\mathbb{F}$ and vector $\mathbf{x}\in V$, the product $\alpha\mathbf{x}$ is also in $V$.
- Distributivity over vector addition: $\alpha(\mathbf{x}+\mathbf{y})=\alpha\mathbf{x}+\alpha\mathbf{y}$ for all $\alpha\in\mathbb{F}$ and $\mathbf{x},\mathbf{y}\in V$.
- Distributivity over scalar addition: $(\alpha+\beta)\mathbf{x}=\alpha\mathbf{x}+\beta\mathbf{x}$ for all $\alpha,\beta\in\mathbb{F}$ and $\mathbf{x}\in V$.
- Associativity: $\alpha(\beta\mathbf{x})=(\alpha\beta)\mathbf{x}$ for all $\alpha,\beta\in\mathbb{F}$ and $\mathbf{x}\in V$.
- Scalar Identity: $1\cdot\mathbf{x}=\mathbf{x}$ for all $\mathbf{x}\in V$, where $1$ is the multiplicative identity in $\mathbb{F}$. It also follows that $0\cdot\mathbf{x}=\mathbf{0}$.

Standard examples of vector spaces include the familiar Euclidean spaces $\mathbb{R}^n$ and complex spaces $\mathbb{C}^n$. More abstract, yet fundamentally important, examples arise in function spaces. The set of continuous functions on a closed interval $[a,b]$, denoted $C[a,b]$, forms a vector space under pointwise addition and scalar multiplication. Similarly, the Lebesgue spaces $L^p([a,b])$, consisting of functions whose $p$-th power is integrable, are also vector spaces. These function spaces are central to many areas of analysis and its applications.

### 1.2. Inner Product Spaces: Introducing Geometry

While vector spaces provide an algebraic framework, inner product spaces introduce geometric notions like length, distance, and angle. An inner product space is a vector space $V$ over a field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$) equipped with an inner product $\langle\cdot,\cdot\rangle:V\times V\to\mathbb{F}$.

This inner product must satisfy the following axioms for all $\mathbf{x},\mathbf{y},\mathbf{z}\in V$ and $\alpha,\beta\in\mathbb{F}$:
1. Symmetry / Hermitian Symmetry: $\langle\mathbf{x},\mathbf{y}\rangle=\overline{\langle\mathbf{y},\mathbf{x}\rangle}$, where the bar denotes complex conjugation. If the field $\mathbb{F}$ is $\mathbb{R}$, this simplifies to $\langle\mathbf{x},\mathbf{y}\rangle=\langle\mathbf{y},\mathbf{x}\rangle$.
2. Linearity in the first argument: $\langle\alpha\mathbf{x}+\beta\mathbf{y},\mathbf{z}\rangle=\alpha\langle\mathbf{x},\mathbf{z}\rangle+\beta\langle\mathbf{y},\mathbf{z}\rangle$.
3. Conjugate Linearity in the second argument (for complex spaces): $\langle\mathbf{x},\alpha\mathbf{y}+\beta\mathbf{z}\rangle=\bar{\alpha}\langle\mathbf{x},\mathbf{y}\rangle+\bar{\beta}\langle\mathbf{x},\mathbf{z}\rangle$. Note that for real spaces, this reduces to linearity in the second argument as well.
4. Positive Definiteness: $\langle\mathbf{x},\mathbf{x}\rangle\geq 0$, and $\langle\mathbf{x},\mathbf{x}\rangle=0$ if and only if $\mathbf{x}=\mathbf{0}$.

The inner product naturally induces a norm on the vector space, defined as $\|\mathbf{x}\|=\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$. This norm represents the "length" or "magnitude" of the vector $\mathbf{x}$.

Common examples include $\mathbb{C}^n$ with the standard inner product $\langle\mathbf{a},\mathbf{b}\rangle=\sum_{i=1}^n a_i\overline{b_i}$. The use of the complex conjugate in the second term is crucial to ensure the positive definiteness property $\langle\mathbf{a},\mathbf{a}\rangle=\sum|a_i|^2\geq 0$. For real function spaces, such as functions defined on an interval $[a,b]$, a common inner product is $\langle f,g\rangle=\int_a^b f(x)g(x)dx$. This inner product induces the $L^2$ norm.

### 1.3. Fundamental Inequalities: Quantifying Relationships

Inner product spaces satisfy several fundamental inequalities that are essential tools in analysis.

**Cauchy-Schwarz Inequality**: This inequality states that for any vectors $\mathbf{x},\mathbf{y}$ in an inner product space, $|\langle\mathbf{x},\mathbf{y}\rangle|\leq\|\mathbf{x}\|\|\mathbf{y}\|$. Equality holds if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent. Geometrically, this inequality relates the inner product to the magnitudes of the vectors, essentially stating that the projection of one vector onto another cannot exceed its length.

**Triangle Inequality**: Derived from the Cauchy-Schwarz inequality, the triangle inequality states that for any vectors $\mathbf{x},\mathbf{y}$, $\|\mathbf{x}+\mathbf{y}\|\leq\|\mathbf{x}\|+\|\mathbf{y}\|$. This confirms that the norm induced by an inner product behaves like a measure of length, satisfying the intuitive geometric property that the length of one side of a triangle cannot exceed the sum of the lengths of the other two sides.

**Parallelogram Law**: This law states that for any vectors $\mathbf{x},\mathbf{y}$, $\|\mathbf{x}+\mathbf{y}\|^2+\|\mathbf{x}-\mathbf{y}\|^2=2(\|\mathbf{x}\|^2+\|\mathbf{y}\|^2)$. Geometrically, it relates the sum of the squares of the lengths of the diagonals of a parallelogram to the sum of the squares of the lengths of its four sides. Significantly, this law holds if and only if the norm is induced by an inner product. This makes it a powerful tool for determining whether a given normed space can also be equipped with an inner product that generates the norm. Spaces like $L^p$ for $p\neq 2$ do not satisfy the parallelogram law, indicating their norms do not arise from an inner product.

### 1.4. Normed Vector Spaces: Measuring Magnitude

A normed vector space is a vector space $V$ equipped with a function $\|\cdot\|:V\to\mathbb{R}_{\geq 0}$, called a norm, satisfying three axioms for all $\mathbf{x},\mathbf{y}\in V$ and scalar $\alpha\in\mathbb{F}$:
1. Definiteness: $\|\mathbf{x}\|\geq 0$, and $\|\mathbf{x}\|=0$ if and only if $\mathbf{x}=\mathbf{0}$.
2. Positive Scalability (Homogeneity): $\|\alpha\mathbf{x}\|=|\alpha|\|\mathbf{x}\|$.
3. Triangle Inequality: $\|\mathbf{x}+\mathbf{y}\|\leq\|\mathbf{x}\|+\|\mathbf{y}\|$.

The norm generalizes the concept of length or magnitude to abstract vector spaces. Every normed space naturally becomes a metric space by defining the distance (metric) between two vectors $\mathbf{x}$ and $\mathbf{y}$ as $\rho(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|$. This induced metric satisfies the standard metric space axioms: non-negativity ($\rho(\mathbf{x},\mathbf{y})\geq 0$, with equality iff $\mathbf{x}=\mathbf{y}$), symmetry ($\rho(\mathbf{x},\mathbf{y})=\rho(\mathbf{y},\mathbf{x})$), and the triangle inequality ($\rho(\mathbf{x},\mathbf{y})\leq\rho(\mathbf{x},\mathbf{z})+\rho(\mathbf{z},\mathbf{y})$). This allows us to discuss concepts like convergence and continuity within normed spaces using the familiar framework of metric spaces.

### 1.5. Convergence, Cauchy Sequences, and Completeness

The introduction of a norm (and thus a metric) allows for the definition of convergence in a vector space. A sequence of vectors $\{\mathbf{x}_n\}_{n=1}^\infty$ in a normed space $V$ converges to a limit $\mathbf{x}\in V$ if, for any $\epsilon>0$, there exists an integer $N$ such that for all $n>N$, $\|\mathbf{x}_n-\mathbf{x}\|<\epsilon$. This is often written as $\lim_{n\to\infty}\|\mathbf{x}_n-\mathbf{x}\|=0$.

A related concept is that of a Cauchy sequence. A sequence $\{\mathbf{x}_n\}$ is Cauchy if its terms eventually become arbitrarily close to each other. Formally, for every $\epsilon>0$, there exists an integer $N$ such that for all $m,n>N$, $\|\mathbf{x}_m-\mathbf{x}_n\|<\epsilon$. Every convergent sequence is necessarily a Cauchy sequence. However, the converse is not always true; a Cauchy sequence might not converge to a limit within the space $V$.

This leads to the crucial concept of completeness. A normed vector space is called complete if every Cauchy sequence in the space converges to a limit that is also in the space. Completeness is a fundamental property in analysis, ensuring that limiting processes yield results within the space itself. Without completeness, operations like integration or solving differential equations might lead outside the original space, complicating analysis.

### 1.6. Banach Spaces: Complete Normed Spaces

A Banach space is defined as a normed vector space that is complete with respect to the metric induced by its norm. Banach spaces provide a robust framework for functional analysis, combining algebraic structure (vector space) with a notion of magnitude (norm) and the analytical power of completeness.

Key examples include the finite-dimensional spaces $\mathbb{R}^n$ and $\mathbb{C}^n$, which are always complete. More significant infinite-dimensional examples include the sequence spaces $\ell^p$ for $1\leq p\leq\infty$, defined as sequences $\mathbf{x}=(x_1,x_2,\ldots)$ such that $\sum_{i=1}^\infty |x_i|^p<\infty$, with the norm $\|\mathbf{x}\|_{\ell^p}=(\sum_{i=1}^\infty |x_i|^p)^{1/p}$. Similarly, the function spaces $L^p([a,b])$ for $1\leq p\leq\infty$, consisting of functions $f$ such that $\int_a^b |f(x)|^p dx<\infty$, equipped with the norm $\|f\|_{L^p}=(\int_a^b |f(x)|^p dx)^{1/p}$, are also Banach spaces. The completeness of these spaces is a cornerstone of modern analysis and its applications. The requirement of completeness ensures that sequences that 'ought' to converge (because their terms get arbitrarily close) actually do converge to an element within the space, preventing analytical procedures from leading to undefined results.

## 2. Hilbert Spaces: Geometry and Structure

Building upon the foundations of vector spaces, norms, and completeness, Hilbert spaces introduce the rich geometric structure of an inner product, providing a powerful framework particularly suited for applications involving orthogonality, projections, and spectral theory.

### 2.1. Definition and Relationship to Other Spaces

A Hilbert space is defined as an inner product space that is also complete with respect to the norm induced by its inner product. Recall that the inner product $\langle\cdot,\cdot\rangle$ induces a norm $\|\mathbf{x}\|=\sqrt{\langle\mathbf{x},\mathbf{x}\rangle}$. Therefore, a Hilbert space $\mathcal{H}$ is a vector space equipped with an inner product $\langle\cdot,\cdot\rangle_{\mathcal{H}}$ such that the metric $\rho(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_{\mathcal{H}}=\sqrt{\langle\mathbf{x}-\mathbf{y},\mathbf{x}-\mathbf{y}\rangle_{\mathcal{H}}}$ makes $\mathcal{H}$ a complete metric space.

This definition places Hilbert spaces at the intersection of several important mathematical structures. Every Hilbert space is inherently an inner product space, possessing geometric notions like angles and orthogonality. Since the inner product induces a norm satisfying the required axioms (including the triangle inequality, proven via Cauchy-Schwarz), every Hilbert space is also a normed vector space. Furthermore, due to the completeness requirement, every Hilbert space is a Banach space. The additional structure provided by the inner product, particularly the Parallelogram Law (Section 1.3), distinguishes Hilbert spaces from general Banach spaces and enables many powerful geometric and analytical results.

A space equipped with an inner product but lacking completeness is termed a pre-Hilbert space. A fundamental result in functional analysis states that any pre-Hilbert space can be uniquely "completed" into a Hilbert space. This completion process involves adding the limit points of all Cauchy sequences that do not converge within the original space. A standard example is the space of continuous functions on $[a,b]$, $C[a,b]$, equipped with the $L^2$ inner product $\langle f,g\rangle=\int_a^b f(x)g(x)dx$. This space is a pre-Hilbert space but is not complete. Its completion is the space $L^2([a,b])$, which includes functions that are limits of sequences of continuous functions in the $L^2$ norm, even if the limit function itself is not continuous. This completeness is crucial for many applications, ensuring that operations like projections onto closed subspaces are well-defined.

### 2.2. Orthogonality and Orthonormal Systems

The inner product structure allows for the definition of orthogonality: two vectors $\mathbf{x},\mathbf{y}$ in an inner product space $\mathcal{H}$ are orthogonal if $\langle\mathbf{x},\mathbf{y}\rangle=0$, denoted $\mathbf{x}\perp\mathbf{y}$. This generalizes the concept of perpendicularity from Euclidean geometry.

An orthonormal set $\{e_i\}_{i\in I}$ in $\mathcal{H}$ is a collection of vectors such that each vector has unit norm ($\|e_i\|=1$) and any two distinct vectors are orthogonal ($\langle e_i,e_j\rangle=0$ for $i\neq j$). This can be compactly written as $\langle e_i,e_j\rangle=\delta_{ij}$, where $\delta_{ij}$ is the Kronecker delta.

In finite-dimensional inner product spaces (like $\mathbb{R}^n$ or $\mathbb{C}^n$), any orthonormal set is automatically linearly independent. If the number of vectors in the set equals the dimension of the space, it forms an orthonormal basis. In this case, any vector $\mathbf{x}$ in the space can be uniquely represented as a finite linear combination of these basis vectors: $\mathbf{x}=\sum_{i=1}^n c_i e_i$, where the coefficients (Fourier coefficients) are given by $c_i=\langle\mathbf{x},e_i\rangle$.

In infinite-dimensional Hilbert spaces, the situation is more nuanced. An orthonormal set might be maximal (meaning no non-zero vector is orthogonal to all elements of the set) but still not span the entire space through finite linear combinations. We instead consider convergence in the norm topology. An orthonormal set $\{e_i\}_{i=1}^\infty$ is called a complete orthonormal system (or Hilbert basis) if every vector $\mathbf{x}\in\mathcal{H}$ can be represented as an infinite series $\mathbf{x}=\sum_{i=1}^\infty \langle\mathbf{x},e_i\rangle e_i$, where the convergence is in the norm of $\mathcal{H}$ (i.e., $\|\mathbf{x}-\sum_{i=1}^N \langle\mathbf{x},e_i\rangle e_i\|\to 0$ as $N\to\infty$). The existence of such a basis is guaranteed for separable Hilbert spaces. The distinction arises because infinite sums require convergence, which relies on the completeness of the space. An example illustrating this subtlety involves approximating a discontinuous function (like a step function or "notch") within $L^2()$ using a sequence of continuous functions (e.g., from $C$). The sequence might converge in the $L^2$ norm to the discontinuous function, which is in $L^2()$ but not in the original space $C$.

### 2.3. Bessel's Inequality and Parseval's Identity

Given an orthonormal set $\{e_i\}_{i=1}^\infty$ (possibly finite) in a Hilbert space $\mathcal{H}$, and any vector $\mathbf{f}\in\mathcal{H}$, the scalars $c_i=\langle\mathbf{f},e_i\rangle$ are called the Fourier coefficients of $\mathbf{f}$ with respect to this set.

**Bessel's Inequality** provides a fundamental relationship between the norm of a vector and its Fourier coefficients. It states that for any orthonormal set $\{e_i\}$, the sum of the squared magnitudes of the Fourier coefficients is bounded by the squared norm of the vector:

$$
\sum_{i=1}^\infty |\langle\mathbf{f},e_i\rangle|^2 \leq \|\mathbf{f}\|^2
$$

This inequality holds regardless of whether the orthonormal set is complete. It essentially says that the "energy" contained in the projection of $\mathbf{f}$ onto the subspace spanned by the orthonormal vectors cannot exceed the total energy of $\mathbf{f}$. The proof involves considering the norm squared of the difference between $\mathbf{f}$ and its projection onto the span of the first $N$ vectors, $\|\mathbf{f}-\sum_{i=1}^N \langle\mathbf{f},e_i\rangle e_i\|^2$, which must be non-negative. Expanding this expression leads directly to Bessel's inequality as $N\to\infty$.

**Parseval's Identity** (also known as the completeness relation) strengthens Bessel's inequality to an equality:

$$
\|\mathbf{f}\|^2 = \sum_{i=1}^\infty |\langle\mathbf{f},e_i\rangle|^2
$$

This identity holds for all $\mathbf{f}\in\mathcal{H}$ if and only if the orthonormal set $\{e_i\}$ is a complete orthonormal system (a Hilbert basis) for $\mathcal{H}$. It signifies that the entire "energy" of the vector $\mathbf{f}$ is captured by its components along the basis vectors. Parseval's identity is thus a defining characteristic of a complete orthonormal basis and is crucial for establishing isomorphisms between Hilbert spaces.

### 2.4. Canonical Hilbert Spaces: $\ell^2$ and $L^2$

Two cornerstone examples of infinite-dimensional Hilbert spaces are the sequence space $\ell^2$ and the function space $L^2$.

**The Sequence Space $\ell^2$**: This space consists of all infinite sequences $\mathbf{x}=(x_1,x_2,\ldots)$ of complex (or real) numbers such that the sum of the squares of their magnitudes converges: $\sum_{i=1}^\infty |x_i|^2<\infty$. The inner product is defined as $\langle\mathbf{x},\mathbf{y}\rangle=\sum_{i=1}^\infty x_i\overline{y_i}$, and the induced norm is $\|\mathbf{x}\|_{\ell^2}=(\sum_{i=1}^\infty |x_i|^2)^{1/2}$. The completeness of $\ell^2$ is a standard result in functional analysis, often proven by showing that any Cauchy sequence $(\mathbf{x}^{(n)})$ in $\ell^2$ converges component-wise to a limit sequence $\mathbf{x}=(x_1,x_2,\ldots)$, and then demonstrating that this limit sequence $\mathbf{x}$ is indeed in $\ell^2$ (i.e., $\sum|x_i|^2<\infty$) and that $\mathbf{x}^{(n)}\to\mathbf{x}$ in the $\ell^2$ norm.

**The Function Space $L^2([a,b])$**: This space consists of (equivalence classes of) measurable functions $f:[a,b]\to\mathbb{C}$ (or $\mathbb{R}$) such that the Lebesgue integral of the squared magnitude is finite: $\int_a^b |f(x)|^2 dx<\infty$. The inner product is defined as $\langle f,g\rangle=\int_a^b f(x)\overline{g(x)}dx$, and the induced norm is $\|f\|_{L^2}=(\int_a^b |f(x)|^2 dx)^{1/2}$. The completeness of $L^2([a,b])$ is a fundamental result, often referred to as the Riesz-Fischer theorem. It guarantees that every Cauchy sequence of $L^2$ functions converges (in the $L^2$ norm) to a limit function that is also in $L^2([a,b])$. Convergence in $L^2$ norm, $\|f_n-f\|_{L^2}\to 0$, implies convergence in measure, and it guarantees the existence of a subsequence $\{f_{n_k}\}$ that converges pointwise almost everywhere to $f$.

### 2.5. Isomorphism of Separable Hilbert Spaces ($\ell^2$ and $L^2$)

A profound result in Hilbert space theory is that all infinite-dimensional separable Hilbert spaces are isometrically isomorphic to each other, and thus, to the sequence space $\ell^2$. A Hilbert space is separable if it contains a countable dense subset. $L^2([a,b])$ is a prime example of a separable Hilbert space.

The isomorphism between $L^2([a,b])$ and $\ell^2$ is established via Fourier series expansion. If $\{\phi_k(x)\}_{k=1}^\infty$ is any complete orthonormal basis for $L^2([a,b])$ (e.g., the trigonometric functions for $L^2([-\pi,\pi])$ or Legendre polynomials for $L^2([-1,1])$), then any function $f\in L^2([a,b])$ can be uniquely represented by its sequence of Fourier coefficients $\{c_k\}_{k=1}^\infty$, where $c_k=\langle f,\phi_k\rangle=\int_a^b f(x)\overline{\phi_k(x)}dx$.

The mapping $f\mapsto\{c_k\}$ defines a linear isometry from $L^2([a,b])$ to $\ell^2$. Parseval's identity, $\|f\|_{L^2}^2=\sum_{k=1}^\infty |c_k|^2$, ensures that this mapping preserves the norm (and thus the inner product) and that the sequence $\{c_k\}$ is indeed in $\ell^2$. This isomorphism implies that, from an abstract structural perspective, studying $L^2$ spaces is equivalent to studying the conceptually simpler sequence space $\ell^2$. This unification simplifies many theoretical considerations in functional analysis and its applications.

## 3. Reproducing Kernel Hilbert Spaces (RKHS): Bridging Functions and Data

Reproducing Kernel Hilbert Spaces (RKHS) form a special class of Hilbert spaces of functions that possess properties particularly useful in machine learning, statistics, and approximation theory. Their defining characteristic relates function evaluation to the inner product structure of the space.

### 3.1. Motivation: Point Evaluation Functional

In many function spaces, such as the standard $L^2$ spaces, evaluating a function at a specific point is not a well-defined or continuous operation. For $L^2([a,b])$, functions are defined only up to sets of measure zero, meaning the value at a single point is ambiguous. Even for spaces of continuous functions like $C[a,b]$, while point evaluation $f\mapsto f(x)$ is well-defined, it might not be continuous with respect to certain norms (like the $L^2$ norm). This poses a problem in machine learning contexts where algorithms often rely on evaluating learned functions at specific data points. RKHS provides a framework where point evaluations are guaranteed to be continuous linear functionals, making them particularly suitable for such applications.

### 3.2. The Reproducing Kernel

An RKHS $\mathcal{H}$ over a set $\mathcal{X}$ is a Hilbert space of functions $f:\mathcal{X}\to\mathbb{F}$ (where $\mathbb{F}$ is typically $\mathbb{R}$ or $\mathbb{C}$) such that for every $x\in\mathcal{X}$, the evaluation functional $\delta_x:\mathcal{H}\to\mathbb{F}$, defined by $\delta_x(f)=f(x)$, is a continuous linear functional on $\mathcal{H}$.

By the Riesz Representation Theorem, for each $x\in\mathcal{X}$, the continuity of $\delta_x$ guarantees the existence of a unique element $K_x\in\mathcal{H}$ such that $f(x)=\delta_x(f)=\langle f,K_x\rangle_{\mathcal{H}}$ for all $f\in\mathcal{H}$. The function $K_x$ is called the "representer of evaluation" at $x$.

The reproducing kernel $K:\mathcal{X}\times\mathcal{X}\to\mathbb{F}$ is then defined by $K(x,y)=\langle K_y,K_x\rangle_{\mathcal{H}}$. Substituting $f=K_y$ into the representer property gives $K_y(x)=\langle K_y,K_x\rangle_{\mathcal{H}}$, which means $K_x$ is the function $K(\cdot,x)$. Therefore, $K_x$ is the function $K(\cdot,x)$.

The fundamental property, known as the reproducing property, is:

$$
f(x) = \langle f, K(\cdot, x) \rangle_{\mathcal{H}} \quad \forall f \in \mathcal{H}, \forall x \in \mathcal{X}
$$

This property directly links point evaluation of any function in the space to an inner product with a specific kernel function associated with that point.

A function $K:\mathcal{X}\times\mathcal{X}\to\mathbb{F}$ can serve as a reproducing kernel if and only if it is symmetric (i.e., $K(x,y)=\overline{K(y,x)}$; for real-valued functions, $K(x,y)=K(y,x)$) and positive definite (or non-negative definite). Positive definiteness means that for any finite set of points $\{x_1,\ldots,x_n\}\subset\mathcal{X}$ and any coefficients $\{c_1,\ldots,c_n\}\subset\mathbb{F}$, the sum $\sum_{i=1}^n \sum_{j=1}^n c_i\overline{c_j}K(x_i,x_j)\geq 0$. The Moore-Aronszajn theorem establishes a one-to-one correspondence between such positive definite kernels and RKHSs. The choice of kernel $K$ thus uniquely determines the Hilbert space $\mathcal{H}$ and its associated norm, which implicitly defines a notion of smoothness or complexity for functions within that space.

### 3.3. The Representer Theorem

The Representer Theorem is a cornerstone result that makes RKHS particularly powerful in machine learning, especially for regularization-based methods. It addresses problems of the form: find a function $f$ within an RKHS $\mathcal{H}$ that minimizes a combination of an empirical loss term (measuring fit to data) and a regularization term (penalizing complexity).

Specifically, consider the problem:

$$
\min_{f \in \mathcal{H}} \left( \frac{1}{N} \sum_{n=1}^N L(y_n, f(x_n)) + \lambda \|f\|_{\mathcal{H}}^2 \right)
$$

where $\{(x_n,y_n)\}_{n=1}^N$ is the training data, $L$ is a loss function (e.g., squared error $L(y,\hat{y})=(y-\hat{y})^2$), $\lambda>0$ is the regularization parameter controlling the trade-off between data fit and smoothness, and $\|f\|_{\mathcal{H}}^2=\langle f,f\rangle_{\mathcal{H}}$ is the squared norm in the RKHS, acting as a complexity penalty.

The Representer Theorem states that, under general conditions on the loss function $L$, the solution $f^*$ to this minimization problem lies within the finite-dimensional subspace spanned by the kernel functions evaluated at the training data points. That is, the optimal solution can always be expressed in the form:

$$
f^*(x) = \sum_{i=1}^N \alpha_i K(x,x_i)
$$

for some coefficients $\alpha_1,\ldots,\alpha_N$.

This result is remarkable because it transforms an optimization problem over an potentially infinite-dimensional function space $\mathcal{H}$ into a finite-dimensional problem of finding the optimal coefficients $\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_N)^T$. For the specific case of squared loss, substituting the form of $f^*$ into the objective function and minimizing with respect to $\boldsymbol{\alpha}$ leads to a system of linear equations:

$$
(\mathbf{K} + N\lambda\mathbf{I})\boldsymbol{\alpha} = \mathbf{y}
$$

where $\mathbf{K}$ is the $N\times N$ Gram matrix with entries $K_{ij}=K(x_i,x_j)$, $\mathbf{I}$ is the identity matrix, and $\mathbf{y}=(y_1,\ldots,y_N)^T$. This system can be solved efficiently using standard linear algebra techniques.

The regularization term $\lambda\|f\|_{\mathcal{H}}^2$ plays a crucial role. It prevents overfitting by penalizing functions that are too complex according to the norm defined by the kernel. The choice of kernel $K$ implicitly defines the notion of "smoothness" or "complexity" being penalized; different kernels lead to different function spaces and thus different types of solutions.

### 3.4. Applications

The framework of RKHS and the Representer Theorem underpins many powerful machine learning algorithms, collectively known as kernel methods. Prominent examples include Support Vector Machines (SVMs), Gaussian Processes (GPs) for regression and classification, kernel Principal Component Analysis (kPCA), and many others. These methods allow algorithms originally designed for linear models in feature space to be applied implicitly to potentially infinite-dimensional feature spaces defined by the kernel, enabling the modeling of complex, non-linear relationships in data.

## 4. Sobolev Spaces: Quantifying Smoothness

While RKHS provide a framework linked to point evaluation and kernel methods, Sobolev spaces offer a different, powerful way to analyze and quantify the smoothness of functions, particularly relevant in the study of partial differential equations (PDEs) and approximation theory.

### 4.1. Introduction and Motivation

In many scientific and engineering applications, particularly those involving PDEs, it is essential to work with functions that possess certain degrees of differentiability or smoothness. Classical function spaces, like the space $C^k(\Omega)$ of $k$-times continuously differentiable functions on a domain $\Omega$, are often insufficient. A key limitation is their lack of completeness with respect to norms involving derivatives. For instance, a sequence of smooth functions might converge (in an integral sense) to a function that is not differentiable, or whose derivatives are not well-behaved.

Sobolev spaces, denoted $W^{k,p}(\Omega)$ or $H^k(\Omega)$ for the special case $p=2$, address this by incorporating derivatives in a generalized sense (weak derivatives) and providing a complete Banach (or Hilbert) space structure. This makes them the natural setting for the modern theory of PDEs and variational problems.

### 4.2. Background: Lebesgue and Classical Spaces

Before defining Sobolev spaces formally, it's useful to recall the spaces they build upon:

- **Classical Smooth Functions $C^k(\Omega)$**: The space of functions whose derivatives up to order $k$ exist and are continuous on the domain $\Omega$. A potential norm involves summing the $L^p$ norms of all derivatives up to order $k$, i.e., $(\sum_{|\alpha|\leq k} \int_\Omega |D^\alpha f(x)|^p dx)^{1/p}$. However, $C^k(\Omega)$ is generally not complete under such norms.

- **Lebesgue Spaces $L^p(\Omega)$**: The space of measurable functions $f$ for which $|f|^p$ is Lebesgue integrable over $\Omega$. Equipped with the norm $\|f\|_{L^p}=(\int_\Omega |f(x)|^p dx)^{1/p}$ (for $1\leq p<\infty$), $L^p(\Omega)$ is a Banach space. For $p=2$, $L^2(\Omega)$ is a Hilbert space with the inner product $\langle f,g\rangle_{L^2}=\int_\Omega f(x)\overline{g(x)}dx$. These spaces handle integrability well but don't inherently incorporate differentiability information.

### 4.3. Weak Derivatives

The concept of weak derivatives is central to Sobolev spaces. It extends the notion of differentiation to functions that may not be smooth enough for classical derivatives to exist everywhere.

The definition is motivated by the integration by parts formula. For smooth functions $f$ and $\phi$, where $\phi$ has compact support within the domain $\Omega$ (denoted $\phi\in C_c^\infty(\Omega)$, the space of "test functions"), integration by parts yields:

$$
\int_\Omega f(x) (D^\alpha \phi)(x) dx = (-1)^{|\alpha|} \int_\Omega (D^\alpha f)(x) \phi(x) dx
$$

where $D^\alpha$ represents a partial derivative of multi-index order $\alpha=(\alpha_1,\ldots,\alpha_d)$ and $|\alpha|=\sum \alpha_i$.

A function $g\in L_{\text{loc}}^1(\Omega)$ (locally integrable) is called the $\alpha$-th weak derivative of $f\in L_{\text{loc}}^1(\Omega)$, denoted $g=D^\alpha f$, if the following integral identity holds for all test functions $\phi\in C_c^\infty(\Omega)$:

$$
\int_\Omega f(x) D^\alpha \phi(x) dx = (-1)^{|\alpha|} \int_\Omega g(x) \phi(x) dx
$$

This definition effectively transfers the differentiation operation onto the smooth test function $\phi$. If a function $f$ has a classical derivative $D^\alpha f$, then this classical derivative is also its weak derivative. However, functions that are not classically differentiable can still possess weak derivatives.

For example, consider the absolute value function $f(x)=|x|$ on $\Omega=(-1,1)$. It's not differentiable at $x=0$. Its weak derivative is the sign function, $g(x)=\text{sgn}(x)$, which is $-1$ for $x<0$ and $+1$ for $x>0$. This can be verified using the definition: $\int_{-1}^1 |x|\phi'(x)dx = -\int_{-1}^0 (-x)\phi'(x)dx - \int_0^1 x\phi'(x)dx$. Integrating by parts yields $\int_{-1}^0 \phi(x)dx - \int_0^1 \phi(x)dx = -\int_{-1}^1 \text{sgn}(x)\phi(x)dx$, confirming $g(x)=\text{sgn}(x)$ is the weak derivative.

### 4.4. Sobolev Spaces $W^{k,p}(\Omega)$ and $H^k(\Omega)$

With the concept of weak derivatives established, Sobolev spaces can be defined.

The Sobolev space $W^{k,p}(\Omega)$, for an integer $k\geq 0$ and $1\leq p\leq\infty$, consists of all functions $f\in L^p(\Omega)$ such that for every multi-index $\alpha$ with $|\alpha|\leq k$, the weak derivative $D^\alpha f$ exists and belongs to $L^p(\Omega)$.

These spaces are equipped with the Sobolev norm:

$$
\|f\|_{W^{k,p}(\Omega)} = \left( \sum_{|\alpha| \leq k} \|D^\alpha f\|_{L^p(\Omega)}^p \right)^{1/p}
$$

for $1\leq p<\infty$, and $\|f\|_{W^{k,\infty}(\Omega)}=\max_{|\alpha|\leq k} \|D^\alpha f\|_{L^\infty(\Omega)}$ for $p=\infty$.

A crucial property is that $W^{k,p}(\Omega)$ is a Banach space (i.e., it is complete) under this norm. Intuitively, these spaces contain functions whose derivatives up to order $k$, in a generalized (weak) sense, have finite $L^p$ norms. They can be viewed as the completion of the space of smooth functions (like $C^k(\Omega)$ or even $C^\infty(\Omega)$) under the Sobolev norm.

The most commonly used Sobolev spaces, especially in applications involving variational methods and PDEs, are the Hilbert Sobolev spaces $H^k(\Omega)$, which correspond to the case $p=2$:

$$
H^k(\Omega) = W^{k,2}(\Omega)
$$

These are Hilbert spaces equipped with the inner product:

$$
\langle f, g \rangle_{H^k(\Omega)} = \sum_{|\alpha| \le k} \langle D^\alpha f, D^\alpha g \rangle_{L^2(\Omega)} = \sum_{|\alpha| \le k} \int_{\Omega} D^\alpha f(x) \overline{D^\alpha g(x)} dx
$$

The associated norm is $\|f\|_{H^k(\Omega)}=\sqrt{\langle f,f\rangle_{H^k(\Omega)}}$. These spaces combine the Hilbert space structure (completeness, inner product) with control over the function's derivatives up to order $k$ in an average ($L^2$) sense.

## 5. Function Approximation and the Curse of Dimensionality

Sobolev spaces provide a natural framework for analyzing the approximation of functions, particularly relevant in machine learning where the goal is often to estimate an unknown function from data. The smoothness of a function, as measured by its Sobolev norm, dictates how well it can be approximated by simpler functions, such as polynomials or trigonometric series.

### 5.1. Machine Learning Context

In supervised learning, we often assume the data $(x_i,y_i)$ is generated via $y_i = f(x_i) + \epsilon_i$, where $f$ is an unknown target function and $\epsilon_i$ is noise. The goal is to find an estimator $\hat{f}$ based on the data that is close to $f$. Approximation theory provides tools to understand how well functions from a certain class (e.g., functions with a bounded Sobolev norm) can be approximated by functions from a simpler, finite-dimensional class (e.g., polynomials or functions constructed from basis elements). The rate at which the approximation error decreases as the complexity of the approximating class increases is crucial for understanding the performance of learning algorithms.

### 5.2. Fourier Series and Sobolev Spaces

A powerful tool for analyzing functions, especially periodic ones or those defined on bounded intervals (which can often be extended periodically), is the Fourier series. For functions in $L^2([-\pi,\pi])$, the Fourier series representation is given by $f(x)=\sum_{k=-\infty}^\infty c_k e^{ikx}$, where $c_k=\frac{1}{2\pi}\int_{-\pi}^\pi f(x)e^{-ikx}dx$ are the Fourier coefficients.

Parseval's identity relates the $L^2$ norm of the function to its coefficients: $\|f\|_{L^2}^2=2\pi \sum_{k=-\infty}^\infty |c_k|^2$. Crucially, the smoothness of the function $f$ is directly reflected in the decay rate of its Fourier coefficients $|c_k|$ as $|k|\to\infty$. This relationship can be quantified using Sobolev norms. For integer smoothness $s\geq 0$, the squared norm in the Sobolev space $H^s([-\pi,\pi])$ (equivalent to $W^{s,2}$ in this context) is related to the Fourier coefficients by:

$$
\|f\|_{H^s}^2 \approx \sum_{k=-\infty}^\infty (1 + k^2)^s |c_k|^2
$$

A function is smoother (belongs to $H^s$ with larger $s$) if its Fourier coefficients decay faster.

### 5.3. Approximation Error and Smoothness

Consider approximating a function $f\in L^2([-\pi,\pi])$ by a trigonometric polynomial of degree $n$, which is the partial sum of its Fourier series: $f_n(x)=\sum_{k=-n}^n c_k e^{ikx}$. This $f_n$ is known to be the best approximation to $f$ in the $L^2$ norm among all trigonometric polynomials of degree at most $n$ (elements of the subspace $\mathcal{H}_n=\text{span}\{e^{ikx}:|k|\leq n\}$).

The squared $L^2$ approximation error is given by:

$$
\epsilon_n[f]^2 = \|f - f_n\|_{L^2}^2 = \sum_{|k|=n+1}^\infty |c_k|^2
$$

If the function $f$ belongs to the Sobolev space $H^s([-\pi,\pi])$ (meaning it has $s$ square-integrable weak derivatives), we can bound this error. Using the relationship between the Sobolev norm and Fourier coefficients, we have $\sum_{k=-\infty}^\infty k^{2s}|c_k|^2 \lesssim \|f\|_{H^s}^2$. The tail sum can then be bounded:

$$
\epsilon_n[f]^2 = \sum_{|k|=n+1}^\infty |c_k|^2 = \sum_{|k|=n+1}^\infty \frac{k^{2s}|c_k|^2}{k^{2s}} \le \frac{1}{(n+1)^{2s}} \sum_{|k|=n+1}^\infty k^{2s} |c_k|^2 \lesssim n^{-2s} \|f\|_{H^s}^2
$$

Thus, the $L^2$ approximation error decreases as $\epsilon_n[f] \propto n^{-s}$. This demonstrates a fundamental principle: smoother functions (larger $s$) can be approximated more accurately (faster decay rate) by finite-dimensional representations.

### 5.4. The Curse of Dimensionality

The relationship between smoothness and approximation rate becomes significantly worse in higher dimensions. Consider functions defined on a $d$-dimensional hypercube, say $[-\pi,\pi]^d$. A function $f$ can be represented by a multi-dimensional Fourier series $f(\mathbf{x})=\sum_{\mathbf{k}\in\mathbb{Z}^d} c_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{x}}$. The Sobolev norm $H^s$ involves sums of squared $L^2$ norms of all partial derivatives up to order $s$. The corresponding relationship between the norm and coefficients becomes $\|f\|_{H^s}^2 \approx \sum_{\mathbf{k}\in\mathbb{Z}^d} (1 + \|\mathbf{k}\|^2)^s |c_{\mathbf{k}}|^2$.

If we approximate $f$ using trigonometric polynomials of degree $n$ in each dimension (a total of approximately $(2n+1)^d \approx (2n)^d$ basis functions), the $L^2$ approximation error $\epsilon_n[f]$ can be shown to decay as:

$$
\epsilon_n[f] \lesssim (2n)^{-s/d} \|f\|_{H^s}
$$

The convergence rate is now $O(n^{-s/d})$. This rate deteriorates rapidly as the dimension $d$ increases. To achieve a desired accuracy $\epsilon$, the number of basis functions (or parameters) required, $N \approx (2n)^d$, scales as $N \propto \epsilon^{-d/s}$. This exponential dependence on dimensionality is known as the **curse of dimensionality**. It implies that for high-dimensional problems, achieving a good approximation requires an infeasibly large number of basis functions or, equivalently, data points, unless the function possesses additional structure (like sparsity or low-dimensional embeddings) that can be exploited.

### 5.5. Implications

The connection between smoothness (Sobolev norms), approximation rates, and the curse of dimensionality has profound implications for machine learning. It highlights why high-dimensional function estimation is inherently difficult. Methods like kernel machines and deep neural networks can be viewed as attempts to implicitly or explicitly adapt to underlying low-dimensional structures or specific smoothness properties of the target function to mitigate the curse of dimensionality. Understanding these trade-offs between function complexity (smoothness), available data (sample size, related to $N$), and dimensionality ($d$) is crucial for designing and analyzing learning algorithms.

## 6. Measure Theory in Infinite Dimensions and Bayesian Inference

Extending probability and integration theory from finite-dimensional Euclidean spaces ($\mathbb{R}^n$) to infinite-dimensional spaces, such as Hilbert spaces of functions, presents significant challenges. This extension is crucial for areas like Bayesian nonparametrics and the study of stochastic processes, where probability distributions are defined over function spaces.

### 6.1. Foundations of Measure Theory Recap

Before tackling infinite dimensions, it's useful to recall key concepts from standard measure theory:

- **$\sigma$-Algebra ($\mathcal{F}$)**: A collection of subsets of a space $X$ that includes the empty set, is closed under complementation, and is closed under countable unions. It defines the sets considered "measurable".
- **Topology ($\tau$)**: A collection of open sets defining nearness and continuity.
- **Borel $\sigma$-Algebra ($\mathcal{B}(X)$)**: The smallest $\sigma$-algebra containing all open sets defined by the topology $\tau$. It represents the standard collection of measurable sets in topological spaces like $\mathbb{R}^n$.
- **Measure ($\mu$)**: A function $\mu:\mathcal{F}\to[0,\infty]$ assigning a non-negative value (like length, area, volume, or probability) to each measurable set, satisfying $\mu(\emptyset)=0$ and countable additivity: $\mu(\cup_{n=1}^\infty A_n)=\sum_{n=1}^\infty \mu(A_n)$ for disjoint sets $A_n\in\mathcal{F}$.
- **Outer Measure ($\mu^*$)**: A function defined on all subsets, often used as a stepping stone to constructing a measure. It satisfies monotonicity and countable sub-additivity.
- **Carathéodory Measurability**: A set $A$ is measurable with respect to an outer measure $\mu^*$ if for every set $E$, $\mu^*(E)=\mu^*(E\cap A)+\mu^*(E\setminus A)$.
- **Lebesgue Measure on $\mathbb{R}^n$**: Constructed using an outer measure based on the volume of covering rectangles, then restricted to the Carathéodory measurable sets (which include the Borel sets). It is the standard measure for volume in $\mathbb{R}^n$ and possesses crucial properties like translation invariance: $\mu(A+\mathbf{x})=\mu(A)$ for any measurable set $A$ and vector $\mathbf{x}\in\mathbb{R}^n$.

### 6.2. The Challenge of Infinite Dimensions: No Lebesgue Measure

A fundamental difficulty arises when attempting to extend the concept of Lebesgue measure to infinite-dimensional Hilbert spaces (or Banach spaces). It turns out that no non-trivial, $\sigma$-finite, translation-invariant measure exists on an infinite-dimensional separable Hilbert space.

The standard proof argument proceeds by contradiction:
1. Assume such a measure $\mu$ exists. Let $\mathcal{H}$ be an infinite-dimensional separable Hilbert space.
2. Consider a countably infinite orthonormal set $\{e_n\}_{n=1}^\infty$ in $\mathcal{H}$.
3. Let $B_r=B(\mathbf{0},r)$ be the open ball of radius $r$ centered at the origin. Assume, without loss of generality, that $0 < \mu(B_{1/4}) < \infty$. (If $\mu(B_{1/4})=0$, then $\mu(\mathcal{H})=0$ due to $\sigma$-finiteness, making it trivial. If $\mu(B_{1/4})=\infty$, consider smaller balls).
4. Consider the sequence of balls $B_n=B(e_n,1/4)$. These balls are disjoint because if $\mathbf{x}\in B_n\cap B_m$ for $n\neq m$, then $\|e_n - e_m\| \leq \|\mathbf{x}-e_n\| + \|\mathbf{x}-e_m\| < 1/4 + 1/4 = 1/2$. However, for an orthonormal set, $\|e_n - e_m\|^2 = \langle e_n - e_m, e_n - e_m \rangle = \|e_n\|^2 + \|e_m\|^2 = 1 + 1 = 2$, so $\|e_n - e_m\|=\sqrt{2}$. This contradiction shows the balls must be disjoint.
5. If $\mu$ is translation-invariant, then $\mu(B_n)=\mu(B(e_n,1/4))=\mu(B(\mathbf{0},1/4))=c>0$ for all $n$.
6. All these disjoint balls $B_n$ are contained within a larger ball, for example, $B(\mathbf{0},2)$, since $\|e_n\|=1$ and for $\mathbf{x}\in B_n$, $\|\mathbf{x}\| \leq \|\mathbf{x}-e_n\| + \|e_n\| < 1/4 + 1 = 5/4 < 2$.
7. By countable additivity, $\mu(B(\mathbf{0},2)) \geq \mu(\cup_{n=1}^\infty B_n) = \sum_{n=1}^\infty \mu(B_n) = \sum_{n=1}^\infty c$. Since $c>0$, this sum diverges to infinity.
8. This contradicts the assumption that $\mu$ is $\sigma$-finite (meaning $\mathcal{H}$ can be covered by a countable union of sets with finite measure, which implies finite-radius balls must have finite measure).

Therefore, no measure with the desirable properties of Lebesgue measure (translation invariance and assigning finite measure to bounded sets) exists in infinite dimensions. This has significant implications, notably making it impossible to define probability density functions with respect to a standard background measure in the same way as in $\mathbb{R}^n$.

### 6.3. Gaussian Measures as a Replacement

Despite the non-existence of a Lebesgue measure, probability measures can still be defined on infinite-dimensional spaces. Gaussian measures are particularly important and widely used.

One way to construct a Gaussian measure on an infinite-dimensional Hilbert space $\mathcal{H}$ (assumed separable with orthonormal basis $\{e_k\}$) is via an infinite product measure. Consider the standard Gaussian measure $\gamma_1$ on $\mathbb{R}$, defined by 

$$d\gamma_1(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}dx$$

 We can attempt to define a product measure $\mu=\bigotimes_{k=1}^\infty \gamma_k$, where each $\gamma_k$ is a copy of $\gamma_1$ associated with the coordinate $x_k$ in the expansion $\mathbf{x}=\sum x_k e_k$. However, this standard Gaussian measure is concentrated on sequences for which $\sum x_k^2$ diverges almost surely, meaning it assigns measure 0 to the Hilbert space $\ell^2$ itself.

To construct a measure supported on a Hilbert space, one typically uses a weighted product measure. Consider a sequence of positive weights $\{a_k\}$ such that $\sum_{k=1}^\infty a_k < \infty$. Define the weighted Hilbert space

$$\ell^2_{\mathbf{a}}=\{\mathbf{x}=(x_k)\in\mathbb{R}^\mathbb{N} : \sum_{k=1}^\infty a_k x_k^2 < \infty\},$$

with inner product $\langle \mathbf{x},\mathbf{y} \rangle_{\mathbf{a}} = \sum_{k=1}^\infty a_k x_k y_k$. Now, consider the product measure $\mu = \bigotimes_{k=1}^\infty \mu_k$, where $\mu_k$ is the Gaussian measure $N(0,1/a_k)$ on the $k$-th coordinate (variance $1/a_k$). This measure $\mu$ can be shown to be concentrated on $\ell^2_{\mathbf{a}}$, meaning $\mu(\ell^2_{\mathbf{a}})=1$. This construction provides a way to define Gaussian measures on specific infinite-dimensional Hilbert spaces.

### 6.4. Gaussian Measures on Hilbert Spaces ($\mathcal{H}$)

More generally, a Gaussian measure $\mu$ on a Hilbert space $\mathcal{H}$ is characterized by its mean element $\mathbf{m}\in\mathcal{H}$ and its covariance operator $C:\mathcal{H}\to\mathcal{H}$. The covariance operator $C$ is a linear, bounded, self-adjoint, positive semi-definite operator. For $\mu$ to be a well-defined Borel probability measure on an infinite-dimensional Hilbert space $\mathcal{H}$, the covariance operator $C$ must be **trace-class**. This means that for any (and hence all) orthonormal basis $\{e_n\}$ of $\mathcal{H}$, the sum of the eigenvalues (or equivalently, the sum of the diagonal elements $\langle e_n, C e_n \rangle$) must be finite:

$$\text{Tr}(C)=\sum_{n=1}^\infty \langle e_n, C e_n \rangle < \infty$$
This condition ensures that the measure does not "spread out too much" in infinitely many directions.

The covariance operator is intrinsically linked to the covariance function $c(x,y)$ often used in Gaussian Processes (GPs). If $u(x)$ is a random function (a realization from the Gaussian measure) defined on some domain $\mathcal{X}$, its covariance function is 

$$c(x,y)=\mathbb{E}[u(x)u(y)] \quad \text{(assuming a zero mean process)}$$

The covariance operator $C$ acts on functions $g\in\mathcal{H}$ as 

$$(Cg)(x)=\int_\mathcal{X} c(x,y)g(y)dy$$

The relationship 

$$\langle f, C g \rangle = \mathbb{E}[\langle f, u \rangle \langle u, g \rangle] = \iint f(x)c(x,y)g(y)dydx$$
connects the operator view with the function view common in GP literature.

### 6.5. Properties and Relationships

Gaussian measures on Hilbert spaces, denoted $N(\mathbf{m},C)$, serve as fundamental reference measures, much like the Lebesgue measure in finite dimensions, particularly in Bayesian inference. However, unlike Lebesgue measure, they are generally not translation-invariant. Instead, they exhibit quasi-invariance under certain shifts. A measure $\mu$ is quasi-invariant under a shift $\mathbf{h}\in\mathcal{H}$ if the translated measure $\mu_{\mathbf{h}}(A)=\mu(A-\mathbf{h})$ is equivalent to $\mu$, meaning they have the same null sets ($\mu(A)=0 \iff \mu_{\mathbf{h}}(A)=0$).

If two measures $\mu_1$ and $\mu_2$ are equivalent (mutually absolutely continuous), the Radon-Nikodym theorem guarantees the existence of a density function (Radon-Nikodym derivative) $\frac{d\mu_2}{d\mu_1}$ such that 

$$\mu_2(A)=\int_A \frac{d\mu_2}{d\mu_1}(\mathbf{x})d\mu_1(\mathbf{x})$$ 
for any measurable set $A$. For Gaussian measures $N(0,\sigma^2)$ and $N(m,\sigma^2)$ on $\mathbb{R}$, they are always equivalent, and the Radon-Nikodym derivative is 

$$\frac{dN(m,\sigma^2)}{dN(0,\sigma^2)}(x)=\exp\left(-\frac{m^2}{2\sigma^2} + \frac{mx}{\sigma^2}\right)$$

This concept extends, with modifications, to infinite dimensions.

### 6.6. Equivalence and Singularity: Feldman-Hajek Theorem

In infinite-dimensional Hilbert spaces, the relationship between two Gaussian measures $N(0,C)$ and $N(\mathbf{v},C)$ (same covariance, different means) is starkly different from the finite-dimensional case. The Feldman-Hajek Theorem states that these two measures are either equivalent (mutually absolutely continuous) or singular (supported on disjoint sets). There is no intermediate possibility.

### 6.7. Cameron-Martin Space ($\mathcal{H}_{\text{CM}}$)

The condition determining equivalence versus singularity is precisely characterized by the Cameron-Martin space (CMS) associated with the covariance operator $C$. The Cameron-Martin space, denoted $\mathcal{H}_{\text{CM}}$ or sometimes $\mathcal{H}(C)$, is defined as the image of the square root of the covariance operator, $C^{1/2}$:

$$
\mathcal{H}_{\text{CM}} = \text{Im}(C^{1/2}) = \{ C^{1/2} \mathbf{x} \mid \mathbf{x} \in \mathcal{H} \}
$$

Equipped with the inner product $\langle \mathbf{u}, \mathbf{v} \rangle_{\text{CM}} = \langle C^{-1/2} \mathbf{u}, C^{-1/2} \mathbf{v} \rangle_{\mathcal{H}}$ (defined properly on the range space), $\mathcal{H}_{\text{CM}}$ is itself a Hilbert space, typically much smaller (denser) than the original space $\mathcal{H}$.

The Feldman-Hajek theorem states that the translated Gaussian measure $N(\mathbf{v},C)$ is equivalent to the centered Gaussian measure $N(0,C)$ if and only if the shift vector $\mathbf{v}$ belongs to the Cameron-Martin space $\mathcal{H}_{\text{CM}}$. If $\mathbf{v}\notin\mathcal{H}_{\text{CM}}$, the measures $N(0,C)$ and $N(\mathbf{v},C)$ are singular.

Intuitively, the Cameron-Martin space represents the directions in which the Gaussian measure has "room to move". Shifts within $\mathcal{H}_{\text{CM}}$ are permissible without making the measures disjoint, while shifts outside $\mathcal{H}_{\text{CM}}$ move the measure onto a set that has zero measure under the original distribution. The operator $C^{1/2}$ acts as a smoothing operator, mapping $\mathcal{H}$ onto the typically smoother functions in $\mathcal{H}_{\text{CM}}$. For example, if $C$ corresponds to integrating against a smooth kernel, $\mathcal{H}_{\text{CM}}$ will often be a Sobolev space of higher regularity. The connection to RKHS is also profound: the RKHS associated with the covariance kernel $c(x,y)$ is precisely the Cameron-Martin space of the Gaussian measure $N(0,C)$.

### 6.8. Bayesian Inference in Hilbert Space

The lack of a Lebesgue-like reference measure in infinite dimensions complicates the standard formulation of Bayes' theorem, which relies on probability densities defined with respect to such a measure. However, Bayesian inference can be rigorously formulated using Radon-Nikodym derivatives.

Let $\mu_0$ be a prior probability measure on a function space (often a Hilbert space $\mathcal{H}$), typically chosen as a Gaussian measure $N(0,C)$ in Bayesian nonparametrics (e.g., Gaussian Process priors). Let $P(y|\mathbf{u})$ represent the likelihood of observing data $y$ given a function $\mathbf{u}\in\mathcal{H}$. The goal is to find the posterior measure $\mu_y$, which represents the updated belief about $\mathbf{u}$ after observing $y$.

Instead of using densities, Bayes' theorem is expressed in terms of the relationship between the posterior measure $\mu_y$ and the prior measure $\mu_0$. Assuming the likelihood defines a measure $\nu_y$ that is absolutely continuous with respect to $\mu_0$, the posterior measure $\mu_y$ is also absolutely continuous with respect to $\mu_0$, and its Radon-Nikodym derivative is proportional to the likelihood function:

$$
\frac{d\mu_y}{d\mu_0}(\mathbf{u}) \propto P(y|\mathbf{u})
$$

This formulation avoids the need for a base Lebesgue-type measure. The likelihood $P(y|\mathbf{u})$ essentially describes how to re-weight the prior measure $\mu_0$ to obtain the posterior measure $\mu_y$. The Feldman-Hajek theorem and the concept of the Cameron-Martin space become relevant when analyzing how the posterior measure (often another Gaussian measure in conjugate cases, like GP regression with Gaussian noise) relates to the prior, particularly concerning the shift in the mean function.

## Conclusion

This exploration, guided by the structure of Mark Girolami's lectures, has traversed fundamental concepts from functional analysis to their application in modern machine learning and statistics. Starting with the basic algebraic structure of vector spaces, we introduced norms and inner products, leading to the crucial concepts of Banach and Hilbert spaces, emphasizing the importance of completeness for analytical rigor. The geometric properties endowed by inner products, captured by inequalities like Cauchy-Schwarz and the Parallelogram Law, and the utility of orthonormal bases via Bessel's inequality and Parseval's identity, were highlighted.

We then examined specific function spaces vital for applications. Reproducing Kernel Hilbert Spaces (RKHS) provide a framework where point evaluation is well-behaved, enabling kernel methods through the powerful Representer Theorem, which bridges infinite-dimensional function spaces and finite-dimensional computation. Sobolev spaces offer a means to rigorously quantify function smoothness using weak derivatives, crucial for understanding approximation properties and PDE theory. The analysis of function approximation in Sobolev spaces revealed the connection between smoothness and convergence rates, starkly illustrating the "curse of dimensionality" inherent in high-dimensional problems.

Finally, the challenges of extending measure theory to infinite-dimensional Hilbert spaces were addressed. The absence of a direct analogue to Lebesgue measure necessitates alternative frameworks, with Gaussian measures playing a central role. The critical requirement for covariance operators to be trace-class, the concept of the Cameron-Martin space, and the Feldman-Hajek dichotomy governing the equivalence or singularity of shifted Gaussian measures were discussed. These concepts underpin the rigorous formulation of Bayesian inference in function spaces, particularly for Gaussian process models, using Radon-Nikodym derivatives to relate prior and posterior distributions.

Collectively, these topics form a mathematical bedrock for understanding and developing sophisticated models and algorithms capable of handling complex, often infinite-dimensional, data structures encountered in modern machine learning and statistical inference.