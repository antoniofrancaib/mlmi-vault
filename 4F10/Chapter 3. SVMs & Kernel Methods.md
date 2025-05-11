## Part I: Linear Support Vector Machines: Foundations and Formulation

### 1. Introduction to Binary Classification

#### 1.1. Problem Setup

**Dataset**: We are given a dataset $D=\{(\mathbf{x}_n, t_n)\}_{n=1}^N$. This represents a collection of $N$ observations.

**Input Space**: Each input $\mathbf{x}_n$ is a vector in a $d$-dimensional real space, $\mathbf{x}_n \in \mathbb{R}^d$. These are often called feature vectors, where each dimension represents a measured characteristic of the observation.

**Target Labels**: Each corresponding target $t_n$ is a scalar belonging to the set $\{-1, +1\}$. This signifies that each observation belongs to one of two distinct classes, conventionally labeled as negative ($-1$) and positive ($+1$). This framing defines the problem as binary classification.

![[decision-border.png]]

#### 1.2. Learning Objective

The goal is to learn a function, the classifier, denoted by $y: \mathbb{R}^d \rightarrow \mathbb{R}$. This function takes an input vector $\mathbf{x}$ and produces a real-valued output.

**Classification Rule**: The sign of the output $y(\mathbf{x})$ determines the predicted class label for $\mathbf{x}$. Specifically:

- If $y(\mathbf{x}) \geq 0$, we predict the class label $\hat{t} = +1$.
- If $y(\mathbf{x}) < 0$, we predict the class label $\hat{t} = -1$.

The magnitude $|y(\mathbf{x})|$ can often be interpreted as a measure of confidence in the prediction, although this is not strictly required by the classification rule itself.

#### 1.3. Decision Boundary

**Definition**: The decision boundary is the set of points in the input space $\mathbb{R}^d$ where the classifier function transitions between predicting one class and the other. Mathematically, it is defined by the level set where the classifier output is exactly zero:

$$\{\mathbf{x} \in \mathbb{R}^d \mid y(\mathbf{x}) = 0\}$$

This boundary implicitly partitions the input space $\mathbb{R}^d$ into two regions, one corresponding to the prediction $+1$ ($y(\mathbf{x}) > 0$) and the other to the prediction $-1$ ($y(\mathbf{x}) < 0$).

**Condition for Correct Classification**: A new input $\mathbf{x}'$ with true label $t'$ is correctly classified by $y(\mathbf{x})$ if the predicted label matches the true label. This occurs if and only if $t'$ and $y(\mathbf{x}')$ have the same sign. Mathematically, this condition is concisely expressed as:

$$t' \cdot y(\mathbf{x}') > 0$$

If $t' \cdot y(\mathbf{x}') \leq 0$, the point is misclassified (or lies exactly on the boundary if equal to zero, which is typically considered ambiguous or incorrect depending on convention).

#### 1.4. Core Intuition

The fundamental task is to find a function $y(\mathbf{x})$ such that its zero-level set (the decision boundary) effectively separates the data points belonging to the class $+1$ from those belonging to the class $-1$.

### 2. Linear Classifiers

#### 2.1. Functional Form

A linear classifier is a specific type of classifier where the function $y(\mathbf{x})$ is an affine function of the input $\mathbf{x}$.

**Equation**: It is expressed as:

$$y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

**Components**:

- $\mathbf{w} \in \mathbb{R}^d$: This is the weight vector. It determines the orientation of the decision boundary.
- $b \in \mathbb{R}$: This is the bias or intercept term. It determines the position of the decision boundary relative to the origin.

**Note**: If we augment the input vector $\mathbf{x}$ to $\tilde{\mathbf{x}} = (\mathbf{x}, 1)^T \in \mathbb{R}^{d+1}$ and the weight vector to $\tilde{\mathbf{w}} = (\mathbf{w}, b)^T \in \mathbb{R}^{d+1}$, the linear classifier can be written compactly as $y(\mathbf{x}) = \tilde{\mathbf{w}}^T \tilde{\mathbf{x}}$. However, keeping $\mathbf{w}$ and $b$ separate is often notationally and conceptually convenient, especially when discussing margins.

#### 2.2. Geometric Interpretation

**Decision Boundary**: The decision boundary is defined by $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = 0$. This equation defines a $(d-1)$-dimensional hyperplane in the $d$-dimensional input space $\mathbb{R}^d$.

**Role of $\mathbf{w}$**: The weight vector $\mathbf{w}$ is normal (orthogonal) to the decision hyperplane. To see this, consider two points $\mathbf{x}_1, \mathbf{x}_2$ lying on the hyperplane. Then $\mathbf{w}^T \mathbf{x}_1 + b = 0$ and $\mathbf{w}^T \mathbf{x}_2 + b = 0$. Subtracting these gives $\mathbf{w}^T (\mathbf{x}_1 - \mathbf{x}_2) = 0$. Since $(\mathbf{x}_1 - \mathbf{x}_2)$ is a vector lying within the hyperplane, this shows $\mathbf{w}$ is orthogonal to any vector within the hyperplane. The orientation of the hyperplane is solely determined by the direction of $\mathbf{w}$.

**Role of $b$**: The bias term $b$ controls the position of the hyperplane. Changing $b$ shifts the hyperplane parallel to itself without changing its orientation. The perpendicular distance from the origin to the hyperplane is $|b| / \|\mathbf{w}\|$. Specifically, the point on the hyperplane closest to the origin is given by $\mathbf{x}_p = -b \mathbf{w} / \|\mathbf{w}\|^2$.

#### 2.3. Linear Separability

**Definition**: A dataset $D = \{(\mathbf{x}_n, t_n)\}_{n=1}^N$ is said to be linearly separable if there exists at least one pair $(\mathbf{w}, b)$ defining a linear classifier $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ such that all training points are classified correctly. That is, $\exists (\mathbf{w}, b)$ such that:

$$t_n \cdot (\mathbf{w}^T \mathbf{x}_n + b) > 0 \quad \text{for all } n = 1, \ldots, N$$

![[linear-separability.png]]

**Challenge**: If a dataset is linearly separable, there are typically infinitely many such hyperplanes $(\mathbf{w}, b)$ that achieve zero classification error on the training set. For example, any small perturbation of a valid $(\mathbf{w}, b)$ that doesn't cause the hyperplane to cross any data point will also be a valid separating hyperplane. The central question then becomes: which of these separating hyperplanes is optimal, particularly in terms of its ability to generalize to unseen data?

### 3. The Maximum Margin Principle

#### 3.1. Concept of Margin

**Definition**: For a given separating hyperplane defined by $(\mathbf{w}, b)$, the margin is the minimum perpendicular distance from the hyperplane $y(\mathbf{x}) = 0$ to the closest data point(s) from either class.

Let $d(\mathbf{x}, H)$ denote the perpendicular distance of a point $\mathbf{x}$ to the hyperplane $H = \{\mathbf{z} \mid \mathbf{w}^T \mathbf{z} + b = 0\}$. This distance is given by $d(\mathbf{x}, H) = |\mathbf{w}^T \mathbf{x} + b| / \|\mathbf{w}\|$.

The margin $\rho$ is then defined as:

$$\rho = \min_{n=1,\ldots,N} d(\mathbf{x}_n, H) = \min_{n=1,\ldots,N} \frac{|\mathbf{w}^T \mathbf{x}_n + b|}{\|\mathbf{w}\|}$$

**Motivation**: The intuition, supported by statistical learning theory (e.g., VC dimension bounds), is that a classifier with a larger margin is likely to *generalize better* to unseen data. A larger margin implies a greater separation between the classes, making the classifier less sensitive to small changes in the input data or potential noise.

#### 3.2. Addressing Scale Invariance

**Observation**: Consider a linear classifier $y(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$. If we scale the parameters $(\mathbf{w}, b)$ by a positive constant $c > 0$, we get a new classifier $y'(\mathbf{x}) = (c\mathbf{w})^T \mathbf{x} + (cb) = c(\mathbf{w}^T \mathbf{x} + b)$.

The decision boundary defined by $y'(\mathbf{x}) = 0$ is identical to the one defined by $y(\mathbf{x}) = 0$. However, the magnitude of the output changes. 

Furthermore, the sign of $y'(\mathbf{x})$ is the same as the sign of $y(\mathbf{x})$ (since $c > 0$). Thus, the classification rule remains unchanged. Consequently, the value which defines the margin is not affected by this scaling:

$$\frac{|(c\mathbf{w})^T \mathbf{x}_n + cb|}{\|c\mathbf{w}\|} = \frac{|c(\mathbf{w}^T \mathbf{x}_n + b)|}{|c| \|\mathbf{w}\|} = \frac{|\mathbf{w}^T \mathbf{x}_n + b|}{\|\mathbf{w}\|}$$

The definition of the margin $\rho$ itself is *unique for a given hyperplane geometry*, but the parameters $(\mathbf{w}, b)$ representing that geometry are not unique due to scaling. The goal is to fix this scaling ambiguity.

#### 3.3. Canonical Hyperplane and Support Vectors

**Normalization Strategy**: To remove the scaling ambiguity in $(\mathbf{w}, b)$, we impose a specific normalization constraint. We demand that for the data point(s) closest to the hyperplane (which will ultimately define the margin), the output of the classifier $y(\mathbf{x})$ should have a magnitude of exactly 1.

Let $\mathbf{x}_{\text{closest}}$ be a point closest to the hyperplane $y(\mathbf{x}) = 0$. We rescale $(\mathbf{w}, b)$ such that $|\mathbf{w}^T \mathbf{x}_{\text{closest}} + b| = 1$. Since we require correct classification $t_n (\mathbf{w}^T \mathbf{x}_n + b) > 0$, this implies $t_n (\mathbf{w}^T \mathbf{x}_n + b) = 1$ for these closest points.

**Support Vectors**: The data points $\mathbf{x}_n$ for which this equality holds, i.e., $t_n (\mathbf{w}^T \mathbf{x}_n + b) = 1$, are called support vectors. These are the critical points that lie exactly on the margin boundaries.

For a positive support vector $\mathbf{x}_+$ ($t_n = +1$): $\mathbf{w}^T \mathbf{x}_+ + b = +1$. This defines the positive margin boundary hyperplane $H_+$.

For a negative support vector $\mathbf{x}_-$ ($t_n = -1$): $\mathbf{w}^T \mathbf{x}_- + b = -1$. This defines the negative margin boundary hyperplane $H_-$.

Under this normalization, all other data points must lie on or outside these margin boundaries: $t_n (\mathbf{w}^T \mathbf{x}_n + b) \geq 1$ for all $n$.

#### 3.4. Margin Calculation under Normalization

With the canonical normalization $t_n (\mathbf{w}^T \mathbf{x}_n + b) = 1$ for support vectors, let's calculate the margin. The margin is the perpendicular distance between the decision boundary $H_0: \mathbf{w}^T \mathbf{x} + b = 0$ and the margin boundary $H_+: \mathbf{w}^T \mathbf{x} + b = 1$ (or $H_-$).

Consider a point $\mathbf{x}_+$ on $H_+$. Its distance to $H_0$ is $\frac{|\mathbf{w}^T \mathbf{x}_+ + b|}{\|\mathbf{w}\|} = \frac{|1|}{\|\mathbf{w}\|} = \frac{1}{\|\mathbf{w}\|}$.

Similarly, for a point $\mathbf{x}_-$ on $H_-$, its distance to $H_0$ is $\frac{|\mathbf{w}^T \mathbf{x}_- + b|}{\|\mathbf{w}\|} = \frac{|-1|}{\|\mathbf{w}\|} = \frac{1}{\|\mathbf{w}\|}$.

The total width of the margin (distance between $H_+$ and $H_-$) is the sum of these distances, $2 / \|\mathbf{w}\|$. The margin $\rho$ (distance from decision boundary to closest point) is therefore $\rho = 1 / \|\mathbf{w}\|$.

**Alternative Derivation from Notes**: The margin can also be seen as half the perpendicular distance between the support hyperplanes $H_+$ and $H_-$. The vector between a positive support vector $\mathbf{x}_+$ and a negative support vector $\mathbf{x}_-$ is $(\mathbf{x}_+ - \mathbf{x}_-)$. The projection of this vector onto the normal vector $\mathbf{w}$ gives the distance between the hyperplanes along the normal direction.

Distance = $\frac{|\mathbf{w}^T (\mathbf{x}_+ - \mathbf{x}_-)|}{\|\mathbf{w}\|} = \frac{|(\mathbf{w}^T \mathbf{x}_+ + b) - (\mathbf{w}^T \mathbf{x}_- + b)|}{\|\mathbf{w}\|} = \frac{|(+1) - (-1)|}{\|\mathbf{w}\|} = \frac{2}{\|\mathbf{w}\|}$.

The margin (distance to the central hyperplane) is half of this: $\text{Margin} = \frac{1}{2} \times \frac{2}{\|\mathbf{w}\|} = \frac{1}{\|\mathbf{w}\|}$.

![[margin.png]]

#### 3.5. Optimization Goal: Choosing the Optimal Hyperplane

**Objective**: We seek the hyperplane $(\mathbf{w}, b)$ that maximizes the margin $\rho = 1 / \|\mathbf{w}\|$, subject to the constraint that all data points are correctly classified and lie outside or on the margin boundaries defined by the canonical normalization.

**Equivalent Objective**: Maximizing $1 / \|\mathbf{w}\|$ is equivalent to minimizing $\|\mathbf{w}\|$. For mathematical convenience (differentiability at zero, convexity), we typically minimize the squared norm $\|\mathbf{w}\|^2 = \mathbf{w}^T \mathbf{w}$, or equivalently, $\frac{1}{2} \|\mathbf{w}\|^2$. The factor $\frac{1}{2}$ simplifies the gradient calculation later.

**Key Insight**: The optimal hyperplane $(\mathbf{w}^*, b^*)$ is determined entirely by the support vectors. Points far away from the boundary do not influence its position, as long as they satisfy the constraint $t_n (\mathbf{w}^T \mathbf{x}_n + b) \geq 1$. This sparsity is a key property of SVMs.

### 4. Primal Formulation of the Hard-Margin SVM

#### 4.1. Formal Optimization Problem

Based on the goal of maximizing the margin $\frac{1}{\|\mathbf{w}\|}$ under the canonical normalization constraints, we formulate the *primal optimization problem* for the hard-margin SVM (assuming linear separability):

$$\min_{\mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}} \frac{1}{2} \|\mathbf{w}\|^2$$

subject to $t_n (\mathbf{w}^T \mathbf{x}_n + b) \geq 1$, for all $n = 1, \ldots, N$

This is a convex quadratic programming (QP) problem because the objective function $\frac{1}{2} \|\mathbf{w}\|^2$ is convex, and the constraints $1 - t_n (\mathbf{w}^T \mathbf{x}_n + b) \leq 0$ are linear (and thus convex) functions of $\mathbf{w}$ and $b$. This guarantees that any **local minimum found is also a global minimum**.

![[lagrange-optimization.png]]
#### 4.2. Background: Constrained Optimization and Lagrange Multipliers

**Generic Formulation**: Consider a general optimization problem with inequality constraints:

$$\max_z f(z)$$

subject to $g_j(z) \geq 0$, $j = 1, \ldots, M$

(Note: SVM is minimization with $\geq$ constraints, equivalent to maximizing $-f$ subject to $g \geq 0$, or minimizing $f$ subject to $-g \leq 0$. We adapt the standard KKT formulation).

**Lagrangian Function**: We introduce non-negative Lagrange multipliers $\lambda_j \geq 0$, one for each constraint, and form the *Lagrangian*:

$$L(z, \lambda) = f(z) + \sum_{j=1}^M \lambda_j g_j(z)$$

For a minimization problem $\min f(z)$ subject to $g_j(z) \geq 0$, the Lagrangian is often written as $L(z, \lambda) = f(z) - \sum_{j=1}^M \lambda_j g_j(z)$ with $\lambda_j \geq 0$. Let's stick to the notes' convention for SVM which uses this latter form (associated with KKT).

**Stationarity (Equality Constraints)**: For an equality constraint $h(z) = 0$, the condition at an optimal point $z^*$ is that the gradient of the objective function is parallel to the gradient of the constraint function: $\nabla_z f(z^*) = -\mu \nabla_z h(z^*)$ for some multiplier $\mu$. This means $\nabla_z (f(z^*) + \mu h(z^*)) = 0$.

**Inequality Constraints - Case Analysis**: Consider $\max f$ s.t. $g(z) \geq 0$ with multiplier $\lambda \geq 0$. At the optimum $z^*$:

- **Inactive Constraint**: If $g(z^*) > 0$, the constraint is not limiting the optimum. The optimum behaves as if the constraint wasn't there, occurring where $\nabla_z f(z^*) = 0$. To satisfy the general KKT conditions (specifically complementary slackness, see below), this requires $\lambda = 0$.
- **Active Constraint**: If $g(z^*) = 0$, the constraint is binding. The optimum lies on the boundary $g(z) = 0$. The condition becomes similar to the equality case: $\nabla_z f(z^*)$ must be proportional to $-\nabla_z g(z^*)$ (pointing "inwards" or tangentially for a max problem), i.e., $\nabla_z f(z^*) = -\lambda \nabla_z g(z^*)$ with $\lambda \geq 0$. Stationarity of the Lagrangian $\nabla_z L(z^*, \lambda) = \nabla_z f(z^*) + \lambda \nabla_z g(z^*) = 0$ holds if we use $L = f + \lambda g$. (Need to be careful with signs depending on max/min and $g \geq 0$ or $g \leq 0$). Let's align with the SVM standard formulation using $\min f$ s.t. $g_n \geq 0$.

#### 4.3. Karush-Kuhn-Tucker (KKT) Conditions

For the optimization problem $\min_z f(z)$ subject to $g_j(z) \geq 0$ for $j = 1, \ldots, M$ and $h_k(z) = 0$ for $k = 1, \ldots, L$, the KKT conditions provide necessary conditions for optimality at a point $z^*$ (and sufficient conditions if the problem is convex, like SVM). Let $\lambda_j \geq 0$ be multipliers for $g_j$ and $\mu_k$ for $h_k$. Define the Lagrangian $L(z, \lambda, \mu) = f(z) - \sum_j \lambda_j g_j(z) - \sum_k \mu_k h_k(z)$. The KKT conditions are:

1. **Stationarity**: $\nabla_z L(z^*, \lambda^*, \mu^*) = 0$.
2. **Primal Feasibility**: $g_j(z^*) \geq 0$ for all $j$, and $h_k(z^*) = 0$ for all $k$.
3. **Dual Feasibility**: $\lambda_j^* \geq 0$ for all $j$. (No sign constraint on $\mu_k$).
4. **Complementary Slackness**: $\lambda_j^* g_j(z^*) = 0$ for all $j$.

**Applying to SVM**: Our primal problem is $\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$ subject to $g_n(\mathbf{w}, b) = t_n (\mathbf{w}^T \mathbf{x}_n + b) - 1 \geq 0$. We introduce multipliers $a_n \geq 0$. The Lagrangian is $L(\mathbf{w}, b, a) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{n=1}^N a_n [t_n (\mathbf{w}^T \mathbf{x}_n + b) - 1]$. The KKT conditions at the optimal solution $(\mathbf{w}^*, b^*, a^*)$ are:

1. **Stationarity**: $\nabla_{\mathbf{w}} L = 0$ and $\frac{\partial L}{\partial b} = 0$ (derived in next section).
2. **Primal Feasibility**: $t_n (\mathbf{w}^{*T} \mathbf{x}_n + b^*) \geq 1$ for all $n$.
3. **Dual Feasibility**: $a_n^* \geq 0$ for all $n$.
4. **Complementary Slackness**: $a_n^* [t_n (\mathbf{w}^{*T} \mathbf{x}_n + b^*) - 1] = 0$ for all $n$.

**Implication of Complementary Slackness**: This condition is crucial. It states that for any data point $\mathbf{x}_n$:

- If $a_n^* > 0$, then it must be that $t_n (\mathbf{w}^{*T} \mathbf{x}_n + b^*) - 1 = 0$, meaning $\mathbf{x}_n$ is a support vector lying exactly on the margin boundary.
- If $t_n (\mathbf{w}^{*T} \mathbf{x}_n + b^*) - 1 > 0$ (the point lies strictly outside the margin boundary), then it must be that $a_n^* = 0$.

This mathematically confirms that only the support vectors can have non-zero Lagrange multipliers.

### 5. Dual Formulation of the Hard-Margin SVM

#### 5.1. Constructing the Lagrangian

As defined above for the KKT conditions, the Lagrangian for the primal SVM problem is:

$$L(\mathbf{w}, b, a) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{n=1}^N a_n \{t_n (\mathbf{w}^T \mathbf{x}_n + b) - 1\}$$

Here, $\mathbf{a} = (a_1, \ldots, a_N)^T$ is the vector of Lagrange multipliers, with the constraint $a_n \geq 0$. The dual problem involves maximizing this Lagrangian with respect to $\mathbf{a}$ after minimizing it with respect to the primal variables $\mathbf{w}$ and $b$.

#### 5.2. Deriving Stationarity Conditions

We find the minimum of $L$ with respect to $\mathbf{w}$ and $b$ by setting the respective gradients to zero.

**Gradient w.r.t. $\mathbf{w}$**:

$$\nabla_{\mathbf{w}} L = \nabla_{\mathbf{w}} \left( \frac{1}{2} \mathbf{w}^T \mathbf{w} \right) - \sum_{n=1}^N a_n t_n \nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{x}_n + b)$$

$$= \mathbf{w} - \sum_{n=1}^N a_n t_n \mathbf{x}_n$$

Setting $\nabla_{\mathbf{w}} L = 0$ yields the condition:

$$\mathbf{w} = \sum_{n=1}^N a_n t_n \mathbf{x}_n$$

This shows that the optimal weight vector $\mathbf{w}^*$ is a linear combination of the input vectors $\mathbf{x}_n$, weighted by their corresponding Lagrange multipliers $a_n$ and class labels $t_n$. Due to complementary slackness, only support vectors (with $a_n > 0$) contribute to this sum.

**Gradient w.r.t. $b$**:

$$\frac{\partial L}{\partial b} = -\sum_{n=1}^N a_n t_n = 0 \implies \sum_{n=1}^N a_n t_n = 0$$

This is a *constraint* that the optimal Lagrange multipliers $\mathbf{a}^*$ must satisfy.

#### 5.3. Substituting back into the Lagrangian to find the Dual Objective

We substitute the stationarity conditions ($\mathbf{w} = \sum_m a_m t_m \mathbf{x}_m$ and $\sum_n a_n t_n = 0$) back into the Lagrangian $L(\mathbf{w}, b, a)$ to eliminate $\mathbf{w}$ and $b$, obtaining the dual objective function $W(a)$ (sometimes denoted $L_D(a)$):

$$W(a) = L(\mathbf{w}(a), b(a), a) = \frac{1}{2} \|\mathbf{w(a)}\|^2 - \sum_{n=1}^N a_n \{t_n (\mathbf{w(a)}^T \mathbf{x}_n + b(a)) - 1\}$$

Let's examine the terms of $L$:

**Term 1**: $\frac{1}{2} \|\mathbf{w}\|^2 = \frac{1}{2} \left( \sum_{n=1}^N a_n t_n \mathbf{x}_n \right)^T \left( \sum_{m=1}^N a_m t_m \mathbf{x}_m \right) = \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (\mathbf{x}_n^T \mathbf{x}_m)$.

**Term 2 (part 1)**: $-\sum_{n=1}^N a_n t_n (\mathbf{w}^T \mathbf{x}_n) = -\sum_{n=1}^N a_n t_n \left( \sum_{m=1}^N a_m t_m \mathbf{x}_m^T \right) \mathbf{x}_n = -\sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (\mathbf{x}_m^T \mathbf{x}_n)$. Since $\mathbf{x}_m^T \mathbf{x}_n = \mathbf{x}_n^T \mathbf{x}_m$, this term is $-2 \times (\text{Term 1})$.

**Term 2 (part 2)**: $-\sum_{n=1}^N a_n t_n b = -b \sum_{n=1}^N a_n t_n$. Using the stationarity condition $\sum_n a_n t_n = 0$, this term becomes $0$.

**Term 3**: $-\sum_{n=1}^N a_n (-1) = \sum_{n=1}^N a_n$.

**Combining Terms**:

$$W(a) = \left( \frac{1}{2} \sum_{n,m} a_n a_m t_n t_m \mathbf{x}_n^T \mathbf{x}_m \right) - \left( \sum_{n,m} a_n a_m t_n t_m \mathbf{x}_n^T \mathbf{x}_m \right) + 0 + \left( \sum_{n=1}^N a_n \right)$$

$$W(a) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (\mathbf{x}_n^T \mathbf{x}_m)$$

#### 5.4. The Dual Optimization Problem

The dual problem is to maximize the dual objective $W(a)$ subject to the constraints derived from stationarity ($\sum a_n t_n = 0$) and the original multiplier constraints ($a_n \geq 0$):

$$\max_{\mathbf{a} \in \mathbb{R}^N} \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (\mathbf{x}_n^T \mathbf{x}_m)$$

subject to $\sum_{n=1}^N a_n t_n = 0$

$a_n \geq 0$, for all $n = 1, \ldots, N$

**Properties**: This is also a convex quadratic programming problem (maximizing a concave function subject to linear constraints). Strong duality holds for the SVM primal problem (since it's convex and satisfies Slater's condition if the data is linearly separable), meaning the optimal value of the dual problem equals the optimal value of the primal problem. Solving this dual problem yields the optimal Lagrange multipliers $\mathbf{a}^*$.

#### 5.5. Prediction using the Dual Solution

**Finding Optimal $a_n^*$**: Solve the dual QP problem to find the optimal vector $\mathbf{a}^* = (a_1^*, \ldots, a_N^*)$.

**Identifying Support Vectors**: The set of indices corresponding to support vectors is $S = \{n \mid a_n^* > 0\}$.

**Classifier Function**: The optimal weight vector is reconstructed using the stationarity condition: $\mathbf{w}^* = \sum_{n=1}^N a_n^* t_n \mathbf{x}_n = \sum_{n \in S} a_n^* t_n \mathbf{x}_n$. The prediction for a new input $\mathbf{x}$ is:

$$y(\mathbf{x}) = (\mathbf{w}^*)^T \mathbf{x} + b^* = \left( \sum_{n \in S} a_n^* t_n \mathbf{x}_n^T \right) \mathbf{x} + b^* = \sum_{n \in S} a_n^* t_n (\mathbf{x}_n^T \mathbf{x}) + b^*$$

Notice the prediction depends only on the inner products between the new input $\mathbf{x}$ and the support vectors $\mathbf{x}_n$. This is a crucial observation for kernel methods.

**Determining the Bias $b^*$**: The bias $b^*$ can be determined using the complementary slackness condition $a_n^* [t_n ((\mathbf{w}^*)^T \mathbf{x}_n + b^*) - 1] = 0$. For any support vector $\mathbf{x}_k$ (where $a_k^* > 0$), we must have $t_k ((\mathbf{w}^*)^T \mathbf{x}_k + b^*) = 1$. We can rearrange this to solve for $b^*$:

$$b^* = \frac{1}{t_k} - (\mathbf{w}^*)^T \mathbf{x}_k = t_k - \sum_{n \in S} a_n^* t_n (\mathbf{x}_n^T \mathbf{x}_k)$$

(using $1 / t_k = t_k$ since $t_k \in \{-1, +1\}$).

**Numerical Stability**: In practice, it's more robust to compute $b^*$ using this equation for all support vectors $\mathbf{x}_k$ for which $0 < a_k^* < C$ (in the soft-margin case, see below) and then average the resulting values of $b^*$.

### 6. Soft Margin SVM: Handling Non-Separable Data

#### 6.1. Motivation

The hard-margin SVM requires the data to be linearly separable. If the data is not linearly separable, the primal constraints $t_n(\mathbf{w}^T\mathbf{x}_n + b) \geq 1$ cannot all be satisfied simultaneously, and the feasible region is empty.

Even if the data is separable, a hyperplane that perfectly separates the training data might be overly sensitive to noise or outliers, leading to a very small margin and poor generalization (overfitting).

![[soft-margin-motivation.png]]

#### 6.2. Introducing Slack Variables

To allow for some misclassifications or points within the margin, we introduce slack variables $\xi_n \geq 0$, one for each data point $\mathbf{x}_n$.

**Modified Constraints:** The constraints are relaxed to:

$$t_n(\mathbf{w}^T\mathbf{x}_n + b) \geq 1 - \xi_n, \text{with } \xi_n \geq 0, \text{for } n=1,\ldots,N$$

**Interpretation of $\xi_n$:**

- If $\xi_n = 0$: The point $\mathbf{x}_n$ is correctly classified and lies on or outside the correct margin boundary ($t_n y(\mathbf{x}_n) \geq 1$). This is the same as the hard-margin case.
- If $0 < \xi_n \leq 1$: The point $\mathbf{x}_n$ is correctly classified ($t_n y(\mathbf{x}_n) > 0$) but lies inside the margin ($0 \leq t_n y(\mathbf{x}_n) < 1$). It violates the margin requirement but not the classification itself.
- If $\xi_n > 1$: The point $\mathbf{x}_n$ is misclassified ($t_n y(\mathbf{x}_n) \leq 0$).

The value $\xi_n$ quantifies the degree of violation of the original margin constraint $t_n y(\mathbf{x}_n) \geq 1$.

#### 6.3. Modified Primal Optimization Problem

We want to still maximize the margin (minimize $\|\mathbf{w}\|^2$) but also minimize the total amount of slack needed. This leads to the soft-margin SVM primal problem:

$$
\min_{\mathbf{w},b,\xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{n=1}^N \xi_n
$$

subject to:
$$t_n(\mathbf{w}^T\mathbf{x}_n + b) \geq 1 - \xi_n, n=1,\ldots,N$$
$$\xi_n \geq 0, n=1,\ldots,N$$

**Regularization Parameter $C$:** $C > 0$ is a hyperparameter that controls the trade-off between maximizing the margin (minimizing $\frac{1}{2}\|\mathbf{w}\|^2$) and minimizing the classification/margin errors (minimizing $\sum \xi_n$).

- **Large $C$:** Places a high penalty on slack variables ($\xi_n$). The optimization tries hard to minimize $\sum \xi_n$, forcing most $\xi_n$ towards zero. This behaves similarly to the hard-margin SVM, potentially leading to a smaller margin if needed to classify points correctly. Corresponds to low tolerance for errors/violations.
- **Small $C$:** Places a lower penalty on slack variables. The optimization prioritizes a larger margin (smaller $\|\mathbf{w}\|^2$) even if it means allowing more points to have $\xi_n > 0$ (i.e., be inside the margin or misclassified). Corresponds to high tolerance for errors/violations.

The term $\sum \xi_n$ acts as a measure of the total training error/margin violation. $C$ balances margin size against this error term.

#### 6.4. Dual Formulation of the Soft Margin SVM

We form the Lagrangian, now including multipliers $\mu_n \geq 0$ for the constraints $-\xi_n \leq 0$ (i.e., $\xi_n \geq 0$).

$$
L(\mathbf{w},b,\xi,\mathbf{a},\mu) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{n=1}^N \xi_n - \sum_{n=1}^N a_n[t_n(\mathbf{w}^T\mathbf{x}_n + b) - 1 + \xi_n] - \sum_{n=1}^N \mu_n \xi_n
$$

where $a_n \geq 0$ and $\mu_n \geq 0$.

**Stationarity Conditions:**

1. $\nabla_{\mathbf{w}}L = \mathbf{w} - \sum_n a_n t_n \mathbf{x}_n = 0 \implies \mathbf{w} = \sum_n a_n t_n \mathbf{x}_n$ (Same as before)
2. $\frac{\partial L}{\partial b} = -\sum_n a_n t_n = 0 \implies \sum_n a_n t_n = 0$ (Same as before)
3. $\frac{\partial L}{\partial \xi_n} = C - a_n - \mu_n = 0 \implies a_n = C - \mu_n$

**Dual Constraints:** From $a_n \geq 0$ and $\mu_n \geq 0$, the condition $a_n = C - \mu_n$ implies $a_n \leq C$. Combining with $a_n \geq 0$, we get the crucial box constraints: $0 \leq a_n \leq C$.

**Resulting Dual Problem:**

$$
\max_{\mathbf{a} \in \mathbb{R}^N} \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (\mathbf{x}_n^T \mathbf{x}_m)
$$

subject to:
$$\sum_{n=1}^N a_n t_n = 0$$
$$0 \leq a_n \leq C, \text{for all } n=1,\ldots,N$$

The only difference from the hard-margin dual is the upper bound $C$ on the Lagrange multipliers $a_n$.

#### 6.5. Prediction Function

The prediction function still takes the same form, using the optimal $\mathbf{a}^*$ found by solving the soft-margin dual:

$$
y(\mathbf{x}) = \sum_{n=1}^N a_n^* t_n (\mathbf{x}_n^T \mathbf{x}) + b^* = \sum_{n \in S} a_n^* t_n (\mathbf{x}_n^T \mathbf{x}) + b^*
$$

The set of support vectors $S = \{n | a_n^* > 0\}$ now includes:

1. Points exactly on the margin ($t_n y(\mathbf{x}_n) = 1, \xi_n = 0$). These typically have $0 < a_n^* < C$.
2. Points inside the margin ($0 \leq t_n y(\mathbf{x}_n) < 1, 0 < \xi_n \leq 1$). These typically have $a_n^* = C$.
3. Misclassified points ($t_n y(\mathbf{x}_n) < 0, \xi_n > 1$). These also typically have $a_n^* = C$.

Points with $a_n^* = 0$ are correctly classified and outside the margin ($t_n y(\mathbf{x}_n) > 1, \xi_n = 0$).

The bias $b^*$ is typically calculated using support vectors for which $0 < a_n^* < C$, as these points satisfy $t_n y(\mathbf{x}_n) = 1$ exactly.

#### 6.6. Practical Aspects

**Choosing $C$:** The value of $C$ is critical for model performance and generalization. It is typically chosen using techniques like *k-fold cross-validation* on the training data or by evaluating performance on a separate validation dataset. A common approach is to try values over a logarithmic scale (e.g., $10^{-3}, 10^{-2}, \ldots, 10^3$).

**Computational Cost:** Solving the dual QP problem is the main computational bottleneck. Standard QP solvers have complexities roughly between $O(N^2d + N^3)$ and $O(N^3)$ (depending on the algorithm and sparsity). Specialized algorithms like Sequential Minimal Optimization (SMO) are often used for SVMs and can be more efficient, particularly when $N$ is large, but the dependence on $N$ remains significant (often closer to $O(N^2)$ in practice). This makes training SVMs challenging for datasets with very large $N$ (e.g., $N > 10^5$ or $10^6$).

## Part II: Kernel Methods for Non-Linear Classification

### 8. Motivation for Non-Linearity

#### 8.1. Limitations of Linear Classifiers: 
Many real-world datasets are not linearly separable in their original input space $\mathbb{R}^d$. A linear decision boundary (hyperplane) may be fundamentally incapable of accurately separating the classes, leading to high training and test error regardless of the margin. Examples include data arranged in concentric circles or checkerboard patterns.

![[non-linear-separable-space.png]]

#### 8.2. The Core Idea: Feature Space Transformation

The key idea is to map the original input data $\mathbf{x}\in\mathbb{R}^d$ into a potentially much higher-dimensional space, called the feature space $\mathcal{H}$, using a non-linear mapping function $\phi$:

$$\phi:\mathbb{R}^d\rightarrow\mathcal{H}$$

where $\mathcal{H}$ is often a Hilbert space (a complete inner product space). For simplicity, we often consider $\mathcal{H}\cong\mathbb{R}^M$ where $M$ can be much larger than $d$, possibly infinite.

**Transformed Data**: The dataset becomes $D_\phi=\{(\phi(\mathbf{x}_n),t_n)\}_{n=1}^N$.

**Classifier in Feature Space**: We then learn a linear classifier in this new feature space $\mathcal{H}$. The classifier function takes the form:

$$y(\mathbf{x})=\mathbf{w}^T\phi(\mathbf{x})+b$$

Here, $\mathbf{w}$ is now a weight vector in the feature space $\mathcal{H}$. If $\mathcal{H}=\mathbb{R}^M$, then $\mathbf{w}\in\mathbb{R}^M$.

**Result**: Although the decision boundary $\mathbf{w}^T\phi(\mathbf{x})+b=0$ defines a hyperplane in the feature space $\mathcal{H}$, its pre-image in the original input space $\mathbb{R}^d$, $\{\mathbf{x}\in\mathbb{R}^d\mid\mathbf{w}^T\phi(\mathbf{x})+b=0\}$, is generally non-linear. This allows the method to learn complex, non-linear decision boundaries in the original space.

![[feature-map.png]]

### 9. Explicit Non-Linear Feature Mappings

#### 9.1. Example: Fixed Basis Functions

One way to construct $\phi$ is by using a set of $M$ fixed non-linear basis functions $\phi_m:\mathbb{R}^d\rightarrow\mathbb{R}$. The mapping is then $\phi(\mathbf{x})=(\phi_1(\mathbf{x}),\phi_2(\mathbf{x}),\ldots,\phi_M(\mathbf{x}))^T\in\mathbb{R}^M$.

**Gaussian Basis Function Example**: A common choice is Gaussian (or RBF) basis functions centered at locations $\mathbf{c}_m\in\mathbb{R}^d$:

$$\phi_m(\mathbf{x})=\exp\left\{-\frac{\|\mathbf{x}-\mathbf{c}_m\|^2}{2s^2}\right\}$$

**Interpretation**: Each $\phi_m(\mathbf{x})$ measures the similarity (proximity) of the input $\mathbf{x}$ to a specific center $\mathbf{c}_m$. The parameter $\gamma$ (related to $s$) controls the width or "reach" of the basis function. The feature space representation $\phi(\mathbf{x})$ encodes the similarity of $\mathbf{x}$ to a predefined set of prototypes $\{\mathbf{c}_m\}$.

![[gaussian-basis-function.png]]

#### 9.2. Implications

**Potential for Linear Separability**: By mapping data to a sufficiently high-dimensional feature space using appropriate non-linear functions, data that was non-linear-ly separable in $\mathbb{R}^d$ might become linearly separable (or nearly so) in $\mathcal{H}$. Cover's Theorem suggests this is likely if the feature space dimension is high enough.

**Computational Challenge**: Explicitly computing and storing the feature vectors $\phi(\mathbf{x}_n)$ can be computationally prohibitive if the dimension $M$ of the feature space is very large. For some desirable mappings (like the one implicitly defined by the Gaussian kernel), $M$ can even be infinite.

### 10. The Gram Matrix in Feature Space

#### 10.1. SVM Dual Formulation with Feature Maps

Recall the SVM dual objective function (for both hard and soft margins):

$$W(\mathbf{a})=\sum_{n=1}^N a_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N a_n a_m t_n t_m (\mathbf{x}_n^T\mathbf{x}_m)$$

If we apply the SVM algorithm in the feature space $\mathcal{H}$ using the mapped data $\phi(\mathbf{x}_n)$, the objective function becomes:

$$W(\mathbf{a})=\sum_{n=1}^N a_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N a_n a_m t_n t_m \langle\phi(\mathbf{x}_n),\phi(\mathbf{x}_m)\rangle$$

where $\langle\cdot,\cdot\rangle$ denotes the inner product in the feature space $\mathcal{H}$. If $\mathcal{H}=\mathbb{R}^M$, this is simply the dot product $\phi(\mathbf{x}_n)^T\phi(\mathbf{x}_m)$.

The constraints remain the same: $\sum a_n t_n=0$ and $0\leq a_n(\leq C)$.

The prediction function also involves inner products in $\mathcal{H}$:

$$y(\mathbf{x})=\sum_{n\in S} a_n^* t_n \langle\phi(\mathbf{x}_n),\phi(\mathbf{x})\rangle + b^* \quad \text{$\mathcal{S:}$ indices of support vectors}$$

#### 10.2. Definition of the Gram Matrix $K$

The crucial observation is that both the dual objective function and the prediction function depend on the mapped data $\phi(\mathbf{x}_n)$ *only through their inner products* $\langle\phi(\mathbf{x}_n),\phi(\mathbf{x}_m)\rangle$.

We define the Gram matrix (or kernel matrix) $K$ as the $N\times N$ matrix whose entries are these pairwise inner products:

$$K_{nm}=\langle\phi(\mathbf{x}_n),\phi(\mathbf{x}_m)\rangle$$

The dual objective can be written compactly using $K$:

$$W(\mathbf{a})=\mathbf{a}^T\mathbf{1}-\frac{1}{2}\mathbf{a}^T\text{diag}(\mathbf{t})K\text{diag}(\mathbf{t})\mathbf{a}$$

where $\mathbf{1}$ is a vector of ones, $\mathbf{t}=(t_1,...,t_N)^T$, and $\text{diag}(\mathbf{t})$ is the diagonal matrix with $\mathbf{t}$ on the diagonal.

#### 10.3. Computational Advantage

If we can compute each entry $K_{nm}=\langle\phi(\mathbf{x}_n),\phi(\mathbf{x}_m)\rangle$ efficiently, without explicitly computing the high-dimensional vectors $\phi(\mathbf{x}_n)$ and $\phi(\mathbf{x}_m)$, then we can solve the dual SVM problem and make predictions without ever working directly in the potentially huge feature space $\mathcal{H}$.

The computational cost of solving the dual QP now depends primarily on $N$ (the number of data points, as we need to handle the $N\times N$ matrix $K$), rather than the dimension $M$ of the feature space.

### 11. Kernel Functions and the Kernel Trick

#### 11.1. Motivation: 
To realize the computational advantage identified above, we need a way to compute the inner product $\langle\phi(\mathbf{x}_n),\phi(\mathbf{x}_m)\rangle$ directly from the original inputs $\mathbf{x}_n$ and $\mathbf{x}_m$.

#### 11.2. Kernel Function Definition: 
A kernel function (or simply kernel) $k:\mathbb{R}^d\times\mathbb{R}^d\rightarrow\mathbb{R}$ is a function that corresponds to an inner product in some feature space $\mathcal{H}$ associated with a mapping $\phi:\mathbb{R}^d\rightarrow\mathcal{H}$. That is:

$$k(\mathbf{x},\mathbf{x}')=\langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle$$

The function $k$ takes two points in the original input space and returns a scalar representing their similarity (inner product) in the feature space.

#### 11.3. The Kernel Trick

**Concept**: The kernel trick is the key idea of replacing every instance of the inner product $\langle\phi(\mathbf{x}_n),\phi(\mathbf{x}_m)\rangle$ in a machine learning algorithm (that relies only on such inner products) with the evaluation of a chosen kernel function $k(\mathbf{x}_n,\mathbf{x}_m)$.

**SVM Application**:

**Dual Objective**:

$$W(\mathbf{a})=\sum_{n=1}^N a_n - \frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N a_n a_m t_n t_m k(\mathbf{x}_n,\mathbf{x}_m)$$

**Prediction Function**:

$$y(\mathbf{x})=\sum_{n\in S} a_n^* t_n k(\mathbf{x}_n,\mathbf{x}) + b^*$$

**Advantages**:

- **Efficiency**: We avoid the potentially expensive or impossible task of explicitly computing $\phi(\mathbf{x})$. We only need to compute the $N\times N$ Gram matrix $K$ where $K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m)$.
- **Generality**: It allows us to implicitly use feature spaces that are extremely high-dimensional or even infinite-dimensional, provided we have a corresponding kernel function that can be computed efficiently.
- **Applicability**: The kernel trick can be applied to any algorithm whose formulation can be expressed solely in terms of inner products between data points.

### 12. Examples of Kernel Functions

#### 12.1. Linear Kernel:

$$k(\mathbf{x}_n,\mathbf{x}_m)=\mathbf{x}_n^T\mathbf{x}_m$$

This corresponds to the identity mapping $\phi(\mathbf{x})=\mathbf{x}$ (or $\mathcal{H}=\mathbb{R}^d$). Using this kernel recovers the original linear SVM discussed in Part I.

#### 12.2. Polynomial Kernel

**General Form**: $k(\mathbf{x}_n,\mathbf{x}_m)=(\gamma\mathbf{x}_n^T\mathbf{x}_m + c)^d$, where $d$ (degree) is typically a positive integer, $\gamma>0$, and $c\geq 0$ are hyperparameters. The notes used $\gamma=1,c=1$.

**Example Feature Map (d=2, c=1, $\gamma=1$, input $\mathbf{x}\in\mathbb{R}^2$):
Consider $k(\mathbf{x},\mathbf{z})=(1+\mathbf{x}^T\mathbf{z})^2=(1+x_1 z_1 + x_2 z_2)^2$.
Expanding this: $1+(x_1 z_1)^2+(x_2 z_2)^2+2x_1 z_1+2x_2 z_2+2(x_1 z_1)(x_2 z_2)$.

This can be written as an inner product $\phi(\mathbf{x})^T\phi(\mathbf{z})$ if we define the feature map $\phi:\mathbb{R}^2\rightarrow\mathbb{R}^6$ as:

$$\phi(\mathbf{x})=(1,\sqrt{2}x_1,\sqrt{2}x_2,x_1^2,x_2^2,\sqrt{2}x_1 x_2)^T$$

**Application: Solving the XOR Problem**:

**Dataset**: $D=\{((1,1),+1),((-1,-1),+1),((-1,1),-1),((1,-1),-1)\}$

This dataset is not linearly separable in $\mathbb{R}^2$.

Using the polynomial kernel $k(\mathbf{x},\mathbf{z})=(1+\mathbf{x}^T\mathbf{z})^2$, we implicitly map the data to the 6D feature space defined by $\phi(\mathbf{x})$ above. Let's compute the mapped points:

$\phi(1,1)=(1,\sqrt{2},\sqrt{2},1,1,\sqrt{2})^T$

$\phi(-1,-1)=(1,-\sqrt{2},-\sqrt{2},1,1,\sqrt{2})^T$

$\phi(-1,1)=(1,-\sqrt{2},\sqrt{2},1,1,-\sqrt{2})^T$

$\phi(1,-1)=(1,\sqrt{2},-\sqrt{2},1,1,-\sqrt{2})^T$

In this 6D space, the points are linearly separable. For example, a hyperplane focusing on the last coordinate ($\sqrt{2}x_1 x_2$) can separate the positive examples (where $x_1 x_2=1$) from the negative examples (where $x_1 x_2=-1$). The SVM algorithm operating with the polynomial kernel will find such a separating hyperplane in the feature space, resulting in a non-linear (quadratic) boundary in the original $\mathbb{R}^2$ space.

![[xor-problem.png]]
#### 12.3. Gaussian (Radial Basis Function - RBF) Kernel

**Function**: $k(\mathbf{x}_n,\mathbf{x}_m)=\exp\left(-\frac{\|\mathbf{x}_n-\mathbf{x}_m\|^2}{2s^2}\right)=\exp(-\gamma\|\mathbf{x}_n-\mathbf{x}_m\|^2)$, where $\gamma=1/(2s^2)>0$ is the kernel width parameter (or $s$ is the length scale).

**Feature Space**: The feature space corresponding to the Gaussian kernel is infinite-dimensional. This kernel is very powerful and widely used, capable of representing complex decision boundaries.

**Interpretation**: It measures similarity based on Euclidean distance. Points that are close in $\mathbb{R}^d$ have a kernel value close to 1; points that are far apart have a kernel value close to 0. The parameter $\gamma$ controls how quickly the similarity decays with distance.

### 13. Conditions for Valid Kernels: Mercer's Theorem

#### 13.1. Requirement: 
For the kernel trick to be mathematically sound (e.g., ensuring the dual QP is convex), the chosen function $k(\mathbf{x},\mathbf{x}')$ must **correspond to a valid inner product in some Hilbert space $\mathcal{H}$**. That is, there must exist a mapping $\phi:\mathbb{R}^d\rightarrow\mathcal{H}$ such that $k(\mathbf{x},\mathbf{x}')=\langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle_\mathcal{H}$. Such functions are called positive semi-definite (PSD) kernels or Mercer kernels.

#### 13.2. Mercer's Theorem (Simplified Statement): 
A continuous function $k:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ (where $\mathcal{X}$ is a compact subset of $\mathbb{R}^d$, though extensions exist) is a valid Mercer kernel if and only if it satisfies two conditions:

1. **Symmetry**: $k(\mathbf{x},\mathbf{x}')=k(\mathbf{x}',\mathbf{x})$ for all $\mathbf{x},\mathbf{x}'\in\mathcal{X}$.
2. **Positive Semi-Definiteness**: For any finite set of points $\{\mathbf{x}_1,\ldots,\mathbf{x}_N\}\subset\mathcal{X}$ and any real coefficients $c_1,\ldots,c_N\in\mathbb{R}$, the following quadratic form must be non-negative:

$$\sum_{n=1}^N\sum_{m=1}^N c_n c_m k(\mathbf{x}_n,\mathbf{x}_m)\geq 0$$

Equivalently, the Gram matrix $K$ with entries $K_{nm}=k(\mathbf{x}_n,\mathbf{x}_m)$ must be positive semi-definite ($\mathbf{c}^T K\mathbf{c}\geq 0$ for all $\mathbf{c}\in\mathbb{R}^N$).

#### 13.3. Importance: 
Mercer's theorem guarantees that if a function $k$ satisfies these conditions, then a feature space $\mathcal{H}$ and map $\phi$ exist (even if we don't know them explicitly). This ensures that using $k$ in the dual SVM formulation leads to a well-defined, convex optimization problem (specifically, maximizing a concave objective). Commonly used kernels like Linear, Polynomial (with appropriate parameters), and Gaussian are all valid Mercer kernels.

### 14. Gaussian Kernel Parameter $s$ (or $\gamma=1/(2s^2)$)

#### 14.1. Role: 
The parameter $\gamma$ (or $s$) in the Gaussian kernel $k(\mathbf{x},\mathbf{x}')=\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$ controls the "width" or "locality" of the kernel. It determines how quickly the similarity measure $k(\mathbf{x},\mathbf{x}')$ drops off as the distance $\|\mathbf{x}-\mathbf{x}'\|$ increases.

#### 14.2. Effect of Small $s$ (Large $\gamma$):

If $s$ is very small ($\gamma$ is very large), the exponential term decays extremely rapidly. $k(\mathbf{x},\mathbf{x}')$ will be close to 1 only if $\mathbf{x}$ and $\mathbf{x}'$ are nearly identical, and close to 0 otherwise.

**Gram Matrix**: The Gram matrix $K$ will become nearly diagonal (identity matrix), as $K_{nm}\approx\delta_{nm}$ (Kronecker delta).

**Decision Boundary**: The decision boundary $y(\mathbf{x})=\sum a_n^* t_n k(\mathbf{x}_n,\mathbf{x})+b^*=0$ becomes highly sensitive to individual data points. Each support vector $\mathbf{x}_n$ essentially creates a small "bubble" of influence around itself. The boundary can become very complex and "wiggly," fitting the training data very closely.

**Risk**: High risk of overfitting the training data, leading to poor generalization on unseen data. The model has high variance.

#### 14.3. Effect of Large $s$ (Small $\gamma$):

If $s$ is very large ($\gamma$ is very small), the exponential term decays very slowly. $k(\mathbf{x},\mathbf{x}')$ will be close to 1 even for points $\mathbf{x},\mathbf{x}'$ that are far apart.

**Gram Matrix**: All entries of the Gram matrix $K$ will be close to 1 ($K_{nm}\approx 1$). The matrix becomes close to a rank-one matrix ($\mathbf{1}\mathbf{1}^T$).

**Decision Boundary**: The influence of each support vector extends very far. The resulting decision boundary tends to be very smooth and may fail to capture the underlying structure of the data.

**Risk**: High risk of underfitting the training data. The model has high bias.

#### 14.4. Practical Tuning: 
Choosing the optimal value for $\gamma$ (and the SVM regularization parameter $C$) is crucial for good performance. This is typically done simultaneously using model selection techniques like grid search with cross-validation. The goal is to find the $(\gamma,C)$ pair that yields the best performance on validation sets, balancing the bias-variance trade-off.

![[gaussian-kernel.png]]
![[gaussian-kernel-2.png]]

### 15. Kernelization Example: Kernel Ridge Regression (Least Squares)

#### 15.1. Primal Objective (in Feature Space): 
Consider standard ridge regression, but performed in the feature space $\mathcal{H}$ associated with a kernel $k$. We want to find $\mathbf{w}\in\mathcal{H}$ that minimizes:

$$J(\mathbf{w})=\frac{1}{2}\sum_{n=1}^N (t_n-(\mathbf{w}^T\phi(\mathbf{x}_n)+b))^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

(Assuming $b=0$ or data is centered for simplicity, often absorbed into $\mathbf{w}$ via augmented features). Let's use the formulation matching the notes:

$$J(\mathbf{w})=\frac{1}{2}\|\Phi\mathbf{w}-\mathbf{t}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$$

where $\Phi$ is the $N\times M$ matrix whose rows are $\phi(\mathbf{x}_n)^T$, $\mathbf{t}=(t_1,\ldots,t_N)^T$, and $\mathbf{w}\in\mathbb{R}^M$ (assuming $\mathcal{H}=\mathbb{R}^M$).

#### 15.2. Dual Representation of Solution: 
The Representer Theorem states that for many regularized loss minimization problems (including this one), the optimal solution $\mathbf{w}^*$ can be expressed as a linear combination of the mapped training data points:

$$\mathbf{w}^*=\sum_{m=1}^N a_m \phi(\mathbf{x}_m)=\Phi^T\mathbf{a}$$

for some coefficient vector $\mathbf{a}=(a_1,\ldots,a_N)^T\in\mathbb{R}^N$.

#### 15.3. Solution for Dual Coefficients $\mathbf{a}$: 
Substitute $\mathbf{w}=\Phi^T\mathbf{a}$ into the objective function $J(\mathbf{w})$:

$$J(\mathbf{a})=\frac{1}{2}\|\Phi(\Phi^T\mathbf{a})-\mathbf{t}\|^2 + \frac{\lambda}{2}\|\Phi^T\mathbf{a}\|^2$$

$$J(\mathbf{a})=\frac{1}{2}\|K\mathbf{a}-\mathbf{t}\|^2 + \frac{\lambda}{2}(\Phi^T\mathbf{a})^T(\Phi^T\mathbf{a})=\frac{1}{2}\|K\mathbf{a}-\mathbf{t}\|^2 + \frac{\lambda}{2}\mathbf{a}^T\Phi\Phi^T\mathbf{a}$$

$$J(\mathbf{a})=\frac{1}{2}(K\mathbf{a}-\mathbf{t})^T(K\mathbf{a}-\mathbf{t}) + \frac{\lambda}{2}\mathbf{a}^T K\mathbf{a}$$

where $K=\Phi\Phi^T$ is the $N\times N$ Gram matrix, $K_{nm}=\phi(\mathbf{x}_n)^T\phi(\mathbf{x}_m)=k(\mathbf{x}_n,\mathbf{x}_m)$.
To minimize $J(\mathbf{a})$, we set the gradient w.r.t. $\mathbf{a}$ to zero:

$$\nabla_\mathbf{a} J(\mathbf{a})=K^T(K\mathbf{a}-\mathbf{t})+\lambda K\mathbf{a}=0$$

Since $K$ is symmetric ($K^T=K$):

$$K(K\mathbf{a}-\mathbf{t})+\lambda K\mathbf{a}=0$$

$$K K\mathbf{a}-K\mathbf{t}+\lambda K\mathbf{a}=0$$

$$K(K+\lambda I)\mathbf{a}=K\mathbf{t}$$

Assuming $K$ is invertible (often requires adding a small ridge $\epsilon I$ if not), we might get $(K+\lambda I)\mathbf{a}=\mathbf{t}$. Let's re-check the standard derivation or the notes.
Standard Derivation: The solution to the primal ridge regression $\min_\mathbf{w} \frac{1}{2}\|\Phi\mathbf{w}-\mathbf{t}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$ is $\mathbf{w}^*=(\Phi^T\Phi+\lambda I_M)^{-1}\Phi^T\mathbf{t}$. Using the substitution $\mathbf{w}^*=\Phi^T\mathbf{a}^*$ and matrix identities (like Woodbury), one arrives at:

$$\mathbf{a}^*=(\Phi\Phi^T+\lambda I_N)^{-1}\mathbf{t}=(K+\lambda I)^{-1}\mathbf{t}$$

This matches the formula in the notes. The derivation involves setting $\nabla_\mathbf{w} J=0\implies\Phi^T(\Phi\mathbf{w}-\mathbf{t})+\lambda\mathbf{w}=0$. Substitute $\mathbf{w}=\Phi^T\mathbf{a}$: $\Phi^T(\Phi\Phi^T\mathbf{a}-\mathbf{t})+\lambda\Phi^T\mathbf{a}=0\implies\Phi^T(K\mathbf{a}-\mathbf{t}+\lambda\mathbf{a})=0\implies\Phi^T((K+\lambda I)\mathbf{a}-\mathbf{t})=0$. If $\Phi$ has full row rank, this implies $(K+\lambda I)\mathbf{a}=\mathbf{t}$.

#### 15.4. Prediction Rule: 
The prediction for a new data point $\mathbf{x}$ is:

$$y(\mathbf{x})=(\mathbf{w}^*)^T\phi(\mathbf{x})=(\Phi^T\mathbf{a}^*)^T\phi(\mathbf{x})=(\mathbf{a}^*)^T\Phi\phi(\mathbf{x})$$

Let $\mathbf{k}(\mathbf{x})=\Phi\phi(\mathbf{x})$ be the vector of kernel evaluations between $\mathbf{x}$ and all training points, i.e., $[\mathbf{k}(\mathbf{x})]_n=\langle\phi(\mathbf{x}_n),\phi(\mathbf{x})\rangle=k(\mathbf{x}_n,\mathbf{x})$. Then:

$$y(\mathbf{x})=(\mathbf{a}^*)^T\mathbf{k}(\mathbf{x})=\sum_{n=1}^N a_n^* k(\mathbf{x}_n,\mathbf{x})$$

#### 15.5. Key Insight: 
Kernel Ridge Regression allows performing ridge regression in a potentially very high (or infinite) dimensional feature space implicitly, by solving an $N\times N$ linear system involving the Gram matrix $K$ to find the dual coefficients $\mathbf{a}^*$. The prediction only requires computing kernel functions between the new point and the training points.

### 16. Multi-Class Support Vector Machines2

#### 16.1. Problem Setting: 
Standard SVM is defined for binary classification ($t_n\in\{-1,+1\}$). We need strategies to handle problems with $K>2$ classes, where $t_n\in\{1,2,\ldots,K\}$.

#### 16.2. One-vs-All (OvA / One-vs-Rest, OvR)

**Training**: Train $K$ independent binary SVM classifiers. For each class $k\in\{1,\ldots,K\}$, train a classifier $y_k(\mathbf{x})=\mathbf{w}_k^T\phi(\mathbf{x})+b_k$. The training data for classifier $k$ consists of all points $(\mathbf{x}_n,t_n')$ where $t_n'=+1$ if the original label $t_n=k$, and $t_n'=-1$ if $t_n\neq k$.

**Prediction**: Given a new input $\mathbf{x}$, evaluate all $K$ classifiers: $y_1(\mathbf{x}),\ldots,y_K(\mathbf{x})$. Assign $\mathbf{x}$ to the class whose classifier gives the highest output value (often interpreted as highest confidence):

$$\hat{t}(\mathbf{x})=\arg\max_{k\in\{1,\ldots,K\}} y_k(\mathbf{x})=\arg\max_{k\in\{1,\ldots,K\}} (\mathbf{w}_k^T\phi(\mathbf{x})+b_k)$$

**Pros**: Relatively simple to implement using existing binary SVM solvers. Training can be parallelized.

**Cons**:
- **Class Imbalance**: Each binary training problem is inherently imbalanced (one class vs $K-1$ classes).
- **Ambiguity/Scaling**: The output scores $y_k(\mathbf{x})$ from different classifiers might not be directly comparable or well-calibrated, potentially leading to ambiguity in regions where multiple classifiers give similar high scores.

![[multi-svm.png]]

#### 16.3. Simultaneous Learning (e.g., Weston-Watkins, Crammer-Singer)

**Training**: Instead of independent binary problems, formulate a single, larger optimization problem that jointly learns all $K$ decision functions $y_k(\mathbf{x})=\mathbf{w}_k^T\phi(\mathbf{x})+b_k$. The goal is to ensure that for each training point $\mathbf{x}_n$ with true class $t_n$, the score for the correct class $y_{t_n}(\mathbf{x}_n)$ is sufficiently larger than the scores for all incorrect classes $y_j(\mathbf{x}_n)$ where $j\neq t_n$.

**Example Margin Constraints (Weston-Watkins)**: For each training example $(\mathbf{x}_n,t_n)$, and for each incorrect class $j\neq t_n$, enforce the constraint:

$$(\mathbf{w}_{t_n}^T\phi(\mathbf{x}_n)+b_{t_n}) - (\mathbf{w}_j^T\phi(\mathbf{x}_n)+b_j) \geq 1 - \xi_n$$

Similarly, the objective minimizes a combination of regularization terms and a penalty on the slack variables 

$$
\min_{\{\mathbf{w}_k, b_k, \xi_{n,j}\}} \quad \frac{1}{2} \sum_{k=1}^{K} \|\mathbf{w}_k\|^2 + C \sum_{\substack{n=1 \\ j \ne t_n}}^{N} \xi_{n,j}
$$
subject to
$$
\begin{cases}
\mathbf{w}_{t_n}^\top \mathbf{x}_n + b_{t_n} \geq \mathbf{w}_j^\top \mathbf{x}_n + b_j + 1 - \xi_{n,j}, & \forall n, \, j \ne t_n \\
\xi_{n,j} \geq 0
\end{cases}
$$

**Pros**: Directly optimizes a multi-class margin objective, potentially leading to better theoretical properties and performance. Avoids the scaling issue of OvA.

**Cons**: Computationally much more expensive. The resulting optimization problem (primal or dual) involves significantly more variables ($K$ weight vectors, $K$ biases, potentially $N(K-1)$ constraints/slack variables) and is more complex to solve than $K$ independent binary problems.

**Practicality**: Due to the computational cost, OvA (or its close relative One-vs-One, OvO) is often preferred in practice, especially for large datasets or large $K$.

## Part III: Advanced Kernels for Structured Data

### 18. Motivation: Beyond Vectorial Data

#### 18.1. Data Types: 
Many important real-world datasets do not naturally reside in $\mathbb{R}^d$. Instead, they consist of structured objects like:

- **Strings/Sequences**: DNA, RNA, protein sequences; text documents (sequences of words/characters).
- **Graphs**: Molecules (atoms as nodes, bonds as edges); social networks; citation networks; program dependency graphs.
- **Trees**: Parse trees in natural language processing; phylogenetic trees in biology.

![[graph-structured-data.png]]

#### 18.2. Challenge: 
Standard kernel methods (Linear, Poly, RBF) operate on fixed-dimensional vectors $\mathbf{x}\in\mathbb{R}^d$. How can we apply the powerful framework of kernel machines (like SVM) to these structured, non-vectorial data types?

#### 18.3. General Approach: 
The core idea remains the same: define a notion of similarity (an inner product) between pairs of structured objects. This is achieved by designing specific kernel functions $k(O_1,O_2)$ that operate directly on the structured objects $O_1,O_2$. These kernels implicitly define a feature map $\phi(O)$ into some (often abstract or very high-dimensional) feature space $\mathcal{H}$, such that $k(O_1,O_2)=\langle\phi(O_1),\phi(O_2)\rangle$. The process usually involves:

1. **Feature Extraction (Implicit or Explicit)**: Define relevant features or patterns within the structured objects (e.g., substrings, walks, subgraphs, generative model parameters).
2. **Kernel Computation**: Define the kernel function based on comparing these features (e.g., counting common features, measuring similarity of feature distributions).

### 19. Kernels for Strings

Let $A$ be a finite alphabet (e.g., $\{A,C,G,T\}$ for DNA, or the set of ASCII characters). Let $A^*$ denote the set of all finite-length strings over $A$. We want kernels $k:A^* \times A^* \rightarrow \mathbb{R}$.

#### 19.1. General String Kernel Framework

**Feature Map Idea:** Consider an indexed set of possible features $S=\{s_1,s_2,\ldots\}$, where each $s_i$ is typically a substring (or subsequence) from $A^*$. Define a feature map $\phi:A^* \rightarrow H$ where the component corresponding to feature $s$ is $\phi_s(\mathbf{x})$.

**Feature Value $\phi_s(\mathbf{x})$:** This value quantifies the presence or frequency of feature $s$ within the string $\mathbf{x}$. Common choices include:

- Binary indicator: $\phi_s(\mathbf{x})=1$ if $s$ occurs in $\mathbf{x}$, $0$ otherwise.
- Count: $\phi_s(\mathbf{x}) = \text{number of times } s \text{ occurs in } \mathbf{x}$. 
- Weighted count: $\phi_s(\mathbf{x}) = \sum_{i \in \text{Occ}(s, \mathbf{x})} w(i)$ sum of weights for each occurrence of $s$ in $\mathbf{x}$ (e.g., gap weighting).

**Kernel:** The kernel is the inner product in this feature space:

$$k(\mathbf{x},\mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle = \sum_{s \in S} \phi_s(\mathbf{x}) \phi_s(\mathbf{x}')$$

This measures the similarity between $\mathbf{x}$ and $\mathbf{x}'$ based on the features (substrings/subsequences) they share.

#### 19.2. Gap-Weighted Subsequence Kernel

**Concept:** Considers subsequences rather than just contiguous substrings. A subsequence $s=s_1s_2\ldots s_k$ occurs in $\mathbf{x}$ if the characters $s_1,\ldots,s_k$ appear in $\mathbf{x}$ in that order, but not necessarily contiguously.

**Weighting:** To penalize non-contiguous matches, each occurrence of a subsequence $s$ in $\mathbf{x}$ is assigned a weight $\lambda^{l(\text{occ})}$, where $l(\text{occ})$ is the total length of the gaps in that specific occurrence (i.e., the number of characters in $\mathbf{x}$ between the first character $s_1$ and the last character $s_k$, minus $k-1$). The parameter $0 < \lambda \leq 1$ controls the gap penalty (smaller $\lambda$ means higher penalty). $\lambda=1$ treats all occurrences equally.

**Feature Value:** $\phi_s(\mathbf{x}) = \sum_{\text{occ of } s \text{ in } \mathbf{x}} \lambda^{l(\text{occ})}$.

**Computation:** The feature space is enormous (all possible subsequences). However, the kernel $k(\mathbf{x},\mathbf{x}') = \sum_s \phi_s(\mathbf{x}) \phi_s(\mathbf{x}')$ can be computed efficiently using dynamic programming algorithms without explicit enumeration, often in $O(k|\mathbf{x}||\mathbf{x}'|)$ time for subsequences of length $k$.

**Example**: For the list of substrings `ca`, `ct`, `cr`, `ar`, `rt`, `ba`, `br`:

|                     | $\phi_{\text{c--a}}$ | $\phi_{\text{c--t}}$ | $\phi_{\text{c--r}}$ | $\phi_{\text{a--r}}$ | $\phi_{\text{r--t}}$ | $\phi_{\text{b--a}}$ | $\phi_{\text{b--r}}$ |
| ------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| $\phi(\text{cat})$  | $1$                  | $\lambda$            | $0$                  | $0$                  | $0$                  | $0$                  | $0$                  |
| $\phi(\text{cart})$ | $1$                  | $\lambda^2$          | $\lambda$            | $1$                  | $1$                  | $0$                  | $0$                  |
| $\phi(\text{bar})$  | $0$                  | $0$                  | $0$                  | $1$                  | $0$                  | $1$                  | $\lambda$            |

$$
k(\text{cat}, \text{cart}) = 1 + \lambda^3, \quad k(\text{cat}, \text{bar}) = 0, \quad k(\text{cart}, \text{bar}) = 1
$$

#### 19.3. k-Spectrum Kernel

**Idea:** A simpler approach that restricts the features to contiguous substrings of a fixed length $k$, often called k-mers.

**Feature Representation:** The feature space is indexed by all possible strings $s \in A^k$ of length exactly $k$. The feature map $\phi^{(k)}:A^* \rightarrow \mathbb{R}^{|A|^k}$ is defined by:

$$[\phi^{(k)}(\mathbf{x})]_s = \text{count}(\mathbf{x},s) = \text{number of times substring } s \text{ occurs in } \mathbf{x}$$

**Kernel:** The k-spectrum kernel is the inner product of these count vectors:

$$k_k(\mathbf{x},\mathbf{x}') = \langle \phi^{(k)}(\mathbf{x}), \phi^{(k)}(\mathbf{x}') \rangle = \sum_{s \in A^k} \text{count}(\mathbf{x},s) \cdot \text{count}(\mathbf{x}',s)$$

**Example**: consider the classification problem for DNA sequences 
![[dna-seq-ex.png]]

**Efficient Counting:** The counts $\phi^{(k)}(\mathbf{x})$ for all $k$-mers $s$ in a string $\mathbf{x}$ can be computed efficiently using data structures like suffix trees or suffix arrays in time roughly linear in the length of $\mathbf{x}$ (e.g., $O(|\mathbf{x}|)$ or $O(|\mathbf{x}|\log|A|)$). Once the count vectors are computed, the kernel calculation is straightforward.

**Parameter $k$:** The choice of $k$ is crucial.

- Small $k$: Captures only local sequence information, may not be discriminative enough.
- Large $k$: Captures longer patterns, more specific, but the feature space becomes huge ($|A|^k$), and counts become sparse (many k-mers won't appear), risking overfitting or poor statistical estimation. Needs careful tuning based on the application and data length.

**Special Case - Bag-of-Words:** When $k=1$, the features are just the counts of individual characters (or words, if the alphabet $A$ is the vocabulary). This is the standard bag-of-words representation commonly used in document classification. $k_1(\mathbf{x},\mathbf{x}')$ simply measures the co-occurrence counts of characters/words.

### 20. Kernels for Graphs

Let $G=(V,E)$ be a graph, where $V$ is the set of nodes (vertices) and $E \subseteq V \times V$ (for directed graphs) or $E \subseteq \{\{u,v\} \mid u,v \in V\}$ (for undirected graphs) is the set of edges. Graphs may have node labels $l(v)$ and edge labels $l(e)$. We want kernels $k(G,G')$.

#### 20.1. Representing Graphs

Common representations include adjacency lists or the adjacency matrix $A$, where $A_{ij}=1$ if there is an edge from node $i$ to node $j$, and $A_{ij}=0$ otherwise (for unweighted graphs). For undirected graphs, $A$ is symmetric.

#### 20.2. Graph Walks

**Definition:** A walk of length $k$ in $G$ is a sequence of $k+1$ vertices $(v_0,v_1,\ldots,v_k)$ such that $(v_{i-1},v_i) \in E$ for all $i=1,\ldots,k$. The walk starts at $v_0$ and ends at $v_k$. Unlike paths, walks can revisit nodes and edges.

**Counting Walks:** The number of distinct walks of length exactly $k$ starting at node $i$ and ending at node $j$ is given by the $(i,j)$-th entry of the $k$-th power of the adjacency matrix, $[A^k]_{ij}$.

**Proof by induction:** Base case $k=1$ is true by definition of $A$. Assume true for $k$. Then $[A^{k+1}]_{ij} = [A^k A]_{ij} = \sum_{s \in V} [A^k]_{is} A_{sj}$. This sums the number of $k$-length walks ending at any neighbor $s$ of $j$, which is exactly the number of $(k+1)$-length walks ending at $j$.

#### 20.3. Random Walk Graph Kernel

**Idea:** Define similarity between graphs $G$ and $G'$ based on the number of matching walks they have in common. Two walks $(v_0,\ldots,v_k)$ in $G$ and $(v_0',\ldots,v_k')$ in $G'$ match if they have the same sequence of node/edge labels (if labels exist). For unlabeled graphs, the structure itself is compared.

**Method using Direct Product Graph $G_\times$:**

1. Construct a direct product graph $G_\times = (V_\times, E_\times)$ from $G=(V,E)$ and $G'=(V',E')$.
   - **Node Set:** $V_\times = V \times V' = \{(u,u') \mid u \in V, u' \in V'\}$. A node in $G_\times$ represents a pair of nodes, one from $G$ and one from $G'$.
   - **Edge Set:** There is an edge from $(u,u')$ to $(v,v')$ in $G_\times$ if and only if there is an edge $(u,v) \in E$ and an edge $(u',v') \in E'$. (Assumes unlabeled edges for simplicity; labels can be incorporated).
2. **Adjacency Matrix of $G_\times$:** If $A$ and $A'$ are the adjacency matrices of $G$ and $G'$, the adjacency matrix $A_\times$ of $G_\times$ is given by the Kronecker product: $A_\times = A \otimes A'$. If $|V|=n, |V'|=n'$, then $A_\times$ is $(nn') \times (nn')$.
3. **Walks in $G_\times$:** A walk in $G_\times$ corresponds precisely to a pair of simultaneous walks, one in $G$ and one in $G'$. The number of walks of length $k$ in $G_\times$ is $\sum_{i,j} [A_\times^k]_{ij} = \mathbf{1}^T A_\times^k \mathbf{1}$.

![[direct-product-graph.png]]

**Kernel Formula:** The random walk kernel sums up contributions from matching walks of all lengths, usually with a decay factor $\lambda > 0$ to ensure convergence and down-weight longer walks:

$$k_{RW}(G,G') = \sum_{k=0}^\infty \lambda^k (\mathbf{1}^T A_\times^k \mathbf{1}) = \mathbf{1}^T \left( \sum_{k=0}^\infty (\lambda A_\times)^k \right) \mathbf{1}$$

If the spectral radius $\rho(\lambda A_\times) < 1$ (which requires $\lambda$ to be sufficiently small, e.g., $\lambda < 1/\rho(A_\times)$), this geometric series converges to:

$$k_{RW}(G,G') = \mathbf{1}^T (I - \lambda A_\times)^{-1} \mathbf{1}$$

**Computation:** Computing the matrix inverse $(I - \lambda A_\times)^{-1}$ is computationally expensive ($O((nn')^3)$). Iterative methods or solving the linear system $(I - \lambda A_\times)x = \mathbf{1}$ can be more efficient. The notes mention an iterative method $x^{(t+1)} = \mathbf{1} + \lambda A_\times x^{(t)}$, which converges to the solution $x = (I - \lambda A_\times)^{-1} \mathbf{1}$, and the kernel is $\mathbf{1}^T x$. Exploiting the Kronecker structure using $\text{vec}$ operations can sometimes speed this up.

**Limitations:**

- **Tottering:** Walks can go back and forth along edges, leading to structurally dissimilar graphs potentially having high kernel values if they share small cyclic structures.
- **Computational Cost:** Still high, especially for large graphs ($n,n'$ large).
- **Label Handling:** Basic version doesn't handle labels well; extensions exist.

#### 20.4. Weisfeiler-Lehman (WL) Graph Kernel

**Key Idea:** Based on the Weisfeiler-Lehman test for graph isomorphism, it iteratively refines node labels by aggregating labels from neighbors. Similarity is then measured based on the counts of these refined labels.

**Incorporation of Node Labels:** Starts with initial discrete node labels (e.g., atom types in molecules, or node degrees if no labels exist).

**Iterative Relabeling Process (WL subtree kernel):**

1. **Initialization (h=0):** Set initial label $l_0(v)$ for each node $v$. Store the multiset of labels in the graph.
2. **Iteration h = 1 to M:** For each node $v \in V$:
   - Collect the multiset of labels of its neighbors at the previous iteration: $N_h(v) = \{\{ l_{h-1}(u) \mid u \text{ is a neighbor of } v \}\}$.
   - Create a combined signature string: $s_h(v) = \text{sort}(l_{h-1}(v) || N_h(v))$. (Concatenate the node's own label with the sorted multiset of neighbor labels).
   - Compress this signature $s_h(v)$ into a new, unique, discrete label $l_h(v)$ using a mapping function (e.g., hashing or indexing into a growing dictionary of observed signatures). Ensure nodes with the same signature get the same new label.
   - Store the multiset of these new labels $l_h(v)$ for all $v$.

**Final Kernel Computation:**

The feature map $\phi_{WL}(G)$ for a graph $G$ is a vector containing the counts of all distinct labels generated across all iterations $h=0,\ldots,M$. That is, $[\phi_{WL}(G)]_{\text{label}} = \sum_{h=0}^M \sum_{v \in V} I(l_h(v) == \text{label})$.

The WL kernel is typically the linear kernel (dot product) between these high-dimensional count vectors:

$$k_{WL}(G,G') = \langle \phi_{WL}(G), \phi_{WL}(G') \rangle$$

**Example**:

![[wl-example.png]]

**Advantages:**

- **Efficiency:** The relabeling process takes time roughly proportional to the number of edges per iteration. Computing the kernel involves a dot product of sparse vectors. Often much faster than random walk kernels, potentially quasi-linear in graph size.
- **Effectiveness:** Captures local neighborhood structures effectively. Often performs very well in practice on various graph classification tasks.

### 21. The Fisher Kernel

#### 21.1. Approach

Provides a bridge between generative probability models and discriminative classifiers like SVMs. Instead of defining features based directly on combinatorial structures (like substrings or walks), it defines features based on how a data point interacts with a learned generative model.

#### 21.2. Step 1: Train a Generative Model

Assume a parametric generative model $p(\mathbf{x}|\theta)$ that can model the distribution of the data (which can be structured, e.g., sequences via HMMs, images via GMMs on features). Here $\theta \in \mathbb{R}^P$ is the vector of model parameters.

Train this model on the available dataset $D = \{\mathbf{x}_1,\ldots,\mathbf{x}_N\}$ to obtain parameter estimates, typically via Maximum Likelihood Estimation (MLE):

$$\theta_{\text{MLE}} = \arg\max_\theta \sum_{n=1}^N \log p(\mathbf{x}_n | \theta)$$

#### 21.3. Step 2: Compute Fisher Score Vector

For each data point $\mathbf{x}_n$, the Fisher score vector $u(\mathbf{x}_n)$ is defined as the gradient of the log-likelihood of that point with respect to the model parameters, evaluated at the estimated parameters $\theta_{\text{MLE}}$:

$$u(\mathbf{x}_n) = \nabla_\theta \log p(\mathbf{x}_n | \theta) \big|_{\theta = \theta_{\text{MLE}}}$$

**Interpretation:** The Fisher score $u(\mathbf{x}_n) \in \mathbb{R}^P$ indicates how sensitive the log-likelihood of observing $\mathbf{x}_n$ is to small changes in each parameter $\theta_i$. It captures how $\mathbf{x}_n$ "pulls" the parameters during estimation. It effectively maps the potentially complex, structured data point $\mathbf{x}_n$ into a fixed-dimensional vector space $\mathbb{R}^P$ (the parameter space gradient). This vector $u(\mathbf{x}_n)$ is used as the feature vector: $\phi_{\text{Fisher}}(\mathbf{x}_n) = u(\mathbf{x}_n)$.

#### 21.4. Step 3: Define the Fisher Kernel

The Fisher kernel measures the similarity between two data points $\mathbf{x}_n$ and $\mathbf{x}_m$ based on the similarity of their Fisher score vectors.

**Simplified Form (as in notes):**

$$k_{\text{Fisher}}(\mathbf{x}_n, \mathbf{x}_m) = u(\mathbf{x}_n)^T u(\mathbf{x}_m) = \phi_{\text{Fisher}}(\mathbf{x}_n)^T \phi_{\text{Fisher}}(\mathbf{x}_m)$$

This is simply the linear kernel applied to the Fisher score vectors.

**Full Form (often preferred):** The geometry of the parameter space is often non-Euclidean. The Fisher Information Matrix $F$ captures this geometry:

$$F = \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}|\theta)} [u(\mathbf{x}) u(\mathbf{x})^T]$$

It acts as a metric tensor on the statistical manifold. The proper Fisher kernel uses $F^{-1}$ (or an approximation) to define the inner product:

$$k_{\text{Fisher}}(\mathbf{x}_n, \mathbf{x}_m) = u(\mathbf{x}_n)^T F^{-1} u(\mathbf{x}_m)$$

This accounts for the curvature and parameter correlations. However, computing and inverting $F$ can be challenging, so the simplified linear kernel on scores is often used.

#### 21.5. Example: Mixture Model

Consider a Gaussian Mixture Model (GMM): $p(\mathbf{x}|\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\mu_k, \Sigma_k)$, where $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$.

The parameters include mixing coefficients $\pi_k$, means $\mu_k$, and covariances $\Sigma_k$.

**Score w.r.t. $\pi_k$ (under sum constraint $\sum \pi_j = 1$):** The gradient computation involves the posterior probability (responsibility) of component $k$ generating $\mathbf{x}$: $p(k|\mathbf{x},\theta) = \frac{\pi_k p(\mathbf{x}|\theta_k)}{\sum_j \pi_j p(\mathbf{x}|\theta_j)}$. 

The score component related to $\pi_k$ is 

$$
\left[ \phi(\mathbf{x}_n) \right]_{\pi_k} = 
\left. \frac{\partial \log p(\mathbf{x} \mid \theta)}{\partial \pi_k} \right|_{\theta = \theta_{\text{MLE}}}
= 
\frac{p(\mathbf{x} \mid \theta_k^{\text{MLE}})}{\sum_k p(\mathbf{x} \mid \theta_k^{\text{MLE}}) \, \pi_k^{\text{MLE}}}
$$

**Intuition:** The Fisher score vector $u(\mathbf{x})$ captures which components of the mixture model are most responsible for generating $\mathbf{x}$ and how sensitive the likelihood is to changes in their parameters (mean, variance, weight). Two points $\mathbf{x}_n, \mathbf{x}_m$ are similar under the Fisher kernel if they exert similar influences on the generative model's parameters.
