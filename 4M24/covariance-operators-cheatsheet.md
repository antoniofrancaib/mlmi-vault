# Covariance Operators Cheat-Sheet

---

## 1. Definition

**Setting:** Real separable Hilbert space $\mathcal{H}$.  
**Random element:** $X \sim \mu$ with mean zero.  
**Covariance operator:** $C: \mathcal{H} \to \mathcal{H}$

$$
C u = \int_{\mathcal{H}} \langle X, u \rangle X \, d\mu(X),
$$

equivalently

$$
\langle u, C v \rangle = \mathbb{E}[\langle X, u \rangle \langle X, v \rangle].
$$

---

## 2. Key Properties

- **Self-adjoint:** $\langle u, C v \rangle = \langle C u, v \rangle$  
- **Positive semidefinite:** $\langle u, C u \rangle \geq 0$  
- **Trace-class (for Gaussian measures):**  
  $$
  \mathrm{tr}(C) = \sum_k \langle e_k, C e_k \rangle < \infty
  $$
- **Nuclear norm:** $\|C\|_1 = \mathrm{tr}(C)$ finite ⇒ Gaussian lives in $\mathcal{H}$

---

## 3. Diagonal (Coordinate-wise) Case

- **Basis:** $\{ e_k \}$ orthonormal  
- **Diagonal covariance:** $C e_k = \lambda_k e_k$  
- **Action:**

  $$
  C(x_1, x_2, \dots) = (\lambda_1 x_1, \lambda_2 x_2, \dots)
  $$

- **Gaussian coordinates:**  
  $$
  \langle X, e_k \rangle \sim \mathcal{N}(0, \lambda_k), \quad \text{independent}
  $$

---

## 4. Spectral / Eigendecomposition

**Mercer / Karhunen–Loève:**

$$
C e_k = \lambda_k e_k, \quad \lambda_1 \geq \lambda_2 \geq \dots \geq 0
$$

**KL Expansion of $X$:**

$$
X = \sum_{k=1}^\infty \sqrt{\lambda_k} Z_k e_k, \quad Z_k \sim \text{iid } \mathcal{N}(0,1)
$$

**Approximation:** Truncate to first $n$ modes for finite-rank approximation.

---

## 5. Radon–Nikodym & Changes of Covariance

**Finite-dimensional:**  
If $\mu = \mathcal{N}(0, C_1)$ and $\nu = \mathcal{N}(0, C_2)$ on $\mathbb{R}^n$,

$$
\frac{d\nu}{d\mu}(x) = \frac{\det C_1}{\det C_2} \exp\left(-\frac{1}{2} x^\top (C_2^{-1} - C_1^{-1}) x \right)
$$

**Infinite-dimensional (diagonal):**  
$C_1 = \mathrm{diag}(1,1,\dots)$, $C_2 = \mathrm{diag}(\lambda_k)$

$$
\frac{d\nu}{d\mu}(x) = \prod_{k=1}^\infty \lambda_k^{-1/2} \exp\left( \frac{1}{2} (1 - \lambda_k^{-1}) x_k^2 \right)
$$

---

## 6. Relation to RKHS / Kernels

Covariance operator $C$ of a Gaussian process with kernel $k$ on domain $D$:

$$
(Cf)(x) = \int_D k(x, y) f(y)\, dy
$$

**Eigenfunctions of $C$ ↔ Mercer expansion:**

$$
k(x, y) = \sum \lambda_i \, \phi_i(x) \phi_i(y)
$$

---

## 7. Practical Notes

- **Estimation:** Empirical covariance  
  $$
  \hat{C} = \frac{1}{N} \sum_i X_i \otimes X_i
  $$
- **Regularization:** Add $\alpha I$ to make $C + \alpha I$ invertible  
- **Dimensionality-reduction:** PCA in $\mathcal{H}$ via top eigenpairs of $C$  
- **Whitening:** Transform $X \mapsto C^{-1/2} X$ to get unit covariance
