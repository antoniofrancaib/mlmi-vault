# Gaussian Processes

Probabilistic machine learning offers a framework that explicitly models uncertainty and variability in data and predictions. In this blog post, I will introduce one of the most powerful non-parametric methods in this field—the Gaussian process (GP). 

## From Scalar Gaussians to Gaussian Processes

At the very core of probabilistic modeling lies the scalar Gaussian distribution. A scalar random variable $\mathbf{x}$ is said to follow a Gaussian (or normal) distribution if its probability density is given by

$$
p(\mathbf{x} \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\mathbf{x} - \mu)^2}{2\sigma^2}\right),
$$

where $\mu$ is the mean and $\sigma^2$ is the variance. This one-dimensional formulation extends naturally to multiple dimensions.

In the multivariate Gaussian distribution, we consider a vector $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N]^\top$ whose joint density is

$$
p(\mathbf{x} \mid \mu, \Sigma) = \frac{1}{(2\pi)^{N/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu)\right),
$$

with $\mu$ as the mean vector and $\Sigma$ as the covariance matrix. The key property of multivariate Gaussians is that any subset of variables (i.e., any marginal) also follows a Gaussian distribution, and conditioning on some variables results in another Gaussian. These properties are central to the construction and analysis of Gaussian processes.

A Gaussian process (GP) generalizes the concept of multivariate Gaussians to infinitely many random variables. Formally, a Gaussian process is defined as a collection of random variables, any finite number of which have a joint Gaussian distribution. This can be interpreted as viewing a function $f(\mathbf{x})$ as an infinitely long vector whose entries correspond to the function’s values at each input $\mathbf{x}$.

A GP is completely specified by its mean function and covariance function:

$$
m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})],
$$

$$
k(\mathbf{x}, \mathbf{x}') = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))].
$$

Thus, we write

$$
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')).
$$

The infinite-dimensional nature of GPs means that we can work with finite slices (via marginalization) without ever needing to handle the infinite-dimensional object directly. This is achieved by leveraging the fact that if we select a finite set of inputs $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$, the corresponding function values $\mathbf{f} = [f(\mathbf{x}_1), \dots, f(\mathbf{x}_N)]^\top$ follow a multivariate Gaussian distribution with mean $\mathbf{m}$ and covariance matrix $\mathbf{K}$, where $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.

## Gaussian Processes as Distributions Over Functions

The idea of a GP is to place a probability measure directly over functions. Rather than parameterizing a model by a finite set of weights or coefficients, a GP defines a distribution over functions. For example, consider the GP defined by

$$
f(\mathbf{x}) \sim \mathcal{GP}\left(0, \exp\left(-\frac{(\mathbf{x} - \mathbf{x}')^2}{2}\right)\right).
$$

For any finite set of inputs $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$, the function values $\mathbf{f} = [f(\mathbf{x}_1), \dots, f(\mathbf{x}_N)]^\top$ follow

$$
\mathbf{f} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}), \quad \text{with} \quad \mathbf{K}_{ij} = \exp\left(-\frac{(\mathbf{x}_i - \mathbf{x}_j)^2}{2}\right).
$$

This view encourages us to imagine sampling entire functions from the GP prior—each sample being a plausible function drawn from the infinite-dimensional Gaussian distribution. Importantly, the marginal and conditional properties of Gaussians guarantee that these finite samples are consistent with the overall GP.

To visualize a GP, one common approach is to sample from the multivariate Gaussian distribution corresponding to a finite set of inputs. The procedure is as follows:

1. **Select Inputs**: Choose a set $\{\mathbf{x}_i\}_{i=1}^N$.
2. **Compute the Covariance Matrix**: Form the matrix $\mathbf{K}$ with entries $k(\mathbf{x}_i, \mathbf{x}_j)$.
3. **Generate Samples**: Draw a vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ of independent standard normal variates. Then, using the Cholesky decomposition $\mathbf{K} = \mathbf{L}\mathbf{L}^\top$, compute

$$
\mathbf{f} = \mathbf{L}^\top \mathbf{z} + \mathbf{m},
$$

where $\mathbf{m}$ is the mean vector. This method guarantees that $\mathbf{f}$ has the desired mean and covariance.

## Gaussian Processes and Data: Conditioning on Observations

In practical machine learning problems, we have observed data $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$. The underlying assumption is that the observations are noisy measurements of a latent function:

$$
\mathbf{y}_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2).
$$

Given a GP prior on $f(\mathbf{x})$, the joint distribution over the latent function values $\mathbf{f}$ and the observations $\mathbf{y}$ is

$$
p(\mathbf{f}, \mathbf{y}) = p(\mathbf{f}) \, p(\mathbf{y} \mid \mathbf{f}).
$$

Because both $p(\mathbf{f})$ (the GP prior) and $p(\mathbf{y} \mid \mathbf{f})$ (the Gaussian likelihood) are Gaussian, the posterior $p(\mathbf{f} \mid \mathbf{y})$ remains Gaussian. The resulting posterior has mean

$$
m_{\mid \mathbf{y}}(\mathbf{x}) = k(\mathbf{x}, \mathbf{x})[\mathbf{K}(\mathbf{x}, \mathbf{x}) + \sigma_n^2 \mathbf{I}]^{-1} \mathbf{y},
$$

and covariance

$$
k_{\mid \mathbf{y}}(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}') - k(\mathbf{x}, \mathbf{x})[\mathbf{K}(\mathbf{x}, \mathbf{x}) + \sigma_n^2 \mathbf{I}]^{-1} k(\mathbf{x}, \mathbf{x}').
$$

These expressions illustrate that the prediction at a new input $\mathbf{x}$ is a weighted sum of the training outputs. The weight vector is determined by the covariance between the new input and the training inputs, adjusted for the correlations among training points and observation noise.

The predictive distribution for a new observation $\mathbf{y}^*$ at $\mathbf{x}^*$ is then given by

$$
p(\mathbf{y}^* \mid \mathbf{x}^*, \mathbf{x}, \mathbf{y}) \sim \mathcal{N}\left(\mu(\mathbf{x}^*), \sigma^2(\mathbf{x}^*)\right),
$$

with

$$
\mu(\mathbf{x}^*) = k(\mathbf{x}^*, \mathbf{x})^\top [\mathbf{K}(\mathbf{x}, \mathbf{x}) + \sigma_n^2 \mathbf{I}]^{-1} \mathbf{y},
$$

$$
\sigma^2(\mathbf{x}^*) = k(\mathbf{x}^*, \mathbf{x}^*) + \sigma_n^2 - k(\mathbf{x}^*, \mathbf{x})^\top [\mathbf{K}(\mathbf{x}, \mathbf{x}) + \sigma_n^2 \mathbf{I}]^{-1} k(\mathbf{x}^*, \mathbf{x}).
$$

Here, the predictive mean can be seen as a weighted sum—either directly in terms of the observed outputs or through kernel evaluations—and the predictive variance quantifies the residual uncertainty after accounting for the data.

## Gaussian Process Marginal Likelihood and Hyperparameters

A key aspect of GP modeling is the selection of hyperparameters, which govern the behavior of the covariance function. The marginal likelihood (or evidence) of the observed data under the GP model is obtained by integrating over the latent functions:

$$
p(\mathbf{y} \mid \mathbf{x}) = \int p(\mathbf{y} \mid \mathbf{f}) \, p(\mathbf{f}) \, d\mathbf{f}.
$$

Because both the prior $p(\mathbf{f})$ and the likelihood $p(\mathbf{y} \mid \mathbf{f})$ are Gaussian, the marginal likelihood is also Gaussian:

$$
p(\mathbf{y} \mid \mathbf{x}) = \mathcal{N}(\mathbf{y}; \mathbf{m}, \mathbf{K} + \sigma_n^2 \mathbf{I}).
$$

Taking the logarithm, we obtain the log marginal likelihood:

$$
\log p(\mathbf{y} \mid \mathbf{x}) = -\frac{1}{2} (\mathbf{y} - \mathbf{m})^\top [\mathbf{K} + \sigma_n^2 \mathbf{I}]^{-1} (\mathbf{y} - \mathbf{m}) - \frac{1}{2} \log |\mathbf{K} + \sigma_n^2 \mathbf{I}| - \frac{N}{2} \log(2\pi).
$$

This expression comprises three terms:

1. The data fit term, which is a Mahalanobis distance measuring how well the model explains the data.
2. A complexity penalty term that involves the determinant of the covariance matrix, embodying the principle of Occam’s Razor.
3. A normalization constant.

Maximizing the log marginal likelihood with respect to the hyperparameters (e.g., length-scale $\ell$, signal variance $\sigma_f^2$, and noise variance $\sigma_n^2$) automatically balances model fit and complexity.

## Correspondence Between Linear Models and Gaussian Processes

Gaussian processes and linear models share a deep connection. Consider a linear model defined as

$$
f(\mathbf{x}) = \sum_{m=1}^M w_m \, \phi_m(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x}),
$$

where the weights $w_m$ are given a Gaussian prior, $w_m \sim \mathcal{N}(0, \sigma_w^2)$. The prior over the weights induces a distribution over functions. The mean function of this model is

$$
m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})] = \mathbf{w}^\top \phi(\mathbf{x}) = 0,
$$

assuming a zero-mean prior on $\mathbf{w}$. The covariance function, derived from the weight prior, is

$$
k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \mathbf{A} \, \phi(\mathbf{x}'),
$$

with $\mathbf{A} = \sigma_w^2 \mathbf{I}$ in the simplest case. In this scenario, the inner product $\phi(\mathbf{x})^\top \phi(\mathbf{x}')$ measures the similarity between the features of $\mathbf{x}$ and $\mathbf{x}'$.

Conversely, any GP with a covariance function of the form

$$
k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \mathbf{A} \, \phi(\mathbf{x}')
$$

can be interpreted as arising from a linear model with an appropriate basis function expansion. Mercer's theorem further generalizes this correspondence by showing that many covariance functions correspond to an inner product in an infinite-dimensional feature space. This duality is a cornerstone of kernel methods in machine learning.

## Covariance Functions in Gaussian Processes

The covariance function $k(\mathbf{x}, \mathbf{x}')$ lies at the heart of GP modeling. It encodes assumptions about the smoothness, periodicity, and overall structure of the functions being modeled. Many common choices exist:

### Squared Exponential and Stationary Covariance Functions

One of the most widely used is the squared exponential (or radial basis function) covariance function:

$$
k_{\text{SE}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{(\mathbf{x} - \mathbf{x}')^2}{2\ell^2}\right),
$$

where $\ell$ is the length-scale controlling the function’s smoothness and $\sigma_f^2$ is the signal variance.

### Rational Quadratic Covariance Function

The rational quadratic covariance function is given by

$$
k_{\text{RQ}}(r) = \left(1 + \frac{r^2}{2\alpha \ell^2}\right)^{-\alpha},
$$

with $r = |\mathbf{x} - \mathbf{x}'|$ and $\alpha > 0$. This function can be viewed as a scale mixture of squared exponential functions, allowing for multiple length-scales in the data.

### Matérn Covariance Functions

The Matérn class of covariance functions introduces a parameter $\nu$ that directly controls the smoothness of the sampled functions:

$$
k_\nu(\mathbf{x}, \mathbf{x}') = \frac{1}{\Gamma(\nu) 2^{\nu-1}} \left(\frac{\sqrt{2\nu} \|\mathbf{x} - \mathbf{x}'\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} \|\mathbf{x} - \mathbf{x}'\|}{\ell}\right),
$$

where $K_\nu$ is the modified Bessel function of the second kind. Special cases include:

- $\nu = \frac{1}{2}$: Exponential covariance (leading to non-differentiable functions).
- $\nu = \frac{3}{2}$ and $\nu = \frac{5}{2}$: Functions with increasing degrees of smoothness.
- As $\nu \to \infty$, the Matérn covariance converges to the squared exponential.

### Periodic and Composite Covariance Functions

For modeling periodic phenomena, the periodic covariance function is often used:

$$
k_{\text{per}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{2 \sin^2\left(\pi \frac{|\mathbf{x} - \mathbf{x}'|}{p}\right)}{\ell^2}\right),
$$

where $p$ represents the period. Furthermore, covariance functions can be combined via summation or multiplication to capture complex data structures while retaining the positive definiteness required for valid kernels.

## Finite and Infinite Basis Representations

A central question in probabilistic modeling is whether to use a finite or an infinite number of basis functions. Finite models, such as linear models with a limited set of basis functions, impose strong assumptions and may suffer from underfitting or overfitting in regions with sparse data. In contrast, Gaussian processes represent an infinite model, effectively employing an infinite number of basis functions to capture the underlying function space.

Consider the following representation:

$$
f(\mathbf{x}) = \lim_{N \to \infty} \frac{1}{N} \sum_{n=-N/2}^{N/2} \gamma_n \exp\left(-\left(\mathbf{x} - \frac{n}{N}\right)^2\right), \quad \gamma_n \sim \mathcal{N}(0, 1).
$$

As $N \to \infty$, the summation converges to an integral

$$
f(\mathbf{x}) = \int_{-\infty}^\infty \gamma(u) \exp\left(-(\mathbf{x} - u)^2\right) \, du,
$$

with $\gamma(u) \sim \mathcal{N}(0, 1)$. This continuous formulation underpins the GP framework and shows that the celebrated squared exponential covariance function emerges naturally from an infinite sum of Gaussian basis functions. The resulting GP offers a robust, flexible, and smooth model that generalizes well, even in regions where data is sparse.

In conclusion, Gaussian processes are truly a game-changer in the landscape of machine learning. Their most compelling feature is the ability to quantify uncertainty directly in predictions—a quality that many other models lack. 

Furthermore, Gaussian processes represent a striking synthesis of mathematical elegance and practical adaptability. Their non-parametric nature means that, instead of being confined to a fixed number of parameters, they can flexibly model complex functions by considering an infinite number of basis functions. This adaptability allows GPs to naturally balance between underfitting and overfitting by letting the data inform the complexity of the model through the choice of the kernel and the optimization of hyperparameters. The result is a framework that not only fits the data well but also generalizes gracefully to unseen scenarios.

In essence, Gaussian processes open a window into a more nuanced understanding of learning from data. They shift our perspective from rigid parameter estimation to a fluid, probabilistic approach that mirrors the inherent uncertainties of the real world. This paradigm not only enhances the robustness of our models but also provides a richer narrative of how data informs decisions. As we continue to explore their theoretical depths and practical applications, it becomes clear that GPs are much more than a tool—they embody a powerful philosophy of embracing uncertainty as a pathway to deeper insights and more resilient models.