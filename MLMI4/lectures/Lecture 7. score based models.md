Lecture by Aliaksandra Shysheya

# Score-Based Generative Models: Masterclass Notes

## 1. Introduction

Generative modeling aims to learn the underlying data distribution $p(\mathbf{x})$ from a set of training samples $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$ so that new samples can be generated. In score-based generative models, the key idea is to learn the score—the gradient of the log density—which bypasses the need to compute the often intractable normalization constant.

## 2. Energy-Based Models (EBMs)

### 2.1 The Formulation

An energy-based model defines a probability density in terms of an energy function $U_\theta(\mathbf{x})$:

$$
p_\theta(\mathbf{x}) = \frac{\exp(-U_\theta(\mathbf{x}))}{Z_\theta} \quad \text{with} \quad Z_\theta = \int \exp(-U_\theta(\mathbf{x})) \, d\mathbf{x}.
$$

Here, $U_\theta: \mathbb{R}^d \to \mathbb{R}^+$ is parameterized by $\theta$ and the partition function $Z_\theta$ normalizes the density.

### 2.2 Challenges in Training

**Log-Likelihood Maximization:** Training via maximum likelihood involves maximizing

$$
\sum_{i=1}^N \log p_\theta(\mathbf{x}_i) = -\sum_{i=1}^N U_\theta(\mathbf{x}_i) - N \log Z_\theta.
$$

**Intractable Partition Function:** In practice, computing $Z_\theta$ (an integral over $\mathbb{R}^d$) is intractable, complicating both training and sampling.

This challenge motivates an alternative approach—modeling the score directly.

## 3. Score-Based Models

### 3.1 The Score Function

The score is defined as the gradient of the log density:

$$
\mathbf{s}_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x}).
$$

For an EBM, note that

$$
\nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} U_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log Z_\theta,
$$

and since $\nabla_\mathbf{x} \log Z_\theta = 0$ (as it is independent of $\mathbf{x}$), we have

$$
\mathbf{s}_\theta(\mathbf{x}) = -\nabla_\mathbf{x} U_\theta(\mathbf{x}).
$$

This formulation lets us sidestep the partition function issue and instead focus on learning a function that provides the local "direction" of increasing probability density.

### 3.2 Training via Score Matching

In an ideal world, we would train the score network by minimizing the discrepancy

$$
\mathcal{L}_{\text{ESM}}(\mathbf{s}_\theta) = \mathbb{E}_{p(\mathbf{x})}\left[\|\nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x})\|^2\right],
$$

known as the Fisher divergence. However, the true score $\nabla_\mathbf{x} \log p(\mathbf{x})$ is not directly available, which leads us to alternative approaches:

- **Implicit Score Matching (ISM)**
- **Sliced Score Matching (SSM)**

Both methods rely solely on the raw data while avoiding the explicit computation of the true score.

## 4. Sampling via Langevin Dynamics

Once a score network is trained, we can generate samples using Langevin dynamics. The key insight is that if one can compute or approximate the score $\nabla_\mathbf{x} \log p_\theta(\mathbf{x})$, then one can simulate a stochastic differential equation (SDE) whose stationary distribution is $p_\theta(\mathbf{x})$.

### 4.1 The Langevin SDE

Consider the SDE:

$$
d\mathbf{X}_t = \nabla_\mathbf{x} \log p_\theta(\mathbf{X}_t) \, dt + \sqrt{2} \, d\mathbf{W}_t,
$$

where $\mathbf{W}_t$ is a Brownian motion. In the limit $t \to \infty$, the distribution of $\mathbf{X}_t$ becomes proportional to $p_\theta(\mathbf{x})$.

### 4.2 Discretisation and Intuition

Discretizing time with a small step $\gamma$ leads to the iterative update:

$$
\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla_\mathbf{x} \log p_\theta(\mathbf{X}_k) + \sqrt{2\gamma} \, \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(0, \mathbf{I}).
$$

This can be interpreted as:

- **Gradient Ascent:** The term $\gamma \nabla_\mathbf{x} \log p_\theta(\mathbf{X}_k)$ drives the sample toward regions of higher probability.
- **Noise Injection:** The Gaussian noise $\sqrt{2\gamma} \, \mathbf{z}_k$ ensures exploration and helps avoid getting trapped in local modes.

## 5. Training the Score Network

### 5.1 The Challenge

The ideal objective would be to directly match $\mathbf{s}_\theta(\mathbf{x})$ to the unknown true score $\nabla_\mathbf{x} \log p(\mathbf{x})$. Instead, we approximate this match using a score matching objective based on the Fisher divergence. In practice, we often use variants such as:

- **Implicit Score Matching (ISM)**
- **Sliced Score Matching (SSM)**

Both these methods reformulate the loss so that it only involves the score network $\mathbf{s}_\theta(\mathbf{x})$ and the available data $\mathbf{x}$.

### 5.2 Denoising Score Matching (DSM)

An important development is to smooth the data distribution by adding noise. By defining a noising process:

$$
p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}} \mid \mathbf{x}, \sigma^2 \mathbf{I}),
$$

we form a noised distribution $p_\sigma(\tilde{\mathbf{x}}) = \int p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x}$. The denoising score matching objective then becomes

$$
\mathcal{L}_{\text{DSM}}(\mathbf{s}_\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}} \sim p_\sigma(\tilde{\mathbf{x}}, \mathbf{x})}\left[\lambda(\sigma) \|\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) - \mathbf{s}_\theta(\tilde{\mathbf{x}})\|^2\right],
$$

where $\lambda(\sigma)$ is a weighting function. Intuitively, this forces the network to learn how to "denoise" a perturbed sample, thereby capturing the local structure of the data.

## 6. Multiple Noise Scales and Score Matching Langevin Dynamics (SMLD)

### 6.1 Why Multiple Scales?

Adding noise improves score estimation outside the data support, but a single noise level may not be sufficient. By introducing multiple noise levels $\sigma_0 < \sigma_1 < \dots < \sigma_T$, we obtain a sequence of noised distributions:

$$
p_{\sigma_t}(\tilde{\mathbf{x}}) = \int p_{\sigma_t}(\tilde{\mathbf{x}} \mid \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x}.
$$

### 6.2 SMLD Objective

A score network is then parameterized as $\mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma_t)$ and trained using the SMLD loss:

$$
\mathcal{L}_{\text{SMLD}}(\mathbf{s}_\theta) = \mathbb{E}_{t, \mathbf{x}, \tilde{\mathbf{x}} \sim p_{\sigma_t}(\tilde{\mathbf{x}}, \mathbf{x})} 
\left[\lambda(t) \|\nabla_{\tilde{\mathbf{x}}} \log p_{\sigma_t}(\tilde{\mathbf{x}} \mid \mathbf{x}) - \mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma_t)\|^2\right].
$$

This multi-scale approach improves score estimation over a wide range of noise levels.

### 6.3 Sampling: Annealed Langevin Dynamics

Sampling is performed by an annealing process:

1. **Start at high noise:** Begin with a large $\sigma_T$ where the density is smooth.
2. **Gradually reduce noise:** Use the sample from the previous noise level to warm-start sampling at the next lower noise level.
3. **Iterate:** Continue until reaching a very small noise level ($\sigma_0$) so that $p_{\sigma_0} \approx p$.

This sequence of steps helps the sampler traverse from a simple distribution to the complex target distribution.

## 7. Comparison: SMLD vs. DDPM in 6 Steps

Both SMLD and Denoising Diffusion Probabilistic Models (DDPM) share a common noising process and use noise-conditional score networks, but differ in parameterization and loss formulation. For example:

- **Noising Process:**
  - SMLD uses: $\tilde{\mathbf{x}} = \mathbf{x} + \sigma_t \epsilon$ with $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.
  - DDPM often defines a similar process with time-dependent coefficients.
- **Parameterization and Loss:**
  - SMLD directly parameterizes the score $\mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma_t)$.
  - DDPM may predict the noise (or even the original $\mathbf{x}$) and scale the loss appropriately.

In essence, the two frameworks are closely related—differing mainly in the precise discretization and weighting choices.

## 8. Continuous Score-Based Models: ODEs and SDEs

Moving from discrete noise scales to a continuous formulation provides additional theoretical and practical insights.

### 8.1 ODEs vs. SDEs

**Ordinary Differential Equations (ODEs):**
Describe dynamics with a deterministic drift term:

$$
d\mathbf{X}_t = \mathbf{b}(t, \mathbf{X}_t) \, dt.
$$

**Stochastic Differential Equations (SDEs):**
Incorporate both drift and diffusion:

$$
d\mathbf{X}_t = \mathbf{b}(t, \mathbf{X}_t) \, dt + \sigma(t, \mathbf{X}_t) \, d\mathbf{W}_t.
$$

The added noise term allows the process to explore the state space stochastically.

### 8.2 Connection to Discrete Models

Using an Euler–Maruyama discretization, the SDE

$$
d\mathbf{X}_t = \mathbf{b}(t, \mathbf{X}_t) dt + \sigma(t, \mathbf{X}_t) d\mathbf{W}_t
$$

becomes

$$
\mathbf{X}_{t+\Delta t} \approx \mathbf{X}_t + \Delta t \, \mathbf{b}(t, \mathbf{X}_t) + \sqrt{\Delta t} \, \sigma(t, \mathbf{X}_t) \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(0, \mathbf{I}).
$$

This formulation is equivalent in form to the transition kernels used in discrete score-based models (e.g., SMLD, DDPM).

### 8.3 Time Reversal of SDEs

A remarkable theoretical insight is that if one knows the score of the intermediate distributions, the forward SDE can be reversed. Under mild conditions (see, e.g., Cattiaux et al., 2021), the time-reversed process satisfies:

$$
d\mathbf{Y}_t = \left[-\mathbf{b}(T-t, \mathbf{Y}_t) + \frac{1}{2} \sigma^2(T-t) \nabla \log p_{T-t}(\mathbf{Y}_t)\right] dt + \sigma(T-t) \, d\tilde{\mathbf{W}}_t,
$$

with $\mathbf{Y}_0 \sim p_T$. Training the score network in continuous time then proceeds with a loss analogous to denoising score matching but indexed by continuous time.

## 9. Probability Flow ODE

An alternative to the stochastic reverse SDE is to construct an ODE—known as the probability flow ODE—that shares the same marginal distributions as the forward process:

$$
d\mathbf{X}_t = \left[\mathbf{b}(t, \mathbf{X}_t) - \frac{1}{2} \sigma^2(t) \nabla \log p_t(\mathbf{X}_t)\right] dt.
$$

This deterministic formulation has advantages (e.g., using advanced ODE solvers) but may accumulate discretization errors without the corrective effect of noise.

## 10. Diffusion Guidance

For many applications, we wish to generate samples conditioned on additional information $\mathbf{y}$ (for example, class labels or text prompts).

### 10.1 Conditional Score

Using Bayes' rule, the conditional score can be written as:

$$
\nabla_{\mathbf{X}_t} \log p(\mathbf{X}_t \mid \mathbf{y}) = \nabla_{\mathbf{X}_t} \log p(\mathbf{X}_t) + \nabla_{\mathbf{X}_t} \log p(\mathbf{y} \mid \mathbf{X}_t).
$$

In practice, the second term is obtained via a classifier or a discriminative model that estimates $p(\mathbf{y} \mid \mathbf{X}_t)$.

### 10.2 Guidance Methods

**Classifier Guidance:**
Multiply the conditioning gradient by a scaling factor $\gamma > 1$ to steer the sample generation:

$$
\nabla_{\mathbf{X}_t} \log p_\gamma(\mathbf{X}_t \mid \mathbf{y}) = \nabla_{\mathbf{X}_t} \log p(\mathbf{X}_t) + \gamma \nabla_{\mathbf{X}_t} \log p(\mathbf{y} \mid \mathbf{X}_t).
$$

A separate classifier must be trained, and care must be taken because the gradient of the classifier can sometimes yield adversarial directions.

**Classifier-Free Guidance:**
Instead of a separate classifier, train the model with conditioning dropout. Then, one can mix the conditional and unconditional scores:

$$
\nabla_{\mathbf{X}_t} \log p_\gamma(\mathbf{X}_t \mid \mathbf{y}) = (1 - \gamma) \nabla_{\mathbf{X}_t} \log p(\mathbf{X}_t) + \gamma \nabla_{\mathbf{X}_t} \log p(\mathbf{X}_t \mid \mathbf{y}).
$$

This approach avoids the need for a separate classifier while still leveraging conditional information.

## 11. Practical "Tricks" for Diffusion Models

Several practical techniques have been developed to improve the performance of diffusion models:

- **Parameterization Choices:**
  - Predict the noise (as in DDPM) or the clean data $\mathbf{X}_0$.
  - Certain parameterizations can be made equivalent by appropriately choosing loss weightings.
- **Loss Weighting $w(t)$:**
  - Adjusts the importance of different noise levels during training.
- **Timestep Distribution $p(t)$:**
  - Although uniform sampling is common, alternative distributions can sometimes yield better performance.
- **Process Truncation:**
  - As $t \to 0$ the score can diverge (since $\sigma_t \to 0$). Truncating the process (i.e., not going all the way to zero noise) helps stabilize training and sampling.
- **Exponential Moving Average (EMA):**
  - Using an EMA of the parameters during training reduces the high variance associated with the Monte Carlo estimation of the loss.

## 12. Mathematical Foundations and Proof Sketches

While the above sections summarize the main ideas, several derivations underpin the methodology:

- **Implicit and Sliced Score Matching:**
  - The ISM objective can be rewritten using integration by parts (often invoking the divergence theorem) so that it no longer depends explicitly on the true score. Sliced score matching further reduces computational complexity by projecting onto random directions.
- **Denoising Score Matching:**
  - One shows that the DSM objective is equivalent (up to a constant) to explicit score matching on the noised data distribution. This connection is made by integrating over the joint distribution $p(\mathbf{x}, \tilde{\mathbf{x}}})$ and applying properties of the Gaussian noising process.
- **Discretised vs. Continuous Formulations:**
  - By comparing the Euler–Maruyama discretization of the SDE with the transition kernels used in discrete models (such as DDPM), one sees that these methods are different discretizations of the same underlying continuous process.

A careful study of these proofs reinforces the intuition that learning the score—and thus the local geometry of the data distribution—suffices for both generation and likelihood estimation.

## 13. Concluding Remarks

Score-based generative models provide a powerful framework that:

- Circumvents the intractable partition function in energy-based models by focusing on gradients.
- Enables sample generation via Langevin dynamics or continuous reverse-time SDEs/ODEs.
- Can be adapted to conditional generation via classifier guidance or classifier-free techniques.
- Benefits from a number of practical improvements (multi-scale noise, EMA, proper loss weighting) to yield high-quality samples.

Together, these ideas form the state of the art in generative modeling and open many avenues for both theoretical analysis and practical applications.