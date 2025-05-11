# Denoising Diffusion Probabilistic Models (DDPMs)

Denoising Diffusion Probabilistic Models (DDPMs) have emerged as one of the most effective frameworks in generative modeling, capable of producing remarkably realistic images, videos, and even intricate scientific data. In this post, we'll unpack the mathematical foundations of DDPMs, offering clear explanations combined with rigorous derivations and equations.

### Step 1: The Noising Process - Data Augmentation

DDPMs convert a challenging unsupervised generative modeling task into a simpler sequence of supervised regression problems. This is done by progressively adding Gaussian noise to clean data points. Starting from a data point $x^{(0)} \sim q(x^{(0)})$, we iteratively add noise:

$$
x^{(t)} = \lambda_t x^{(t-1)} + \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)
$$

To maintain a stable variance across fidelity levels, we set $\sigma_t^2 = 1 - \lambda_t^2$. This results in:

$$
q(x^{(t)}|x^{(0)}) = \mathcal{N}\left(x^{(t)}; \left(\prod_{i=1}^t \lambda_i\right) x^{(0)},\, 1 - \prod_{i=1}^t \lambda_i^2\right)
$$

This step is crucial since it creates a structured pathway from simple distributions (highly noisy data) back to the complex data distribution.

---

### Step 2: Regression Objective - Maximum Likelihood

Each intermediate step in the diffusion model forms a supervised regression problem. The optimal parameters $\theta$ at each fidelity level $t$ are found by maximizing likelihood:

$$
\theta^*_{t-1} = \arg\max_{\theta_{t-1}} \mathbb{E}_{q(x^{(t-1)}, x^{(t)})}\left[\log p(x^{(t-1)}|x^{(t)}, \theta_{t-1})\right]
$$

For Gaussian regressions, this equates to minimizing the squared difference between predicted and actual means, thus connecting to the classical least squares framework:

$$
\mathcal{L}_{t-1}(\theta) = -\frac{1}{2}\mathbb{E}_{q(x^{(t-1)}, x^{(t)})}\left[\frac{(x^{(t-1)} - \mu_\theta(x^{(t)}, t-1))^2}{\sigma^2_\theta(x^{(t)}, t-1)} + \log(2\pi\sigma^2_\theta(x^{(t)}, t-1))\right]
$$

---

### Step 3: Parameter Sharing and Efficient Learning

To avoid parameter explosion across numerous fidelity levels, DDPMs use parameter sharing. This involves parameterizing a neural network to condition explicitly on the fidelity level $t$:

$$
\mu_\theta(x^{(t)}, t-1) = \text{NeuralNet}(x^{(t)}, t; \theta)
$$

This shared parameterization improves training efficiency and generalization, as adjacent fidelity levels represent closely related regression tasks.

---

### Step 4: Gaussian Model and Variance Reduction

Selecting a Gaussian regression model simplifies both the learning and inference phases. It also allows analytical evaluation of some integrals, significantly reducing the training variance:

$$
q(x^{(t-1)}|x^{(0)}, x^{(t)}) = \mathcal{N}(x^{(t-1)}; \mu_{t-1|0,t}, \sigma^2_{t-1|0,t})
$$

where:

$$
\mu_{t-1|0,t} = a^{(t-1)}x^{(0)} + b^{(t-1)}x^{(t)}, \quad \sigma^2_{t-1|0,t} = \frac{(1 - \prod_{i=1}^{t-1}\lambda_i^2)(1-\lambda_t^2)}{1 - \prod_{i=1}^t\lambda_i^2}
$$

This analytic step significantly reduces Monte Carlo variance in the optimization.

---

### Step 5: Model Parameterization - Predicting Clean Data vs Noise

DDPMs commonly use two parameterizations:

**Predicting the Clean Data:**
$$
\mu_\theta(x^{(t)}, t-1) = a^{(t-1)}\hat{x}^{(0)}_\theta(x^{(t)}, t-1) + b^{(t-1)}x^{(t)}
$$

**Predicting the Noise:**
$$
\mu_\theta(x^{(t)}, t-1) = \frac{a^{(t-1)}}{c^{(t)}}\left(x^{(t)} - d^{(t)}\hat{\epsilon}_\theta(x^{(t)}, t-1)\right) + b^{(t-1)}x^{(t)}
$$

This choice affects optimization and interpretation: predicting noise connects closely to denoising score matching, whereas predicting clean data links to denoising autoencoders.

---

### Step 6: Optimal Noise Scheduling and Signal-to-Noise Ratio (SNR)

The choice of noise scheduling ($\lambda_t$) significantly affects DDPM performance. A useful theoretical tool is the Signal-to-Noise Ratio (SNR), defined as:

$$
\text{SNR}(t) = \frac{\prod_{i=1}^{t}\lambda_i^2}{1-\prod_{i=1}^{t}\lambda_i^2}
$$

The training loss can then be elegantly reparameterized as:

$$
\mathcal{L}(\theta) \approx -\frac{1}{2}\int_{\text{SNR}_\text{min}}^{\text{SNR}_\text{max}} \mathbb{E}\left[(x^{(0)} - \hat{x}^{(0)}_\theta(x^{(u)}, u))^2\right] du
$$

This formulation reveals that only boundary SNR values significantly affect model performance, guiding efficient noise scheduling choices.

---

## Connections to Related Models

DDPMs bridge various modeling approaches:

- **Auto-Regressive Models:** DDPMs autoregress over fidelity levels.
- **Denoising Autoencoders:** DDPMs extend these by modeling intermediate fidelity levels explicitly.
- **Score Matching:** Predicting noise is akin to score-based generative modeling.
- **Variational Autoencoders (VAE):** DDPMs optimize an Evidence Lower Bound (ELBO), though often with fixed approximate posteriors.

---

## Concluding Remarks

DDPMs elegantly blend mathematical theory and practical utility, transforming complex generative tasks into manageable regression problems. By deeply understanding their mathematical underpinnings—from noising schedules to analytic variance reductions—we gain powerful intuitions and practical tools for leveraging DDPMs across diverse applications.

---
---

# Score-based Generative Models

## Introduction: Generative Modeling
Generative modeling aims at learning the probability distribution $p(x)$ of observed training data $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$, enabling the generation of new data points. Score-based generative models specifically approach this challenge by modeling the gradient of the log probability density, known as the **score function**:

$$
s_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x})
$$

## 1. Energy-based Models (EBMs)

An Energy-based Model defines a probability distribution via an energy function $U_\theta(\mathbf{x})$:

$$
p_\theta(\mathbf{x}) = \frac{\exp(-U_\theta(\mathbf{x}))}{Z_\theta}, \quad \text{with } Z_\theta = \int \exp(-U_\theta(\mathbf{x}))\,d\mathbf{x}
$$

Here, $Z_\theta$ is the partition function, typically intractable to compute due to the integral over all possible states.

### Challenges in EBMs
- Computing the partition function $Z_\theta$ is often intractable.
- Direct sampling from EBMs is computationally demanding.

## 2. Transition to Score-based Models

Score-based models circumvent the partition function by directly modeling the score function:

$$
s_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} U_\theta(\mathbf{x})
$$

## 2.1 Langevin Dynamics: Sampling via Scores

### The Langevin SDE
Given the score, Langevin dynamics allow sampling through a stochastic differential equation (SDE):

$$
d\mathbf{X}_t = \nabla_{\mathbf{X}_t} \log p_\theta(\mathbf{X}_t) dt + \sqrt{2}\,d\mathbf{W}_t
$$
where $\mathbf{W}_t$ represents standard Brownian motion.

### Euler–Maruyama Discretization
Practically, we approximate this continuous SDE with discrete updates:

$$
\mathbf{X}_{k+1} = \mathbf{X}_k + \gamma \nabla_{\mathbf{X}_k}\log p_\theta(\mathbf{X}_k) + \sqrt{2\gamma}\, \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(0, \mathbf{I})
$$

This iterative procedure balances deterministic gradient ascent and stochastic exploration.

## 2.2 Score Matching: Training the Score Network

### Explicit Score Matching (ESM)
Direct minimization of the Fisher divergence:

$$
\mathcal{L}_{ESM}(s_\theta) = \mathbb{E}_{p(\mathbf{x})}[\| \nabla_\mathbf{x} \log p(\mathbf{x}) - s_\theta(\mathbf{x}) \|^2]
$$

But the true score $\nabla_\mathbf{x} \log p(\mathbf{x})$ is unknown, thus impractical.

### Implicit Score Matching (ISM)
Introduced by Hyvärinen (2005), it avoids the true score by reformulating the loss:

$$
\mathcal{L}_{ISM}(s_\theta) = \mathbb{E}_{p(\mathbf{x})}\left[\nabla_\mathbf{x} \cdot s_\theta(\mathbf{x}) + \frac{1}{2}\|s_\theta(\mathbf{x})\|^2\right] + \text{const}
$$

### Sliced Score Matching (SSM)
A more scalable alternative that leverages random projections:

$$
\mathcal{L}_{SSM}(s_\theta) = \mathbb{E}_{p(\mathbf{x}),p(\mathbf{v})}\left[\left(\mathbf{v}^T \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{v}^T s_\theta(\mathbf{x})\right)^2\right]
$$

## 2.3 Denoising Score Matching (DSM)

DSM simplifies training by adding Gaussian noise:

$$
p_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x}, \sigma^2\mathbf{I})
$$

The optimization objective becomes a denoising problem:

$$
\mathcal{L}_{DSM}(s_\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\left[\Big\|\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}-s_\theta(\tilde{\mathbf{x}})\Big\|^2\right]
$$

DSM effectively transforms score estimation into supervised learning of denoising.

### Multiple Noise Perturbations
Using multiple noise scales $(\sigma_0<\dots<\sigma_T)$ refines score estimation:

- Annealed Langevin Dynamics gradually reduces noise to enhance sample quality.

## 3. Continuous Score-based Models

### Continuous-time Diffusion Models (SDEs)
Continuous models define the noising process via an SDE:

$$
d\mathbf{X}_t = \mathbf{b}(t,\mathbf{X}_t)dt + \sigma(t)d\mathbf{W}_t
$$

### Reverse-time (Denoising) SDE
Reversing the forward diffusion SDE gives:

$$
d\mathbf{Y}_t = [-\mathbf{b}(T-t,\mathbf{Y}_t)+\sigma(T-t)^2\nabla\log p_{T-t}(\mathbf{Y}_t)]dt + \sigma(T-t)d\mathbf{W}_t
$$

### Probability Flow ODE
An associated deterministic ODE, sharing marginals with the SDE:

$$
d\mathbf{X}_t = \left[\mathbf{b}(t,\mathbf{X}_t)-\frac{1}{2}\sigma(t)^2\nabla\log p_t(\mathbf{X}_t)\right]dt
$$

- ODE sampling is deterministic and numerically stable.
- SDE sampling includes stochastic correction, improving diversity.

## 4. Conditional and Guided Sampling

Conditional sampling from $p(\mathbf{X}|\mathbf{y})$:

- **Classifier Guidance:**

$$
\nabla \log p(\mathbf{X}_t|\mathbf{y})=\nabla\log p(\mathbf{X}_t)+\gamma\nabla\log p(\mathbf{y}|\mathbf{X}_t)
$$

- **Classifier-Free Guidance:** avoids separate classifiers by training conditional models with occasional dropout of conditioning information:

$$
\nabla\log p_\gamma(\mathbf{X}_t|\mathbf{y})=(1-\gamma)\nabla\log p(\mathbf{X}_t)+\gamma\nabla\log p(\mathbf{X}_t|\mathbf{y})
$$

## Practical Considerations
- **Loss Weighting $w(t)$**: Adjusts the importance of different noise scales.
- **Process Truncation**: Stabilizes training by limiting minimum noise scale.
- **Exponential Moving Average (EMA)**: Smooths parameter updates for robustness.

## Conclusion: Key Insights
1. Score-based models eliminate the intractable normalization constant problem.
2. Langevin dynamics leverage score functions for effective sampling.
3. Continuous SDE/ODE frameworks offer flexible modeling and stable sampling.
4. Conditional generation techniques enhance generative control and applicability.

These models and methods provide a powerful mathematical framework for understanding and implementing state-of-the-art generative models.