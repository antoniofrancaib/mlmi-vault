---
#### Question-1+2: <span style="color: green;">Explain the relationship between linear-in-the-parameters models and Gaussian processes</span>  // <span style="color: green;">Explain the relationship between priors over parameters and priors over functions in Bayesian inference</span> 
---
---
- **LIPMs** $\sim$ $f(x) = \sum_{m=1}^{M} w_m \phi_m(x) = \mathbf{w}^\top \boldsymbol{\phi}(x)$ $\rightarrow$ parametric  
- **GPs** $\sim$  $f(x) \sim GP(m(x), k(x, x'))$ $\rightarrow$ non-parametric  

Relationship: 
- **LIPMs $\rightarrow$ GPs**
	- prior $\mathbf{w} \sim \mathcal{N}(0, \mathbf{A})$ + arbitrary set input points $\{x_i\}_{i=1}^{N}$ 

	- $\mathbf{f} = \begin{bmatrix} \mathbf{\phi}(x_1)^T \\ \vdots \\ \mathbf{\phi}(x_n)^T \end{bmatrix} \mathbf{w}  \sim \mathcal{N}(\mathbf{0}, \mathbf{\Phi} \mathbf{A} \mathbf{\Phi}^T)$

	- $f(x) \sim \mathcal{GP}\left(0, \boldsymbol{\phi}(x)^\top \mathbf{A} \boldsymbol{\phi}(x') \right)$ with finite-rank kernel!

- **GPs as Infinite LIPMs**
	- WLOG, consider $f(x) \sim GP(0, k(x, x'))$ 
	
	- Mercer's theorem: $k(x, x') = \sum_{n=1}^\infty \lambda_n \phi_n(x) \phi_n(x').$ (mild conditions)
	
	- $f(x) = \sum_{n=1}^\infty w_n \phi_n(x),$ with $w_n \sim \mathcal{N}(0, \lambda_n).$

- Computational Complexity! **GP:** $\mathcal{O}(N^3)$ *vs* **Linear model:** $\mathcal{O}(NM^2)$

---
#### Question-3: <span style="color: green;">Explain the process and results of marginalizing and conditioning Gaussian processes</span>
--- 
- **GP Prior**: $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$

Marginalization: marginalize $\sim$ sample in practice 
- *Process*: take any finite ${\mathbf{f}} \sim \mathcal{N}(\mathbf{m}, \mathbf{K})$ 
	e.g. confidence Intervals $\mu_*(x_*) \pm 1.96 \sqrt{\text{diag}(\Sigma_*)}$

- *Result*: GP evaluated at our domain of interest! 

Conditioning on data $\approx$ update beliefs:

- $\mathcal{D} = \{x_i, y_i\}_{i=1}^{N}$ with $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, and 

- $\begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{m}(x) \\ \mathbf{m}(x_*) \end{bmatrix}, \begin{bmatrix} \mathbf{K}(x, x) + \sigma_n^2 \mathbf{I} & \mathbf{K}(x, x_*) \\ \mathbf{K}(x_*, x) & \mathbf{K}(x_*, x_*) \end{bmatrix} \right)$

- Result: $p(\mathbf{f}_* \mid \mathbf{y})= \mathcal{N}(\mathbf{m}^{\text{post}}, \mathbf{K}_*^{\text{post}})$ with   

	- $\mathbf{m}^{\text{post}}(x_*) = \mathbf{m}(x_*) + \mathbf{K}(x_*, x) \big(\mathbf{K}(x, x) + \sigma_n^2 \mathbf{I}\big)^{-1} (\mathbf{y} - \mathbf{m}(x))$
	- $\mathbf{K}_*^{\text{post}}(x_*, x_*') = \mathbf{K}(x_*, x_*') - \mathbf{K}(x_*, x) \big(\mathbf{K}(x, x) + \sigma_n^2 \mathbf{I}\big)^{-1} \mathbf{K}(x, x_*')$

<div style="page-break-before: always;"></div>

---
#### Question-4: <span style="color: green;">Discuss the use and role of hyperparameters in Gaussian process covariance functions and give examples</span>
---
- **GPs**: $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$
	- $m(x)$ $\sim$ overall expected trend, or "baseline"
	- $k(x, x')$ $\sim$ shape, smoothness, variability ("wiggle")

- **Ex**: compare them via Marg. Likelihood!
    1. **SE**:  **$k(x, x') = \sigma_f^2 \exp\left(-\dfrac{(x - x')^2}{2\ell^2}\right)$** (how it comes from inf. LIPM!)


    2. **RQ**: **$k(x, x') = \sigma_f^2 \left(1 + \dfrac{(x - x')^2}{2\alpha \ell^2}\right)^{-\alpha}$**

    3. **Periodic**: $k(x, x') = \sigma_f^2 \exp\left(-\dfrac{2\sin^2\left(\dfrac{\pi |x - x'|}{p}\right)}{\ell^2}\right)$
    
    4. combination of any = new cov.! 

---
#### Question-5: <span style="color: green;">Explain the concept of marginal likelihood and its use in Gaussian processes</span> 
---

Marginal Likelihood: EV of likelihood under prior
$\sim$ how well we fit the data 
$\sim$ second level inference: $p(M \mid \mathbf{y}, \mathbf{x}) \propto \text{ML} \times \text{Prior(M)}$ 
- $p(\mathbf{y} \mid \mathbf{X}, M) = \int p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}, M) p(\boldsymbol{\theta} \mid M) d\boldsymbol{\theta}$  $\sim$ parametric!
- $p(\mathbf{y} \mid \mathbf{X}, M) = \int p(\mathbf{y} \mid \mathbf{X}, \mathcal{f}, M) p(\mathcal{f} \mid M) d\mathcal{f}$  $\sim$ non-parametric!

**GPs' Use**: 
- $f(\mathbf{x}) \sim GP(m(\mathbf{x}) \equiv 0, k(\mathbf{x}, \mathbf{x}'))$        +        $y_n = f(\mathbf{x}_n) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$  

- **Expression:**  $p(\mathbf{y} \mid \mathbf{X}, M) = \mathcal{N}(\mathbf{y} \mid \mathbf{0}, \mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I})$  

- **Log-ML:**  $\log p(\mathbf{y} \mid \mathbf{X}) = -\frac{1}{2} \mathbf{y}^\top [\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I}]^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K} + \sigma_{\text{noise}}^2 \mathbf{I}| - \frac{n}{2} \log 2\pi$

	- **Data Fit:** $\sum_{i=1}^n \frac{(\mathbf{u}_i^\top \mathbf{y})^2}{\mu_i}$.  
	- **Complexity:** $\sum_{i=1}^n \log \mu_i$.  
	- **Trade-off**

- **Occam’s Razor**!

<div style="page-break-before: always;"></div>

---
#### Question-6: <span style="color: green;">Explain the properties of and inference in infinitely large linear in the parameters models</span>
---
- LIPM: $f(x) = \sum_{m=1}^{M} w_m \phi_m(x) = \mathbf{w}^\top \boldsymbol{\phi}(x)$ $\sim$ $M$ finite $\rightarrow$ **strong assumptions** 
- Inf. LIPM :  $f(x) = \lim_{M \to \infty} \sum_{m=1}^{M} w_m \phi_m(x)$

Inference: 
- comp. intractable problem in the parameter space! $\mathcal{O}(NM^2)$

- uncorrelated prior $w_m \sim \mathcal{N}(0, \sigma_m^2)$ + arbitrary inputs $\{x_i\}_{i=1}^{N}$ $\rightarrow$ $\mathbf{f}$ is multivariate Gaussian!

- prior $f(x) \sim \mathcal{GP}\left(0, \sum_{m=1}^\infty \sigma_m^2 \phi_m(x) \phi_m(x')\right)$ 

- $\mathcal{D} = \{x_i, y_i\}_{i=1}^{N}$ with $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, and 

- $\begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{m}(x) \\ \mathbf{m}(x_*) \end{bmatrix}, \begin{bmatrix} \mathbf{K}(x, x) + \sigma_n^2 \mathbf{I} & \mathbf{K}(x, x_*) \\ \mathbf{K}(x_*, x) & \mathbf{K}(x_*, x_*) \end{bmatrix} \right)$

- posterior = predictive dist!: $p(\mathbf{f}_* \mid \mathbf{y})= \mathcal{N}(\mathbf{m}^{\text{post}}, \mathbf{K}_*^{\text{post}})$ with   

	- $\mathbf{m}^{\text{post}}(x_*) = \mathbf{m}(x_*) + \mathbf{K}(x_*, x) \big(\mathbf{K}(x, x) + \sigma_n^2 \mathbf{I}\big)^{-1} (\mathbf{y} - \mathbf{m}(x))$
	- $\mathbf{K}_*^{\text{post}}(x_*, x_*') = \mathbf{K}(x_*, x_*') - \mathbf{K}(x_*, x) \big(\mathbf{K}(x, x) + \sigma_n^2 \mathbf{I}\big)^{-1} \mathbf{K}(x, x_*')$

Properties: 
- ✅: Flexibility --- Nonparametric = Interpretable --- Inference in Function Space!
- ❌: $\mathcal{O}(NM^2)$ *vs* $\mathcal{O}(N^3)$.

---
#### Question-7: <span style="color: green;">Explain the Gibbs sampling algorithm and its practical use for inference</span>
---
- MCMC method $\approx$ joint dist. $\rightarrow$ iteratively conditionals 

Steps: 
    1. $x^{(0)} = (x_1^{(0)}, x_2^{(0)}, \dots, x_D^{(0)})$.
    2. $p(x_i \mid x_1^{(t)}, \dots, x_{i-1}^{(t)}, x_{i+1}^{(t-1}, \dots, x_D^{(t-1)})$
    3. Converge to stationary $p(x)$!

Inference: $\mathbb{E}_{\theta \sim p(\theta \mid \mathcal{D})}[g(\theta)] = \int g(\theta) p(\theta \mid \mathcal{D}) \, d\theta$ $\rightarrow$ e.g. mean, variance, prediction...

Example: Gibbs Sampling in TrueSkill 

- $p(y_{ij} = +1) = \int \Phi\left(\frac{w_i - w_j}{\sigma_n}\right)p(w_i, w_j) \, dw_i \, dw_j$ 

- **Approach:** Gibbs + Monte Carlo 
    - Steps:
        1. $p(t_g \mid w_{I_g}, w_{J_g}, y_g) \propto \delta(y_g - \text{sign}(t_g)) \cdot N(t_g \mid w_{I_g} - w_{J_g}, \sigma_n^2)$
        2. $p(w_i \mid t) = N(\mu_{\text{post}}, \Sigma_{\text{post}})$

    - Predict: $p(y_{ij} = +1) \approx \frac{1}{N} \sum_{s=1}^N \Phi\left(\frac{w_i^{(s)} - w_j^{(s)}}{\sigma_n}\right)$

<div style="page-break-before: always;"></div>

---
#### Question-8+9: <span style="color: green;">Explain the semantics and use of factor graphs</span> // **<span style="color: green;">Explain message passing on factor graphs and the sum-product rules</span>

- **FG** $G=(V,F,E)$ $\rightarrow$ bipartite graphical models $\sim$ product structure of a function
    - $V={\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_n}$: variable nodes $\sim$ unknown quantities
    - $F={f_1, f_2, \dots, f_m}$: factor nodes $\sim$ local interactions among variables
    - $E \subseteq V \times F$: edges connecting variables to factors $\sim$ encodes conditional dependencies 

- ex: joint factorization $p(v,w,x,y,z)=f_1(v,w)f_2(w,x)f_3(x,y)f_4(x,z)$ $\sim$ $w$ marginal $\mathcal{O}(K^5)$.

![text](fact-grph.png)

**Message Passing:** computational framework for inference via "messages"!

Sum-product algorithm: 
- $m_{x \to f}(x) = \prod_{f' \in \text{ne}(x) \setminus f} m_{f' \to x}(x) = \frac{p(x)}{m_{f \to x}(x)}$ 

- $m_{f \to t_1}(t_1) = \sum_{t_2} \sum_{t_3} \cdots \sum_{t_n} f(t_1, t_2, \dots, t_n) \prod_{i \neq 1} m_{t_i \to f}(t_i)$

- $p(x) \propto \prod_{f \in \text{ne}(x)} m_{f \to x}(x)$

---
#### Question-10: <span style="color: green;">For factor graphs, explain moment matching approximations and how to approximate inference on graphs with cycles</span>

- **FG** $G=(V,F,E)$ $\rightarrow$ bipartite graphical models $\sim$ product structure of a function
    - $V={\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_n}$: variable nodes $\sim$ unknown quantities
    - $F={f_1, f_2, \dots, f_m}$: factor nodes $\sim$ local interactions among variables
    - $E \subseteq V \times F$: edges connecting variables to factors $\sim$ encodes conditional dependencies 
    
- ex: joint factorization $p(v,w,x,y,z)=f_1(v,w)f_2(w,x)f_3(x,y)f_4(x,z)$ $\sim$ $w$ marginal $\mathcal{O}(K^5)$.

**Message Passing:** computational framework for inference via "messages"!

- **MM Approx**: complex or intractable distributions with Gaussians by matching moments. 

- **Example: updated marginal $p(t)$
    - $p(t) = \frac{1}{Z} \delta(y - \text{sign}(t)) N(t; \mu, \sigma^2)$
    -  $Z = \Phi\left(\frac{y\mu}{\sigma}\right)$
    - **Approximation**: Replace $p(t)$ with Gaussian $q(t)$ matching $E_p[t]$ and $\text{Var}_p[t]$.

**Graphs with Cycles**: same product-rule algorithm with more passes until convergence! 
<div style="page-break-before: always;"></div>

---
#### Question-11:  <span style="color: green;">Explain the beta-binomial conjugate pair and the Dirichlet distribution.</span>
---
Conjugate Prior-Likelihood Pair: tractable inference! 

ex: $p$ of a biased coin landing $H$ $\sim$ Bernoully r.v
- **Beta:** $\text{Beta}(p∣\alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} p^{\alpha-1} (1-p)^{\beta-1}$ $\rightarrow$ continuous on $[0,1]$!
- **Binomial:** $P(k∣n,p) = \binom{n}{k} p^k (1-p)^{n-k}$ $\rightarrow$ nº successes $k$ in $n$ independent Bernoulli trials
- **Posterior**: $p(p∣D) = \text{Beta}(p∣\alpha_{\text{post}},\beta_{\text{post}})$ with 
	- $\alpha_{\text{post}} = \alpha_{\text{prior}} + k$
	- $\beta_{\text{post}} = \beta_{\text{prior}} + n - k$

ex: $\mathbf{p} = [p_1, p_2, \dots, p_6]$ of a biased dice landing a given number $\sim$ Categorical r.v
- **Dirichlet**: $\text{Dir}(p∣\boldsymbol{\alpha}) = \frac{\Gamma\left(\sum_{i=1}^m \alpha_i\right)}{\prod_{i=1}^m \Gamma(\alpha_i)} \prod_{i=1}^m p_i^{\alpha_i-1}$ $\rightarrow$ cont. on simplex $\sum_{i=1}^6 p_i = 1, \quad p_i \geq 0 \quad \forall i$.
- **Multinomial**: $P(\mathbf{k}∣n,\mathbf{p}) = \frac{n!}{k_1!k_2!\dots k_m!} \prod_{i=1}^m p_i^{k_i}$ $\rightarrow$ Binomial generalization to multiple categories!
- **Posterior Distribution:** $p(\mathbf{p}∣\mathbf{k}) = \text{Dir}(\mathbf{p}∣\boldsymbol{\alpha}_{\text{post}})$ with
	- $\alpha_{i,\text{post}} = \alpha_{i,\text{prior}} + k_i$ for each category $i$.

---
#### Question-12: <span style="color: green;">Explain the Expectation Maximisation (EM) algorithm in a model with observations y, latent variables z and parameters θ</span>
---
observations $\{y_i\}_{i=1}^{N}$      +      hidden variables $\{z_j\}_{j=1}^{M}$ $\rightarrow$ Incomplete Data 

**Model:** $p(y|\theta) = \int p(y,z|\theta) dz$  $\sim$ direct optimization difficult! e.g. GMM

Reformulation: 
$$p(z|y,\theta) = \frac{p(y|z,\theta)p(z|\theta)}{p(y|\theta)} \iff p(y|\theta) = \frac{p(y|z,\theta)p(z|\theta)}{p(z|y,\theta)}.$$
$$p(y|\theta) = \frac{p(y|z,\theta)p(z|\theta)}{q(z)} \cdot \frac{q(z)}{p(z|y,\theta)},$$
$$\log p(y|\theta) = \log \frac{p(y|z,\theta)p(z|\theta)}{q(z)} + \log \frac{q(z)}{p(z|y,\theta)},$$
$$\log p(y|\theta) = \int q(z) \log \frac{p(y|z,\theta)p(z|\theta)}{q(z)} dz + \int q(z) \log \frac{q(z)}{p(z|y,\theta)} dz $$
$$\log p(y|\theta) = \mathcal{F}(\mathcal{q}(z), θ) + KL(q(z)\|p(z|y,\theta))$$

Steps:
- **E-Step:** maximize $F(q,\theta^{(t)})$ w.r.t. $q(z)$ $\rightarrow$ $q^{(t+1)}(z) = p(z|y,\theta^{(t)})$

- **M-Step (Maximization):** maximize $F(q^{(t+1)}, \theta)$ w.r.t. $\theta$ $\rightarrow$ $\theta^{(t+1)} = \arg\max_\theta \int q^{(t+1)}(z) \log p(y,z|\theta) dz$

Guarantee! $\log p(y|\theta^{t-1}) \overset{\text{E step}}{=} F(q^t(z), \theta^{t-1}) \overset{\text{M step}}{\leq} F(q^t(z), \theta^t) \overset{\text{lower bound}}{\leq} \log p(y|\theta^t).$

---
#### Question-13+15: <span style="color: green;">Explain the Bayesian mixture of multinomials model, its graphical representation and inference algorithms</span> // <span style="color: green;">Explain the difference between Gibbs sampling and collapsed Gibbs sampling in a mixture of multinomials model</span>

BMM $\sim$ Gen. Model:
![[txt-bmm-scheme.png]]

- **Inference**
	- **Goal**: $p(\{\beta_k\}, \{\theta\}, \{z_{d}\} \mid \{w_{nd}\})$

    - **Gibbs:** 
	    - $p(z_d = k \mid \mathbf{w_d}, \boldsymbol{\theta}, \boldsymbol{\beta}) \propto \theta_k p(\mathbf{w_d} \mid \beta_{z_d}),$
	    
	    - $p(\beta_{km} \mid \mathbf{w}, \mathbf{z}) \propto \text{Dir}(\gamma + \mathbf{c}_{km})$ 
        
        - $p(\boldsymbol{\theta_k} \mid \mathbf{z}, \boldsymbol{\alpha}) \propto \text{Dir}(\boldsymbol{\alpha} + \mathbf{c_k}),$            

    - **Collapsed:** 
	    1. $p(z_d = k \mid \mathbf{w_d}, \mathbf{z}_{-d}, \boldsymbol{\beta}, \boldsymbol{\alpha}) \propto p(\mathbf{w_d} \mid \beta_k) \frac{\alpha + c_{-d,k}}{\sum_{j=1}^K (\alpha + c_{-d,j})}.$
	
        2. $p(\beta_{km} \mid \mathbf{w}, \mathbf{z}) \propto \text{Dir}(\gamma + \mathbf{c}_{km})$ 

    
---
#### Question-14: <span style="color: green;">Explain the Latent Dirichlet Allocation model, its graphical representation and inference algorithms</span>

LDA $\sim$ Gen. Model:
![[txt-lda-scheme.png]]

Inference: 
- **Goal**: $p(\{\beta_k\}, \{\theta_d\}, \{z_{dn}\} \mid \{w_{dn}\}, \alpha, \gamma)$ $\rightarrow$ many $\{z_{nd}\}$ combinations, if only we knew the $\{z_{nd}\}$?

- **Collapsed Gibbs**: 
$$\begin{align}
p(z_{nd} = k \mid \{z_{-nd}\}, \{\mathbf{w}\}, \gamma, \boldsymbol{\alpha}) &\propto p(z_{nd} = k \mid \{z_{-nd}\}, \boldsymbol{\alpha}) \, p(w_{nd} \mid z_{nd} = k, \{\mathbf{w}_{-nd}\}, \{z_{-nd}\}, \gamma), \\
&\propto \frac{\alpha + c_{-nd}^k}{\sum_{j=1}^K (\alpha + c_{-nd}^j)} \cdot \frac{\gamma + \tilde{c}_{-m}^k}{\sum_{m=1}^M (\gamma + \tilde{c}_{-m}^k)}.
\end{align}
$$
- **EV estimation:** 
$$\hat{\theta}_{dk} = \frac{c_k^d + \alpha}{\sum_{j=1}^K \left(c_j^d + \alpha \right)}, \quad
\hat{\beta}_{km} = \frac{\tilde{c}_k^m + \gamma}{\sum_{n=1}^M \left(\tilde{c}_k^n + \gamma \right)}.$$


**Perplexity Minimization**: Choose $K$ with lowest held-out perplexity! $\sim$ $$\exp \left( -\frac{\sum_{d=1}^D \sum_{n=1}^{N_d} \log \left( \sum_{k=1}^K \hat{\theta}_{dk} \hat{\beta}_{kw_{nd}} \right)}{\sum_{d=1}^D N_d} \right).$$
