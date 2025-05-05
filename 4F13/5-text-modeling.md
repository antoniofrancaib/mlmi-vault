# Index 

- [33-Introduction-to-Text-Modeling](#33-Introduction-to-Text-Modeling)
- [34-Discrete-Binary-Distributions](#34-Discrete-Binary-Distributions)
- [35-Discrete-Categorical-Distribution](#35-Discrete-Categorical-Distribution)
- [36-Document-Models ](#36-Document-Models)
- [37-Expectation-Maximization-Algorithm ](#37-Expectation-Maximization-Algorithm)
- [38-Gibbs Sampling for Bayesian Mixture Models  ](#38-Gibbs-Sampling-for-Bayesian-Mixture-Models )
- [39-Latent-Dirichlet-Allocation-(LDA)-for-Topic-Modeling ](#39-Latent-Dirichlet-Allocation-(LDA)-for-Topic-Modeling)


# 33-Introduction-to-Text-Modeling

Text modeling focuses on representing and analyzing textual data using probabilistic models. Understanding text modeling is crucial for tasks such as document classification, topic modeling, and information retrieval.

## Key Concepts
- **Modeling Document Collections**: Approaches to represent and analyze large sets of documents.
- **Probabilistic Models of Text**: Statistical methods to model the generation of text data.
- **Zipf's Law**: An empirical law describing the frequency distribution of words in natural language.
- **Bag-of-Words Representations**: A simplified representation of text that disregards grammar and word order but keeps multiplicity.

## Modeling Text Documents

### Why Model Text Documents?
Text documents are rich sources of information but are inherently unstructured. Modeling text allows us to:
- **Extract Meaningful Patterns**: Identify topics, sentiments, or authorship.
- **Compress Information**: Represent documents in a lower-dimensional space.
- **Facilitate Search and Retrieval**: Improve the efficiency of information retrieval systems.
- **Enable Automatic Categorization**: Classify documents into predefined or discovered categories.

### How to Model a Text Document?
One common approach is to represent a document by the frequency of occurrence of each distinct word it contains. This is known as the bag-of-words model.

- **Bag-of-Words Model**: Represents text as an unordered collection of words, disregarding grammar and word order.
- **Word Counts**: The frequency with which each word appears in the document.

## Word Counts in Text

![[text-data.png]]

### Word Frequency Analysis
By analyzing word frequencies, we observe that for different text collection, similar behaviours:
- **High-Frequency Words**: A small number of words occur very frequently.
- **Low-Frequency Words**: A large number of words occur infrequently.

This phenomenon is described by Zipf's Law.

## Zipf's Law

Zipf's Law states that in a given corpus of natural language, the frequency of any word is inversely proportional to its rank in the frequency table.

Mathematically:
$$f(r) \propto \frac{1}{r}$$

- $f(r)$: Frequency of the word with rank $r$.
- $r$: Rank of the word when ordered by decreasing frequency.

![[text-law.png]]

- The x-axis is the cumulative fraction of distinct words included (from the most frequent to the least frequent).
- The y-axis is the **frequency of the current word** at that point in the ranking.
### Observations from the Datasets
- **Frequency Distribution**: When plotting word frequencies against their ranks on a log-log scale, we obtain a roughly straight line, indicating a power-law distribution.
- **Cumulative Fraction**: A small fraction of the most frequent words accounts for a significant portion of the total word count.

## Automatic Categorization of Documents

**Goal**: Develop an unsupervised learning system to automatically categorize documents based on their content without prior knowledge of categories.

### Challenges
- **Unsupervised Learning**: No labeled data to guide the categorization.
- **Unknown Categories**: The system must discover categories from the data.
- **Definition of Categories**: Need to define what it means for a document to belong to a category.

### Approaches
- **Clustering**: Group documents based on similarity in word distributions.
- **Topic Modeling**: Use probabilistic models to discover latent topics (e.g., Latent Dirichlet Allocation).

--- 
# 34-Discrete-Binary-Distributions

Discrete binary distributions are fundamental in modeling binary outcomes, such as coin tosses or binary features in text data.

### Coin Tossing
- **Question**: Given a coin, what is the probability $p$ of getting heads?
- **Challenge**: Estimating $p$ based on observed data (coin toss outcomes).

#### Maximum Likelihood Estimation (MLE)
- **Single Observation**: If we observe one head ($H$), MLE suggests $p=1$.
- **Limitations**: With limited data, MLE can give extreme estimates.

#### Need for More Data
- **Additional Observations**: Suppose we observe $HHTH$.
- **MLE Estimate**: $p=\frac{3}{4}$.
- **Intuition**: Estimates become more reliable with more data.

### Bernoulli Distribution

The Bernoulli distribution models a single binary trial.

- **Random Variable**: $X \in \{0,1\}$.
  - $X=1$ represents success (e.g., heads).
  - $X=0$ represents failure (e.g., tails).
- **Parameter**: $p$, the probability of success.

#### Probability Mass Function (PMF)
$$P(X=x \mid p) = p^x (1-p)^{1-x}$$

- For $x=1$: $P(X=1 \mid p) = p$.
- For $x=0$: $P(X=0 \mid p) = 1-p$.

#### Maximum Likelihood Estimation
Given data $D = \{x_1, x_2, \dots, x_n\}$:

- **Likelihood Function**:
  $$L(p) = \prod_{i=1}^{n} p^{x_i} (1-p)^{1-x_i}$$

- **Log-Likelihood**:
  $$\ell(p) = \sum_{i=1}^{n} \left[x_i \log p + (1-x_i) \log (1-p)\right]$$

- **MLE Estimate**:
  $$\hat{p}_{\text{MLE}} = \frac{\sum_{i=1}^{n} x_i}{n}$$

### Binomial Distribution

#### Definition
The binomial distribution models the number of successes $k$ in $n$ independent Bernoulli trials.

- **Parameters**:
  - $n$: Number of trials.
  - $p$: Probability of success in each trial.

#### Probability Mass Function
$$P(k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

where the **Binomial Coefficient**  $\binom{n}{k} = \frac{n!}{k!(n-k)!}$

#### Interpretation
- **Order Independence**: The binomial distribution considers all possible sequences with $k$ successes equally likely.
- **Use Case**: When only counts of successes are important, not the specific sequence.

#### Naming of discrete distributions
![[text-vars.png]]

### Bayesian Inference and Priors

#### Limitations of MLE
- **Overconfidence**: MLE can give extreme estimates with limited data.
- **No Incorporation of Prior Knowledge**: MLE relies solely on observed data.

#### Bayesian Approach
- **Incorporate Prior Beliefs**: Use prior distributions to represent initial beliefs about parameters.
- **Update Beliefs with Data**: Compute the posterior distribution using Bayes' theorem.

#### Priors and Pseudo-Counts
- **Pseudo-Counts**: Represent prior beliefs as if we have observed additional data.
  - E.g., believing the coin is fair corresponds to pseudo-counts of $\alpha=\beta=1$.
- **Strength of Belief**: Larger pseudo-counts ($\alpha=\beta=1000$) indicate stronger prior beliefs.

### Beta Distribution

#### Definition
The Beta distribution is a continuous probability distribution defined on the interval $[0, 1]$, suitable for modeling probabilities.

- **Parameters**: Shape parameters $\alpha > 0$ and $\beta > 0$.

- **Probability Density Function (PDF)**:
  $$\text{Beta}(p \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} p^{\alpha-1} (1-p)^{\beta-1}$$

  where $\Gamma(\cdot)$ is the gamma function, which generalizes the factorial function.

![[text-bta.png]]

#### Properties
- **Mean**:
  $$E[p] = \frac{\alpha}{\alpha + \beta}$$
- **Conjugate Prior**: The Beta distribution is conjugate to the Bernoulli and binomial distributions.

### Posterior Distribution
Given observed data $D$ with $k$ successes and $n-k$ failures:

- **Posterior Parameters**:
  $$\alpha_{\text{post}} = \alpha_{\text{prior}} + k$$
  $$\beta_{\text{post}} = \beta_{\text{prior}} + n - k$$

- **Posterior Distribution**:
  $$p(p \mid D) = \text{Beta}(p \mid \alpha_{\text{post}}, \beta_{\text{post}})$$

#### Interpretation
- **Updating Beliefs**: The posterior Beta distribution combines prior beliefs with observed data.
- **Flexibility**: By adjusting $\alpha$ and $\beta$, we can represent different levels of certainty.

![[text-priors.png]]

### Making Predictions

#### Bayesian Predictive Distribution
To predict the probability of success in the next trial:
$$P(X_{\text{next}}=1 \mid D) = E[p \mid D] = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}}$$

With the Bayesian approach, average over all possible parameter settings. The prediction for heads happens to correspond to the mean of the posterior distribution. 

Given the posterior distribution, we can also answer other questions such as “what is the probability that π > 0.5 given the observed data?”.

## Model Comparison

### Comparing Models
Suppose we have two models for the coin:
- **Fair Coin Model**:
  - Assumes $p=0.5$.
  - No parameters to estimate.
- **Bent Coin Model**:
  - Assumes $p$ is unknown and uniformly distributed over $[0, 1]$.
  - Requires estimating $p$.

![[text-probs.png]]

### Bayesian Model Comparison
- **Prior Probabilities**:
  $$P(\text{Fair}) = 0.8$$
  $$P(\text{Bent}) = 0.2$$

- **Compute Evidence**:
  - For the Fair Coin Model:
    $$P(D \mid \text{Fair}) = (0.5)^n$$
  - For the Bent Coin Model:
    $$P(D \mid \text{Bent}) = \int_0^1 P(D \mid p) P(p \mid \text{Bent}) dp$$

    Since $P(p \mid \text{Bent})$ is uniform, this simplifies to the Beta function.

- **Posterior Probabilities**:
  $$P(\text{Fair} \mid D) = \frac{P(D \mid \text{Fair}) P(\text{Fair})}{P(D)}$$
  $$P(\text{Bent} \mid D) = \frac{P(D \mid \text{Bent}) P(\text{Bent})}{P(D)}$$

  where $P(D)$ is the total probability of the data under both models.

### Decision Making
- **Model Selection**: Choose the model with the higher posterior probability.
- **Predictive Distribution**: Combine predictions from both models weighted by their posterior probabilities.


# 35-Discrete-Categorical-Distribution

We extend the concepts from binary variables to multiple discrete outcomes, which is essential in modeling categorical data such as word frequencies in text documents.

### Key Concepts
- **Discrete and Multinomial Distributions**: Modeling counts of multiple categories.
- **Dirichlet Distribution**: Prior distribution over multinomial parameters.

### Multinomial Distribution

#### Definition
The multinomial distribution generalizes the binomial distribution to more than two outcomes.

- **Parameters**:
  - $n$: Number of trials.
  - $p = [p_1, p_2, \dots, p_m]$: Probabilities of each category, where $\sum_{i=1}^m p_i = 1$.

#### Probability Mass Function
Given counts $k = [k_1, k_2, \dots, k_m]$ with $\sum_{i=1}^m k_i = n$:
$$P(k \mid n, p) = \frac{n!}{k_1! k_2! \dots k_m!} \prod_{i=1}^m p_i^{k_i}$$

#### Example: Rolling a Die
- **Outcomes**: $m=6$ faces.
- **Counts**: Number of times each face appears in $n$ rolls.

### Dirichlet Distribution

The Dirichlet distribution is a continuous multivariate probability distribution over the simplex of $m$-dimensional probability vectors $p$.

- **Parameters**: Concentration parameters $\alpha = [\alpha_1, \alpha_2, \dots, \alpha_m]$, with $\alpha_i > 0$.

#### Probability Density Function
$$\text{Dirichlet}(p \mid \alpha) = \frac{\Gamma\left(\sum_{i=1}^m \alpha_i\right)}{\prod_{i=1}^m \Gamma(\alpha_i)} \prod_{i=1}^m p_i^{\alpha_i - 1}$$

![[txt-dirichlet.png]]
#### Properties
- **Conjugate Prior**: The Dirichlet distribution is the conjugate prior for the multinomial distribution.
- **Mean**:
  $$E[p_i] = \frac{\alpha_i}{\sum_{j=1}^m \alpha_j}$$

#### Posterior Distribution
Given observed counts $k$:
- **Posterior Parameters**:
  $$\alpha_{i, \text{post}} = \alpha_{i, \text{prior}} + k_i$$
- **Posterior Distribution**:
  $$p(p \mid k) = \text{Dirichlet}(p \mid \alpha_{\text{post}})$$

#### Symmetric Dirichlet Distribution
- **Definition**: A Dirichlet distribution where all concentration parameters are equal: $\alpha_i = \alpha$ for all $i$.
- **Interpretation**:
  - Small $\alpha$: Distributions are more variable; samples are likely to be sparse.
  - Large $\alpha$: Distributions are more uniform; samples are more balanced.

### Sampling from the Dirichlet Distribution
- **Method**:
  - Sample $m$ independent gamma random variables $g_i$ with shape parameter $\alpha$ and scale parameter 1.
  - Normalize:
    $$p_i = \frac{g_i}{\sum_{j=1}^m g_j}$$

## Application: Word Counts in Text

### Modeling Word Frequencies
- **Vocabulary**: Set of $m$ distinct words.
- **Document Representation**: Counts $k = [k_1, k_2, \dots, k_m]$ of each word in a document.
- **Multinomial Distribution**: Models the probability of observing these counts given word probabilities $p$.

### Prior Over Word Probabilities
- **Dirichlet Prior**: Encodes prior beliefs about word probabilities.
- **Posterior Updating**: Update the Dirichlet parameters based on observed word counts.

### Implications for Text Modeling
- **Flexibility**: Can model documents with varying word distributions.
- **Bayesian Inference**: Allows incorporation of prior knowledge and uncertainty.


# 36-Document-Models 

## A Simple Document Model

In the simplest form, we can model a collection of documents by considering each word in the documents as being drawn independently from a fixed vocabulary according to some probability distribution.

### Notation and Definitions
- **Vocabulary Size ($M$):** The number of unique words in our vocabulary.
- **Number of Documents ($D$):** The total number of documents in our collection.
- **Number of Words in Document $d$ ($N_d$):** Each document may contain a different number of words.
- **Word Index:** Each word in the vocabulary is assigned an index from $1$ to $M$.
- **Word in Document $d$ at Position $n$ ($w_{dn}$):** The $n$-th word in document $d$, where $w_{dn} \in \{1, 2, ..., M\}$.
- **Word Distribution ($\beta$):** A vector of probabilities $\beta = [\beta_1, \beta_2, ..., \beta_M]^T$, where $\beta_m$ is the probability of word $m$ being selected.
![[text-simpl.png]]

### The Categorical Distribution
The categorical distribution is a discrete probability distribution that describes the probability of a single trial resulting in one of $M$ possible outcomes.

**Probability Mass Function (PMF):**
$$P(w_{dn} = m \mid \beta) = \beta_m$$

**Properties:**
1. $\beta_m \geq 0$ for all $m$.
2. $\sum_{m=1}^M \beta_m = 1$.

### Generative Process
Under this model, each word $w_{dn}$ in document $d$ is generated independently by:
1. **Sampling a Word:** For each position $n$ in document $d$, sample $w_{dn}$ from the categorical distribution $\text{Cat}(\beta)$.

### Maximum Likelihood Estimation (MLE)
To estimate the word distribution $\beta$ from data, we use Maximum Likelihood Estimation.

#### Likelihood Function
The likelihood of observing the entire collection of documents $W = \{w_{dn}\}$ given $\beta$ is:
$$L(\beta; W) = \prod_{d=1}^D \prod_{n=1}^{N_d} P(w_{dn} \mid \beta) = \prod_{d=1}^D \prod_{n=1}^{N_d} \beta_{w_{dn}}$$
We can fit $\beta$ by maximising the likelihood: 
$$\hat{\beta} = \arg\max_{\beta} \prod_{d=1}^{D} \prod_{n=1}^{N_d} \text{Cat}(w_{nd}|\beta) = \arg\max_{\beta} \text{Mult}(c_1, \dots, c_M | \beta, N)$$

#### Log-Likelihood
Taking the logarithm simplifies the product into a sum:
$$\log L(\beta; W) = \sum_{d=1}^D \sum_{n=1}^{N_d} \log \beta_{w_{dn}}$$

#### Sufficient Statistics
Define $c_m$ as the total count of word $m$ across all documents:
$$c_m = \sum_{d=1}^D \sum_{n=1}^{N_d} \delta(w_{dn}, m)$$
where $\delta(a, b)$ is the Kronecker delta function, which is $1$ if $a = b$ and $0$ otherwise.

Then, the log-likelihood becomes:
$$\log L(\beta; W) = \sum_{m=1}^M c_m \log \beta_m$$

#### Constraint
Since $\beta$ is a probability distribution, the parameters must satisfy:
$$\sum_{m=1}^M \beta_m = 1$$

#### Optimization Using Lagrange Multipliers
To maximize the log-likelihood under the constraint, we set up the Lagrangian $F$:
$$F(\beta, \lambda) = \sum_{m=1}^M c_m \log \beta_m + \lambda \left(1 - \sum_{m=1}^M \beta_m\right)$$

Taking the derivative of $F$ with respect to $\beta_m$ and setting it to zero:
$$\frac{\partial F}{\partial \beta_m} = \frac{c_m}{\beta_m} - \lambda = 0 \implies \beta_m = \frac{c_m}{\lambda}$$

Using the constraint $\sum_{m=1}^M \beta_m = 1$:
$$\lambda = \sum_{m=1}^M c_m = N$$
where $N$ is the total number of words across all documents.

Therefore, the Maximum Likelihood Estimate (MLE) of $\beta_m$ is:
$$\hat{\beta}_m = \frac{c_m}{N}$$

#### Interpretation
The estimated probability $\hat{\beta}_m$ is the relative frequency of word $m$ in the entire collection.

### Limitations of the Simple Model
- **Lack of Specialization:** The model does not account for differences between documents. All documents are assumed to have the same distribution over words.
- **No Topic Modeling:** There is no mechanism to capture different topics or categories that might be present in the document collection.
- **Assumption of Independence:** Words are assumed to be independent, which ignores syntactic and semantic relationships.

## Mixture (of Categorical) Models for Documents

### Motivation
To overcome the limitations of the simple model, we introduce a mixture model that allows for documents to belong to different categories or topics, each with its own word distribution.

### Generative Process
1. **Document Category Assignment:**  
   For each document $d$, assign it to a category $z_d$ by sampling from a categorical distribution:
   $$z_d \sim \text{Cat}(\theta)$$
   where $\theta = [\theta_1, \theta_2, ..., \theta_K]^T$ and $\theta_k = P(z_d = k)$ is the probability of a document belonging to category $k$.

2. **Word Generation:**  
   For each word position $n$ in document $d$, sample $w_{dn}$ from the categorical distribution corresponding to category $z_d$:
   $$w_{dn} \sim \text{Cat}(\beta_{z_d})$$
   where $\beta_{z_d}$ is the word distribution for category $z_d$.

![[word-doc.png]]

### Model Parameters
- **Category Probabilities ($\theta$):** Parameters of the categorical distribution over document categories.
- **Word Distributions ($\beta_k$):** For each category $k$, $\beta_k = [\beta_{k1}, \beta_{k2}, ..., \beta_{kM}]^T$ represents the probability distribution over words.

### Latent Variables
- **Document Category ($z_d$):** A hidden variable indicating the category of document $d$.  
  **Purpose:** Capturing the idea that different documents may discuss different topics, each with its own characteristic word distribution.

### Likelihood Function
The likelihood of observing the entire collection $W$ given the parameters $\theta$ and $\{\beta_k\}$ is:
$$P(W \mid \theta, \{\beta_k\}) = \prod_{d=1}^D P(w_d \mid \theta, \{\beta_k\}) = \prod_{d=1}^D \sum_{k=1}^K P(z_d = k \mid \theta) P(w_d \mid z_d = k, \beta_k)$$
where:
$$P(w_d \mid z_d = k, \beta_k) = \prod_{n=1}^{N_d} P(w_{dn} \mid z_d = k, \beta_k) = \prod_{n=1}^{N_d} \beta_{k w_{dn}}$$

### Challenges
- **Latent Variables:** The document categories $\{z_d\}$ are not observed.
- **Nonlinear Optimization:** The presence of the sum over $k$ inside the product over $d$ complicates direct maximization of the likelihood.

## Fitting the Mixture Model with the EM Algorithm

The Expectation-Maximization (EM) algorithm is an iterative method used for finding Maximum Likelihood Estimates (MLE) in models with latent variables.

1. **E-Step (Expectation):** Estimate the posterior distribution of the latent variables given the current parameters.
2. **M-Step (Maximization):** Maximize the expected log-likelihood with respect to the parameters, using the estimated posterior from the E-Step.

### Applying EM to the Mixture Model

#### E-Step
For each document $d$, compute the posterior probability (responsibility) that it belongs to category $k$:
$$r_{kd} = P(z_d = k \mid w_d, \theta, \{\beta_k\}) = \frac{\theta_k P(w_d \mid \beta_k)}{\sum_{k'=1}^K \theta_{k'} P(w_d \mid \beta_{k'})}$$
where:
$$P(w_d \mid \beta_k) = \prod_{n=1}^{N_d} \beta_{k w_{dn}} = \text{Mult}(\{c_{md}\} \mid \beta_k, N_d)$$
and $c_{md}$ is the count of word $m$ in document $d$.  
$\text{Mult}(\{c_{md}\} \mid \beta_k, N_d)$ is the multinomial probability of the word counts in document $d$ under category $k$.

#### M-Step
Update the parameters $\theta$ and $\{\beta_k\}$ by maximizing the expected log-likelihood:

$$\sum_{d=1}^{D} \sum_{k=1}^{K} q(z_d = k) \log p(w, z_d) = \sum_{k, d} r_{kd} \log \left[ p(z_d = k | \theta) \prod_{n=1}^{N_d} p(w_{n,d} | \beta_k, w_{n,d}) \right]$$
$$= \sum_{k, d} r_{kd} \left( \log \prod_{m=1}^{M} \beta_{km}^{c_{md}} + \log \theta_k \right) = \sum_{k, d} r_{kd} \left( \sum_{m=1}^{M} c_{md} \log \beta_{km} + \log \theta_k \right)$$
$$= \sum_{k, d} r_{kd} \left( \sum_{m=1}^{M} c_{md} \log \beta_{km} + \log \theta_k \right) \overset{\text{def}}{=} F(R, \theta, \beta)$$

**Subject to the constraints:**
$$\sum_{k=1}^K \theta_k = 1, \quad \sum_{m=1}^M \beta_{km} = 1 \quad \forall k$$

**Updating $\theta_k$:**
$$\hat{\theta}_k \leftarrow \arg\max_{\theta_k} F(R, \theta, \beta) + \lambda \left( 1 - \sum_{k'=1}^K \theta_{k'} \right)$$
$$= \frac{\sum_{d=1}^D r_{kd}}{\sum_{k'=1}^K \sum_{d=1}^D r_{k'd}} = \frac{\sum_{d=1}^D r_{kd}}{D}$$

**Updating $\beta_{km}$:
$$\hat{\beta}_{km} \leftarrow \arg\max_{\beta_{km}} F(R, \theta, \beta) + \sum_{k'=1}^K \lambda_{k'} \left( 1 - \sum_{m'=1}^M \beta_{k'm'} \right)$$
$$= \frac{\sum_{d=1}^D r_{kd} c_{md}}{\sum_{m'=1}^M \sum_{d=1}^D r_{kd} c_{m'd}}$$

### Interpretation
- **E-Step:** Calculates the expected assignment of documents to categories based on current parameter estimates.
- **M-Step:** Updates the parameter estimates to maximize the likelihood, weighted by the expected assignments.

### Convergence
- The EM algorithm is guaranteed to not decrease the likelihood at each iteration.
- It converges to a local maximum of the likelihood function.

## Bayesian Mixture of Categoricals Model

### Motivation
The EM algorithm provides point estimates of the parameters $\pi$ and $\{\beta_k\}$. A Bayesian approach introduces prior distributions over the parameters, allowing us to incorporate prior beliefs and quantify uncertainty.

### Priors
- **Dirichlet Prior for $\pi$:**
  $$\pi \sim \text{Dir}(\alpha)$$
  where $\alpha = [\alpha_1, \alpha_2, ..., \alpha_K]^T$ are the concentration parameters.

- **Dirichlet Prior for $\beta_k$:**
  $$\beta_k \sim \text{Dir}(\gamma)$$
  where $\gamma = [\gamma_1, \gamma_2, ..., \gamma_M]^T$.

### Bayesian Inference
- **Posterior Distributions:** Instead of point estimates, we aim to compute the posterior distributions:
  $$P(\pi, \{\beta_k\} \mid W)$$
- **Inference Methods:** Exact inference is often intractable; we may use approximate methods such as Variational Inference or Markov Chain Monte Carlo (MCMC).

### Benefits
- **Uncertainty Quantification:** Provides a measure of confidence in the parameter estimates.
- **Regularization:** The priors can prevent overfitting, especially in cases with limited data.
- **Incorporation of Prior Knowledge:** Prior beliefs about the distribution of topics or word frequencies can be included.

---

## 37-Expectation-Maximization-Algorithm

### Notation
- **Observed Data ($y$):** The data we can observe directly.
- **Latent Variables ($z$):** Hidden variables that are not directly observed.
- **Parameters ($\theta$):** The parameters of the model we wish to estimate.
- **Complete Data:** The combination of observed data and latent variables $(y, z)$.
- **Likelihood:**
  $$p(y \mid \theta) = \int p(y, z \mid \theta) \, dz$$

### Derivation of the Lower Bound for $\log p(y \mid \theta)$

Using **Bayes' Rule**, the log-likelihood can be decomposed as:

$$
p(z \mid y, \theta) = \frac{p(y \mid z, \theta)p(z \mid \theta)}{p(y \mid \theta)} \quad \implies \quad p(y \mid \theta) = \frac{p(y \mid z, \theta)p(z \mid \theta)}{p(z \mid y, \theta)}.
$$

Multiply and divide the marginal likelihood $p(y \mid \theta)$ by an arbitrary (non-zero) distribution $q(z)$, 
$$
p(y \mid \theta) = \frac{p(y \mid z, \theta)p(z \mid \theta)}{q(z)} \frac{q(z)}{p(z \mid y, \theta)}.
$$

Applying the logarithm to $p(y \mid \theta)$:
$$
\log p(y \mid \theta) = \log \frac{p(y \mid z, \theta)p(z \mid \theta)}{q(z)} + \log \frac{q(z)}{p(z \mid y, \theta)}.
$$

Taking the expectation with respect to $q(z)$:
$$
\log p(y \mid \theta) = \int q(z) \log \frac{p(y \mid z, \theta)p(z \mid \theta)}{q(z)} \, dz + \int q(z) \log \frac{q(z)}{p(z \mid y, \theta)} \, dz.
$$

The equation can now be split into two terms:
- **Lower bound functional $\mathcal{F}(q(z), \theta)$**:
  $$
  \mathcal{F}(q(z), \theta) = \int q(z) \log \frac{p(y \mid z, \theta)p(z \mid \theta)}{q(z)} \, dz.
  $$
- **KL Divergence $\text{KL}(q(z) \parallel p(z \mid y, \theta))$**:
  $$
  \text{KL}(q(z) \parallel p(z \mid y, \theta)) = \int q(z) \log \frac{q(z)}{p(z \mid y, \theta)} \, dz.
  $$

This leads to:
$$
\log p(y \mid \theta) = \mathcal{F}(q(z), \theta) + \text{KL}(q(z) \parallel p(z \mid y, \theta)).
$$

### The EM Algorithm

The Expectation-Maximization (EM) algorithm iteratively maximizes the marginal likelihood of the observed data, $\log p(y \mid \theta)$, by alternately updating the posterior distribution of the latent variables and the model parameters.

#### Procedure:

1. **Initialization**:
   Start with initial (random) parameters $\theta^{t=0}$.

2. **Iterative Steps**:
   For $t = 1, \dots, T$, repeat the following two steps:

   **E-step** (Expectation):
   - For fixed $\theta^{t-1}$, maximize the lower bound $\mathcal{F}(q(z), \theta^{t-1})$ with respect to $q(z)$.
   - Since the log-likelihood $\log p(y \mid \theta)$ is independent of $q(z)$, maximizing the lower bound is equivalent to minimizing the KL divergence:
    $$\text{KL}(q(z) \parallel p(z \mid y, \theta^{t-1})).$$
   - The optimal $q^t(z)$ is:
    $$q^t(z) = p(z \mid y, \theta^{t-1}).$$

   **M-step** (Maximization):
   - For fixed $q^t(z)$, maximize the lower bound $\mathcal{F}(q^k(z), \theta)$ with respect to $\theta$.
   - The lower bound is:
    $$\mathcal{F}(q(z), \theta) = \int q(z) \log (p(y \mid z, \theta)p(z \mid \theta)) \, dz - \int q(z) \log q(z) \, dz,$$
     where the second term is the entropy of $q(z)$, independent of $\theta$.
   - The M-step updates $\theta^t$ as:
    $$\theta^t = \arg \max_{\theta} \int q^t(z) \log (p(y \mid z, \theta)p(z \mid \theta)) \, dz.$$

#### Key Insight:
- Although each step works with the lower bound $\mathcal{F}(q(z), \theta)$, the algorithm ensures that the log-likelihood $\log p(y \mid \theta)$ does not decrease after each iteration.

At iteration $t$:
$$
\log p(y \mid \theta^{t-1}) = 
\underbrace{\mathcal{F}(q^t(z), \theta^{t-1})}_{\text{E-step}} 
\leq 
\underbrace{\mathcal{F}(q^t(z), \theta^t)}_{\text{M-step}} 
\leq 
\log p(y \mid \theta^t).
$$
Thus, the EM algorithm guarantees non-decreasing log-likelihood values during the optimization process.


## Example: Gaussian Mixture Model (GMM)

### Model Definition
- **Observations ($y_i$):** Real-valued data points.
- **Latent Variables ($z_i$):** Indicate which Gaussian component generated $y_i$, where $z_i \in \{1, 2, ..., K\}$.

**Parameters:**
- Mixing coefficients: $\pi_j = P(z_i = j)$
- Means: $\mu_j$
- Variances: $\sigma_j^2$

### Generative Process
For each data point $i$:
1. Sample $z_i \sim \text{Cat}(\pi)$.
2. Sample $y_i \sim \mathcal{N}(\mu_{z_i}, \sigma_{z_i}^2)$.

### Applying EM to GMM

#### E-Step
Compute the responsibilities $r_{ij}$:
$$r_{ij} = P(z_i = j \mid y_i, \theta^{(t)}) = \frac{\pi_j^{(t)} \mathcal{N}(y_i \mid \mu_j^{(t)}, \sigma_j^{2(t)})}{\sum_{k=1}^K \pi_k^{(t)} \mathcal{N}(y_i \mid \mu_k^{(t)}, \sigma_k^{2(t)})}$$

#### M-Step
Update the parameters using the responsibilities:

- **Update Mixing Coefficients:**
  $$\pi_j^{(t+1)} = \frac{1}{N} \sum_{i=1}^N r_{ij}$$

- **Update Means:**
  $$\mu_j^{(t+1)} = \frac{\sum_{i=1}^N r_{ij} y_i}{\sum_{i=1}^N r_{ij}}$$

- **Update Variances:**
  $$\sigma_j^{2(t+1)} = \frac{\sum_{i=1}^N r_{ij} (y_i - \mu_j^{(t+1)})^2}{\sum_{i=1}^N r_{ij}}$$

### Interpretation
- **Responsibilities:** Measure the probability that data point $i$ was generated by component $j$.
- **Parameter Updates:** Weighted averages using the responsibilities as weights.

## Appendix: Kullback-Leibler (KL) Divergence

### Properties
1. **Non-negativity:** $\text{KL}(q(x) \parallel p(x)) \geq 0$.
2. **Zero Minimum:** $\text{KL}(q(x) \parallel p(x)) = 0$ if and only if $q(x) = p(x)$ almost everywhere.
3. **Asymmetry:** $\text{KL}(q(x) \parallel p(x)) \neq \text{KL}(p(x) \parallel q(x))$ in general.

### Role in EM Algorithm
The KL divergence measures how close the approximate distribution $q(z)$ is to the true posterior $p(z \mid y, \theta)$. Minimizing the KL divergence in the E-Step ensures that $q(z)$ is the best approximation to the true posterior given the current parameters.



# 38-Gibbs-Sampling-for-Bayesian-Mixture-Models 

In the realm of probabilistic modeling, understanding how to infer latent structures in data is crucial. Mixture models and topic models like Latent Dirichlet Allocation (LDA) are powerful tools for uncovering hidden patterns in complex datasets, such as collections of documents. 

## Bayesian Mixture Models for Documents

#### Model Components
- **Observations ($w_d$):** The words in document $d$, where $d=1,2,\dots,D$.
- **Parameters ($\beta_k$ and $\theta$):**
  - $\beta_k$: The parameters of the categorical distribution over words for component $k$, with prior $p(\beta_k)$.
  - $\theta$: The mixing proportions (mixture weights) for the components, with prior $p(\theta)$.
- **Latent Variables ($z_d$):** The component assignments for each document $d$, where $z_d \in \{1,2,\dots,K\}$.

#### Generative Process
1. **Mixing Proportions:**
   Draw mixing proportions $\theta$ from a Dirichlet prior:
   $$\theta \sim \text{Dirichlet}(\alpha).$$
2. **Component Parameters:**
   For each component $k=1,2,\dots,K$, draw word distribution parameters $\beta_k$ from a Dirichlet prior:
   $$\beta_k \sim \text{Dirichlet}(\gamma).$$
3. **Document Assignments:**
   For each document $d$, assign it to a component $z_d$ by sampling from the categorical distribution:
   $$z_d \sim \text{Categorical}(\theta).$$
4. **Word Generation:**
   For each word position $n$ in document $d$, sample word $w_{dn}$ from the categorical distribution corresponding to component $z_d$:
   $$w_{dn} \sim \text{Categorical}(\beta_{z_d}).$$

#### Model Representation
The model can be visually represented with a graphical model where:
- **Nodes:** Random variables (observed and latent).
- **Edges:** Dependencies between variables.
- **Plates:** Repetition over indices (documents, words, components).

![[txt-bmm.png]]
### Latent Conditional Posterior
The latent conditional posterior for $z_d$ is:
$$
p(z_d = k \mid w_d, \boldsymbol{\theta}, \boldsymbol{\beta}) \propto p(z_d = k \mid \boldsymbol{\theta}) p(w_d \mid z_d = k, \boldsymbol{\beta}),
$$
which simplifies to:
$$
p(z_d = k \mid w_d, \boldsymbol{\theta}, \boldsymbol{\beta}) \propto \theta_k p(w_d \mid \boldsymbol{\beta}_{z_d}),
$$
a discrete distribution with $K$ possible outcomes.

## Gibbs Sampling for Bayesian Mixture Models

In the context of Bayesian mixture models for documents, we have observed data (words in documents) and latent variables (component assignments for documents, mixing proportions, and component parameters). Our goal is to approximate the posterior distribution of these latent variables given the observed data.

### Gibbs Sampling Steps

##### 1. Sampling Component Parameters ($\beta_k$)

**Goal:** Sample $\beta_k$ from $p(\beta_k \mid \{w_d\}, \{z_d\})$. Given the current component assignments $\{z_d\}$ and the observed words $\{w_d\}$, we sample each $\beta_k$ from its posterior distribution.

**Derivation:**

- The prior for $\beta_k$ is a Dirichlet distribution: $\beta_k \sim \text{Dirichlet}(\gamma)$.

- The likelihood of the words in documents assigned to component $k$ is:
  $$p(\{w_d : z_d = k\} \mid \beta_k) = \prod_{d : z_d = k} \prod_{n=1}^{N_d} p(w_{dn} \mid \beta_k),$$

  where $N_d$ is the number of words in document $d$, and $w_{dn}$ is the $n$-th word in document $d$.

- Since the words are generated from a categorical distribution parameterized by $\beta_k$, the likelihood for $\beta_k$ is multinomial.

- The posterior distribution of $\beta_k$ is proportional to the product of the prior and the likelihood:
  $$p(\beta_k \mid \{w_d\}, \{z_d\}) \propto p(\beta_k) \prod_{d : z_d = k} p(w_d \mid \beta_k) .$$

- Because both the prior and the likelihood are conjugate (Dirichlet prior and multinomial likelihood), the posterior is also a Dirichlet distribution.

**Calculations:**
- Let $c_{mk}$ be the total count of word $m$ in all documents assigned to component $k$:
  $$c_{mk} = \sum_{d : z_d = k} \sum_{n=1}^{N_d} \delta(w_{dn} = m),$$
  where $\delta(\cdot)$ is the Kronecker delta function.

- The posterior is:
$$\beta_k \mid \{w_d\}, \{z_d\} \sim \text{Dirichlet}(\gamma + c_{\cdot k}),$$

  where $c_{\cdot k} = [c_{1k}, c_{2k}, \dots, c_{Mk}]^T$ is the vector of word counts for component $k$, and $M$ is the size of the vocabulary.


##### 2. Sampling Mixing Proportions ($\theta$)

**Goal:** Sample $\theta$ from $p(\theta \mid \{z_d\}, \alpha)$. Given the current component assignments $\{z_d\}$, we sample $\theta$ from its posterior distribution.

**Derivation:**

- The prior for $\theta$ is a Dirichlet distribution: $\theta \sim \text{Dirichlet}(\alpha)$.

- The likelihood of the component assignments $\{z_d\}$ given $\theta$ is:
$$p(\{z_d\} \mid \theta) = \prod_{d=1}^D p(z_d \mid \theta) = \prod_{d=1}^D \theta_{z_d}.$$

- The posterior distribution of $\theta$ is proportional to the product of the prior and the likelihood:
$$p(\theta \mid \{z_d\}, \alpha) \propto p(\theta) \prod_{d=1}^D \theta_{z_d}.$$

- Since the prior is Dirichlet and the likelihood is multinomial, the posterior is also Dirichlet.

**Calculations:**

- Let $c_k$ be the number of documents assigned to component $k$:
$$c_k = \sum_{d=1}^D \delta(z_d = k).$$

- The posterior is:
$$\theta \mid \{z_d\}, \alpha \sim \text{Dirichlet}(\alpha + c),$$

  where $c = [c_1, c_2, \dots, c_K]^T$ is the vector of component counts.

##### 3. Sampling Component Assignments ($z_d$)

**Goal:** For each document $d$, sample $z_d$ from $p(z_d \mid w_d, \theta, \{\beta_k\})$. For each document $d$, given the current $\theta$ and $\{\beta_k\}$, we sample $z_d$ from its conditional distribution.

**Derivation:**

- The conditional distribution for $z_d$ is proportional to:
$$p(z_d = k \mid w_d, \theta, \{\beta_k\}) \propto p(z_d = k \mid \theta) p(w_d \mid z_d = k, \beta_k).$$

- The prior probability of $z_d = k$ is $\theta_k$.

- The likelihood of the words in document $d$ given $z_d = k$ is:
$$p(w_d \mid z_d = k, \beta_k) = \prod_{n=1}^{N_d} p(w_{dn} \mid \beta_k).$$

**Calculations:**

- Compute the unnormalized probability for each component $k$:
$$\tilde{p}_k = \theta_k \prod_{n=1}^{N_d} \beta_{k, w_{dn}},$$

  where $\beta_{k, w_{dn}}$ is the probability of word $w_{dn}$ under component $k$.

- Normalize to obtain a probability distribution:
$$p(z_d = k \mid w_d, \theta, \{\beta_k\}) = \frac{\tilde{p}_k}{\sum_{k'=1}^K \tilde{p}_{k'}}.$$

- Sample $z_d$ from this categorical distribution.

## Collapsed Gibbs Sampling

#### Motivation

In standard Gibbs sampling, we sample all variables, including the mixing proportions $\theta$. However, we can improve the efficiency of the sampler by integrating out (collapsing) some variables analytically.

- **Collapsing $\theta$:** By integrating out $\theta$, we reduce the number of parameters to sample and potentially reduce the variance of our estimates.

- **Benefit:** Collapsed Gibbs sampling can lead to faster convergence and more accurate approximations of the posterior distribution.

### Collapsed Gibbs Sampling Steps

We will integrate out $\theta$ and derive a new conditional distribution for $z_d$ that accounts for this.

###### 1. Integrating Out $\theta$

**Goal:** Derive $p(z_d = k \mid \{z_{-d}\}, \alpha)$, where $\{z_{-d}\}$ denotes all component assignments except for document $d$.

**Derivation:**

We start with:

$$p(z_d = k \mid \{z_{-d}\}, \alpha) = \int p(z_d = k \mid \theta) p(\theta \mid \{z_{-d}\}, \alpha) d\theta.$$

- We know that:
$$p(z_d = k \mid \theta) = \theta_k.$$
$$p(\theta \mid \{z_{-d}\}, \alpha) \sim \text{Dirichlet}(\alpha + c_{-d}),$$

  where $c_{-d} = [c_{1, -d}, c_{2, -d}, \dots, c_{K, -d}]^T$ and $c_{k, -d}$ is the number of documents assigned to component $k$ excluding document $d$.

**Calculations:**

- The expected value of $\theta_k$ under the Dirichlet distribution is:

$$E[\theta_k \mid \{z_{-d}\}, \alpha] = \frac{\alpha_k + c_{k, -d}}{\sum_{j=1}^K (\alpha_j + c_{j, -d})}.$$

- However, we need the full distribution, not just the expectation. Since $\theta$ follows a Dirichlet distribution, and $\theta_k$ is a beta-distributed marginal, we compute the integral:

$$p(z_d = k \mid \{z_{-d}\}, \alpha) = \int \theta_k \cdot \text{Dirichlet}(\theta \mid \alpha + c_{-d}) d\theta.$$

- The integral simplifies due to the properties of the Dirichlet distribution:

  $$p(z_d = k \mid \{z_{-d}\}, \alpha) = \frac{\alpha_k + c_{k, -d}}{\sum_{j=1}^K (\alpha_j + c_{j, -d})}.$$

**Explanation:**

- This expression gives the probability of assigning document $d$ to component $k$ based on the counts of component assignments excluding $d$.

- The term $\alpha_k + c_{k, -d}$ combines the prior information ($\alpha_k$) and the observed data (counts $c_{k, -d}$).

###### 2. Updating the Conditional Distribution for $z_d$

**Derivation:**

- The conditional probability is proportional to:

  $$p(z_d = k \mid \{z_{-d}\}, \alpha, \{\beta_k\}, w_d) \propto p(z_d = k \mid \{z_{-d}\}, \alpha) \cdot p(w_d \mid z_d = k, \{\beta_k\}).$$

- We have already derived $p(z_d = k \mid \{z_{-d}\}, \alpha)$.

- The likelihood $p(w_d \mid z_d = k, \{\beta_k\})$ is:
$$p(w_d \mid z_d = k, \beta_k) = \prod_{n=1}^{N_d} \beta_{k, w_{dn}}.$$

**Calculations:**

- Compute the unnormalized probability for each component $k$:
$$\tilde{p}_k = (\alpha_k + c_{k, -d}) \cdot \prod_{n=1}^{N_d} \beta_{k, w_{dn}}.$$

- Normalize to obtain a probability distribution:
$$p(z_d = k \mid \{z_{-d}\}, \alpha, \{\beta_k\}, w_d) = \frac{\tilde{p}_k}{\sum_{k'=1}^K \tilde{p}_{k'}}.$$

- Sample $z_d$ from this categorical distribution.

###### 3. Sampling $\beta_k$ Remains the Same

Since we have only integrated out $\theta$, the sampling of $\beta_k$ remains as before:

- Use the counts $c_{mk}$, which now include the updated component assignments $\{z_d\}$.

- Sample $\beta_k$ from:
$$\beta_k \mid \{w_d : z_d = k\}, \gamma \sim \text{Dirichlet}(\gamma + c_{\cdot k}).$$

### Collapsed Gibbs Sampling Algorithm

1. **Initialization:**

   - Initialize $\{z_d\}$ randomly.
   - Initialize $\{\beta_k\}$ accordingly.

2. **Iterative Sampling:**

   - For each iteration:

     a. Sample $z_d$ for each $d$ using the probabilities:
     $$p(z_d = k \mid \{z_{-d}\}, \alpha, \{\beta_k\}, w_d) \propto (\alpha_k + c_{k, -d}) \cdot \prod_{n=1}^{N_d} \beta_{k, w_{dn}}.$$

     b. Update counts $c_{k, -d}$ after each $z_d$ is sampled.

     c. Sample $\beta_k$ for each $k$ as before.

3. **Convergence:**

   - After sufficient iterations, the samples approximate the posterior distribution of the component assignments and component parameters, marginalized over $\theta$.

#### Understanding the Dependency

###### Why Do Component Assignments Become Dependent?

- In standard Gibbs sampling, $\theta$ acts as a bridge between the component assignments $\{z_d\}$. Given $\theta$, the $z_d$ are conditionally independent.

- When we integrate out $\theta$, the $z_d$ become dependent because they now directly influence each other through the counts $c_{k, -d}$ in the computation of $p(z_d = k \mid \{z_{-d}\}, \alpha)$.

- The probability of assigning document $d$ to component $k$ depends on how many other documents are assigned to $k$ (excluding $d$).


###### The "Rich Get Richer" Phenomenon

- Components with more documents assigned (higher $c_{k, -d}$) have higher probabilities for new documents to be assigned to them.

- This self-reinforcing behavior is sometimes called the "rich get richer" effect.

### Benefits of Collapsed Gibbs Sampling

1. **Reduced Dimensionality:** By integrating out $\theta$, we reduce the number of variables we need to sample, which can speed up convergence.

2. **Reduced Variance:** Marginalizing over $\theta$ can lead to more accurate estimates because we're averaging over its uncertainty rather than relying on sampled values.

3. **Faster Convergence:** Empirically, collapsed Gibbs samplers often converge faster than their uncollapsed counterparts.


---
# 39-Latent-Dirichlet-Allocation-(LDA)-for-Topic-Modeling

### Limitations of the Mixture of Categoricals Model

#### Model Overview
- In the mixture of categoricals model:
  - Each document $d$ is assigned to a single topic $z_d$.
  - All words $w_{dn}$ in document $d$ are drawn from the word distribution $\beta_{z_d}$ of topic $z_d$.
#### Limitations
1. **Single Topic Assumption:** Assumes each document is exclusively about one topic, which is unrealistic for documents covering multiple topics.
2. **Blurred Topics:** When documents span multiple topics, the model tends to learn topics that are a blend of multiple true topics, reducing interpretability.

### Motivation for LDA
Latent Dirichlet Allocation (LDA) addresses the limitations by allowing documents to exhibit multiple topics in varying proportions.
- **Flexibility:** Documents can be composed of multiple topics, with each word potentially drawn from a different topic.
- **Interpretability:** Topics are more coherent and distinct, improving the quality of topic modeling.

#### Key Differences:
1. **Per-Word Topic Assignment**: In LDA, each word $w_{nd}$ has its own topic assignment $z_{nd}$, allowing for multiple topics within a single document.
2. **Document-Specific Topic Distributions**: Each document $d$ has its own distribution over topics $\theta_d$, drawn from a Dirichlet prior.

## Latent Dirichlet Allocation (LDA)
LDA is a hierarchical Bayesian model. The generative process for LDA is as follows:

1. **Topic Distributions for Documents**:
   For each document $d$, draw a distribution over topics $\theta_d$ from a Dirichlet prior:
   $$\theta_d \sim Dirichlet(\alpha)$$

2. **Word Distributions for Topics**:
   For each topic $k$, draw a distribution over words $\beta_k$ from a Dirichlet prior:
   $$\beta_k \sim Dirichlet(\gamma)$$

3. **Topic Assignments for Words**:
   For each word position $n$ in document $d$, draw a topic assignment $z_{nd}$ from the categorical distribution over topics $\theta_d$:
   $$z_{nd} \sim Categorical(\theta_d)$$

4. **Word Generation**:
   Given the topic assignment $z_{nd}$, draw word $w_{nd}$ from the corresponding topic's word distribution $\beta_{z_{nd}}$:
   $$w_{nd} \sim Categorical(\beta_{z_{nd}})$$

LDA assumes that documents are mixtures of topics, and topics are distributions over words. Each word in a document is generated by first selecting a topic (according to the document's topic distribution) and then selecting a word from that topic's word distribution.

![[txr-lda.png]]

## The LDA Inference Problem

In Latent Dirichlet Allocation (LDA), our primary objective is to uncover the hidden thematic structure within a corpus of documents. Specifically, we aim to estimate:

Topic-Word Distributions ($\beta_{1:K}$): For each topic $k$, we want to determine the probability distribution over the vocabulary, indicating how likely each word is to appear in that topic.

Document-Topic Distributions ($\theta_{1:D}$): For each document $d$, we want to estimate the proportion of topics it contains.

Topic Assignments for Words (${z_{nd}}$): For each word $w_{nd}$ in each document, we aim to infer the topic $z_{nd}$ it is associated with.

Given the observed words (${w_{nd}}$), we want to compute the posterior distribution:

$$p(\beta_{1:K}, \theta_{1:D}, \{z_{nd}\} \mid \{w_{nd}\}, \alpha, \gamma)$$

Where $\alpha$ and $\gamma$ are hyperparameters of the Dirichlet priors.

## The Inference Challenge

### Why Is Exact Inference Intractable?

Computing the posterior distribution exactly is computationally infeasible due to several reasons:

#### Marginalization Over Latent Variables:

- We need to consider all possible topic assignments (${z_{nd}}$) for all words in all documents.
- For each word, there are $K$ possible topics it could be assigned to.
- If a document has $N_d$ words, the number of possible topic assignment configurations for that document is $K^{N_d}$.

#### Combinatorial Explosion:

- The total number of possible configurations across all documents becomes astronomical.
- For $D$ documents, the total configurations are $\prod_{d=1}^D K^{N_d}$.

#### Integration Over Continuous Parameters:

- We need to integrate over the continuous Dirichlet-distributed parameters $\theta_d$ and $\beta_k$.
- These integrals are high-dimensional and do not have closed-form solutions due to the dependencies introduced by the latent variables (${z_{nd}}$).

### Mathematical Formulation of the Problem

The joint distribution of all variables in the LDA model is:

$$p(\beta_{1:K}, \theta_{1:D}, \{z_{nd}\}, \{w_{nd}\} \mid \gamma, \alpha) = \left[\prod_{k=1}^K p(\beta_k \mid \gamma)\right] \left[\prod_{d=1}^D p(\theta_d \mid \alpha) \prod_{n=1}^{N_d} p(z_{nd} \mid \theta_d) p(w_{nd} \mid \beta_{1:K}, z_{nd})\right]$$

This does not follow a simple, tractable distribution. Instead, it's a complex combination of Dirichlet priors, multinomial distributions, and latent topic assignments. To compute the posterior, we need to compute the evidence (marginal likelihood):

$$p(\{w_{nd}\} \mid \alpha, \gamma) = \int \int \sum_{\{z_{nd}\}} \left[\prod_{k=1}^K p(\beta_k \mid \gamma)\right] \left[\prod_{d=1}^D p(\theta_d \mid \alpha) \prod_{n=1}^{N_d} p(z_{nd} \mid \theta_d) p(w_{nd} \mid \beta_{1:K}, z_{nd})\right] d\beta_{1:K} d\theta_{1:D}$$

This involves:

1. Summing over all possible topic assignments (${z_{nd}}$).
2. Integrating over all possible values of $\beta_{1:K}$ and $\theta_{1:D}$.

### Why This Is Intractable

1. **Exponential Number of Configurations**: The sum over (${z_{nd}}$) involves $K^{\sum_{d=1}^D N_d}$ terms, which is computationally infeasible for realistic corpus sizes.
2. **High-Dimensional Integrals**: The integrals over $\beta_{1:K}$ and $\theta_{1:D}$ are over high-dimensional Dirichlet distributions, which, coupled with the latent variables, make analytical solutions impossible.
3. **Dependencies Between Variables**: The latent variables and parameters are interdependent, preventing factorization that could simplify the computations.

### Implications of Intractability

Due to these computational challenges:

- **Exact Inference Is Not Feasible**: We cannot compute the posterior distribution directly.
- **Need for Approximate Methods**: We must resort to approximate inference techniques to estimate the posterior.

## Gibbs Sampling for LDA

In LDA, we can use Gibbs sampling to sample the latent topic assignments $\{z_{nd}\}$ and parameters $\theta_d$ and $\beta_k$ from their conditional distributions.

However, sampling $\theta_d$ and $\beta_k$ at each iteration can be computationally intensive. An alternative is to integrate out these parameters analytically, leading to a **collapsed Gibbs sampler**.

## Refresher on Beta and Dirichlet Distributions

Given a prior $\pi \sim Beta(\alpha, \beta)$ and observed data with $k$ successes and $n - k$ failures, the **posterior** is:
$$\pi \mid n, k \sim Beta(\alpha + k, \beta + n - k)$$

and, the predictive probability of success in the next trial is the expected value of $\pi$ under the posterior:
$$p(success \mid n, k) = \mathbb{E}[\pi \mid n, k] = \frac{\alpha + k}{\alpha + \beta + n}$$

Analogously, given a prior $\pi \sim Dirichlet(\alpha)$ and observed counts $\mathbf{c} = [c_1, c_2, \dots, c_K]$, the posterior is:
$$\pi \mid \mathbf{c} \sim Dirichlet(\alpha_1 + c_1, \alpha_2 + c_2, \dots, \alpha_K + c_K)$$

and, the predictive probability for category $j$ is:
$$p(j \mid \mathbf{c}) = \mathbb{E}[\pi_j \mid \mathbf{c}] = \frac{\alpha_j + c_j}{\sum_{i=1}^K (\alpha_i + c_i)}$$

## Collapsed Gibbs Sampling: An Approximate Inference Solution

To address the intractability, we employ Collapsed Gibbs Sampling, which involves:

- **Integrating Out Parameters**: We analytically integrate out the parameters $\theta_d$ (document-topic distributions) and $\beta_k$ (topic-word distributions) from the joint distribution.

- **Focusing on Latent Variables**: By collapsing these parameters, we concentrate on sampling the latent topic assignments $\{z_{nd}\}$.

#### Benefits of Collapsing
1. **Reduced Dimensionality**: Fewer variables need to be sampled, simplifying computations.
2. **Improved Efficiency**: Often leads to faster convergence due to reduced variance in estimates.
3. **Enhanced Mixing**: Sampling the latent variables directly can improve the mixing properties of the Markov Chain.

### Integrating Out $\theta_d$ and $\beta_k$

##### The Collapsed Joint Distribution
By integrating out $\theta_d$ and $\beta_k$, the joint distribution becomes dependent only on the latent variables $\{z_{nd}\}$ and observed words $\{w_{nd}\}$:

$$
p(\{z_{nd}\}, \{w_{nd}\} \mid \alpha, \gamma) = \int \int p(\beta_{1:K}, \theta_{1:D}, \{z_{nd}\}, \{w_{nd}\} \mid \alpha, \gamma) \, d\theta_{1:D} \, d\beta_{1:K}
$$

##### Consequences
1. **Interdependence of Topic Assignments**: The topic assignments $\{z_{nd}\}$ are no longer conditionally independent given the data; they become dependent through the integrated-out parameters.
2. **Simplified Sampling Target**: We now aim to sample from $p(\{z_{nd}\} \mid \{w_{nd}\}, \alpha, \gamma)$.

#### Derivation of the Collapsed Gibbs Sampler

Gibbs Sampling is a Markov Chain Monte Carlo (MCMC) method where we iteratively sample each variable from its conditional distribution given all other variables.

In the context of LDA:
- **Variables to Sample**: The latent topic assignments $\{z_{nd}\}$.
- **Conditional Distribution**: For each $z_{nd}$, we compute:

$$
p(z_{nd} \mid \{z_{-nd}\}, \{w_{nd}\}, \alpha, \gamma)
$$

where $\{z_{-nd}\}$ denotes all topic assignments except $z_{nd}$.

##### The Conditional Distribution for $z_{nd}$
The probability of assigning topic $k$ to word $w_{nd}$ is:

$$
p(z_{nd} = k \mid \{z_{-nd}\}, \{w_{nd}\}, \alpha, \gamma) \propto p(z_{nd} = k \mid \{z_{-nd}\}, \alpha) \cdot p(w_{nd} \mid z_{nd} = k, \{w_{-nd}\}, \{z_{-nd}\}, \gamma)
$$

##### Breaking Down the Components

**Topic Assignment Prior**:
$$
p(z_{nd} = k \mid \{z_{-nd}\}, \alpha) = \frac{\alpha + c_{dk}^{-nd}}{\sum_{j=1}^K (\alpha + c_{dj}^{-nd})}
$$

where $c_{dk}^{-nd}$ is the count of words in document $d$ assigned to topic $k$, excluding the current word $w_{nd}$.

**Word Likelihood Given Topic**:
$$
p(w_{nd} \mid z_{nd} = k, \{w_{-nd}\}, \{z_{-nd}\}, \gamma) = \frac{\gamma + c_{kw_{nd}}^{-nd}}{\sum_{m=1}^M (\gamma + c_{km}^{-nd})}
$$

where $c_{kw_{nd}}^{-nd}$ is the count of word $w_{nd}$ assigned to topic $k$ across all documents, excluding $w_{nd}$, and $M$ is the total number of unique words in the vocabulary.

#### The Collapsed Gibbs Sampling Algorithm

##### Step-by-Step Procedure

1. **Initialization**:
    - Randomly assign a topic $z_{nd}$ to each word $w_{nd}$.
    - Initialize counts $c_{dk}$ and $c_{kw}$.

2. **Iteration**:
    For each word $w_{nd}$ in the corpus:
    - Exclude Current Assignment:
      - Decrement counts $c_{dk}$ and $c_{kw}$ for the current topic assignment $z_{nd}$.
    - Compute Conditional Probabilities:
      - For each topic $k$, compute:
        $$
        p(z_{nd} = k \mid \cdot) \propto (\alpha + c_{dk}^{-nd}) \cdot \frac{\gamma + c_{kw_{nd}}^{-nd}}{\sum_{m=1}^M (\gamma + c_{km}^{-nd})}
        $$
    - Sample New Topic Assignment:
      - Use the computed probabilities to sample a new topic $z_{nd}$ for word $w_{nd}$.
    - Update Counts:
      - Increment counts $c_{dk}$ and $c_{kw}$ with the new topic assignment.

3. **Convergence Check**:
   - Repeat the iteration until the topic assignments stabilize.

#### Intuitive Understanding

1. **Topic Popularity in Document**:
   Words are more likely to be assigned to topics that are already prevalent in the document.

2. **Word-Topic Affinity**:
   Words are more likely to be assigned to topics where they frequently occur across the corpus.

### Per Word Perplexity: Evaluating Model Performance

Perplexity measures how well a probability model predicts a sample. For a document with $n$ words and total log-likelihood $\ell$, the per-word perplexity is calculated as:

$$
\text{Perplexity} = \exp\left(-\frac{\ell}{n}\right)
$$

##### Interpretation
- **Uncertainty Measurement**: Reflects the uncertainty in generating observed data.
- **Lower Perplexity is Better**: Indicates better performance.

##### Example
For a fair six-sided die:
- The probability of any sequence of rolls is $(1/6)^n$ for $n$ rolls.
- The log-likelihood is $\ell = n \log(1/6)$.
- The per-word perplexity is:

$$
\text{Perplexity} = \exp\left(-\frac{\ell}{n}\right) = \exp\left(-\frac{n \log(1/6)}{n}\right) = 6
$$

#### Conclusion

1. **Intractability of Exact Inference**: Exact computation is infeasible due to the exponential number of possible topic assignments.
2. **Collapsed Gibbs Sampling**: Simplifies inference by focusing on latent topic assignments.
3. **Perplexity as Evaluation Metric**: Provides a quantitative measure of model performance.

###### Developing Intuition
1. **Understanding Counts**: Counts $c_{dk}$ and $c_{kw}$ encapsulate the state of topic assignments.
2. **Conjugate Priors**: Dirichlet-multinomial conjugacy simplifies computations.
3. **Interplay Between Documents and Corpus**: Balances local (document) and global (corpus) information.
