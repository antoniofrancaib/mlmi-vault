## **Part I: Fundamentals of Classification and Decision Theory**

**(Based on Lectures by Jose Miguel Hernandez Lobato)**

### 1. **Introduction & Motivation**

In numerous scientific and engineering domains, we encounter classification problems where the objective is to assign an observed entity, characterized by a set of features, to one of several predefined categories or classes. Examples include identifying vowels from acoustic features or distinguishing fruit types based on visual attributes. A fundamental challenge in these tasks is the inherent **uncertainty**. Observations rarely provide unambiguous evidence for a single class. Feature distributions for different classes often overlap, meaning identical feature vectors might arise from different underlying classes.

Consider the Peterson & Barney vowel formant data visualized: even for relatively distinct vowels, the distributions of the first two formants (F1, F2) show considerable overlap. Consequently, any decision rule mapping features to class labels will inevitably make errors. The core objective of statistical decision theory in this context is not to achieve perfection (which is typically impossible), but to design decision procedures $f$ that map feature vectors $\mathbf{x}$ to class labels $\omega$ in a way that minimizes the expected "cost" or "loss" associated with potential errors. These notes explore the mathematical framework for constructing such optimal, or near-optimal, decision rules, focusing on the interplay between loss functions, probabilistic models (specifically posterior probabilities), and the resulting decision boundaries.

![[peterson-data.png]]

### 2. **Supervised Data & Evaluation**

We operate within the paradigm of **supervised learning**. This assumes access to a dataset $D$ consisting of $N$ pairs of observations:

$$D = \{ (\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N) \}$$

where:

* $\mathbf{x}_i \in \mathbb{R}^d$ is a $d$-dimensional feature vector representing the $i$-th entity.
* $y_i \in \{\omega_1, \omega_2, \dots, \omega_K\}$ is the corresponding true class label for the $i$-th entity, chosen from a set of $K$ predefined classes.

This dataset is assumed to be a collection of samples drawn independently and identically distributed (i.i.d.) from an underlying, typically unknown, joint probability distribution $p(\mathbf{x}, \omega)$ over the feature space $\mathbb{R}^d$ and the class label set $\{\omega_1, \dots, \omega_K\}$. This joint distribution can be factorized in two ways:

* $p(\mathbf{x}, \omega) = p(\mathbf{x} | \omega) P(\omega)$: Reflects a generative process where a class $\omega$ is chosen according to the prior probability $P(\omega)$, and then features $\mathbf{x}$ are generated according to the class-conditional density $p(\mathbf{x} | \omega)$.
* $p(\mathbf{x}, \omega) = P(\omega | \mathbf{x}) p(\mathbf{x})$: Reflects a process where features $\mathbf{x}$ are drawn from the marginal distribution $p(\mathbf{x})$, and then a label $\omega$ is assigned according to the posterior probability $P(\omega | \mathbf{x})$.

In practice, the ultimate goal is to build a classifier $f(\mathbf{x}; \theta)$, parameterized by $\theta$, that performs well on *new, unseen* data drawn from the same underlying distribution $p(\mathbf{x}, \omega)$. Since we only have access to the finite dataset $D$, we typically partition it:

* **Training Data ($D_{\text{train}}$):** Used to learn or estimate the parameters $\theta$ of the classifier $f$.
* **Evaluation Data ($D_{\text{eval}}$):** Also known as held-out or test data. This subset is *not* used during parameter estimation. It serves to provide an unbiased estimate of the classifier's generalization performance – its expected performance on unseen data drawn from $p(\mathbf{x}, \omega)$.

### 3. **Loss Functions and Error**

To quantify the "cost" of making a potentially incorrect decision, we introduce a **loss function**, $L(\hat{\omega}, \omega_{\text{true}})$. This function assigns a numerical value (cost) when the classifier predicts class $\hat{\omega} = f(\mathbf{x}; \theta)$ but the true class is $\omega_{\text{true}}$.

* **Expected Loss ($L_{\text{act}}$ - Actual or True Loss):** The fundamental measure of a classifier's performance is its average loss over the entire data distribution. For a given classifier $f(\mathbf{x}; \theta)$, the expected loss is defined by integrating the loss incurred for each possible true class $\omega_i$, weighted by the probability of that class occurring given the features $\mathbf{x}$, and averaged over the distribution of features $p(\mathbf{x})$:

$$L_{\text{act}}(\theta) = \mathbb{E}_{\mathbf{x}, \omega} [L(f(\mathbf{x}; \theta), \omega)] = \int_{\mathbb{R}^d} \left[ \sum_{i=1}^K L(f(\mathbf{x}; \theta), \omega_i) P(\omega_i \mid \mathbf{x}) \right] p(\mathbf{x}) \, d\mathbf{x}$$

The goal of classifier design is to find parameters $\theta^*$ that minimize $L_{\text{act}}(\theta)$.

* **Empirical Loss ($L_{\text{emp}}$):** Since $p(\mathbf{x}, \omega)$ is unknown, we cannot compute $L_{\text{act}}$ directly. Instead, we estimate it using the available data. The empirical loss on a dataset $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ is the average loss observed on that set:

$$L_{\text{emp}}(\theta; D) = \frac{1}{N} \sum_{i=1}^N L(f(\mathbf{x}_i; \theta), y_i)$$

By the Law of Large Numbers, as the size of the dataset $N \to \infty$, the empirical loss converges to the expected loss: $L_{\text{emp}}(\theta; D) \to L_{\text{act}}(\theta)$. This justifies using the empirical loss on the evaluation set $D_{\text{eval}}$ as an estimate of the true performance. For finite $N$, typically $L_{\text{act}}(\theta) \geq L_{\text{emp}}(\theta; D_{\text{train}})$, especially if $\theta$ was optimized on $D_{\text{train}}$ (overfitting).

* **0-1 Loss Function:** A very common and intuitive loss function is the 0-1 loss, which penalizes all errors equally:

$$L_{0-1}(\hat{\omega}, \omega_{\text{true}}) = \begin{cases} 0, & \text{if } \hat{\omega} = \omega_{\text{true}} \\ 1, & \text{if } \hat{\omega} \neq \omega_{\text{true}} \end{cases} = I(\hat{\omega} \neq \omega_{\text{true}})$$

where $I(\cdot)$ is the indicator function. Under the 0-1 loss, minimizing the expected loss $L_{\text{act}}$ is equivalent to minimizing the overall probability of misclassification, $P(\text{error})$.

### 4. **Bayes' Decision Rule**

The decision rule that minimizes the expected loss $L_{\text{act}}$ is known as **Bayes' decision rule**. To derive it, consider a specific input feature vector $\mathbf{x}^\star$. We want to choose the predicted class $\hat{\omega} = f(\mathbf{x}^\star; \theta)$ that minimizes the *conditional expected loss* given $\mathbf{x}^\star$:

$$R(\hat{\omega} \mid \mathbf{x}^\star) = \mathbb{E}_{\omega \mid \mathbf{x}^\star} [L(\hat{\omega}, \omega)] = \sum_{i=1}^K L(\hat{\omega}, \omega_i) P(\omega_i \mid \mathbf{x}^\star)$$

The Bayes' decision rule selects the class $\hat{\omega}$ that minimizes this conditional risk for the given $\mathbf{x}^\star$:

$$\hat{\omega}_{\text{Bayes}}(\mathbf{x}^\star) = \arg \min_{\omega_j \in \{\omega_1, \dots, \omega_K\}} R(\omega_j \mid \mathbf{x}^\star) = \arg \min_{\omega_j} \sum_{i=1}^K L(\omega_j, \omega_i) P(\omega_i \mid \mathbf{x}^\star)$$

This rule provides the minimum possible expected loss, $L_{\text{act}}$, achievable for the given problem and loss function; this minimum value is called the **Bayes risk**.

* **Bayes' Rule under 0-1 Loss:** For the specific case of the 0-1 loss, $L_{0-1}(\omega_j, \omega_i) = 1 - \delta_{ji}$ (where $\delta_{ji}$ is the Kronecker delta). The conditional risk becomes:

$$R(\omega_j \mid \mathbf{x}^\star) = \sum_{i=1}^K (1 - \delta_{ji}) P(\omega_i \mid \mathbf{x}^\star) = \sum_{i=1}^K P(\omega_i \mid \mathbf{x}^\star) - P(\omega_j \mid \mathbf{x}^\star) = 1 - P(\omega_j \mid \mathbf{x}^\star)$$

Minimizing $1 - P(\omega_j \mid \mathbf{x}^\star)$ is equivalent to maximizing $P(\omega_j \mid \mathbf{x}^\star)$. Therefore, under 0-1 loss, Bayes' decision rule simplifies to selecting the class with the maximum *a posteriori* probability (MAP):

$$\hat{\omega}_{\text{MAP}}(\mathbf{x}^\star) = \arg \max_{\omega_j \in \{\omega_1, \dots, \omega_K\}} P(\omega_j \mid \mathbf{x}^\star)$$

* **Model Approximation:** In practice, the true posterior probabilities $P(\omega_j \mid \mathbf{x}^\star)$ are usually unknown. We train a model $f(\mathbf{x}; \theta)$ to approximate them. For instance, the model might output scores or probabilities $P(\omega_j \mid \mathbf{x}^\star, \theta)$, and the decision rule becomes:

$$\hat{\omega}(\mathbf{x}^\star; \theta) = \arg \max_{\omega_j} P(\omega_j \mid \mathbf{x}^\star, \theta)$$

The goal of training is to make $P(\omega_j \mid \mathbf{x}, \theta)$ as close as possible to the true $P(\omega_j \mid \mathbf{x})$.

### 5. **Binary Classification Specifics ($K=2$)**

Let the two classes be $\omega_1$ and $\omega_2$. Often, labels $y_i \in \{+1, -1\}$ or $\{1, 0\}$ are used.

* **Empirical Error Rate Estimation:** On an evaluation set $D_{\text{eval}} = \{(\mathbf{x}_i^\star, y_i^\star)\}_{i=1}^{N^\star}$, the empirical error rate (estimate of $P(\text{error})$) under 0-1 loss is:

$$\hat{P}(\text{error}) = L_{\text{eval}} = \frac{1}{N^\star} \sum_{i=1}^{N^\star} I(f(\mathbf{x}_i^\star; \theta) \neq y_i^\star)$$

If labels are $y_i^\star \in \{+1, -1\}$ and the classifier outputs $f(\mathbf{x}_i^\star; \theta) \in \{+1, -1\}$, this can sometimes be written using the property that $|f - y|/2 = 1$ if $f \neq y$ and $0$ if $f = y$:

$$\hat{P}(\text{error}) = \frac{1}{2N^\star} \sum_{i=1}^{N^\star} |f(\mathbf{x}_i^\star; \theta) - y_i^\star|$$

* **True Probability of Error:** Let $\Omega_1 = \{\mathbf{x} \mid f(\mathbf{x}; \theta) = \omega_1\}$ and $\Omega_2 = \{\mathbf{x} \mid f(\mathbf{x}; \theta) = \omega_2\}$ be the decision regions in the feature space where the classifier predicts $\omega_1$ and $\omega_2$, respectively ($\mathbb{R}^d = \Omega_1 \cup \Omega_2$). An error occurs if the true class is $\omega_1$ but $\mathbf{x}$ falls in $\Omega_2$, or if the true class is $\omega_2$ but $\mathbf{x}$ falls in $\Omega_1$. The total probability of error is the sum of probabilities of these two disjoint events:

$$P(\text{error}) = P(\mathbf{x} \in \Omega_2, \omega = \omega_1) + P(\mathbf{x} \in \Omega_1, \omega = \omega_2)$$

Using the definition of joint probability ($p(\mathbf{x}, \omega) = p(\mathbf{x}|\omega)P(\omega)$), this can be written as integrals over the decision regions:

$$P(\text{error}) = \int_{\Omega_2} p(\mathbf{x}, \omega_1) \, d\mathbf{x} + \int_{\Omega_1} p(\mathbf{x}, \omega_2) \, d\mathbf{x} = \int_{\Omega_2} p(\mathbf{x} \mid \omega_1) P(\omega_1) \, d\mathbf{x} + \int_{\Omega_1} p(\mathbf{x} \mid \omega_2) P(\omega_2) \, d\mathbf{x}$$

* **Applying Bayes' Theorem for Decision:** The MAP rule is to decide $\omega_1$ if $P(\omega_1 \mid \mathbf{x}) > P(\omega_2 \mid \mathbf{x})$, and $\omega_2$ otherwise. Using Bayes' theorem:

$$P(\omega_i \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \omega_i) P(\omega_i)}{p(\mathbf{x})}$$

Since $p(\mathbf{x}) = \sum_j p(\mathbf{x} \mid \omega_j) P(\omega_j)$ is positive and common to both posteriors, the comparison $P(\omega_1 \mid \mathbf{x}) > P(\omega_2 \mid \mathbf{x})$ is equivalent to comparing the joint probabilities or the unnormalized posteriors:

$$p(\mathbf{x} \mid \omega_1) P(\omega_1) > p(\mathbf{x} \mid \omega_2) P(\omega_2)$$

This forms the basis for generative classifiers.

![[prob-error.png]]

### 6. **Decision Boundaries**

The **decision boundary** is the set of points $\mathbf{x}$ in the feature space where the classifier is "indifferent" between two or more classes. For the Bayes classifier under 0-1 loss, this occurs where the posterior probabilities for the top competing classes are equal.

* **Binary Case:** The decision boundary is the surface defined by $P(\omega_1 \mid \mathbf{x}) = P(\omega_2 \mid \mathbf{x})$. Assuming $p(\mathbf{x}) > 0$, this is equivalent to $p(\mathbf{x} \mid \omega_1) P(\omega_1) = p(\mathbf{x} \mid \omega_2) P(\omega_2)$. If we use a model approximation $P(\omega_i \mid \mathbf{x}, \theta)$, the boundary is $P(\omega_1 \mid \mathbf{x}, \theta) = P(\omega_2 \mid \mathbf{x}, \theta)$. Taking logarithms (often simplifying calculations, especially with exponential family distributions like Gaussians):

$$\log P(\omega_1 \mid \mathbf{x}, \theta) = \log P(\omega_2 \mid \mathbf{x}, \theta)$$

Or equivalently:

$$\log p(\mathbf{x} \mid \omega_1, \theta_1) + \log P(\omega_1) = \log p(\mathbf{x} \mid \omega_2, \theta_2) + \log P(\omega_2)$$

* **Multivariate Gaussian Case:** Assume the class-conditional densities are multivariate Gaussians: $p(\mathbf{x} \mid \omega_i, \theta_i) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$. The log-likelihood term is $\log p(\mathbf{x} \mid \omega_i, \theta_i) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i) - \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}_i|$. Substituting this into the boundary equation $\log(p_1 P_1) = \log(p_2 P_2)$ and simplifying leads to an equation involving quadratic terms in $\mathbf{x}$:

$$-\frac{1}{2} \mathbf{x}^T \boldsymbol{\Sigma}_1^{-1} \mathbf{x} + \mathbf{x}^T \boldsymbol{\Sigma}_1^{-1} \boldsymbol{\mu}_1 - \frac{1}{2} \boldsymbol{\mu}_1^T \boldsymbol{\Sigma}_1^{-1} \boldsymbol{\mu}_1 - \frac{1}{2}\log|\boldsymbol{\Sigma}_1| + \log P(\omega_1) = \dots (\text{similar terms for class 2})$$

Rearranging this yields a general quadratic decision boundary of the form:

$$\mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c = 0$$

where:

* $A = \frac{1}{2} (\boldsymbol{\Sigma}_2^{-1} - \boldsymbol{\Sigma}_1^{-1})$
* $\mathbf{b} = \boldsymbol{\Sigma}_1^{-1} \boldsymbol{\mu}_1 - \boldsymbol{\Sigma}_2^{-1} \boldsymbol{\mu}_2$
* $c = \frac{1}{2} (\boldsymbol{\mu}_2^T \boldsymbol{\Sigma}_2^{-1} \boldsymbol{\mu}_2 - \boldsymbol{\mu}_1^T \boldsymbol{\Sigma}_1^{-1} \boldsymbol{\mu}_1) + \frac{1}{2} \log \frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} + \log \frac{P(\omega_2)}{P(\omega_1)}$

(Note: The exact definitions of $A, \mathbf{b}, c$ might differ slightly by constant factors depending on algebraic convention, but the quadratic nature remains).

* **Special Case: Equal Covariances ($\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \boldsymbol{\Sigma}$):** If the covariance matrices are identical, the quadratic terms $\mathbf{x}^T A \mathbf{x}$ cancel out ($A=0$). The decision boundary becomes linear (a hyperplane):

$$\mathbf{b}^T \mathbf{x} + c = 0$$

where $\mathbf{b} = \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$ and $c$ simplifies accordingly. This is the basis of Linear Discriminant Analysis (LDA).

### 7. **Classifier Types: Generative vs. Discriminative**

There are two primary approaches to building classifiers:

* **Generative Models:**
    * *Approach:* Model how the data for each class is generated. This involves estimating the class priors $P(\omega_i)$ and the class-conditional densities $p(\mathbf{x} \mid \omega_i, \theta_i)$ for each class $i$. The joint distribution is implicitly modeled: $p(\mathbf{x}, \omega_i \mid \theta_i) = p(\mathbf{x} \mid \omega_i, \theta_i) P(\omega_i)$.
    * *Decision:* Use Bayes' theorem to compute the posterior probabilities needed for the decision rule:

$$P(\omega_i \mid \mathbf{x}, \theta) = \frac{p(\mathbf{x} \mid \omega_i, \theta_i) P(\omega_i)}{\sum_{j=1}^K p(\mathbf{x} \mid \omega_j, \theta_j) P(\omega_j)}$$

Then apply the Bayes decision rule (e.g., MAP for 0-1 loss).
    * *Parameter Estimation:* Typically involves maximizing the likelihood for each class separately using the data belonging to that class, or maximizing the overall joint likelihood: $\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^N \log p(\mathbf{x}_i, y_i \mid \theta)$.
    * *Examples:* Naive Bayes, Gaussian Discriminant Analysis (LDA, QDA), Hidden Markov Models (HMMs).
    * *Probability of Error:* Calculated via $P(\text{error}) = \int_{\Omega_2} p(\mathbf{x} \mid \omega_1) P(\omega_1) \, d\mathbf{x} + \int_{\Omega_1} p(\mathbf{x} \mid \omega_2) P(\omega_2) \, d\mathbf{x}$.

* **Discriminative Models:**
    * *Approach:* Model the posterior probability $P(\omega_i \mid \mathbf{x}, \theta)$ directly, without explicitly modeling the class-conditional densities or the marginal $p(\mathbf{x})$. Alternatively, learn a direct mapping (discriminant function) $f: \mathbb{R}^d \to \{\omega_1, \dots, \omega_K\}$.
    * *Decision:* The model directly outputs the posterior or the class label.
    * *Parameter Estimation:* Typically involves maximizing the conditional likelihood of the labels given the features:

$$\hat{\theta} = \arg \max_\theta \sum_{i=1}^N \log P(y_i \mid \mathbf{x}_i, \theta)$$

    * *Examples:* Logistic Regression, Support Vector Machines (SVMs), Neural Networks (standard feed-forward), Conditional Random Fields (CRFs).
    * *Probability of Error:* Calculated via $P(\text{error}) = \int_{\Omega_2} P(\omega_1 \mid \mathbf{x}, \theta) p(\mathbf{x}) \, d\mathbf{x} + \int_{\Omega_1} P(\omega_2 \mid \mathbf{x}, \theta) p(\mathbf{x}) \, d\mathbf{x}$. Note that $p(\mathbf{x})$ is not modeled.

* *Comparison:* Generative models can model the data distribution (useful for outlier detection, generating new samples), may work better with less data, but make stronger assumptions. Discriminative models focus directly on the decision boundary, are often more flexible, and can achieve higher accuracy if the generative assumptions are wrong, but they don't model $p(\mathbf{x})$.

### 8. **Unequal Loss Functions and ROC Analysis**

The 0-1 loss assumes all errors are equally costly. In many real-world scenarios (medical diagnosis, fraud detection), this is unrealistic.

* **Unequal Loss:** Define specific costs $L(\hat{\omega}_j, \omega_i) = C_{ji}$ for predicting class $j$ when the true class is $i$. Typically $C_{ii}=0$. For binary classification ($\omega_1, \omega_2$):
    * $C_{12}$: Cost of predicting $\omega_1$ when true is $\omega_2$ (False Positive Cost if $\omega_1$ is "positive").
    * $C_{21}$: Cost of predicting $\omega_2$ when true is $\omega_1$ (False Negative Cost if $\omega_1$ is "positive").
* **Modified Bayes' Decision Rule:** Choose $\hat{\omega}$ to minimize conditional expected loss $R(\hat{\omega} \mid \mathbf{x}^\star)$. For binary case, compare the risk of predicting $\omega_1$ vs. $\omega_2$:
    * $R(\omega_1 \mid \mathbf{x}^\star) = L(\omega_1, \omega_1)P(\omega_1|\mathbf{x}^\star) + L(\omega_1, \omega_2)P(\omega_2|\mathbf{x}^\star) = C_{12} P(\omega_2 \mid \mathbf{x}^\star)$.
    * $R(\omega_2 \mid \mathbf{x}^\star) = L(\omega_2, \omega_1)P(\omega_1|\mathbf{x}^\star) + L(\omega_2, \omega_2)P(\omega_2|\mathbf{x}^\star) = C_{21} P(\omega_1 \mid \mathbf{x}^\star)$.
    * Decision: Predict $\omega_1$ if $R(\omega_1 \mid \mathbf{x}^\star) < R(\omega_2 \mid \mathbf{x}^\star)$, i.e., if $C_{12} P(\omega_2 \mid \mathbf{x}^\star) < C_{21} P(\omega_1 \mid \mathbf{x}^\star)$.
    * Decision in terms of Posterior Ratio: Predict $\omega_1$ if

$$\frac{P(\omega_1 \mid \mathbf{x}^\star, \theta)}{P(\omega_2 \mid \mathbf{x}^\star, \theta)} > \frac{C_{12}}{C_{21}}$$

The ratio of costs $C_{12}/C_{21}$ acts as an adjustable **operating threshold** on the likelihood ratio or posterior ratio.

* **Receiver Operating Characteristic (ROC) Curve:**
    * *Context:* Useful for binary classification where $\omega_1$ is considered "positive" and $\omega_2$ "negative".
    * *Metrics:*
        * True Positive Rate (TPR) / Sensitivity / Recall = $P(\hat{\omega}=\omega_1 | \omega_{\text{true}}=\omega_1)$.
        * False Positive Rate (FPR) = $P(\hat{\omega}=\omega_1 | \omega_{\text{true}}=\omega_2) = 1 - \text{Specificity}$.
    * *Curve Generation:* By varying the decision threshold (implicitly, the cost ratio $C_{12}/C_{21}$) from $-\infty$ to $+\infty$, we trace a curve in the (FPR, TPR) space.
    * *Interpretation:* The curve shows the trade-off between sensitivity and specificity. A classifier closer to the top-left corner (TPR=1, FPR=0) is better. The Area Under the Curve (AUC) summarizes overall performance across all thresholds.
    * *Application:* Comparing different classifiers independently of specific cost ratios or class priors. Visualizing performance trade-offs.

![[roc.png]]
### 9. **Training the Decision Process (Revisited)**

* **Supervised Training:** Goal is to find parameters $\theta$ using $D_{\text{train}} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^N$.
* **Discriminative Training:** Maximize conditional likelihood $\mathcal{L}_C(\theta) = \sum_{i=1}^N \log P(y_i \mid \mathbf{x}_i, \theta)$
* **Generative Training:**
    * Estimate parameters $\theta_i$ for each class-conditional density $p(\mathbf{x} \mid \omega_i, \theta_i)$ and priors $P(\omega_i)$.
    * ML for Class-Conditional Density: $\theta_i = \arg\max_{\theta} \sum_{j \text{ s.t. } y_j = \omega_i} \log p(\mathbf{x}_j \mid \omega_i, \theta)$.
    * ML Estimates for Gaussian Parameters:
        * $\hat{\boldsymbol{\mu}}_i = \frac{1}{N_i} \sum_{j: y_j = \omega_i} \mathbf{x}_j$ (Sample mean for class $i$).
        * $\hat{\boldsymbol{\Sigma}}_i = \frac{1}{N_i} \sum_{j: y_j = \omega_i} (\mathbf{x}_j - \hat{\boldsymbol{\mu}}_i) (\mathbf{x}_j - \hat{\boldsymbol{\mu}}_i)^T$ (Sample covariance for class $i$, where $N_i = \sum_{j: y_j = \omega_i} 1$). Note: Sometimes $(N_i-1)$ is used for an unbiased estimate, but ML uses $N_i$.
* **Class Prior Estimation:**
    * Counting (ML Estimate): $\hat{P}(\omega_i) = \frac{N_i}{N}$.
    * Smoothing (e.g., Laplace/Add-One): $\hat{P}(\omega_i) = \frac{N_i + 1}{N + K}$. Prevents zero probabilities, acts as a simple Bayesian prior.

### 10. **Multivariate Gaussian Models (Revisited)**

* **PDF Formula:** Recap of the Gaussian density $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$.
* **Decision Boundary Derivation:** Reiteration that equating log-posteriors (or log-joints + log-priors) for Gaussian densities leads to quadratic boundaries in general, simplifying to linear boundaries when covariance matrices are assumed equal ($\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2$).

### 11. **Evaluating Classifier Performance**

* **Bayes' Optimality:** The Bayes' decision rule achieves the lowest possible expected loss (Bayes risk) *if* the true posterior probabilities $P(\omega_i \mid \mathbf{x})$ are known.
* **Practical Challenges:**
    * *Limited Training Data:* Leads to inaccurate estimates of $P(\omega_i \mid \mathbf{x})$ or model parameters $\theta$. The empirical risk minimizer might not minimize the true risk.
    * *Model Mismatch:* The chosen model family (e.g., Gaussian assumption in LDA/QDA, linearity in logistic regression) may not accurately represent the true underlying data distributions or decision boundary.
    * *Optimization Issues:* The algorithm used to find the optimal parameters $\theta$ (e.g., gradient descent) might converge to a local optimum rather than the global optimum of the objective function (e.g., likelihood).
* **Engineering Trade-offs:** Classifier design involves choices (generative vs. discriminative, model complexity, loss function, regularization, optimization algorithm) that impact performance, robustness, computational cost, and interpretability. Evaluation on held-out data and techniques like cross-validation are crucial for making informed choices and estimating true generalization performance.

---

## **Part II: Graphical Models for Structured Probability Distributions**

**(Based on Lectures by Jose Miguel Hernandez Lobato)**

### 1. **Fundamentals of Probability (Recap)**

Graphical models are built upon the fundamental rules of probability theory.

* **Sum Rule (Marginalization):** Allows computing the distribution of a subset of variables from a joint distribution by summing or integrating over the unwanted variables. For random variables $\mathbf{X}, \mathbf{Y}$:

$$p(\mathbf{X}) = \sum_{\mathbf{Y}} p(\mathbf{X}, \mathbf{Y}) \quad \text{(if } \mathbf{Y} \text{ is discrete)}$$
$$p(\mathbf{X}) = \int p(\mathbf{X}, \mathbf{Y}) \, d\mathbf{Y} \quad \text{(if } \mathbf{Y} \text{ is continuous)}$$

* **Product Rule (Chain Rule):** Expresses the joint distribution in terms of conditional probabilities:

$$p(\mathbf{X}, \mathbf{Y}) = p(\mathbf{Y} \mid \mathbf{X}) p(\mathbf{X}) = p(\mathbf{X} \mid \mathbf{Y}) p(\mathbf{Y})$$

For multiple variables $\mathbf{X}_1, \dots, \mathbf{X}_d$:

$$p(\mathbf{X}_1, \dots, \mathbf{X}_d) = p(\mathbf{X}_1) p(\mathbf{X}_2 | \mathbf{X}_1) p(\mathbf{X}_3 | \mathbf{X}_1, \mathbf{X}_2) \dots p(\mathbf{X}_d | \mathbf{X}_1, \dots, \mathbf{X}_{d-1})$$

* **Bayes' Rule:** Derived from the product rule, allows relating $p(\mathbf{Y} \mid \mathbf{X})$ to $p(\mathbf{X} \mid \mathbf{Y})$:

$$p(\mathbf{Y} \mid \mathbf{X}) = \frac{p(\mathbf{X} \mid \mathbf{Y}) p(\mathbf{Y})}{p(\mathbf{X})} = \frac{p(\mathbf{X} \mid \mathbf{Y}) p(\mathbf{Y})}{\sum_{\mathbf{Y}'} p(\mathbf{X}, \mathbf{Y}')}$$

Here, $p(\mathbf{Y} \mid \mathbf{X})$ is the posterior, $p(\mathbf{X} \mid \mathbf{Y})$ is the likelihood, $p(\mathbf{Y})$ is the prior, and $p(\mathbf{X})$ is the evidence or marginal likelihood.

### 2. **Independence and Conditional Independence**

These concepts are crucial for simplifying complex probability distributions.

* **Independence:** Two variables $\mathbf{X}$ and $\mathbf{Y}$ are independent, denoted $\mathbf{X} \perp \mathbf{Y}$, if their joint distribution factorizes into the product of their marginal distributions:

$$\mathbf{X} \perp \mathbf{Y} \iff p(\mathbf{X}, \mathbf{Y}) = p(\mathbf{X}) p(\mathbf{Y})$$

Equivalently, $p(\mathbf{Y} \mid \mathbf{X}) = p(\mathbf{Y})$ or $p(\mathbf{X} \mid \mathbf{Y}) = p(\mathbf{X})$. Knowing the value of one variable provides no information about the other.

![[independence-example.png]]

* **Conditional Independence:** $\mathbf{X}$ and $\mathbf{Y}$ are conditionally independent given $\mathbf{Z}$, denoted $\mathbf{X} \perp \mathbf{Y} \mid \mathbf{Z}$, if their joint conditional distribution factorizes given $\mathbf{Z}$:

$$\mathbf{X} \perp \mathbf{Y} \mid \mathbf{Z} \iff p(\mathbf{X}, \mathbf{Y} \mid \mathbf{Z}) = p(\mathbf{X} \mid \mathbf{Z}) p(\mathbf{Y} \mid \mathbf{Z})$$

Equivalently, $p(\mathbf{Y} \mid \mathbf{X}, \mathbf{Z}) = p(\mathbf{Y} \mid \mathbf{Z})$ or $p(\mathbf{X} \mid \mathbf{Y}, \mathbf{Z}) = p(\mathbf{X} \mid \mathbf{Z})$. Once the value of $\mathbf{Z}$ is known, $\mathbf{X}$ provides no *additional* information about $\mathbf{Y}$ (and vice versa).

* **Examples:**
    * *Independence:* A uniform distribution $p(X, Y)=1$ over the unit square $[0,1]\times[0,1]$ shows independence ($p(X)=1, p(Y)=1$). A density like $p(X, Y) = 2 \cdot I[X+Y < 1]$ over the same square is not independent.
    * *Conditional Independence:* Consider a mixture model $p(X, Y, Z) = p(Z) p(X | Z) p(Y | Z)$. Given $Z$, $X$ and $Y$ are independent. For example, let $Z \in \{0, 1\}$ with $p(Z=1)=0.5$. If $p(X, Y|Z=1) = \mathcal{N}(X|2,1)\mathcal{N}(Y|2,1)$ and $p(X, Y|Z=0) = \mathcal{N}(X|-2,1)\mathcal{N}(Y|-2,1)$, then $X \perp Y \mid Z$. However, $X$ and $Y$ are *not* marginally independent, as they are correlated (both tend to be positive or both negative). The more complex mixture involving four Gaussians presented in the notes shows a case where $X$ and $Y$ are *not* conditionally independent given $Z$.

![[cond-independence-example.png]]
### 3. **Motivation: Factorization and Compact Representation**

Modeling the joint distribution $p(\mathbf{X}_1, \dots, \mathbf{X}_d)$ directly is often intractable. If each $\mathbf{X}_i$ can take $M$ values, a full joint probability table requires $M^d - 1$ parameters. Conditional independence allows factorization, drastically reducing complexity.

* **Example Factorization:** $p(A, B, C, D) = p(A \mid C) \, p(B \mid C) \, p(C) \, p(D)$. Assuming binary variables ($M=2$):
    * The full joint requires $2^4 - 1 = 15$ parameters.
    * The factored form requires parameters for: $p(C)$ (1), $p(D)$ (1), $p(A|C)$ (2 parameters, one for $C=0$, one for $C=1$), $p(B|C)$ (2 parameters). Total = $1+1+2+2 = 6$ parameters.

![[motivation-cond-independence.png]]

* **Implied Independencies:** The factorization structure *encodes* conditional independence assumptions. From $p(A, B, C, D) = p(A \mid C) \, p(B \mid C) \, p(C) \, p(D)$, we can infer:
    * $D$ is independent of $A, B, C$: $D \perp \{A, B, C\}$. (Since $p(A,B,C,D) = p(A,B,C)p(D)$).
    * Given $C$, $A$ and $B$ are independent: $A \perp B \mid C$. (Since $p(A, B \mid C) = p(A, B, C)/p(C) = [p(A|C)p(B|C)p(C)]/p(C) = p(A|C)p(B|C)$).

* **Application to Language Models:**
    * The chain rule gives $p(W_1, \dots, W_T) = \prod_{t=1}^T p(W_t | W_1, \dots, W_{t-1})$.
    * The number of possible conditioning contexts $W_1, \dots, W_{t-1}$ grows exponentially, making estimation impossible (data sparsity).
    * The **Markov assumption** introduces conditional independence: $p(W_t | W_1, \dots, W_{t-1}) \approx p(W_t | W_{t-N+1}, \dots, W_{t-1})$. This assumes the next word depends only on the $N-1$ preceding words (N-gram model).
        * *First Order (Bigram):* $p(W_1, \dots, W_T) \approx p(W_1) \prod_{t=2}^T p(W_t \mid W_{t-1})$.
        * *Second Order (Trigram):* $p(W_1, \dots, W_T) \approx p(W_1) p(W_2|W_1) \prod_{t=3}^T p(W_t \mid W_{t-1}, W_{t-2})$.
    * This factorization allows probabilities to be estimated from counts of N-grams.

### 4. **Structured Distributions and the Big Picture**

Graphical models provide a framework to represent and work with these structured factorizations. They offer:

* A visual language to represent conditional independence assumptions.
* A way to define complex distributions using simpler, local interactions.
* Algorithms for efficient inference (e.g., computing marginals $p(X_i)$ or conditionals $p(X_i | \text{evidence})$) that exploit the factorization.

### 5. **Directed Graphical Models (Bayesian Networks - BNs)**

* **Definition:** A BN consists of:
    * A Directed Acyclic Graph (DAG) $\mathcal{G} = (V, E)$, where nodes $V = \{1, \dots, d\}$ represent random variables $X_1, \dots, X_d$.
    * A set of conditional probability distributions (CPDs), $p(X_i | \text{Pa}_{\mathcal{G}}(X_i))$, for each node $X_i$, where $\text{Pa}_{\mathcal{G}}(X_i)$ are the parents of $X_i$ in the DAG $\mathcal{G}$.

* **Factorization:** The graph $\mathcal{G}$ implies a factorization of the joint distribution:

$$p(X_1, \dots, X_d) = \prod_{i=1}^d p(X_i \mid \text{Pa}_{\mathcal{G}}(X_i))$$

* **Conditional Independence Property:** The factorization implies specific conditional independencies. A key one is: Each node $X_i$ is conditionally independent of its non-descendants, given its parents: $X_i \perp \text{ND}(X_i) \mid \text{Pa}_{\mathcal{G}}(X_i)$. (More general independence queries can be answered using the concept of *d-separation* in the graph).

* **Example:** For the graph S->F, S->H, F->C, H->C, F->M:
    * Factorization: $p(S, F, H, C, M) = p(S) p(F \mid S) p(H \mid S) p(C \mid F, H) p(M \mid F)$.
    * Implied Independencies: $F \perp H \mid S$; $C \perp S \mid \{F, H\}$; $M \perp \{S, H, C\} \mid F$.

* **Efficient Marginalization (Variable Elimination):** Computing marginals like $p(D)$ from $p(A,B,C,D) = p(A)p(B|A)p(C|B)p(D|C)$ can be done efficiently by pushing sums inwards:

$$p(D) = \sum_C p(D|C) \sum_B p(C|B) \sum_A p(B|A) p(A)$$

The complexity scales with the size of the largest intermediate factor created (related to graph treewidth), often much better than the exponential cost of summing over the full joint table.

### 6. **Undirected Graphical Models (Markov Networks / Markov Random Fields - MRFs)**

* **Motivation:** Used when dependencies are naturally symmetric, without a clear causal direction (e.g., pixels in an image, variables in certain physical systems). Representing the precision matrix (inverse covariance) of a Gaussian: if $\Lambda_{ij}=0$, then $X_i \perp X_j \mid \text{Rest}$. This conditional independence structure is naturally represented by an undirected edge $(i,j)$ being absent.

* **Definition:** An MRF consists of:
    * An Undirected Graph $\mathcal{G} = (V, E)$.
    * A set of non-negative potential functions (or factors) $\phi_C(X_C)$ defined over cliques $C$ of the graph. A clique is a subset of nodes where every pair is connected by an edge. Often, maximal cliques are used.

* **Factorization (Hammersley-Clifford Theorem):** The joint distribution factorizes over the (maximal) cliques of the graph:

$$p(X_1, \dots, X_d) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \phi_C(X_C)$$

where $\mathcal{C}$ is the set of (maximal) cliques, and $X_C$ denotes the variables in clique $C$.

* **Partition Function:** $Z = \sum_{x_1, \dots, x_d} \prod_{C \in \mathcal{C}} \phi_C(x_C)$ is the normalization constant, ensuring the distribution sums/integrates to 1. Computing $Z$ is generally computationally hard (#P-complete). Potentials are often represented in log-space: $\phi_C(X_C) = \exp(-\psi_C(X_C))$, where $\psi_C$ is an energy function.

* **Conditional Independence Property (Global Markov Property):** For disjoint sets of nodes $A, B, C$, if $C$ separates $A$ from $B$ in the graph (i.e., all paths from any node in $A$ to any node in $B$ must pass through $C$), then $A \perp B \mid C$. Other related properties (Local Markov, Pairwise Markov) also hold.

* **Example:** Graph A-B, B-C, C-D, D-A (a square). Maximal cliques are pairs of connected nodes. Factorization over edges (pairwise MRF):

$$p(A, B, C, D) = \frac{1}{Z} \phi_{AB}(A, B) \, \phi_{BC}(B, C) \, \phi_{CD}(C, D) \, \phi_{DA}(D, A)$$

Implied Independencies: $A \perp C \mid \{B, D\}$ (removing B and D disconnects A and C). Similarly, $B \perp D \mid \{A, C\}$.

### 7. **The Potts Model (Example MRF)**

* **Application:** Commonly used for image segmentation, clustering, modeling interacting spin systems in physics.
* **Model:** Variables $x_i \in \{1, \dots, K\}$ represent the label/state of node $i$ (e.g., pixel label). Typically defined on a grid graph where nodes are pixels and edges connect neighbors.
* **Joint Distribution (Pairwise):**

$$p(x_1, \dots, x_n) = \frac{1}{Z} \exp \left( \sum_{(i,j) \in E} J_{ij} \delta(x_i, x_j) + \sum_i h_i(x_i) \right)$$

* The first term encourages neighboring nodes $(i,j)$ to have the same label if $J_{ij} > 0$ (ferromagnetic coupling). The interaction potential is $\phi_{ij}(x_i, x_j) = \exp(J_{ij} \delta(x_i, x_j))$. The notes use $\log \phi_{ij}(x_i, x_j) = \beta$ if $x_i = x_j$, equivalent to $J_{ij}=\beta$.
* The second term $h_i(x_i)$ represents an external field or node-specific potential (e.g., related to the observed pixel color likelihood for label $x_i$).
* **Intuition:** The model balances adherence to local data (via $h_i$) with spatial smoothness (via $J_{ij}$), encouraging connected regions of the same label. Inference (finding the most likely labeling or marginals) is often done using approximate methods like Belief Propagation or MCMC.

---

## **Part III: Generative Models, Mixture Models, and the EM Algorithm**

**(Based on Lectures by Andrew Fitzgibbon)**

![[latent-generative-models.png]]
### 1. **From Discriminative to Generative Modeling**

Discriminative classifiers (Part I) directly model $P(\omega | \mathbf{x})$ or learn a boundary function. While often effective for classification, they have limitations:

* They **do not model the input distribution $p(\mathbf{x})$**. This means they cannot easily detect outliers or atypical inputs, nor can they generate new samples resembling the training data. The question arises: *"How do you know what you don't know?"*
* They might struggle if the underlying structure of $p(\mathbf{x})$ is informative for classification but not easily captured by the chosen discriminative model form.

Generative models, by contrast, aim to model the joint distribution $p(\mathbf{x}, \omega) = p(\mathbf{x} | \omega) P(\omega)$, which implicitly includes modeling $p(\mathbf{x}) = \sum_\omega p(\mathbf{x} | \omega) P(\omega)$. This allows addressing the limitations above but often requires stronger assumptions (like the Gaussian forms in LDA/QDA). When simple assumptions are insufficient, more flexible generative models like **Mixtures of Gaussians (GMMs)** can be employed.

### 2. **Mixture of Gaussians (GMMs)**

A GMM represents the overall data density $p(\mathbf{x})$ as a weighted sum of $K$ individual Gaussian components:

$$p(\mathbf{x} | \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where:

* $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_k|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)\right)$ is the PDF of the $k$-th Gaussian component with mean $\boldsymbol{\mu}_k$ and covariance $\boldsymbol{\Sigma}_k$.
* $\pi_k$ are the **mixing coefficients**, representing the prior probability that a data point was generated by component $k$. They must satisfy $\pi_k \ge 0$ and $\sum_{k=1}^K \pi_k = 1$.
* $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K$ represents the set of all parameters.

**Latent Variable Interpretation:** A GMM can be viewed as a generative process involving a discrete latent variable $z \in \{1, \dots, K\}$, indicating which component generated a data point $\mathbf{x}$:

1. Choose a component $z \sim \text{Categorical}(\pi_1, \dots, \pi_K)$, i.e., $P(z=k) = \pi_k$.
2. Generate $\mathbf{x}$ from the chosen Gaussian component: $\mathbf{x} | (z=k) \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

The marginal distribution $p(\mathbf{x}) = \sum_{k=1}^K P(z=k) p(\mathbf{x}|z=k)$ recovers the GMM formula. The key challenge is that $z$ is *latent* (unobserved).

### 3. **Maximum Likelihood Estimation (MLE) for GMMs**

Given an i.i.d. dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$, the goal is to find parameters $\boldsymbol{\theta}$ that maximize the likelihood (or log-likelihood) of the data:

$$p(\mathcal{D} | \boldsymbol{\theta}) = \prod_{i=1}^N p(\mathbf{x}_i | \boldsymbol{\theta}) = \prod_{i=1}^N \left[ \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]$$
$$\mathcal{L}(\boldsymbol{\theta}) = \log p(\mathcal{D} | \boldsymbol{\theta}) = \sum_{i=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$$

Direct maximization is difficult due to the **sum inside the logarithm**. Taking derivatives w.r.t. parameters and setting to zero does not yield closed-form solutions. For example, differentiating w.r.t. $\boldsymbol{\mu}_k$ yields:

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_k} = \sum_{i=1}^N \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} \boldsymbol{\Sigma}_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) = 0$$

Let $r_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$. This term, $r_{ik}$, is the **responsibility** that component $k$ takes for explaining data point $\mathbf{x}_i$. It represents the posterior probability $P(z_i=k | \mathbf{x}_i, \boldsymbol{\theta})$. Setting the derivative to zero gives:

$$\boldsymbol{\mu}_k = \frac{\sum_{i=1}^N r_{ik} \mathbf{x}_i}{\sum_{i=1}^N r_{ik}}$$

Similarly, updates for $\boldsymbol{\Sigma}_k$ and $\pi_k$ also depend on the responsibilities $r_{ik}$:

$$\boldsymbol{\Sigma}_k = \frac{\sum_{i=1}^N r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_{i=1}^N r_{ik}}$$
$$\pi_k = \frac{\sum_{i=1}^N r_{ik}}{N}$$

These equations are not solutions, as the parameters ($\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k$) appear on both sides (implicitly within $r_{ik}$). This suggests an iterative approach: the **Expectation-Maximization (EM) algorithm**.

### 4. **The Expectation-Maximization (EM) Algorithm**

EM is a general iterative algorithm for finding MLE (or MAP) estimates for models with latent variables.

* **Intuition:** If we *knew* the latent assignments $z_i$ for each $\mathbf{x}_i$, maximizing the *complete-data log-likelihood* $\log p(\mathcal{D}, \mathbf{Z} | \boldsymbol{\theta}) = \sum_i \log p(\mathbf{x}_i, z_i | \boldsymbol{\theta}) = \sum_i [\log P(z_i=k) + \log p(\mathbf{x}_i | z_i=k)]$ would be easy (like fitting $K$ separate Gaussians). Since $\mathbf{Z}=\{z_i\}$ is unknown, EM iterates between:
    * **E-step:** Estimating the *expected* complete-data log-likelihood, given the current parameters and observed data. This involves computing the posterior probabilities (responsibilities) of the latent variables.
    * **M-step:** Maximizing this expected quantity w.r.t. the parameters to get the next parameter estimate.

* **Derivation using Jensen's Inequality / Lower Bound:**

Let $\mathbf{X} = \{\mathbf{x}_i\}$ be observed data, $\mathbf{Z} = \{z_i\}$ be latent variables. We want to maximize $\mathcal{L}(\theta) = \log p(\mathbf{X} | \theta) = \log \sum_{\mathbf{Z}} p(\mathbf{X}, \mathbf{Z} | \theta)$.

Introduce an arbitrary distribution $q(\mathbf{Z})$ over the latent variables.

$$\mathcal{L}(\theta) = \log \sum_{\mathbf{Z}} q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}$$

Since $\log$ is concave, by Jensen's inequality ($\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$):

$$\mathcal{L}(\theta) \ge \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})} \equiv \mathcal{F}(q, \theta)$$

This function $\mathcal{F}(q, \theta)$ is the **Evidence Lower Bound (ELBO)**. The difference is the KL divergence: $\mathcal{L}(\theta) - \mathcal{F}(q, \theta) = \text{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z} | \mathbf{X}, \theta)) \ge 0$.

EM iteratively maximizes the ELBO $\mathcal{F}(q, \theta)$ by coordinate ascent:

* **E-step:** Fix current parameters $\theta^{(t)}$. Maximize $\mathcal{F}(q, \theta^{(t)})$ w.r.t. $q$. This occurs when $\text{KL}(q \parallel p) = 0$, i.e., when $q(\mathbf{Z})$ is set to the true posterior $p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$. For GMMs, this involves calculating the responsibilities $r_{ik}^{(t)} = p(z_i=k | \mathbf{x}_i, \theta^{(t)})$.
* **M-step:** Fix $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$. Maximize $\mathcal{F}(q, \theta)$ w.r.t. $\theta$ to get $\theta^{(t+1)}$. This is equivalent to maximizing the expected complete-data log-likelihood:

$$Q(\theta | \theta^{(t)}) = \mathbb{E}_{\mathbf{Z} | \mathbf{X}, \theta^{(t)}} [\log p(\mathbf{X}, \mathbf{Z} | \theta)] = \sum_{\mathbf{Z}} p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}) \log p(\mathbf{X}, \mathbf{Z} | \theta)$$

For GMMs, this leads exactly to the update equations for $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k$ derived earlier, using the responsibilities $r_{ik}^{(t)}$ computed in the E-step.

* **Convergence:** Each EM iteration guarantees $\mathcal{L}(\theta^{(t+1)}) \ge \mathcal{L}(\theta^{(t)})$. The algorithm converges to a local maximum (or saddle point) of the likelihood function.

### 5. **Latent Variable Generative Models (Examples)**

GMMs are one example. Other important ones include:

* **Factor Analysis (FA):**
    * *Model:* Continuous latent variable $\mathbf{z} \in \mathbb{R}^p$ (factors, $p \ll D$) and observed variable $\mathbf{x} \in \mathbb{R}^D$.
        * Prior: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        * Conditional (Likelihood): $\mathbf{x} | \mathbf{z} \sim \mathcal{N}(\mathbf{C} \mathbf{z} + \boldsymbol{\mu}, \boldsymbol{\Psi})$ where $\mathbf{C}$ is the $D \times p$ factor loading matrix, and $\boldsymbol{\Psi}$ is a *diagonal* covariance matrix (unique variances). Often $\boldsymbol{\mu}=0$.
    * *Marginal:* $p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z} = \mathcal{N}(\boldsymbol{\mu}, \mathbf{C} \mathbf{C}^T + \boldsymbol{\Psi})$.
    * *Interpretation:* Models the covariance structure of $\mathbf{x}$ as a low-rank part ($\mathbf{C} \mathbf{C}^T$) capturing shared variability plus a diagonal part ($\boldsymbol{\Psi}$) capturing independent noise. It assumes data lies near a $p$-dimensional subspace defined by $\mathbf{C}$.
    * *EM for FA:* E-step computes posterior $p(\mathbf{z} | \mathbf{x}, \theta^{(t)})$, which is Gaussian. M-step updates $\mathbf{C}, \boldsymbol{\Psi}$ using expected sufficient statistics $\mathbb{E}[\mathbf{z}]$ and $\mathbb{E}[\mathbf{z}\mathbf{z}^T]$.
* **Probabilistic PCA (pPCA):** A special case of FA where $\boldsymbol{\Psi} = \sigma^2 \mathbf{I}$ (isotropic noise). The principal subspace found by pPCA corresponds to that of standard PCA as $\sigma^2 \to 0$.
* **Discrete Mixture Models:** E.g., mixture of Bernoullis for binary data, mixture of multinomials (used in document clustering like Latent Dirichlet Allocation).

### 6. **Variational EM Framework (Generalization)**

The ELBO provides a general framework.

$$\log p(\mathbf{X} | \theta) = \underbrace{\sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}}_{\mathcal{F}(q, \theta) \text{ - ELBO}} + \underbrace{\text{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z} | \mathbf{X}, \theta))}_{\ge 0}$$

* **Goal:** Maximize $\log p(\mathbf{X} | \theta)$ by maximizing the lower bound $\mathcal{F}(q, \theta)$.
* **EM Algorithm (Variational View):**
    * E-step: Fix $\theta^{(t)}$, maximize $\mathcal{F}(q, \theta^{(t)})$ w.r.t. $q$. Achieved by setting $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$, which makes KL=0.
    * M-step: Fix $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$, maximize $\mathcal{F}(q, \theta)$ w.r.t. $\theta$ to get $\theta^{(t+1)}$.
* **Variational Inference / Generalized EM:** When the exact posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$ is intractable, we restrict $q(\mathbf{Z})$ to a simpler family $\mathcal{Q}$ (e.g., factorized distributions - mean-field approximation $q(\mathbf{Z}) = \prod_i q_i(Z_i)$).
    * E-step (Variational): Fix $\theta^{(t)}$, find $q^* \in \mathcal{Q}$ that *minimizes* $\text{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z} | \mathbf{X}, \theta^{(t)}))$ (equivalent to maximizing ELBO within $\mathcal{Q}$).
    * M-step: Fix $q^*$, maximize $\mathcal{F}(q^*, \theta)$ w.r.t. $\theta$ to get $\theta^{(t+1)}$.
    This still increases the ELBO at each iteration but may not reach the true likelihood optimum if $\mathcal{Q}$ is too restricted.

### 7. **Kullback-Leibler (KL) Divergence**

A fundamental measure of dissimilarity between two probability distributions $p$ and $q$.

* **Definition:**

$$\text{KL}(p \parallel q) = \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x} \quad \text{(continuous)}$$
$$\text{KL}(p \parallel q) = \sum_{\mathbf{x}} p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \quad \text{(discrete)}$$

* **Properties:**
    * $\text{KL}(p \parallel q) \ge 0$.
    * $\text{KL}(p \parallel q) = 0 \iff p(\mathbf{x}) = q(\mathbf{x})$ almost everywhere.
    * *Not symmetric:* $\text{KL}(p \parallel q) \neq \text{KL}(q \parallel p)$ generally.
* **Derivation of Non-negativity:** Using Jensen's inequality for the concave $\log$ function:

$$-\text{KL}(p \parallel q) = \int p(\mathbf{x}) \log \frac{q(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x} \le \log \int p(\mathbf{x}) \frac{q(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x} = \log \int q(\mathbf{x}) \, d\mathbf{x} = \log 1 = 0$$

Hence, $\text{KL}(p \parallel q) \ge 0$.
* **KL for Gaussians:** For $p = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$:

$$\text{KL}(p \parallel q) = \frac{1}{2} \left( \text{tr}(\boldsymbol{\Sigma}_2^{-1} \boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - d + \log \frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} \right)$$

* **Role in Variational EM:** Measures the approximation error introduced by using $q(\mathbf{Z})$ instead of the true posterior $p(\mathbf{Z} | \mathbf{X}, \theta)$. Minimizing KL is equivalent to maximizing the ELBO.

---

## **Part IV: Sequence Modeling: From HMMs to CRFs**

**(Based on Lectures by Andrew Fitzgibbon)**

### 1. **Introduction: Sequences in Deep Learning**

Sequential data is prevalent across many fields:

* Time series (Finance: stock prices, Economics: GDP).
* Biological sequences (DNA, RNA, proteins).
* Audio and Speech (waveforms, phoneme sequences).
* Natural Language Text (words, characters).
* Video (sequences of frames).
* Sensor readings, trajectories (robotics, navigation).

A core task in sequence modeling is **prediction**: given a sequence history, predict the next element(s). This underlies generation, translation, tagging, etc.

### 2. **Language Modeling and Autoregressive Data Generation**

* **Language Modeling Goal:** Assign a probability to a sequence of words (or other tokens) $\mathbf{W} = (w_1, w_2, \ldots, w_T)$, where each $w_t$ belongs to a vocabulary $V$. Using the chain rule of probability:

$$P(\mathbf{W}) = P(w_1) P(w_2 | w_1) P(w_3 | w_1, w_2) \dots P(w_T | w_1, \ldots, w_{T-1}) = \prod_{t=1}^T P(w_t | w_{<t})$$

where $w_{<t} = (w_1, \ldots, w_{t-1})$. The task is to estimate the conditional probabilities $P(w_t | w_{<t})$.

* **Autoregressive Generation:** A model that computes $P(w_t | w_{<t})$ can generate new sequences:
    1. Start with an initial context (e.g., a start-of-sequence token $\langle s \rangle$).
    2. Sample $w_1 \sim P(w | \langle s \rangle)$.
    3. Sample $w_2 \sim P(w | \langle s \rangle, w_1)$.
    4. Sample $w_t \sim P(w | \langle s \rangle, w_1, \ldots, w_{t-1})$.
    5. Continue until an end-of-sequence token is generated or a maximum length is reached.

* **Model Complexity:** The number of possible histories $w_{<t}$ grows exponentially with $t$. If $|V|=2^{16}$ and we condition on $L-1=7$ previous words, there are $(2^{16})^7 = 2^{112}$ possible contexts. Storing parameters for each context is infeasible. We need methods to represent the history compactly.

### 3. **Compact History Representations: N-gram Models**

The simplest approach is to assume the probability of the next word depends only on a fixed number $N-1$ of preceding words (Markov assumption of order $N-1$).

$$P(w_t | w_1, \ldots, w_{t-1}) \approx P(w_t | w_{t-N+1}, \ldots, w_{t-1})$$

* $N=1$: Unigram model (context-independent word probabilities).
* $N=2$: Bigram model ($P(w_t | w_{t-1})$).
* $N=3$: Trigram model ($P(w_t | w_{t-2}, w_{t-1})$).

* **Estimation:** Probabilities are estimated from counts in a large corpus:

$$\hat{P}(w_t | w_{t-N+1}, \ldots, w_{t-1}) = \frac{\text{Count}(w_{t-N+1}, \ldots, w_{t-1}, w_t)}{\text{Count}(w_{t-N+1}, \ldots, w_{t-1})}$$

* **Challenges:**
    * *Sparsity:* Many valid N-grams may not appear in the training corpus (zero counts). Requires **smoothing** techniques (e.g., Add-k, Kneser-Ney) to assign non-zero probabilities to unseen N-grams.
    * *Limited Context:* Cannot capture long-range dependencies beyond $N-1$ words.

### 4. **Latent Variable Models for Sequences**

Instead of truncating history, use a latent (hidden) state variable $h_t$ that summarizes the relevant information from the past $w_{<t}$ to predict the future $w_{\ge t}$.

$$P(w_t | w_{<t}) \approx P(w_t | h_t)$$

The state itself evolves over time, $h_t = f(h_{t-1}, w_{t-1})$.

* **Discrete Latent State:** Hidden Markov Models (HMMs).
* **Continuous Latent State:** Linear Dynamical Systems (Kalman Filters), Recurrent Neural Networks (RNNs).

### 5. **Hidden Markov Models (HMMs)**

An HMM is a generative model for sequences $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$ assuming they are generated by an underlying sequence of discrete hidden states $(\mathbf{q}_0, \mathbf{q}_1, \ldots, \mathbf{q}_T)$.

* **Components:**
    * Finite set of states $S = \{s_1, \ldots, s_N\}$.
    * Initial state probabilities: $\boldsymbol{\pi} = [\pi_i]$, where $\pi_i = P(\mathbf{q}_1 = s_i)$. (Sometimes includes start state $q_0$).
    * State transition probabilities: $A = [a_{ij}]$, where $a_{ij} = P(\mathbf{q}_t = s_j | \mathbf{q}_{t-1} = s_i)$. ($\sum_j a_{ij} = 1$).
    * Emission probabilities: $B = [b_j(k)]$, where $b_j(k) = P(\mathbf{x}_t = v_k | \mathbf{q}_t = s_j)$ if observations are discrete from vocabulary $V=\{v_k\}$, or $b_j(\mathbf{x}) = p(\mathbf{x}_t = \mathbf{x} | \mathbf{q}_t = s_j)$ if continuous.
* **Key Assumptions:**
    1. **Markov Property:** The current state $\mathbf{q}_t$ depends only on the previous state $\mathbf{q}_{t-1}$: $P(\mathbf{q}_t | \mathbf{q}_{<t}) = P(\mathbf{q}_t | \mathbf{q}_{t-1})$.
    2. **Output Independence:** The current observation $\mathbf{x}_t$ depends only on the current state $\mathbf{q}_t$: $P(\mathbf{x}_t | \mathbf{q}_{\le t}, \mathbf{x}_{<t}) = P(\mathbf{x}_t | \mathbf{q}_t)$.
* **Joint Likelihood:** The probability of an observed sequence $\mathbf{X} = (\mathbf{x}_1, \ldots, \mathbf{x}_T)$ involves summing over all possible hidden state sequences $\mathbf{Q} = (\mathbf{q}_1, \ldots, \mathbf{q}_T)$:

$$P(\mathbf{X} | \boldsymbol{\lambda}) = \sum_{\mathbf{Q}} P(\mathbf{X}, \mathbf{Q} | \boldsymbol{\lambda}) = \sum_{q_1, \ldots, q_T} \pi_{q_1} b_{q_1}(\mathbf{x}_1) \prod_{t=2}^T a_{q_{t-1} q_t} b_{q_t}(\mathbf{x}_t)$$

where $\boldsymbol{\lambda} = (\boldsymbol{\pi}, A, B)$ represents the HMM parameters.
* **Training:** Parameters $\boldsymbol{\lambda}$ are typically learned using the **Baum-Welch algorithm**, which is an instance of the EM algorithm for HMMs.

### 6. **Discrete Kalman Filters (Linear Dynamical Systems)**

Models sequences with continuous latent states $\mathbf{z}_t \in \mathbb{R}^p$ and observations $\mathbf{x}_t \in \mathbb{R}^D$.

* **Model Equations:** Assumes linear dynamics with Gaussian noise:
    * State Transition (Process Model): $\mathbf{z}_t = A \mathbf{z}_{t-1} + \mathbf{w}_t$, where $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q)$ (process noise).
    * Observation Model (Measurement Model): $\mathbf{x}_t = C \mathbf{z}_t + \mathbf{v}_t$, where $\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R)$ (measurement noise).
* **Properties:** The distribution $p(\mathbf{z}_t | \mathbf{x}_{1:t})$ (filtering) and $p(\mathbf{z}_t | \mathbf{x}_{1:T})$ (smoothing) remain Gaussian and can be computed exactly and efficiently using the recursive **Kalman filter** and **Kalman smoother** (RTS smoother) algorithms.

### 7. **Inference Algorithms for HMMs**

Efficient algorithms based on dynamic programming are used to answer key questions about HMMs.

* **Evaluation Problem:** Compute the likelihood of an observation sequence $P(\mathbf{X} | \boldsymbol{\lambda})$. Solved by the **Forward Algorithm**.
* **Decoding Problem:** Find the most likely sequence of hidden states given the observations: $\mathbf{Q}^* = \arg \max_{\mathbf{Q}} P(\mathbf{Q} | \mathbf{X}, \boldsymbol{\lambda})$. Solved by the **Viterbi Algorithm**.
* **Learning Problem:** Estimate the HMM parameters $\boldsymbol{\lambda}$ from data. Solved by the **Baum-Welch (EM) Algorithm**, which uses the Forward-Backward algorithm internally.

* **Forward Algorithm:** Computes forward probabilities $\alpha_j(t) = P(\mathbf{x}_1, \ldots, \mathbf{x}_t, \mathbf{q}_t = s_j | \boldsymbol{\lambda})$.
    * Initialization: $\alpha_j(1) = \pi_j b_j(\mathbf{x}_1)$.
    * Recursion: $\alpha_j(t) = \left[ \sum_{i=1}^N \alpha_i(t-1) a_{ij} \right] b_j(\mathbf{x}_t)$.
    * Termination: $P(\mathbf{X} | \boldsymbol{\lambda}) = \sum_{j=1}^N \alpha_j(T)$.
    (Often computed in log-space using log-sum-exp for numerical stability).
``
* **Viterbi Algorithm:** Finds the single best state sequence. Computes $\delta_j(t) = \max_{q_1, \dots, q_{t-1}} P(q_1, \dots, q_{t-1}, \mathbf{q}_t=s_j, \mathbf{x}_1, \dots, \mathbf{x}_t | \boldsymbol{\lambda})$.
    * Initialization: $\delta_j(1) = \pi_j b_j(\mathbf{x}_1)$. Store backpointer $\psi_j(1)=0$.
    * Recursion: $\delta_j(t) = \left[ \max_{1 \le i \le N} \delta_i(t-1) a_{ij} \right] b_j(\mathbf{x}_t)$. Store backpointer $\psi_j(t) = \arg \max_{1 \le i \le N} \delta_i(t-1) a_{ij}$.
    * Termination: $P(\mathbf{Q}^*, \mathbf{X} | \boldsymbol{\lambda}) = \max_{j} \delta_j(T)$. Find final state $q_T^* = \arg \max_j \delta_j(T)$.
    * Backtracking: Recover path $q_{t}^* = \psi_{q_{t+1}^*}(t+1)$ for $t=T-1, \dots, 1$.
    (Usually implemented in log-domain: use $\max$ instead of $\sum$, and $+$ instead of $\times$).

* **Forward–Backward Algorithm (for EM):** Computes forward variables $\alpha_j(t)$ and backward variables $\beta_j(t) = P(\mathbf{x}_{t+1}, \ldots, \mathbf{x}_T | \mathbf{q}_t = s_j, \boldsymbol{\lambda})$.
    * Backward Recursion: $\beta_i(t) = \sum_{j=1}^N a_{ij} b_j(\mathbf{x}_{t+1}) \beta_j(t+1)$. (Initialization $\beta_i(T)=1$).
    * Used to compute posterior marginals needed for EM: $P(\mathbf{q}_t = s_i | \mathbf{X}, \boldsymbol{\lambda}) = \frac{\alpha_i(t) \beta_i(t)}{P(\mathbf{X} | \boldsymbol{\lambda})}$ and pairwise marginals $P(\mathbf{q}_{t-1}=s_i, \mathbf{q}_t=s_j | \mathbf{X}, \boldsymbol{\lambda})$.

### 8. **From Generative to Discriminative Models: CRFs**

HMMs are generative ($P(\mathbf{X}, \mathbf{Q})$). Discriminative models directly model the conditional probability $P(\mathbf{Q} | \mathbf{X})$.

* **Analogy:** Logistic Regression is discriminative version of Naive Bayes.
* **Maximum Entropy Markov Models (MEMMs):** Use logistic regression (MaxEnt) for each state transition probability $P(\mathbf{q}_t | \mathbf{q}_{t-1}, \mathbf{X})$. Uses features $f_i(\mathbf{q}_t, \mathbf{q}_{t-1}, \mathbf{X})$:

$$P(\mathbf{q}_t | \mathbf{q}_{t-1}, \mathbf{X}) = \frac{\exp(\sum_i \lambda_i f_i(\mathbf{q}_t, \mathbf{q}_{t-1}, \mathbf{X}))}{\sum_{q'} \exp(\sum_i \lambda_i f_i(q', \mathbf{q}_{t-1}, \mathbf{X}))}$$

Suffers from the **label bias problem**: states with low-entropy next-state distributions effectively ignore the observation. Normalization is local at each step $t$.

* **Conditional Random Fields (CRFs):** Globally normalized discriminative model. For a linear-chain CRF:

$$P(\mathbf{Q} | \mathbf{X}, \boldsymbol{\lambda}) = \frac{1}{Z(\mathbf{X}, \boldsymbol{\lambda})} \exp \left( \sum_{t=1}^T \sum_k \lambda_k f_k(\mathbf{q}_t, \mathbf{q}_{t-1}, \mathbf{X}, t) \right)$$

* $f_k$: Feature functions depending on current state, previous state, observations $\mathbf{X}$, and time $t$. Can be complex (e.g., "is word $\mathbf{x}_t$ capitalized AND is state $\mathbf{q}_t$ 'Proper Noun'?").
* $\boldsymbol{\lambda}$: Feature weights learned from data.
* $Z(\mathbf{X}, \boldsymbol{\lambda})$: Global partition function, sum over *all possible* state sequences $\mathbf{Q}$:

$$Z(\mathbf{X}, \boldsymbol{\lambda}) = \sum_{\mathbf{Q}'} \exp \left( \sum_{t=1}^T \sum_k \lambda_k f_k(\mathbf{q}'_t, \mathbf{q}'_{t-1}, \mathbf{X}, t) \right)$$

* **Advantages:** Avoids label bias, allows arbitrary features of observations $\mathbf{X}$.
* **Training:** Maximize conditional log-likelihood $\sum_n \log P(\mathbf{Q}^{(n)} | \mathbf{X}^{(n)}, \boldsymbol{\lambda})$ using gradient-based methods. Requires computing $Z$ and marginals using forward-backward algorithm adapted to the CRF potential structure. Inference (Viterbi) is similar to HMMs but uses CRF potentials.