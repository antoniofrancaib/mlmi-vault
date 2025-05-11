# Few-Shot Image Classification on Mini-Imagenet

## Task Definition:
The task is a standard N-way, K-shot image classification problem. In the experiments, the authors focus on 5-way 1-shot and 5-way 5-shot settings. In each task, five classes are randomly selected from Mini-Imagenet, and for each class:

- **Training (Support Set):** K examples per class ($K = 1$ or $5$)
- **Meta-update (Query Set):** Unseen examples from the same five classes are used for computing the meta-gradient.

## Dataset Details:
- **Training Classes:** 64  
- **Validation Classes:** 12  
- **Test Classes:** 24  

## Network Architecture:
The same architecture is used for both CAVIA and the baseline (MAML), with variations in the number of filters to test scaling:

- **Input:** Images of size $84\times84\times3$.
- **Convolutional Backbone:**
  - Four convolutional layers.
  - Each layer uses a $3\times3$ kernel with padding $1$.
  - The number of filters per layer is varied between $32$ and $512$.
  - After each convolution, there is a batch normalization, a max-pooling (kernel size $2$), and a ReLU nonlinearity.

### Context Conditioning via FiLM:
A FiLM layer is introduced after the third convolution.

The FiLM transformation is computed as follows: given the context vector $\mathbf{\phi} \in \mathbb{R}^{100}$ (fixed size of $100$), a fully connected layer produces a $256$-dimensional output which is split into two $128$-dimensional vectors $\mathbf{\gamma}$ and $\mathbf{\beta}$. These are used to modulate the convolutional feature maps $\mathbf{h}$ by:

$$\text{FiLM}(\mathbf{h}) = \mathbf{\gamma} \odot \mathbf{h} + \mathbf{\beta},$$

where $\odot$ denotes elementwise multiplication.

### Final Classifier:
The output from the four modules is a feature map of size $5\times5\times$(number of filters) which is then flattened and passed through one fully connected layer to produce 5 logits (one per class).

## Meta-Learning Formulation (Supervised Setting):
CAVIA splits the model parameters into:

- **Shared parameters** $\mathbf{\theta}$: These include all weights and biases except the context-conditioning weights.
- **Task-specific context parameters** $\mathbf{\phi}$:
  - Initially set to $\mathbf{\phi}_0 = 0$.
  - Updated for each task using one (or more) gradient descent steps.

The inner-loop (task-specific) update is given by:

$$\mathbf{\phi}_i = \mathbf{\phi}_0 - \alpha \nabla_{\mathbf{\phi}} \frac{1}{M_{\text{train}}} \sum_{(x,y) \in \mathcal{D}_{\text{train}}^i} L_i(f_{\mathbf{\phi}_0,\mathbf{\theta}}(x), y),$$

where:
- $\mathcal{D}_{\text{train}}^i$ is the support set for task $i$,
- $\alpha$ is the inner-loop learning rate,
- $L_i$ is the classification loss (typically cross-entropy).

The outer-loop (meta-update) then optimizes $\mathbf{\theta}$ over tasks:

$$\mathbf{\theta} \gets \mathbf{\theta} - \beta \nabla_{\mathbf{\theta}} \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M_{\text{test}}} \sum_{(x,y) \in \mathcal{D}_{\text{test}}^i} L_i(f_{\mathbf{\phi}_i,\mathbf{\theta}}(x), y),$$

where:
- $\mathcal{D}_{\text{test}}^i$ is the query set for task $i$,
- $\beta$ is the outer-loop learning rate.

## Training Details:

### Inner Loop:
- Two gradient steps are taken per task during training.
- Inner-loop learning rate is set to $0.1$ (with experiments testing alternatives such as $1.0$ and $0.01$).

### Outer Loop:
- The meta-update is performed using the Adam optimizer with an initial learning rate of $0.001$.
- This learning rate is annealed every $5,000$ steps by multiplying it by $0.9$.

### Meta-Batch Configuration:
- Meta-batch sizes are set to 4 tasks for 1-shot and 2 tasks for 5-shot classification.

### Batch Normalization:
- Batch norm statistics are computed using the current batch (even at test time, where the batch size equals the number of classes, i.e., $5$).

### Meta-Training Duration:
- Experiments are run for $60,000$ meta-iterations with the best-performing model (on the validation set) selected for evaluation on the test set.

## Experimental Results (Table 3):
The paper reports the following 5-way classification accuracies (with 95% confidence intervals over 1000 randomly sampled tasks):

| Method | 1-shot (%) | 5-shot (%) |
|--------|------------|------------|
| Matching Nets (Vinyals et al.) | 46.6 | 60.0 |
| Meta LSTM (Ravi & Larochelle) | 43.44 ± 0.77 | 60.60 ± 0.71 |
| Prototypical Networks | 46.61 ± 0.78 | 65.77 ± 0.70 |
| Meta-SGD | 50.47 ± 1.87 | 64.03 ± 0.94 |
| REPTILE | 49.97 ± 0.32 | 65.99 ± 0.58 |
| MT-NET | 51.70 ± 1.84 | – |
| VERSA | 53.40 ± 1.82 | 67.37 ± 0.86 |
| MAML (32 filters) | 48.07 ± 1.75 | 63.15 ± 0.91 |
| MAML (64 filters) | 44.70 ± 1.69 | 61.87 ± 0.93 |
| CAVIA (32 filters) | 47.24 ± 0.65 | 59.05 ± 0.54 |
| CAVIA (128 filters) | 49.84 ± 0.68 | 64.63 ± 0.54 |
| CAVIA (512 filters) | 51.82 ± 0.65 | 65.85 ± 0.55 |
| CAVIA (512 filters, first order) | 49.92 ± 0.68 | 63.59 ± 0.57 |

*(Refer to Table 3 in the paper.)*

## Key Observations:

- **Scaling Up:** CAVIA can scale to larger networks (e.g., using 512 filters) without overfitting, in contrast to MAML, which suffers when scaled up.
- **Efficiency:** Even though CAVIA adapts only 100 context parameters at test time (compared to over 30,000 parameters in MAML), it achieves competitive or superior performance.
- **First-Order Approximation:** Using a first-order approximation (i.e., not backpropagating through the inner-loop update for $\mathbf{\theta}$) results in a slight drop in performance, yet still outperforms the corresponding MAML baseline.
