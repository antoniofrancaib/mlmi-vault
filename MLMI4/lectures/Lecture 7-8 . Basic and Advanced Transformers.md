Lectures by Isaac Reid

Transformers, introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. (2017), have revolutionized AI by dramatically improving tasks involving sequences of data, such as language translation, text generation, and image recognition.

## What is a Transformer?

A Transformer is a neural network architecture designed for processing sequential data by efficiently capturing relationships between distant tokens. Unlike traditional recurrent neural networks (RNNs) or convolutional networks (CNNs), Transformers rely solely on self-attention mechanisms, enabling them to effectively handle long-range dependencies.

## Core Components of Transformers

Transformers consist of two main alternating layers:
![[transform-block.png]]
### 1. Self-Attention
Self-attention computes weighted representations of tokens based on their interactions. It uses three key matrices derived from the input tokens:

- **Queries (Q)**: Represent what information a token seeks.
- **Keys (K)**: Represent information that other tokens provide.
- **Values (V)**: Actual data transmitted between tokens.

Mathematically, self-attention is computed as:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
$$

Where:
- $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, $\mathbf{K} \in \mathbb{R}^{n \times d_k}$, $\mathbf{V} \in \mathbb{R}^{n \times d_v}$
- $n$ is the number of tokens, $d_k$ is the dimensionality of queries/keys, $d_v$ is dimensionality of values.

The softmax operation ensures attention weights sum to 1, highlighting the most relevant tokens.

### 2. Multi-Layer Perceptron (MLP)
The MLP processes each token independently after attention, further refining their representations. It typically involves two linear transformations with a non-linear activation (ReLU or GELU):

$$
\text{MLP}(\mathbf{x}) = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

Each token is updated individually, enhancing the network's ability to learn complex representations.

## Multi-Head Attention

Multi-head attention allows the model to attend to different parts of the input simultaneously, capturing richer contextual information. Formally:

$$
\text{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O
$$

Each head $i$ is calculated independently:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)
$$

Here, $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V$ are learnable parameters projecting queries, keys, and values into subspaces, and $\mathbf{W}^O$ combines outputs from all heads.

## Positional Encoding

Transformers inherently treat inputs as unordered sets. To incorporate sequence order, positional encodings are added to token embeddings:

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

This allows Transformers to differentiate between tokens based on their positions.

## Tokenization

Data must be tokenized into numerical vectors:
- **Language Models:** Text is converted into subword embeddings.
- **Vision Transformers (ViT):** Images are divided into fixed-size patches that become tokens.

## Real-World Applications
- **Vision Transformers (ViT):** Classify images by processing patches and using a special classifier token.
- **Language Models:** Predict next words through context-aware autoregressive modeling.
- **Multimodal Models (e.g., CLIP):** Align embeddings of text and images to perform tasks like open-vocabulary classification.

## Improving Efficiency and Scalability

Standard Transformers have computational complexity $O(n^2)$ due to pairwise token interactions. To improve this:

### Sparse Attention
Sparse attention restricts token interactions to local or strategically chosen subsets, significantly reducing computations. Mathematically, this modifies the attention computation by introducing masks that zero out irrelevant interactions:

$$
\text{SparseAttention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T \odot \mathbf{M}}{\sqrt{d_k}}\right)\mathbf{V}
$$

Here, $\mathbf{M}$ is a mask matrix indicating allowed interactions.

### Kernelized (Low-Rank) Attention
Kernelized attention approximates full attention using kernel functions, greatly reducing complexity:

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) \approx \phi(\mathbf{Q})\phi(\mathbf{K})^T\mathbf{V}
$$

where $\phi(\cdot)$ is a feature map approximating the original attention kernel, allowing computations in $O(n)$.

## Generation with Transformers

Generating coherent and diverse outputs requires controlled randomness strategies:
- **Top-k sampling:** Select tokens from the top-k predictions.
- **Nucleus sampling:** Choose tokens from a subset whose probabilities cumulatively surpass a threshold.

## Conclusion

Transformers have profoundly impacted AI, offering a powerful architecture to capture complex patterns in sequential data efficiently. Through the integration of sophisticated mathematical operations, Transformers continue to drive innovation across numerous AI applications, from natural language processing to computer vision and multimodal tasks.