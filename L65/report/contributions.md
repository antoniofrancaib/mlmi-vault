Our final architecture is a universal model for learning sets of geometric graphs that integrates global attention with geometric processing. The design consists of three main modules: a hybrid graph encoder, a universal set aggregator, and a global geometric attention-based decoder. Below is a detailed explanation of the architecture and the contributions we implemented.

### 1. Hybrid Graph Encoder (HybridGeometricEncoderLayer)
Each input graph represents a conformation of an RNA structure. For each graph, every node is endowed with raw features consisting of scalar components $\mathbf{s}_i^{\text{raw}} \in \mathbb{R}^{64}$ and vector components $\mathbf{V}_i^{\text{raw}} \in \mathbb{R}^{4 \times 3}$. Edge features are similarly provided. The encoder processes $C$ conformations independently; formally, the input space is

$$
\{\mathbf{H}_1, \dots, \mathbf{H}_C\} \subset \left((\mathbb{R}^{64} \times \mathbb{R}^{4 \times 3})^N\right)^C,
$$

where each $\mathbf{H}_i$ represents one conformation with $N$ nodes.

#### Enhanced Feature Construction
For each node $i$ in each conformation, we compute an enhanced feature vector that integrates three components:

- **Normalized Scalar Features**: $\mathbf{s}_i \in \mathbb{R}^{128}$ (after applying LayerNorm and a GVP-based embedding).
- **Vector Norms**: Computed from the vector features $\mathbf{V}_i \in \mathbb{R}^{16 \times 3}$, yielding $\|\mathbf{V}_i\| \in \mathbb{R}^{16}$.
- **Aggregated Edge Features**: The edge features incident to node $i$ are aggregated via a linear operator (the adjacency matrix $\mathbf{A}$ or a similar operator) with spectral norm $\|\mathbf{A}\|$, yielding a vector $g(\mathbf{e}_i) \in \mathbb{R}^{d_e}$.

Thus, the enhanced feature for node $i$ is defined as:

$$
\hat{\mathbf{x}}_i = [\mathbf{s}_i; \|\mathbf{V}_i\|; g(\mathbf{e}_i)] \in \mathbb{R}^{d_s + d_v + d_e}.
$$

#### Branching: Local and Global Processing
The encoder processes each node’s enhanced feature via two parallel branches:

- **Local Branch**: A module analogous to the original MultiGVPConvLayer that aggregates information from neighboring nodes through local message passing, preserving fine-grained geometric interactions.
- **Global Branch**: A global multi-head attention mechanism that operates on the enhanced features $\hat{\mathbf{x}}_i$ across all nodes. This branch captures long-range dependencies and global contextual information.

These two branches produce outputs $\mathbf{M}_i$ (local) and $\mathbf{G}_i$ (global) which are then fused with equal weights:

$$
\mathbf{x}_i^{\text{hybrid}} = 0.5 \mathbf{M}_i + 0.5 \mathbf{G}_i.
$$

Subsequently, a residual connection, followed by a pointwise feed-forward network, dropout, and layer normalization, refines the node embeddings.

### 2. Universal Set Aggregator
Given that each conformation is encoded independently, the overall set of embeddings is

$$
\{\mathbf{X}_1, \dots, \mathbf{X}_C\} \subset (\mathbb{R}^d)^C.
$$

To aggregate these into a single representation that is permutation invariant, we leverage the theoretical result by Maron et al. Instead of using simple sum pooling, we compute higher-order tensor moments:

$$
\Phi(\{\mathbf{X}_i\}_{i=1}^C) = \left(\sum_{i=1}^C \mathbf{X}_i, \sum_{i=1}^C \mathbf{X}_i^{\otimes 2}, \dots, \sum_{i=1}^C \mathbf{X}_i^{\otimes p}\right).
$$

Then, an MLP $\psi$ processes $\Phi$ to produce the final set-level representation:

$$
F(\{\mathbf{G}_i\}) = \psi(\Phi(\{\mathbf{X}_i\})).
$$

This construction is universal: any continuous symmetric function on sets can be approximated arbitrarily well by a function of these tensor moments.

### 3. Global Geometric Attention Decoder (GlobalGeometricAttentionDecoderLayer)
For the autoregressive sequence generation in RNA design, the decoder updates node embeddings based on global context. The decoder operates as follows:

#### Enhanced Feature Construction:
For each node, the enhanced feature vector is recomputed as

$$
\hat{\mathbf{x}}_i = [\mathbf{s}_i; \|\mathbf{V}_i\|; g(\mathbf{e}_i)].
$$

#### Global Attention:
Global multi-head attention is applied over all nodes:

$$
\mathbf{Y} = \text{MultiHeadAttention}(\hat{\mathbf{X}}, \hat{\mathbf{X}}, \hat{\mathbf{X}}),
$$

where $\hat{\mathbf{X}}$ is the matrix of enhanced features for all nodes. The output $\mathbf{Y}$ is then projected back to the scalar space:

$$
\mathbf{G}_i = P(\mathbf{Y}_i) \in \mathbb{R}^{d_s}.
$$

#### Residual and Feed-Forward Update:
The projected output is added to the original scalar features via a residual connection:

$$
\mathbf{s}_i^{\text{attn}} = \mathbf{s}_i + \text{Dropout}(\mathbf{G}_i).
$$

A feed-forward network $F$ then refines this result:

$$
\mathbf{s}_i^{\text{final}} = \text{LayerNorm}(\mathbf{s}_i^{\text{attn}} + \text{Dropout}(F(\mathbf{s}_i^{\text{attn}}))).
$$

A similar update (via a dedicated vector branch) can be applied to the vector features.

### Contributions and Rationale
- **Universality through Enhanced Pooling**: By aggregating not only node features but also the norms of the vector features and aggregated edge features, and then applying higher-order moment pooling (inspired by Maron et al.), the architecture is capable of approximating any continuous symmetric function on the set of graphs. This overcomes the limitations of simple deep sets pooling.
- **Integration of Local and Global Information**: The hybrid encoder fuses a local branch (that maintains fine-grained, neighborhood-level geometric interactions) with a global branch (that captures long-range dependencies via global attention). This dual approach ensures that both detailed and holistic geometric information is preserved, which is crucial for accurately modeling RNA structures.
- **Attention-Based Geometric Processing**: We replaced traditional GVP layers with attention-based modules that are designed to be analogous to GVPs. The global attention mechanism operates on enhanced features that encapsulate geometric information, ensuring that the model can exploit both local and global structure in a manner that respects geometric invariances.
- **Universal Set Aggregation**: The use of higher-order tensor pooling to aggregate graph embeddings guarantees universality over sets. This theoretical foundation ensures that our architecture can approximate any continuous function on the space of geometric graphs, given sufficient capacity.
- **Practical Considerations**: The design choices—such as residual connections, dropout, and layer normalization—help control the Lipschitz constants of the network. In particular, when the aggregation operator (e.g., the normalized adjacency matrix) has a small spectral norm (e.g., $\|\mathbf{A}\| = \frac{1}{k}$ for a $k$-regular graph), the overall Lipschitz constant can be driven very low, which has implications for oversmoothing. This careful control of the Lipschitz constant is essential for stability and generalization in deep graph networks.

### Final Summary:
Our final architecture is a universal model for sets of geometric graphs that combines a HybridGeometricEncoderLayer with a GlobalGeometricAttentionDecoderLayer. The encoder processes each conformation independently using a dual-branch strategy that fuses local message passing and global attention on enhanced features (integrating node, vector, and edge information). The resulting graph embeddings are then aggregated using higher-order moment pooling, guaranteeing universality over sets. The decoder, designed for autoregressive sequence generation, updates node embeddings with global attention that respects the geometric structure. This architecture overcomes the limitations of previous methods by ensuring that all relevant geometric information is retained and that the overall mapping is universal, making it capable of approximating any continuous function on the space of geometric graphs.