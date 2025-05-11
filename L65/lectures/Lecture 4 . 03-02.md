Lecture by Petar 
# 1. Permutation-Equivariant Learning on Graphs

## 1.1 From Sets to Graphs
**Context:** In the previous lecture (on Deep Sets), we saw how to build neural networks on sets in a permutation-invariant or permutation-equivariant manner.

**New Aspect:** We now introduce edges between elements of a set, turning it into a graph $\mathbf{G} = (\mathbf{V}, \mathbf{E})$. We can represent $\mathbf{E}$ using an adjacency matrix $\mathbf{A}$, where
$$
\mathbf{A}_{ij} = \begin{cases} 
1 & \text{if } (i,j) \in \mathbf{E}, \\
0 & \text{otherwise}.
\end{cases}
$$

## 1.2 Permutations Now Affect Edges
A permutation $\Pi$ of the node set $\mathbf{V}$ implies a corresponding transformation of the edges. If the matrix representation of $\Pi$ is $\mathbf{P}$, then applying $\mathbf{P}$ to $\mathbf{A}$ yields $\mathbf{P} \mathbf{A} \mathbf{P}^\top$.

**Permutation Invariance:** A function $f$ over graphs is invariant if
$$
f(\mathbf{P} \mathbf{A} \mathbf{P}^\top, \mathbf{P} \mathbf{X}) = f(\mathbf{A}, \mathbf{X}),
$$
where $\mathbf{X}$ is a matrix of node features.

**Permutation Equivariance:** A function $\Phi$ is equivariant if
$$
\Phi(\mathbf{P} \mathbf{A} \mathbf{P}^\top, \mathbf{P} \mathbf{X}) = \mathbf{P} \Phi(\mathbf{A}, \mathbf{X}).
$$
These definitions ensure that if we permute the nodes, the function’s outputs permute in a consistent, predictable way.

# 2. Locality on Graphs: Neighborhoods

## 2.1 The Neighborhood Concept
For a node $v$, define its 1-hop neighborhood, $\mathbf{N}_v$, as
$$
\mathbf{N}_v = \{ u \mid (u,v) \in \mathbf{E} \text{ or } (v,u) \in \mathbf{E} \}.
$$
Intuitively, $\mathbf{N}_v$ consists of all nodes directly connected to $v$.

## 2.2 Neighborhood Features
Given node features $\{ \mathbf{x}_u \mid u \in \mathbf{V} \}$, the features of $\mathbf{N}_v$ can be assembled into a multiset:
$$
\mathbf{X}_{\mathbf{N}_v} = \{ \{ \mathbf{x}_u : u \in \mathbf{N}_v \} \}.
$$
We then consider a local function $\phi(\mathbf{x}_v, \mathbf{X}_{\mathbf{N}_v})$ that processes both the center node’s feature $\mathbf{x}_v$ and its neighborhood’s features.

## 2.3 Building Equivariant Functions from Locality
To construct a graph-level equivariant function $\Phi(\mathbf{A}, \mathbf{X})$, we apply $\phi$ locally and shared across all nodes:
$$
\Phi(\mathbf{A}, \mathbf{X}) = \begin{bmatrix}
\phi(\mathbf{x}_1, \{ \mathbf{x}_u : u \in \mathbf{N}_1 \}) \\
\phi(\mathbf{x}_2, \{ \mathbf{x}_u : u \in \mathbf{N}_2 \}) \\
\vdots \\
\phi(\mathbf{x}_n, \{ \mathbf{x}_u : u \in \mathbf{N}_n \})
\end{bmatrix}.
$$
**Key Condition:** $\phi$ must be permutation-invariant in the neighbor features to preserve the overall equivariance of $\Phi$.

**Exercise:** Prove that if $\phi$ is permutation-invariant on its neighbor arguments, then $\Phi$ is permutation-equivariant at the graph level.

# 3. Recipe for Graph Neural Networks

## 3.1 Core Blueprint
1. Define a local, permutation-invariant function $\phi$ that aggregates information from a node’s neighbors (and possibly the node itself).
2. Apply $\phi$ to each node in the graph, using the same parameters (weight-sharing), to get updated node features across all nodes simultaneously.

![[recipe-GNN.png]]

![[blueprint-GNN.png]]
## 3.2 Common Terminology
- The function $\Phi(\mathbf{A}, \mathbf{X})$ formed by these local operations is called a GNN layer.
- Each local update $\phi$ is often referred to as “diffusion,” “propagation,” or “message passing.”

## 3.3 Three Flavors of GNN Layers
Most common GNN layers can be grouped into three patterns, based on how they aggregate neighbor features:
1. **Convolutional GNNs**
2. **Attentional GNNs**
3. **Message-Passing GNNs**

We will explore each flavor in detail.

![[flavors-GNN.png]]
# 4. Convolutional Graph Neural Networks

## 4.1 Basic Idea
Convolutional GNNs assign fixed weights to neighbor features, typically depending on the graph structure (e.g., node degrees) but not on the neighbor features themselves. A simple (and often quite effective) choice is to sum or average neighbor features with a learnable linear transformation.

## 4.2 Matrix-Form Representation
Suppose we choose to sum neighbor features for each node. If $\mathbf{W}$ is a trainable linear transformation, then in matrix form,
$$
\mathbf{H} = \sigma(\mathbf{A} \mathbf{X} \mathbf{W}),
$$
where $\sigma$ is an activation (e.g., ReLU). Summation can lead to feature explosion if some nodes have high degree; hence, an averaging approach (normalizing by degrees) is typical.

## 4.3 Degree Normalization for Stability
Normalizing by node degrees can be done in two ways:
1. **Random-walk normalisation:** Multiply by $\mathbf{D}^{-1} \mathbf{A}$.
2. **Symmetric normalisation:** Multiply by $\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}$.

In matrix notation (with $\mathbf{D}$ as the degree matrix), this leads to:
$$
\mathbf{H} = \sigma(\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} \mathbf{X} \mathbf{W}).
$$
This operator is closely related to the normalized graph Laplacian, helping to stabilize eigenvalues.

## 4.4 Graph Convolutional Network (GCN)
GCN (Kipf & Welling, ICLR’17) adds an identity to the adjacency ($\mathbf{A} + \mathbf{I}$) for self-loops and normalizes it by the corresponding degree matrix. Thus the GCN update is effectively:
$$
\mathbf{H} = \sigma(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{X} \mathbf{W}), \quad \tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}, \quad \tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}.
$$
**Motivation:** fosters self-connections and controls the largest eigenvalue for stability.

## 4.5 SGC (Simplified Graph Convolution)
One might ask: Do we even need nonlinearities and multi-layer transformations in some tasks? SGC (Wu et al., ICML’19) strips away the MLP layers, applying only repeated adjacency multiplications for $\ell$-hops, then uses a simple linear/logistic regression on top:
$$
\mathbf{H} = (\tilde{\mathbf{A}}^\ell \mathbf{X}) \mathbf{W}.
$$
Surprisingly effective on many homophilous graph benchmarks.

## 4.6 Chebyshev Networks (ChebyNet)
ChebyNet (Defferrard et al., NeurIPS’16) generalizes to multi-hop convolutions, using polynomial filters in the Laplacian:
$$
\mathbf{H} = \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\mathbf{L}}) \mathbf{X},
$$
where $T_k$ is the Chebyshev polynomial of order $k$, and $\tilde{\mathbf{L}}$ is a scaled Laplacian. Offers sparse and scalable multi-hop message passing, performing well on graphs that need deeper propagation in a single layer.

## 4.7 Limitations
For regular grids (e.g., images), ChebyNets become radial filters, failing to replicate general 2D convolution kernels exactly. This indicates that purely structural weighting (without feature-based distinctions) can limit expressivity.

# 5. Attentional Graph Neural Networks

## 5.1 Motivation
We may want different weights for different neighbors, even if they share the same structural pattern. In attentional GNNs, each edge gets an attention coefficient $\alpha_{ij}$ that depends on the features of its endpoints.

## 5.2 From MoNet to Graph Attention
MoNet (Monti et al., CVPR’17) introduced a framework using pseudo-coordinates $\mathbf{u}_{ij}$ and a weighting function $g$ (often Gaussian). For general graphs, MoNet’s pseudo-coordinates could not distinguish structurally identical neighbors without additional feature context.

## 5.3 Graph Attention Networks (GAT)
GAT (Veličković et al., ICLR’18) reframes MoNet’s idea: Let node features themselves serve as “coordinates” to disambiguate neighbors. The attention mechanism $\alpha_{ij} = a(\mathbf{x}_i, \mathbf{x}_j)$ is a learnable function (often an MLP) with a softmax over neighbors:
$$
\alpha_{ij} = \frac{\exp(a(\mathbf{x}_i, \mathbf{x}_j))}{\sum_{k \in \mathbf{N}_i} \exp(a(\mathbf{x}_i, \mathbf{x}_k))}.
$$
The updated feature for node $i$ is:
$$
\mathbf{x}_i' = \sigma\left( \sum_{j \in \mathbf{N}_i} \alpha_{ij} \mathbf{W} \mathbf{x}_j \right).
$$

## 5.4 Static vs. Dynamic Attention
In the original GAT paper, $a$ was chosen to be a simple linear+LeakyReLU function:
$$
a(\mathbf{x}_i, \mathbf{x}_j) = \text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W} \mathbf{x}_i \| \mathbf{W} \mathbf{x}_j]).
$$
This can cause “static” attention, where there might exist a global ranking of nodes (since the receiver’s features have a limited effect on the ordering). GATv2 (Brody et al., ICLR’22) modifies $a$ to:
$$
a(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{a}^\top \text{LeakyReLU}(\mathbf{W} [\mathbf{x}_i \| \mathbf{x}_j]),
$$
effectively making it a two-layer MLP. This enables dynamic attention, allowing the receiver node’s feature to fully reshape the ordering of attention coefficients.

## 5.5 Multi-Head Attention
Inspired by Transformers, GAT often uses multiple attention heads in parallel:
$$
\mathbf{x}_i' = \|_{h=1}^H \sigma\left( \sum_{j \in \mathbf{N}_i} \alpha_{ij}^{(h)} \mathbf{W}^{(h)} \mathbf{x}_j \right).
$$
Each head learns a distinct “interaction mode,” then the results are concatenated (or averaged).

## 5.6 Interpretability and Applications
Attention coefficients can be visualized, providing some interpretability. GAT and GATv2 often excel in tasks where neighbors should be weighed unequally based on feature importance rather than only structural relations (e.g., many scientific tasks).

# 6. Message-Passing GNNs

## 6.1 Generalizing the Aggregation
Message-Passing GNNs (MPNNs) fully empower edges to carry arbitrary vector messages. A node $i$ gathers messages $\mathbf{m}_{ij}$ from each neighbor $j$, with each message depending on $(\mathbf{x}_i, \mathbf{x}_j)$:
$$
\mathbf{m}_{ij} = \psi(\mathbf{x}_i, \mathbf{x}_j).
$$
The node then aggregates (e.g., sums) all incoming messages and updates its feature:
$$
\mathbf{x}_i' = \phi\left( \mathbf{x}_i, \sum_{j \in \mathbf{N}_i} \mathbf{m}_{ij} \right).
$$
This is the most generic local form—both convolutional and attentional GNNs can be viewed as special cases.

## 6.2 Historical Drivers
- **Physics Simulations:** Interaction Networks (Battaglia et al., NeurIPS’16) used MPNNs to model n-body systems, collisions, or strings.
- **Computational Chemistry:** Molecules as graphs with atoms as nodes and bonds as edges. Early GNNs (e.g., Kireev et al., 1995) recognized the need for flexible edge-based interactions.

## 6.3 Neural Message Passing (Gilmer et al., ICML’17)
Defines a broad MPNN framework unifying many GNN approaches. Achieves near chemical accuracy on tasks like molecular property prediction (QM9 dataset) by systematically exploring architecture variants.

# 7. Graph Networks (The Most General Blueprint)

## 7.1 Attributed Graphs
Real-world graphs can carry attributes at multiple levels:
- Node features $\mathbf{x}_v$.
- Edge features $\mathbf{x}_{uv}$.
- Global graph features $\mathbf{x}_G$.

We can keep latent versions of each (e.g., $\mathbf{h}_v, \mathbf{h}_{uv}, \mathbf{h}_G$) that get updated as we pass messages.

## 7.2 Graph Network (GN) Architecture
According to Battaglia et al. (2018), a GN block updates edge, node, and graph features in three sequential steps:

1. **Edge Update:**
   $$
   \mathbf{h}_{uv} = \phi_e(\mathbf{h}_u, \mathbf{h}_v, \mathbf{h}_{uv}, \mathbf{h}_G).
   $$

2. **Node Update:**
   $$
   \mathbf{h}_v = \phi_v\left( \mathbf{h}_v, \bigoplus_{u \in \mathbf{N}_v} \mathbf{h}_{uv}, \mathbf{h}_G \right),
   $$
   where $\oplus$ is a permutation-invariant aggregator (e.g., sum).

3. **Global (Graph) Update:**
   $$
   \mathbf{h}_G = \phi_g\left( \bigoplus_{v \in \mathbf{V}} \mathbf{h}_v, \bigoplus_{(u,v) \in \mathbf{E}} \mathbf{h}_{uv}, \mathbf{h}_G \right).
   $$
   Each $\phi$ is a learnable function (e.g., an MLP).

**Key Observations:**
- **Permutation Equivariance:** The node and edge updates are applied identically across all nodes/edges.
- **Skip Connections:** GNs often employ skip connections to preserve original features, aiding in deeper architectures.

## 7.3 Connections to Other GNN Flavors
By restricting $\phi_e, \phi_v, \phi_g$ to simpler forms (e.g., ignoring edge features or using trivial edge updates), one can recover:
- GCN
- GAT / MoNet
- MPNN
- …and other popular variants.

# 8. Applications and Illustrations

## 8.1 Social Networks
A node might represent a person, edges represent friendships or interactions. GNNs (especially attentional GNNs) can highlight which neighbors most strongly influence a user’s behavior (e.g., retweets).

## 8.2 Physics Simulations
Interaction Networks treat each edge as an interaction force or potential. GNs excel at predicting future states in dynamical systems.

## 8.3 Computational Chemistry
Molecules are graphs where nodes are atoms and edges are bonds. Node features might include atomic number, formal charge; edge features might indicate bond type. GNNs can approximate quantum properties (e.g., energies, dipole moments) or predict docking/interaction properties.

# 9. Food for Thought: Expressive Power
We showed for sets that any permutation-invariant function can be approximated by a Deep Sets architecture. For graphs, the question of universal approximation with GNNs is more intricate.

**Open Question:** Can any permutation-equivariant function on graphs be realized by the three GNN flavors or by Graph Networks? This is a major research topic, with numerous partial results on GNN expressivity and universality under certain constraints.

# 10. Conclusion and Next Steps

**Summary:**
- We explored the notion of permutation-invariant and permutation-equivariant learning on graphs.
- We introduced local neighborhood aggregation as the core principle behind GNNs.
- We covered three major flavors of GNN layers: convolutional, attentional, and message-passing.
- We concluded with Graph Networks (GN), a unifying framework that handles node, edge, and global features simultaneously.

**What’s Next:**
- Deeper investigation into how GNNs handle large graphs, long-range dependencies, heterogeneity, or dynamic structures.
- Exploring advanced attention mechanisms, deeper architectures, and the interplay with fields like NLP and classical algorithms.
- Upcoming practical session (lab) will delve into implementing these GNN primitives and exploring real-world tasks.

**Exercises (recap):**
1. Prove that if $\phi$ is permutation-invariant over the neighborhood features, $\Phi$ is permutation-equivariant at the graph level.
2. Implement GAT attention with $O(|\mathbf{V}|)$ memory, hinting at efficient handling of large edge sets.
3. Specify the $\phi_e, \phi_v, \phi_g$ functions in the GN framework to recover standard GCN, GAT, or MPNN layers.

These notes should serve as a robust foundation for a masterclass discussion on GNNs—highlighting their core theoretical motivations, common design patterns, and important applications—while providing enough technical detail and intuition to guide advanced research and practice.