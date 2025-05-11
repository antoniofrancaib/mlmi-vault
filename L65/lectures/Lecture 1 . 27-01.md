# GEOMETRIC DEEP LEARNING (L65)

https://geometricdeeplearning.com/

Graph Representation Learning by William L. Hamilton (Book)


# --FIRST HALF--

#### Why Geometric Deep Learning?
Modern deep learning architecture design often hinges on some notion of geometry or symmetry in the data. Even in domains where geometry may appear absent—like natural language—careful inspection reveals implicit geometric constraints (e.g., self-attention in Transformers imposes particular structural choices on how tokens can interact).

**Key motivations:**
- Data arising from nature frequently possess geometric structure (e.g., grids in images, molecular graphs in chemistry, manifolds in physics).
- Many models of nature embed geometric assumptions (e.g., isometries, rotations, permutations).
- Understanding and applying symmetry-based principles can unify and categorize existing architectures (CNNs, GNNs, Transformers, etc.) by the types of symmetry constraints they enforce.
- Geometric perspectives often yield more principled reasoning about how models generalize and handle domain structure.

#### Historical Context: The “One True Geometry?”
- **Euclid (~300 B.C.):** Laid the foundations of geometry, focusing on Euclidean planes and parallel lines.
- **Lobachevsky & Bolyai (1800s):** Explored non-Euclidean geometries (hyperbolic geometries).
- **Riemann (mid-1800s):** Provided a broad generalization of geometry that influenced modern differential geometry.
- **Felix Klein’s Erlangen Program (1872):** Proposed a blueprint to unify many geometries through the concept of invariances under certain groups of transformations. “Geometry” can be characterized by which transformations preserve its key properties.
- **Cartan (1920s):** Further unified geometric theories.
- **Noether (1918):** Demonstrated how physical conservation laws arise from symmetries, providing deep insight into why symmetry is so fundamental.
- **Eilenberg & Mac Lane (1945):** Developed Category Theory, another abstract framework that can be connected with geometric ideas.

This lineage of geometric thinking profoundly influenced mathematics, physics, and now deep learning. In modern AI, we often ask if there is a “one true architecture,” akin to the historical quest for the “one true geometry.” Geometric Deep Learning suggests that many deep architectures can be understood and unified through symmetry groups, invariances, and equivariances.

---
### 1. Introduction to Groups and Representations

The content below explains the formal mathematical underpinnings of Geometric Deep Learning: domains, group actions, representations, and the crucial notions of invariance and equivariance.

#### 1.1. Signals and Domains
A signal on a set $Ω$ is a function

$$x:Ω→V,$$

where:
- $Ω$ is the domain. This could be a discrete set of pixels (in an image), nodes (in a graph), or continuous points (in a manifold), among other possibilities.
- $V$ is a vector space describing the features (often called channels) of each element of $Ω$. For example, an RGB image has $V=R^3$.

In Geometric Deep Learning, we often deal with finite, discrete $Ω$, so a signal $x$ can be represented by a matrix of size $|Ω|×C$, where $|Ω|$ is the number of domain elements and $C$ is the number of channels.

**Vector space structure of signals:**
- We can add two signals $x$ and $y$ or scale a signal by a scalar, implying the space of all signals $X(Ω)$ naturally forms a vector space.
- An inner product can also be defined (when a measure on $Ω$ is available), giving rise to a Hilbert space structure.

#### 1.2. Symmetries and Groups
A symmetry of an object is a transformation that leaves that object unchanged. Historically, group theory arose from studying these symmetry transformations (e.g., rotating a triangle or flipping it).

Formally, a group $G$ is a set equipped with a binary operation (often called composition) such that:
1. **Associativity:** $(g_1∘g_2)∘g_3=g_1∘(g_2∘g_3)$.
2. **Identity:** There exists an identity element $e∈G$ such that $e∘g=g∘e=g$ for all $g$.
3. **Inverse:** Each element $g∈G$ has a unique inverse $g^{-1}$ satisfying $g∘g^{-1}=g^{-1}∘g=e$.
4. **Closure:** For any $g_1,g_2∈G$, we have $g_1∘g_2∈G$.

A symmetry group for a domain $Ω$ is a group of transformations $ϕ:Ω→Ω$ whose elements are invertible maps that preserve some structure on $Ω$. For example, the set of all rotations of a square forms a finite group of symmetries under composition of rotations.

#### 1.3. Group Actions
An abstract group $G$ is often given without specifying how it acts on the domain $Ω$. A group action formally links the group’s elements to transformations on $Ω$. Specifically, an action of $G$ on $Ω$ is a map

$$(g,ω)↦g⋅ω\text{ where }g∈G,ω∈Ω,$$

such that:
1. $e⋅ω=ω$ for all $ω$ (identity acts trivially).
2. $(g_1∘g_2)⋅ω=g_1⋅(g_2⋅ω)$ for all $g_1,g_2∈G$.

When $Ω$ is discrete and finite, the group elements can be viewed as permutation-like operations on the domain indices.

**Extension to signals:** Once we have a group action on $Ω$, we can naturally extend it to signals $x(ω)$. For $g∈G$:

$$(g⋅x)(ω)=x(g^{-1}⋅ω).$$

One can check that this extended action on signals is linear if the signal space is a vector space. This is crucial since deep learning typically relies on linear algebraic operations.

#### 1.4. Group Representations
A representation of a group $G$ on a finite-dimensional vector space $V$ is a function

$$ρ:G→GL(V)$$

that assigns to each $g∈G$ an invertible matrix $ρ(g)$ (or invertible linear map) such that:

$$ρ(g_1∘g_2)=ρ(g_1)ρ(g_2).$$

In other words, the group operation corresponds to matrix multiplication in the representation space.

- **Faithful representation:** If $ρ$ is injective (different group elements give different matrices), it is called faithful.
- Often, $dim(V)$ can be quite large, e.g., if $V$ corresponds to signals on $Ω$ with $|Ω|×C$ dimensions.

**Example:** Consider a discrete shift group $G=\{\text{shift by }k\}$ on a 1D ring of 5 elements. The matrix for the “shift by 1” element can be written in a permutation form that cyclically moves indices. Higher shifts are powers of this permutation matrix.

#### 1.5. Invariance and Equivariance
These are central concepts in Geometric Deep Learning:
- **Invariance:** A function $f$ is group-invariant if applying any group transformation to its input does not change its output. Formally, if $x$ is a signal on $Ω$,

$$f(g⋅x)=f(x)\text{ for all }g∈G.$$

- **Equivariance:** A function $Φ$ is group-equivariant if applying the group transformation before or after the function yields commensurate results. Formally,

$$Φ(g⋅x)=g⋅Φ(x)\text{ for all }g∈G.$$

**Why this matters:**
- Invariance is helpful for tasks where a single global label is desired (e.g., “Is there a cat in the image?”), because we do not want the classification to depend on how the cat is shifted or rotated.
- Equivariance is more flexible for tasks that require a structured output that must “follow” the transformations in the input (e.g., segmentation masks or detection bounding boxes).

However, a purely invariant intermediate representation can lose crucial relational information (e.g., the relative positions of parts of an object). Hence, modern designs often favor equivariant layers internally and adopt a final, invariant operation near the output.

#### 1.6. Orbits and Equivalence Relations
Given a group $G$ acting on $Ω$, the orbit of an element $ω∈Ω$ is the set $\{g⋅ω:g∈G\}$. Two elements $ω_1$ and $ω_2$ are said to be $G$-equivalent if one can be reached from the other by a group transformation. This defines an equivalence relation (reflexive, symmetric, and transitive). Invariance can be viewed as producing a constant value on each entire orbit; equivariance ensures consistent movement within each orbit.

---

### 2. The Blueprint of Geometric Deep Learning
Having established the basic group-theoretic machinery, we can now articulate the standard GDL “blueprint.” Let $Ω$ be a domain equipped with a group $G$ of symmetries.

A Geometric Deep Learning architecture typically consists of:
1. **Linear $G$-Equivariant Layers $Λ$:**

$$Λ:X(Ω,R^{C_{in}})→X(Ω,R^{C_{out}})$$

satisfying (for all $g∈G$):

$$Λ(g⋅x)=g⋅Λ(x).$$

In essence, these layers are linear maps that commute with group actions.

2. **Elementwise Nonlinearities $σ$:**

$$σ(z(ω))=σ_{elem}(z(ω)),$$

applied identically to every element of the domain or to every channel.

3. **Pooling / Coarsening $P$:**
   - Local pooling might aggregate or “downsample” features within local neighborhoods, respecting domain structure.
   - Global pooling is an aggregator over the entire domain $Ω$ to yield a single $G$-invariant feature (e.g., for classification tasks).

4. **Optional $G$-Invariant Layer $Γ$:**

$$Γ:X(Ω,R^{C_{in}})→R,$$

satisfying:

$$Γ(g⋅x)=Γ(x),\forall g∈G.$$

This is often used in the final stage of a network to produce a single global output (like a class label).

#### Common Examples of GDL Architectures
- **Convolutional Neural Networks (CNNs):** Grid domain with translation symmetry.
- **Spherical CNNs:** Spherical domain (e.g., 360° images) with rotation symmetry on the sphere ($SO(3)$).
- **Graph Neural Networks (GNNs):** Node/edge domains with permutation symmetry (any re-labelling of nodes leaves the graph unchanged).
- **Transformers/Deep Sets:** Set-like domains with permutation symmetry.
- **Intrinsic CNNs / Mesh CNNs:** Manifold domains with isometry (or gauge) symmetries.

Each is an instantiation of the same high-level principle:
**Constrain the learned function to respect certain transformations—leading to better generalization and data efficiency.**

---
# --SECOND HALF--

# 1. Introduction to Graphs and Networks
## 1.1. Why Study Graphs?
Graphs are among the most expressive data structures for representing real-world systems. They provide a formal way to capture entities (nodes) and their interactions (edges). Real-life phenomena—such as social networks, biological networks, communication networks, and collaborative relationships—can be mapped to graph structures.

**Examples:**
- An actor-movie network from IMDb, where nodes can be actors or movies.
- A retweet network on Twitter, where nodes can be users and edges represent retweets or follows.
- The Internet router network, where nodes represent routers and edges represent physical or logical connections.
- A protein interaction network, where nodes are proteins and edges represent interactions.
- A food web, where nodes can be species and edges represent predator-prey relationships.

Through graphs, we can analyze connectivity, centrality, clustering, communities, and many other rich properties that illuminate the inner workings of these systems.

# 2. Basic Graph Definitions
A graph $G=(V,E)$ is specified by:
- A set of nodes (or vertices), $V$.
- A set of edges (or links), $E \subseteq V \times V$, connecting pairs of nodes.

When dealing with graphs, we often use an adjacency matrix $A$, of size $|V| \times |V|$, where

$$A_{ij} = \begin{cases} 
1 & \text{if there is an edge between nodes } i \text{ and } j, \\
0 & \text{otherwise}
\end{cases}$$

for a simple unweighted graph. Variants include directed graphs ($A$ can be asymmetric) or weighted graphs (entry $A_{ij}$ can be a real-valued weight).

## 2.1. Questions to Ask When Observing a Network
- What do nodes represent? (e.g., individuals, airports, proteins)
- What do edges represent? (e.g., friendships, flights, interactions)
- Are edges directed or undirected?
- Are edges weighted or unweighted? (e.g., frequencies, capacities)
- How do we measure node importance? (e.g., number of connections, centrality metrics)
- Are there visible clusters or communities?

By clarifying these questions, we characterize the network’s topology and function.

# 3. Beyond Simple Graphs: Complex and Multilayer Networks
## 3.1. Complex Networks
Many real-world graphs are complex networks, often exhibiting:
- Multiple types of relationships (e.g., “multiplex” or “multilayer” settings).
- Weighted or directed edges.
- Community structures.
- Hierarchical organization.

A multilayer network can be represented by a “supra-adjacency matrix,” stitching together adjacency matrices for each layer and, potentially, inter-layer edges. This can be viewed as a higher-order tensor structure, encoding more nuanced information than a single adjacency matrix.

## 3.2. Different Types of Walks
In multilayer networks, we can define various walks that traverse across different layers. Restricting repeated nodes, repeated edges, or specifying origin/destination nodes leads to specialized definitions (e.g., multilayer trails or multilayer cycles). These generalize classical graph walks to richer, multi-aspect structures.

# 4. The Graph Laplacian
## 4.1. Motivation
While the adjacency matrix $A$ succinctly encodes which nodes connect to which, certain downstream applications (spectral clustering, diffusion processes, etc.) benefit from the graph Laplacian, $L$. The Laplacian reorganizes the adjacency information into a matrix with beneficial mathematical properties.

## 4.2. Definitions
Let $D$ be the diagonal degree matrix, where $D_{ii} = \sum_{j} A_{ij}$ (for undirected graphs, this is the number of neighbors of node $i$).

- **Unnormalized Laplacian:**  
  $$L = D - A.$$

- **Symmetric normalized Laplacian:**  
  $$L_{\text{sym}} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}}.$$

- **Random walk Laplacian:**  
  $$L_{\text{RW}} = D^{-1} L.$$

These variants encode the same essential connectivity information but differ in how they normalize by node degrees.

## 4.3. Key Properties
- For undirected graphs, $L$ is symmetric and positive semi-definite.
- The eigenvalues of $L$ are real and nonnegative. One can order them as $0 = \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_{|V|}$. The existence of a zero eigenvalue corresponds to at least one connected component in the graph.
- A common identity:  
  $$x^T L x = \frac{1}{2} \sum_{(i,j) \in E} (x_i - x_j)^2,$$  
  for any vector $x \in \mathbb{R}^{|V|}$. This identity captures how Laplacians measure “smoothness” of signals on graphs.

# 5. Spectral Clustering
## 5.1. The Core Idea
Spectral clustering uses the eigen-decomposition of $L$ to find community structures. Notably, the second-smallest eigenvalue (and associated eigenvector) captures information about how the graph can be split into two connected components in a way that minimizes certain “cut” metrics.

## 5.2. Two-Way Clustering
We aim to partition the nodes $V$ into two disjoint sets $S$ and $S^c$. A common objective is to minimize:

$$\text{RatioCut}(S) = \frac{\text{Cut}(S)}{|S|} + \frac{\text{Cut}(S^c)}{|S^c|},$$

where

$$\text{Cut}(S) = | \{(i,j) \in E \,|\, i \in S, j \in S^c \} |.$$

Minimizing the cut alone (the number of inter-cluster edges) can yield trivial solutions (tiny sets). RatioCut (and other variants like Normalized Cut) penalizes very small partitions.

## 5.3. Relaxation and the Second Eigenvector
Construct an indicator vector $v$ to represent the cluster: $v_i$ is positive if node $i$ is in $S$, negative if node $i$ is in $S^c$. The problem’s integer constraints make it NP-hard. By allowing $v \in \mathbb{R}^{|V|}$ as a relaxation, the Rayleigh–Ritz theorem implies that the optimal $v$ is the eigenvector of $L$ associated with the second-smallest eigenvalue. After obtaining the eigenvector, we recover a binary partition by placing each node in one subset or the other depending on the sign of its corresponding entry.

## 5.4. Extending to k Clusters
For $k$-way clustering, one uses the first $k$ eigenvectors of $L$, creating a $|V| \times k$ matrix. Each row (one row per node) is treated as a vector in $\mathbb{R}^k$. Clustering (often with $k$-means) on these vectors partitions nodes into $k$ clusters.

## 5.5. Example: Zachary’s Karate Club
A famous social network capturing friendships in a karate club that eventually split into two groups. Spectral clustering on the second-smallest Laplacian eigenvector nearly reproduces the real split, misclassifying only one node.

**Physical Analogy:** This approach is reminiscent of the lowest modes of vibration in physical systems, where the second mode naturally splits the structure into two opposing phases.

# 6. Node (and Edge) Importance Measures
Real networks exhibit heterogeneous roles for their elements. Several centrality measures capture different notions of “importance.”

## 6.1. Degree Centrality
The simplest measure. The degree of a node $i$ is the number of neighbors of $i$. Nodes with significantly higher degree are called hubs. Often, large-degree hubs play critical roles in connectivity.

## 6.2. Closeness Centrality
A node is “close” to others if it has short path distances to them. The closeness of node $i$ is defined as:

$$C_{\text{closeness}}(i) = \frac{1}{\sum_j \ell_{ij}},$$

where $\ell_{ij}$ is the shortest-path distance between $i$ and $j$. A high closeness score means the node can reach others more quickly on average.

## 6.3. Betweenness Centrality
Measures how often a node (or edge) lies on shortest paths between other pairs of nodes:

$$C_{\text{betweenness}}(i) = \sum_{h \neq t} \frac{\sigma_{ht}(i)}{\sigma_{ht}},$$

where $\sigma_{ht}$ is the number of shortest paths from $h$ to $t$, and $\sigma_{ht}(i)$ is the number of those shortest paths passing through $i$. Nodes can have high betweenness without having a high degree, if they serve as crucial “bridges” in the network.

## 6.4. Centrality Distributions
- **Degree Distribution:** The fraction of nodes $p_k$ having degree $k$. In large networks, many real systems exhibit “heavy-tailed” (skewed) degree distributions, meaning some nodes have extremely large degree compared to the majority.
- **Betweenness Distribution:** Similar heavy-tailed behavior can appear for betweenness, where a few nodes handle disproportionately many shortest paths.

## 6.5. Friendship Paradox
In heavy-tailed networks, “Your friends have more friends than you do.” Formally, if you pick an edge at random, it is more likely to connect to a high-degree node. Hence, the average degree of neighbors is higher than the average degree of a randomly chosen node.

# 7. Macroscopic Network Properties
## 7.1. Small-World Effect
Many real networks have surprisingly short path lengths between nodes:
- Even for billions of individuals, the average path length can remain small (the so-called “six degrees of separation”).
- Hubs accelerate the reduction of average distances (they enable “ultra-small” worlds).

## 7.2. Clustering Coefficient
Clustering measures how many triangles exist in a node’s neighborhood. A node’s clustering coefficient is:

$$C(i) = \frac{\text{number of edges among neighbors of } i}{\binom{k_i}{2}},$$

or equivalently:

$$C(i) = \frac{\text{number of triangles through } i}{\binom{k_i}{2}}$$

(for $k_i \geq 2$). High clustering is typical in social or biological networks where neighbors tend to be interconnected.

## 7.3. Robustness
When we remove nodes/edges from a network (intentionally or at random), how does its connectivity deteriorate?
- **Random failure:** Removing random nodes often leaves the network structure largely intact in the presence of hubs—unless the fraction removed is very large.
- **Targeted attack:** Deliberately removing hubs can quickly fragment the network.

## 7.4. k-Core Decomposition
A $k$-core is the maximal subgraph in which every node has at least degree $k$. Iteratively removing all nodes of degree less than $k$ peels away “shells” to uncover progressively denser cores. This helps visualize or simplify large networks by focusing on the most interconnected areas.

# 8. Classic Generative Models of Networks
Real networks are rarely formed by simply linking nodes in arbitrary ways. Several models have been proposed to explain how large-scale structures and heavy-tailed distributions emerge.

## 8.1. Erdős–Rényi (Random) Model
A fundamental but often unrealistic model:
- **Nodes:** $N$ nodes.
- **Edges:** Each pair of nodes is connected with probability $p$, independently of all other pairs.
- **Degree Distribution:** Binomial (or Poisson-like for large $N$ and small $p$), centered around $\langle k \rangle = p(N-1)$.
- **Small-World Property:** Average distances are logarithmic in $N$.
- **Low Clustering:** Since any two neighbors of a node connect with probability $p$, the expected clustering coefficient is $p$. In reality, many real networks have much higher clustering.
- **No Hubs:** The degree distribution is narrow (peaked around $\langle k \rangle$) and does not exhibit heavy tails.

Hence, pure random networks do not capture many real-world phenomena (lack of hubs, low clustering).

## 8.2. Watts–Strogatz (Small-World) Model
Combines a regular ring lattice with random rewiring:
- **Arrange:** $N$ nodes in a ring; each node has degree $k$ by connecting to its nearest neighbors.
- **Rewire:** With probability $p$, rewire each edge to a random node.

**Key Features:**
- For small $p$, distances drastically shorten (because of “shortcuts”), while the network retains relatively high clustering from the underlying lattice.
- **Small-World:** Present for intermediate $p$.
- **High Clustering:** Better than Erdős–Rényi for small $p$.
- **Degree Distribution:** Remains relatively narrow (no heavy tails).

## 8.3. Barabási–Albert (Scale-Free) Model
A breakthrough model capturing hubs via growth + preferential attachment:
- **Growth:** Start with $m_0$ nodes (often a clique). At each step, a new node arrives with $m \leq m_0$ edges to attach.
- **Preferential attachment:** The probability a new node links to an existing node $j$ is proportional to $k_j$ (the current degree of $j$).

Over time, older nodes with higher degrees accumulate edges even faster, creating heavy-tailed (power-law) degree distributions with “hubs.”

**Rich-Get-Richer:** Older, more connected nodes keep winning new links.

**Limitation:** The oldest nodes are always the highest-degree hubs. In reality, a younger node can sometimes overtake older ones.

## 8.4. Extensions of Barabási–Albert
Various refinements allow for differences in node “intrinsic appeal” (fitness) or local triadic closure:
- **Attractiveness Model:** Adds a constant $A$ so that even a node with zero degree can gain edges.
- **Fitness Model:** Each node $j$ has fitness $\eta_j$. When connecting, the probability of choosing $j$ is proportional to $\eta_j k_j$.
- **Random Walk Model:** New edges are formed to neighbors-of-neighbors, naturally creating clustering and effectively implementing a form of preferential attachment.
- **Copy Model:** A new Webpage “copies” the neighbors of existing pages.
- **Rank Model:** Replaces absolute measures of degree with ranks.

These models explain heavy-tailed degree distributions and other emergent phenomena more realistically.

# 9. Motifs, Hierarchies, and Modularity
## 9.1. Network Motifs
A network motif is a small subgraph that appears significantly more often than it would in a randomized version of the network. These can indicate fundamental building blocks or “atomic” interaction patterns in complex systems (e.g., feed-forward loops in biological networks).

## 9.2. Hierarchical Networks
Some real-world graphs exhibit hierarchical organization:
- High clustering within sub-communities.
- A few edges bridging different parts of the hierarchy.

In hierarchical networks, one may observe the clustering coefficient $C(k)$ scaling as $k^{-1}$ (when plotted in a log-log fashion), which is often interpreted as “small-degree nodes live in tightly knit communities, whereas large-degree hubs link different communities.”

# 10. Concluding Remarks
## 10.1. Key Takeaways
- **Graphs are Universal:** They can represent diverse systems (technological, social, biological) with a common mathematical language.
- **Connectivity and Measures:** Adjacency matrices, Laplacians, and centralities measure structural and functional importance of nodes/edges.
- **Spectral Methods:** The graph Laplacian’s eigenvectors are powerful tools for detecting communities (spectral clustering) and understanding network structure.
- **Real-World Networks:** Many real networks are small-world (short distances, high clustering) and scale-free (broad degree distributions, presence of hubs).
- **Growth and Attachment Models:** Classic random graphs miss key real-world features. Barabási–Albert and its extensions provide better approximations of how real networks evolve.
- **Modularity and Hierarchies:** Networks are composed of communities and hierarchical layers, revealing how function is distributed and maintained.

## 10.2. Intuition and Big Picture
- **Hubs:** Facilitate short paths (small-world effect) but also pose vulnerability points (targeted attacks).
- **High Clustering:** Arises from local attachment biases (friends-of-friends, triadic closure).
- **Heavy-Tailed Distributions:** Naturally occur when new connections grow “preferentially” toward already well-connected nodes.
- **Hierarchical Modularity:** Real networks are often fractal-like: dense clusters of nodes revolve around hubs, which link to other dense clusters.

By combining the spectral perspective (Laplacians, eigen-decompositions) with generative models (random, small-world, scale-free) and centrality measures, we gain a robust, multifaceted picture of networks—a foundation for advanced topics in Geometric Deep Learning and
