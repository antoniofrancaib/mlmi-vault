Some problems RNA inverse folding problem 
two geometric the chances of predicting the underlying sequence 


<span style="color: red;"> consider in our inverse folding problem, the set of graphs are constrained to share the same vertices but we might generalize this.</span>

Let $X$ be the space of (compact) sets $\{G_1, \dots, G_n\}$ of geometric graphs, where each $G_i = (V_i, E_i, P_i)$ has:
- $V_i$: finite set of nodes
- $E_i$: finite set of edges
- $P_i \in \mathbb{R}^{|V_i| \times 3}$: 3D coordinates of the nodes via injective mapping $\phi_i: V_i \to P_i$

We consider the group $G = S_n \times SE(3)$. In short, $\sigma \in S_n$ permutes the index of each graph in the set, and $h \in SE(3)$ acts on each $G_i$ by applying the rigid transform $h$ to all node coordinates in $P_i$. $G$ acts on $X$ by: 

$$
  (\sigma,h) \cdot \{G_1, \dots, G_n\} = \{ h \cdot G_{\sigma^{-1}(1)}, \dots, h \cdot G_{\sigma^{-1}(n)} \}.
  $$

In this set up, a function $L: X \to Y$ is $G$-equivariant if, for every $(x,g) \in X \times G$:
$$
  L((\sigma,h) \cdot \{G_1, \dots, G_n\}) = (\sigma,h) \cdot L(\{G_1, \dots, G_n\}),
  $$
for every $\sigma \in S_n$, $h \in SE(3)$.


**Proposition (Universal Architecture):**

Let $X$ be the space of sets of geometric graphs (with 3D node coordinates). Fix a compact subset $K \subset X$ such that
  $$
  K = \bigcup_{g \in G} g K,
  $$
where $G$ is our symmetry group (e.g., $G = S_n \times SE(3)$). 

**Goal**: Find a $G$-equivariant network architecture that is a **universal approximator** of all continuous, $G$-equivariant functions on $K$ (with respect to the $\|\cdot\|_{\infty}$ norm), aka for any $\epsilon > 0$, there exists a finite-depth $G$-equivariant network $\hat{f}$ such that $$
  \sup_{x \in K} \| \hat{f}(x) - f(x) \|_{\infty} < \epsilon.
  $$







### 2. $G$-Invariant Layer

A function $L: X \to Y$ is $G$-invariant if, for every $(x,g) \in X \times G$:
  $$
  L(g \cdot x) = L(x).
  $$
In the same setting, that expands to:
  $$
  L((\sigma,h) \cdot \{G_1, \dots, G_n\}) = L(\{G_1, \dots, G_n\}),
  $$
again for all $\sigma \in S_n$ and $h \in SE(3)$.

### Linear Layers

If $X$ and $Y$ are vector spaces (or modules) carrying representations of $G$, then one often adds the requirement that $L$ be a linear map that also satisfies one of the above symmetry conditions. In that case, $L$ is called a linear $G$-equivariant (or $G$-invariant) layer.

### Summary

- **$G$-equivariant layer**: commutes with the group action, so:
  $$
  L(g \cdot x) = g \cdot L(x).
  $$
- **$G$-invariant layer**: is fixed by the group action, so:
  $$
  L(g \cdot x) = L(x).
  $$

Where $G = S_n \times SE(3)$ acts on a set of 3D geometric graphs by reordering the graphs ($\sigma$) and rigidly transforming their coordinates ($h$).




## 1. $G$-Equivariant Networks

We define a $G$-equivariant network $f$ as a composition of **linear $G$-equivariant layers** $L_i$ and  **pointwise nonlinearities** $\sigma$ (e.g., ReLU). Given $x$ is an element in our input space $X$, the network would have the form: $$
  f = L_k \circ \sigma \circ L_{k-1} \circ \dots \circ \sigma \circ L_1.
  $$
Since each $L_i$ is $G$-equivariant and $\sigma$ is pointwise, one can show inductively that $f$ is itself $G$-equivariant. Note each $L_i$ is specifically designed (or constrained) so that it commutes with every $(\sigma, h) \in S_n \times SE(3)$, meaning it respects both **Permutation equivariance** (reordering the set of input graphs), and **Rigid-motion equivariance** (applying the same rotation/translation to all 3D node coordinates).

## 2. $G$-Invariant Networks

To obtain a $G$-invariant function $g$ from a $G$-equivariant network, we simply add a **$G$-invariant layer** $h$ (invariant readout). 
