For a schematic overview of the model transformations, refer to Appendix~\ref{sec:gnn-transformations-summary}.  Building on this foundation, we introduce several modifications that we refer to as our main contributions. 

First, we conduct a theoretical analysis of the original model to guide our enhancements aimed at improving its expressivity. In particular, we compute an upper bound on the Lipschitz constant of the encoder to assess how contractive the overall mapping might be, with the goal of avoiding underperformance in edge cases.

Let $f = f_L \circ \cdots \circ f_1 \circ f_{\text{emb}}$ denote the overall encoder mapping, where

$$
f_{\text{emb}} = \text{GVP}_{\text{emb}} \circ \text{LN}: \{\mathbf{G}_1, \dots, \mathbf{G}_C\} \to \{\mathbf{H}_1, \dots, \mathbf{H}_C\},
$$

with each $\mathbf{G}_i \in (\mathbb{R}^{64} \times \mathbb{R}^{4 \times 3})^N$ being a conformation and each output $\mathbf{H}_i \in (\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3})^N$. We assume that the layer normalization is 1--Lipschitz. Thus, we can upper-bound the Lipschitz constant of the embedding as $L(f_{\text{emb}}) \leq L_{\text{GVP}_{\text{emb}}}.$

Each subsequent encoder layer $f_l$ (for $l = 1, \dots, L$) consists of two submodules: a message-passing component implemented via a GVP block, with Lipschitz constant $L_{\text{GVP}_{\text{msg}}}^{(l)}$, and a feedforward update (also via a GVP block) in a residual connection, with Lipschitz constant $L_{\text{GVP}_{\text{ff}}}^{(l)}$. In our message passing, the neighboring node features are aggregated through a linear operator $\mathbf{A}$ with spectral norm $\|\mathbf{A}\|$. The residual connection introduces an additive identity term, which results in the Lipschitz constant of the $l$th layer being bounded by
$$
L(f_l) \leq 1 + \left( \|\mathbf{A}\| \, L_{\text{GVP}_{\text{msg}}}^{(l)} \cdot L_{\text{GVP}_{\text{ff}}}^{(l)} \right).
$$
Thus, assuming that the Lipschitz constant of the pooling operation (mean aggregation) is 1, and by the composition rule,
$$
L(f) \leq L_{\text{GVP}_{\text{emb}}} \cdot \prod_{l=1}^L \left( 1 + \left( \|\mathbf{A}\| \, L_{\text{GVP}_{\text{msg}}}^{(l)} \cdot L_{\text{GVP}_{\text{ff}}}^{(l)} \right) \right).
$$
In this context, we recall a large Lipschitz constant can make the model overly sensitive to small input perturbations (instability), whereas a small one may be excessively contractive, failing to distinguish different inputs (oversmoothing). We observe that each new layer raises the Lipschitz constant monotonically, underscoring the risk of instability. On the other side, the only layer that could lead to an **contractive** encoding is the embedding map. For simplicity, we assume that this is not the case in our model.

Motivated by these considerations, our Lipschitz analysis directly informs us to introduce an **attention‐based** layer in parallel with the original GVP‐based message passing. Rather than multiplying the existing Lipschitz factors, which could potentially cause a notorious blow‐up in sensitivity, a parallel aggregator is simply added by fusing the GVP and attention outputs via $\alpha \times \text{GVP} + (1 - \alpha) \times \text{Attention}$. For simplicity, we set $\alpha = 0.5$, leaving the learning of an optimal $\alpha$ to future work.

Because the Lipschitz constant of a sum $f_1 + f_2$ is bounded by $L(f_1) + L(f_2)$ (scaled by the mixing coefficients), this design boosts expressivity—through attention’s adaptive neighbor weighting—without recklessly inflating the overall Lipschitz bound. Note the new Lipschitz constant encoder becomes
$$
L(f) \leq L_{\text{GVP}_{\text{emb}}} \cdot \prod_{l=1}^L \left( 1 + \left( \alpha \|\mathbf{A}\| L_{\text{GVP}_{\text{msg}}}^{(l)} + (1 - \alpha) L_\text{attn}^{(l)} \right) \right).
$$

Unlike a fixed linear operator $\mathbf{A}$, attention learns a data‐dependent weighting matrix (constrained to be row‐stochastic via a softmax).  While the GVP branch enforces strong geometric equivariance and local feature aggregation, the attention branch captures long-range dependencies and global interactions. Our proposed hybrid mechanism can “down‐weight” uninformative neighbors and “up‐weight” salient ones, capturing more nuanced node relationships and thus increasing the model’s capacity to discriminate among different inputs. Their combined output is more expressive and capable of representing a broader class of functions. We achieve this while maintaining a theoretical guarantee that the encoder's Lipschitz constant does not increase significantly. 

Implementation-wise, we replace the original MultiGVPConvLayer with a MultiAttentiveGVPLayer that fuses GVP‐based message passing (which encodes local geometric features) and a multi‐head attention mechanism (which learns a row‐stochastic weight matrix via $\text{softmax}(\mathbf{Q} \mathbf{K}^\top)$). For each node, scalar features are combined with vector norms for attention queries and keys


 The attention branch provides an alternative information pathway that is less contractive, thereby preserving input variations that might otherwise be lost.

2. **Augmented Expressivity:** While the GVP branch enforces strong geometric equivariance and local feature aggregation, the attention branch captures long-range dependencies and global interactions. Their combined output is more expressive and capable of representing a broader class of functions.

Building on this, our Lipschitz analysis directly informs us to **introduce an attention‐based aggregator in parallel** with the original GVP‐based message passing. The rationale is that rather than multiplying the existing Lipschitz factors (which would risk an exponential blow‐up in sensitivity), a parallel aggregator is simply added— by fusing the GVP and attention outputs (e.g.$\ \alpha \times \text{GVP} + (1-\alpha) \times \text{Attention}$). For simplicity we just take $\alpha = 0.5$ and just leave the learning of optimal $\alpha$ for future work. 

Because the Lipschitz constant of a sum $f_1 + f_2$ is bounded by $L(f_1) + L(f_2)$ (scaled by the mixing coefficients), we can boost the model’s expressivity (through attention’s adaptive neighbor weighting) without recklessly inflating the overall Lipschitz bound. 

In the standard GVP aggregator, A\mathbf{A}A is a fixed linear operator (often a normalized adjacency matrix). This lacks flexibility when certain edges are noisy or irrelevant. By contrast, attention learns a **data‐dependent** weighting matrix—often constrained to be row‐stochastic via a softmax—which can “down‐weight” uninformative neighbors and “up‐weight” salient ones. Intuitively, this **adaptive** aggregator can capture more nuanced relationships among nodes, increasing the network’s ability to distinguish different inputs (i.e.\ **increasing expressivity**).






Furthermore, we note that in this architecture, after the embedding layer, each RNA structure graph is represented as an element of

$$
\mathbf{H} = (\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3})^N,
$$

and the input to the encoder is a set

$$
\mathbf{X} = \{\mathbf{H}_1, \dots, \mathbf{H}_C\} \subset \mathbf{H}^C.
$$

The overall function computed by the encoder must respect the symmetry inherent to this input: it must be invariant under any permutation of the $C$ graphs (i.e. invariant with respect to $S_C$) and equivariant with respect to the symmetry group $\mathbf{H}$ that governs the internal geometric structure of each graph. 

A critical component in achieving universality for set functions (in our case sets of geometric graphs) is the design of the pooling operator. Standard pooling strategies (such as simple averaging) are invariant under $S_C$ but are not injective; that is, they can collapse different sets into the same representation. To overcome this limitation, we propose a pooling operator based on higher–order statistics.

Let $\psi:\mathbf{H} \to \mathbf{F}$ be the per–graph feature extractor implemented by our AttentiveGVP–GNN layers , where $\mathbf{F} \subset \mathbb{R}^d$ (with, for instance, $d=128$ for the scalar part and the corresponding vector part in $\mathbb{R}^{16 \times 3}$). We define the pooling operator $\phi$ as

$$
\phi(\mathbf{X}) = \left( \sum_{i=1}^C \psi(\mathbf{H}_i),\; \sum_{i=1}^C \psi(\mathbf{H}_i)^{\odot 2},\; \dots,\; \sum_{i=1}^C \psi(\mathbf{H}_i)^{\odot K},\; C \right).
$$

Here, $\psi(\mathbf{H}_i)^{\odot k}$ denotes the element–wise $k$th power of the feature vector, and $K$ is chosen sufficiently large so that the mapping

$$
\phi: \mathbf{H}^C \to (\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3})^N
$$

injective over the compact domain of interest. It is worth noting that $\phi$ is invariant with respect to $S_C$ because summation is permutation–invariant, and it is equivariant with respect to $\mathbf{H}$ since $\psi$ is built from $\mathbf{H}$–equivariant GVP layers.

We now state the following theorem, which is inspired by and extends results such as those in \cite{Zaheer2017Deepsets}.

**Theorem 1 (Injective Higher–Order Pooling).**  
Let

$$
\mathcal{K} \subset \mathbf{H}^C = \left((\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3})^N\right)^C
$$

be a compact domain that is invariant under the action of

$$
G = S_C \times \mathbf{H},
$$

where $S_C$ permutes the $C$ graphs and $\mathbf{H}$ acts on the geometric features. Suppose that the per–graph mapping $\psi: \mathbf{H} \to \mathbf{F}$ is implemented by universal $\mathbf{H}$–equivariant layers and that for a sufficiently large integer $K$, the pooling operator

$$
\phi(\mathbf{X}) = \left( \sum_{i=1}^C \psi(\mathbf{H}_i),\; \sum_{i=1}^C \psi(\mathbf{H}_i)^{\odot 2},\; \dots,\; \sum_{i=1}^C \psi(\mathbf{H}_i)^{\odot K},\; C \right)
$$

is injective on $\mathcal{K}$. Then, for any continuous $G$–invariant function

$$
f: \mathcal{K} \to (\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3})^N,
$$

and for every $\epsilon > 0$, there exists a multilayer perceptron $M$ (which is a universal approximator by assumption) such that the function

$$
F(\mathbf{X}) = M(\phi(\mathbf{X}))
$$

satisfies

$$
\sup_{\mathbf{X} \in \mathcal{K}} \|F(\mathbf{X}) - f(\mathbf{X})\| < \epsilon.
$$

**Proof (Sketch):**

1. *Equivariant Per–Graph Processing:*  
Each graph $\mathbf{H}_i$ is mapped to a feature representation $\psi(\mathbf{H}_i) \in \mathbf{F}$ by a stack of $\mathbf{H}$–equivariant GVP–GNN layers. By the universal approximation properties of such layers (see, e.g., \cite{Maron2019UniversalGNN}), $\psi$ can approximate any continuous $\mathbf{H}$–equivariant function on $\mathbf{H}$.

2. *Injective Pooling via Higher–Order Moments:*  
The mapping $\phi$ aggregates higher–order moments of the set $\{\psi(\mathbf{H}_1),\dots,\psi(\mathbf{H}_C)\}$. Classical results in symmetric polynomial theory (e.g., Newton’s identities) imply that, if $K$ is chosen sufficiently large relative to the dimension of $\psi(\mathbf{H}_i)$, then the collection

$$
\left\{\sum_{i=1}^{C}\psi(\mathbf{H}_i)^{\odot k} \;:\; k=1,\dots,K\right\} \quad \text{(augmented with } C \text{)}
$$

uniquely determines the multiset $\{\psi(\mathbf{H}_i)\}$. Thus, $\phi$ is injective.

3. *Function Decomposition:*  
Since $f$ is $G$–invariant and $\phi$ is both invariant and injective, there exists a continuous function $\rho$ defined on $\phi(\mathcal{K})$ such that

$$
f(\mathbf{X}) = \rho(\phi(\mathbf{X})).
$$

4. *Universal Approximation:*  
By the classical universal approximation theorem (e.g., \cite{Cybenko1989Approximation, Hornik1989Multilayer}), there exists an MLP $M$ that can approximate $\rho$ uniformly over the compact set $\phi(\mathcal{K})$ to within any desired error $\epsilon$.

5. *Conclusion:*  
Thus, the composite mapping

$$
F(\mathbf{X}) = M(\phi(\mathbf{X}))
$$

approximates $f$ uniformly on $\mathcal{K}$. This proves that the encoder part of our architecture is a universal approximator of continuous $G$–invariant functions, provided that the pooling operator uses higher–order statistics.

**Corollary 1 (Universality of the Full Encoder).**  
Assuming that the GVP layers in the remainder of the network are universal approximators of equivariant functions, it follows that the overall encoder—composed of equivariant per–graph processing, higher–order injective pooling, and a universal MLP—is a universal approximator for continuous functions defined on

$$
\{\mathbf{H}_1, \dots, \mathbf{H}_C\} \subset ((\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3})^N)^C.
$$

This theoretical result motivates our architectural modification: by replacing simple averaging with higher–order tensor moment pooling, we ensure that the aggregated representation retains all necessary information to uniquely characterize the input set. Consequently, this guarantees the universality of the encoder component of our model.


