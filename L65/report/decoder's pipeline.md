# Notation & Initial Setup

## Nodes:
Let the graph have $n$ nodes. For each node $i \in \{1, \dots, n\}$, the encoder provides a representation:

$$
\mathbf{x}_i = (\mathbf{s}_i, \mathbf{v}_i) \quad \text{with} \quad \mathbf{s}_i \in \mathbb{R}^{128} \quad \text{and} \quad \mathbf{v}_i \in \mathbb{R}^{16 \times 3}.
$$

We denote the collection as

$$
\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^n.
$$

## Edges:
For each edge $(j, i) \in E$ (with $j$ a neighbor of $i$), we have an edge embedding

$$
\mathbf{e}_{ji} = (\mathbf{s}_{ji}, \mathbf{v}_{ji}), \quad \mathbf{s}_{ji} \in \mathbb{R}^{32}, \quad \mathbf{v}_{ji} \in \mathbb{R}^{1 \times 3}.
$$

## Sequence Tokens:
The sequence $\text{seq}$ is an array with entries in $\{0, 1, 2, 3\}$ (one per node). There is a learnable token embedding

$$
\mathbf{W}_s: \{0, 1, 2, 3\} \rightarrow \mathbb{R}^4,
$$

so that for any token $t \in \{0, 1, 2, 3\}$ we have an embedding

$$
\boldsymbol{\tau}_t = \mathbf{W}_s(t) \in \mathbb{R}^4.
$$

Initially, before any prediction, the tokens are set to a default (say, $0$), and as we progress, they are updated with the sampled nucleotides.

# Step 1. Augmenting Edge Features with Token Embeddings

During decoding, we condition the message passing on previously decoded tokens. For each edge $(j, i)$:

**Token Embedding for Source Node $j$:**
If node $j$ has already been decoded (i.e. $j < i$ in the autoregressive order), we embed its token:

$$
\boldsymbol{\tau}_j = \mathbf{W}_s(\text{seq}[j]) \in \mathbb{R}^4.
$$

Otherwise, we set the embedding to zero.

**Augmented Edge Scalar:**
The original edge scalar $\mathbf{s}_{ji} \in \mathbb{R}^{32}$ is concatenated with the token embedding:

$$
\tilde{\mathbf{s}}_{ji} = [\mathbf{s}_{ji}; \mathbb{I}(j < i) \cdot \boldsymbol{\tau}_j] \in \mathbb{R}^{36},
$$

where $\mathbb{I}(j < i)$ is 1 if $j < i$ and 0 otherwise.

**Augmented Edge Feature:**
The augmented edge is then given by:

$$
\tilde{\mathbf{e}}_{ji} = (\tilde{\mathbf{s}}_{ji}, \mathbf{v}_{ji}).
$$

This augmented feature is passed into the decoder layers.

# Step 2. Message Passing in a Decoder Layer

Let the decoder consist of $L$ layers. At layer $l$ (with $l = 0$ corresponding to the initial state from the encoder), each node $i$ has a representation:

$$
\mathbf{x}_i^{(l)} = (\mathbf{s}_i^{(l)}, \mathbf{v}_i^{(l)}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.
$$

**For a Fixed Target Node $i$ (Decoding Step $i$)**
Only node $i$’s representation is updated at this step (using a node mask that selects $i$ across all sampled copies). For each incoming edge $(j, i)$ with $j \in N(i)$:

**Gathering Representations:**

- Target node representation: $\mathbf{x}_i^{(l)} = (\mathbf{s}_i^{(l)}, \mathbf{v}_i^{(l)})$.
- Source node representation: $\mathbf{x}_j^{(l)} = (\mathbf{s}_j^{(l)}, \mathbf{v}_j^{(l)})$.
- Augmented edge representation: $\tilde{\mathbf{e}}_{ji} = (\tilde{\mathbf{s}}_{ji}, \mathbf{v}_{ji})$.

**Concatenation for Message Formation:**
The message function first concatenates the scalar parts:

$$
\mathbf{s}_{\text{concat}} = [\mathbf{s}_i^{(l)}; \tilde{\mathbf{s}}_{ji}; \mathbf{s}_j^{(l)}] \in \mathbb{R}^{128 + 36 + 128 = 292},
$$

and the vector parts:

$$
\mathbf{v}_{\text{concat}} = [\mathbf{v}_i^{(l)}; \mathbf{v}_{ji}; \mathbf{v}_j^{(l)}] \in \mathbb{R}^{16 + 1 + 16 \ (\text{channels}) \times 3}.
$$

In the vector branch, the three parts are concatenated along the channel dimension (so the total number of channels becomes $16 + 1 + 16 = 33$), keeping the spatial dimension $3$ unchanged.

**Computing the Norm of the Vectors:**
The vector concatenation is summarized by its norms:

$$
n(\mathbf{v}_{\text{concat}}) \in \mathbb{R}^{33},
$$

where $n(\cdot)$ denotes the $L_2$ norm computed over the $3$-dimensional vectors for each channel.

**Augmenting Scalar Input:**
The scalar concatenation is further augmented with these norms:

$$
\tilde{\mathbf{s}}_{\text{concat}} = [\mathbf{s}_{\text{concat}}; n(\mathbf{v}_{\text{concat}})] \in \mathbb{R}^{292 + 33 = 325}.
$$

**Message Function via GVP:**
The message for edge $(j, i)$ is then computed using a GVP function:

$$
\mathbf{M}_{ji}^{(l)} = \text{GVP}_{\text{msg}}(\tilde{\mathbf{s}}_{\text{concat}}, \mathbf{v}_{\text{concat}}),
$$

where the GVP function internally applies:

- A linear map $\mathbf{W}_{\text{wh}}$ to the vector part:
  $$
  \mathbf{v}_h = \mathbf{W}_{\text{wh}}(\mathbf{v}_{\text{concat}}) \in \mathbb{R}^{33 \times 3},
  $$
- A computation of the norm $n(\mathbf{v}_h)$ and concatenation with the scalar part, followed by a linear mapping $\mathbf{W}_{\text{ws}}$ to produce the scalar message $\mathbf{s}_{\text{msg}} \in \mathbb{R}^{128}$.
- For the vector output, a linear map $\mathbf{W}_{\text{wv}}$ is applied and then gated by a signal derived from $\mathbf{s}_{\text{msg}}$ using a mapping $\mathbf{W}_{\text{wsv}}$ (with a sigmoid activation):
  $$
  \mathbf{v}_{\text{msg}} = \mathbf{W}_{\text{wv}}}(\mathbf{v}_h) \odot \sigma(\mathbf{W}_{\text{wsv}}}(\mathbf{s}_{\text{msg}})) \in \mathbb{R}^{16 \times 3}.
  $$

Thus, the message is:

$$
\mathbf{M}_{ji}^{(l)} = (\mathbf{s}_{\text{msg}}, \mathbf{v}_{\text{msg}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.
$$

**Aggregation Over Neighbors:**
The messages from all neighbors $j$ (with $j < i$ since only decoded tokens contribute) are aggregated (typically by averaging):

$$
\mathbf{M}_i^{(l)} = \frac{1}{|N(i)|} \sum_{j \in N(i)} \mathbf{M}_{ji}^{(l)}.
$$

Here, the aggregation is applied elementwise to the scalar and vector parts.

**Residual Update & Feedforward Transformation:**
The update for node $i$ in layer $l$ is computed in two stages:

1. **Message Update:**
   $$
   \hat{\mathbf{x}}_i^{(l)} = \mathbf{x}_i^{(l)} + \text{Dropout}(\mathbf{M}_i^{(l)}).
   $$

2. **Layer Normalization:**
   $$
   \mathbf{x}_i^{\prime (l)} = \text{LN}(\hat{\mathbf{x}}_i^{(l)}).
   $$

3. **Feedforward (GVP) Mapping:**
   $$
   \Delta \mathbf{x}_i^{(l)} = \text{GVP}_{\text{ff}}}(\mathbf{x}_i^{\prime (l)}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.
   $$

4. **Final Update with Residual & Dropout:**
   $$
   \mathbf{x}_i^{(l+1)} = \text{LN}(\mathbf{x}_i^{\prime (l)} + \text{Dropout}(\Delta \mathbf{x}_i^{(l)})).
   $$

This completes the update of node $i$ for one decoder layer.

# Step 3. Final Projection and Sampling

After processing through all $L$ decoder layers, the updated node representation for node $i$ is:

$$
\mathbf{x}_i^{(L)} = (\mathbf{s}_i^{(L)}, \mathbf{v}_i^{(L)}).
$$

**Final Output Mapping:**
A final GVP mapping projects $\mathbf{x}_i^{(L)}$ into a 4-dimensional space (for the four nucleotide logits):

$$
\text{logits}_i = \mathbf{W}_{\text{out}}}(\mathbf{x}_i^{(L)}) \in \mathbb{R}^4.
$$

Note that $\mathbf{W}_{\text{out}}$ is defined so that its vector branch is absent (i.e. it outputs only scalars).

**Probability Distribution and Sampling:**

- **Softmax with Temperature:**
  The logits are scaled by a temperature $\tau$ (and possibly adjusted by a logit bias $\mathbf{b}_i$):
  $$
  P_i(k) = \frac{\exp((\text{logits}_i)_k / \tau)}{\sum_{k'=0}^3 \exp((\text{logits}_i)_{k'} / \tau)} \quad \text{for} \quad k \in \{0, 1, 2, 3\}.
  $$

- **Sampling:**
  A nucleotide $t_i$ is sampled from the categorical distribution $P_i$:
  $$
  t_i \sim \text{Categorical}(P_i).
  $$

**Feedback into the Decoder:**
The sampled nucleotide $t_i$ is then embedded:

$$
\boldsymbol{\tau}_i = \mathbf{W}_s(t_i) \in \mathbb{R}^4,
$$

and is used to update the sequence tensor. This new token embedding will be used for augmenting the edge features in subsequent decoding steps (for any edge $(i, k)$ with $k > i$).

# Step 4. Iterative Autoregressive Decoding

The entire process is carried out iteratively for nodes $i = 1, 2, \dots, n$:

1. **For each $i$:**
   - (a) Update the augmented edge features: for every edge $(j, i)$, if $j < i$ then use the newly sampled token $\boldsymbol{\tau}_j$; otherwise, use zero.
   - (b) Run the message passing and feedforward update through the decoder layers as described above.
   - (c) Compute $\text{logits}_i$, sample a nucleotide, embed it, and update the sequence.

**Final Output:**
At the end, the complete sequence is

$$
\text{seq} = \{t_1, t_2, \dots, t_n\} \quad \text{with corresponding logits} \quad \{\text{logits}_1, \dots, \text{logits}_n\}.
$$

This iterative loop ensures that each new prediction is conditioned on all previously decoded tokens and the static context provided by the encoder, while each update is mathematically described by linear maps, nonlinear activations, and gating mechanisms as performed by the GVP functions.

# Summary of the Maps and Spaces

- **Input Node Space:**
  $\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$

- **Edge Augmentation:**
  Original edge scalars $\mathbf{s}_{ji} \in \mathbb{R}^{32}$ are mapped to augmented scalars $\tilde{\mathbf{s}}_{ji} \in \mathbb{R}^{36}$ by concatenating token embeddings.

- **Message Function (GVP):**
  Maps concatenated scalar and vector inputs

  $$
  (\tilde{\mathbf{s}}_{\text{concat}} \in \mathbb{R}^{325}, \mathbf{v}_{\text{concat}} \in \mathbb{R}^{33 \times 3})
  $$

  to output messages in

  $$
  \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.
  $$

- **Residual & Feedforward Updates:**
  Operate within the same space, with LayerNorm and dropout applied.

- **Output Projection:**
  Final mapping $\mathbf{W}_{\text{out}}$ projects from

  $$
  \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} \rightarrow \mathbb{R}^4
  $$

  (only the scalar branch is used) to yield nucleotide logits.

- **Token Embedding Map:**
  $\mathbf{W}_s: \{0, 1, 2, 3\} \rightarrow \mathbb{R}^4$ is used to convert the sampled tokens into a vector space that augments edge features for subsequent decoding.

By combining these mappings and transformations, the decoder updates each node’s representation in an autoregressive fashion, and ultimately outputs a categorical distribution over four nucleotides for every node.

This completes the detailed mathematical explanation of the decoder flow with both the words and the maps involved.