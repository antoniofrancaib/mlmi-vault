Lecture by Pietro Lio

# 1. Motivation and Recap

## 1.1 Why Exploit Geometry?
We have observed that real-world data often carries underlying geometric structure (e.g., permutations, translations, rotations). By designing neural networks to respect these symmetries, we gain:

- **Efficiency**: Fewer parameters and better generalization, because the model does not waste capacity on redundant transformations of the data.
- **Regularity**: Local interactions and symmetry-preserving operations impose a natural inductive bias that better captures the data’s structure.
- **Interpretability**: Equivariant or invariant features align more naturally with geometric or physical invariances (e.g., forces rotating with molecules).

In earlier discussions, we also surveyed the significance of graph data and introduced some traditional methods (network science, spectral graph theory, classical graph generative models). We will soon focus on learning geometric functions on structured domains.

## 1.2 Tools and Visualization
Several software packages (e.g., NetworkX, iGraph, SNAP) aid in analyzing and visualizing networks. In parallel, advanced neural architectures on graphs have emerged, and we also mention that some architectures are discovered by random wiring (e.g., using graph generators like Watts–Strogatz).

# 2. Architectures of Interest
We have a variety of neural network families that can be viewed under the umbrella of Geometric Deep Learning (GDL). They correspond to domains $\Omega$ (sets, grids, spheres, graphs) and symmetry groups $G$ (permutations, translations, rotations, etc.):

| Architecture           | Domain $\Omega$       | Symmetry Group $G$          |
|------------------------|-----------------------|-----------------------------|
| Perceptrons            | Any domain            | No specific group           |
| CNNs                   | Grids (images, 1D/2D) | Translations                |
| Spherical CNNs         | Sphere                | Rotations ($SO(3)$)         |
| Intrinsic CNNs (Mesh)  | Manifold              | Isometries (or gauge sym.)  |
| GNNs                   | Graph                 | Permutations ($\Sigma_n$)   |
| Deep Sets / Transformers | Sets / complete graphs | Permutations ($\Sigma_n$)   |
| LSTMs                  | 1D sequences (time)   | Time-warping                |

In today’s focus (Lecture 3), we will concentrate on learning on:

- **Sets** (the “simplest” domain—no connectivity),
- **Grids** (familiar from standard CNNs), and
- **Spheres** (leading to spherical CNNs).

# 3. Symmetries, Invariance, and Equivariance

## 3.1 Symmetries and Groups
A symmetry is a transformation on the domain that leaves some aspect of the problem unchanged. For instance:

- **Translation** leaves an image’s spatial structure unchanged (it just shifts it).
- **Rotation** leaves a sphere’s geometry unchanged.

Formally, the set of symmetries forms a group $G$. Each group element acts on the domain $\Omega$. In a neural network context, we focus on group actions on signals (feature vectors indexed by $\Omega$), which we typically represent via group representation matrices.

## 3.2 Invariance
A function $f$ is invariant under a group action if applying any group transformation to the input does not change the output. Concretely:

$$
f(g \cdot x) = f(x), \quad \forall g \in G.
$$

**Example**: Molecular atomization energy is invariant to rotations—if we rotate the positions of all atoms in 3D space, the energy remains the same.

![[invariant-representation.png]]

***Remark***: sometimes, for complex objects it is not a good idea to build invariant representations, at least not too soon

![[invariance-not-good.png]]

## 3.3 Equivariance
A function $\Phi$ is equivariant if applying a group transformation $g$ to the input transforms the output in a predictable way consistent with $g$. Formally:

$$
\Phi(g \cdot x) = g \cdot \Phi(x).
$$

**Example**: Force vectors in physics rotate consistently with the molecule—if the molecule is rotated, the forces must rotate in the same way.

![[predict-energy.png]]


While invariance can discard valuable information for more complex downstream tasks, equivariance preserves a structured transformation of that information. Hence, equivariant neural architectures are typically desirable when we need fine-grained or per-element predictions.

# 4. Learning on Sets (Deep Sets)

## 4.1 The Setup
A set can be thought of as a “graph with no edges.” Let $\Omega = \{1, 2, \dots, n\}$ index the elements, with feature vectors $\mathbf{x}_i \in \mathbb{R}^d$. Critically, sets have no canonical ordering—any permutation of the elements represents the same set.

Thus, a function $f$ on a set must be permutation-invariant if it outputs a single value for the set. If we want a function that outputs a value per element (while preserving structure), it must be permutation-equivariant.

**Permutation Invariance**

$$
f(PX) = f(X), \quad \forall P \in \{\text{all permutation matrices}\}.
$$

**Permutation Equivariance**

$$
\Phi(PX) = P \Phi(X), \quad \forall P.
$$

## 4.2 Deep Sets
A classic way to achieve permutation-invariance is Deep Sets (Zaheer et al.). The Deep Sets architecture has the form:

$$
f(\{\mathbf{x}_1, \dots, \mathbf{x}_n\}) = \phi\left(\bigoplus_{i=1}^n \rho(\mathbf{x}_i)\right),
$$

where:

- $\rho$ is a learnable function (e.g., MLP) applied to each element individually.
- $\oplus$ is a permutation-invariant reduction such as sum, mean, or max.
- $\phi$ is another learnable function (e.g., an MLP) operating on the aggregated result.

This combination of an elementwise transformation followed by a global pooling enforces permutation invariance.

### Expressivity and Universality
Under fairly general conditions, any continuous permutation-invariant function over finite sets can be approximated by a Deep Sets model (with sufficiently large hidden dimension). The core idea in the proof is that a suitably chosen elementwise mapping $\rho(\mathbf{x}_i)$ can injectively encode the element, and summation aggregates all elements into a single vector that still uniquely identifies the set. Then $\phi$ can map that sum to the desired output.

**Key Takeaway**: In practice, for set-level tasks (classification, outlier detection, etc.), Deep Sets provide a universal and efficient solution to building permutation-invariant neural networks.

## 4.3 Permutation Equivariance
If we want per-element outputs while respecting permutation equivariance, we typically apply the same function to each element without a global sum:

$$
\mathbf{h}_i = \psi(\mathbf{x}_i), \quad \forall i.
$$

This ensures that permuting the input elements results in the same permutation of the outputs.

# 5. Learning on Grids (CNNs)

## 5.1 Grids and Translations
A grid (e.g., 1D sequence, 2D image) can be seen as a special case of a graph with rigid, local connectivity. Crucially, grids come with a natural notion of translation. For a 1D grid of length $n$ with periodic boundary conditions, the symmetry group is the cyclic group $C_n$. In higher dimensions, we deal with the group of discrete translations $\mathbb{Z}^d$.

**Equivariance Requirement**

If we want a neural network that “treats all locations equally” and gracefully handles shifts in the input (e.g., moving a pattern from left to right), we seek:

$$
\Phi(S\mathbf{x}) = S \Phi(\mathbf{x}),
$$

where $S$ is a shift operator (translation in 1D, or 2D/3D for images/volumes).

## 5.2 Convolution Emerges from Translation Symmetry
To enforce translation equivariance and locality (so each element’s features depend mainly on its near neighbors), we arrive at the convolution operation:

$$
(\mathbf{x} * \mathbf{w})_i = \sum_{j \in \text{neighborhood of } i} \mathbf{w}_{i-j} \mathbf{x}_j.
$$

For a 1D sequence, this reduces to the standard discrete convolution. For images (2D grids), we get the familiar CNN filters. Convolution’s weight-sharing is precisely what enforces the translation symmetry.

### Connection to the Fourier Transform
- Convolution can be viewed as multiplication in the Fourier domain.
- In 1D with periodic boundary conditions, the DFT arises from diagonalizing the shift operator.
- In 2D or higher, a similar viewpoint holds, but the relevant group is $\mathbb{Z}^d$.

**Conclusion**: CNNs are a natural consequence of local translation equivariance on a regular grid. This perspective generalizes to continuous domains $\mathbb{R}^d$ as well, yielding standard continuous convolutions.

# 6. Learning on Spheres (Spherical CNNs)

## 6.1 Spherical Domains
Many important data sources inherently live on a sphere $S^2$. Examples:

- Global topographic data (lat-long representation).
- Cosmic microwave background (CMB) maps in astrophysics.
- 360° vision sensors, VR imagery, or medical imaging with 3D surfaces.

On a sphere, the relevant symmetry group is the 3D rotation group $SO(3)$. We want rotation equivariance: if the sphere is rotated, the representation of that spherical signal should transform accordingly.

## 6.2 Why Standard CNNs Fail Here
Any attempt to “unwrap” a spherical surface onto a flat grid introduces distortions. Unlike planar translations, a rotation of the sphere $SO(3)$ cannot be translated to a simple 2D shift operation. Thus, standard CNNs on unwrapped images would break rotation equivariance.

## 6.3 Spherical Convolution
A spherical convolution over a function $\mathbf{x}(u)$, where $u$ is a point on the sphere, can be defined as an integral over rotations (or points on $S^2$):

- **Domain**: The sphere $S^2$.
- **Symmetry group**: $SO(3)$.
- **Group-based Convolution** (qualitative form):

$$
(\mathbf{x} * \mathbf{w})(g) = \int_{u \in S^2} \mathbf{x}(g^{-1} \cdot u) \mathbf{w}(u) \, d\mu(u),
$$

where $g \in SO(3)$ and $d\mu$ is the appropriate measure on the sphere.

The result is a function defined over the group $SO(3)$, so stacking multiple layers means convolving on $SO(3)$ again—leading to deeper Spherical CNNs.

### Practical Caveat
For many symmetries, especially large or continuous groups, direct group convolution can be computationally heavy. In practice, one often uses careful sampling or expansions in spherical harmonics (the spherical analogue of the Fourier basis) to efficiently implement spherical convolutions (similar to how standard CNNs use the Fast Fourier Transform for certain computations).

# 7. Group Convolution: A General View

## 7.1 The Blueprint
All these special cases—Deep Sets, CNNs, Spherical CNNs—fit a single blueprint:

- **Domain** $\Omega$ (e.g., set elements, grid positions, sphere points).
- **Symmetry Group** $G$ acting on $\Omega$ (permutations, translations, rotations).
- A layer that is $G$-equivariant and local (if desired), typically structured as a “convolution” on $G$.

In general, for a domain $\Omega$ with a group $G$ of symmetries, a group convolution has the form:

$$
(\mathbf{x} * \mathbf{w})(g) = \int_\Omega \mathbf{x}(g^{-1} \cdot u) \mathbf{w}(u) \, d\mu(u),
$$

when $\Omega$ itself is isomorphic to the group $G$. Or in more general settings, one can integrate over $G$ with a suitably defined kernel.

### Locality vs. Global Symmetry
- Local operators (e.g., standard convolution) are prized because they scale well to large inputs and control error propagation.
- Some group convolutions require sampling or expansions that can be global in nature (e.g., all rotations in $SO(3)$), but strategies such as restricting kernel supports or using harmonic expansions can make them more tractable.

## 7.2 Peter–Weyl Theorem (Conceptual Remark)
The Peter–Weyl theorem tells us that for a compact group $G$, the space of square-integrable functions on $G$ decomposes into a direct sum of irreducible representations (irreps). This is the ultimate generalization of the Fourier transform to arbitrary compact groups. Convolutions become multiplication in this “group-Fourier domain.” Hence, ideas like discrete Fourier transforms (DFT) and spherical harmonics are all special cases of group representation theory.

**High-level takeaway**: Each “frequency component” or “irrep” corresponds to a fundamental way the group can act on signals, and building equivariant networks can be seen as operating on these representations.

# 8. Putting It All Together

- **Deep Sets** (Sets; Permutation Group $\Sigma_n$)
  - Achieves permutation invariance by aggregating elementwise features via sum/max/mean.
  - A universal approximator for set functions under mild conditions.

- **CNNs** (Grids; Translation Group)
  - Achieves translation equivariance via localized convolution filters.
  - A mainstay in image, audio, text analysis.

- **Spherical CNNs** (Spheres; Rotation Group $SO(3)$)
  - Achieves rotation equivariance by defining convolutions on the sphere or on the group $SO(3)$.
  - Useful for 360° images, planetary data, astrophysics.

- **General Group Convolution**
  - Unifies the concept for any domain $\Omega$ with a compact group $G$.
  - Can be very powerful but faces challenges for large or continuous groups (cost, discretization).

The unifying principle: To build a Geometric Neural Network that respects a symmetry group $G$, we require that each layer is $G$-equivariant, often implemented via local “convolution-like” operations whose weight-sharing encodes the group’s action.

# 9. Key Takeaways and Intuitions

- **Symmetry as a Constraint**: Imposing symmetry drastically narrows the search space of possible functions, enabling more data-efficient and interpretable learning.
- **Locality**: In most practical domains (images, manifolds, graphs), local receptive fields help control complexity and error propagation (e.g., 3×3 filters in CNNs).
- **Invariance vs. Equivariance**:
  - Invariance can be powerful for global tasks (e.g., set classification, molecule energy).
  - Equivariance is essential for tasks needing output structure (e.g., segmenting each point, predicting per-element forces).
- **Universality**: Deep Sets demonstrate how a simple pointwise-MLP-plus-pooling model can represent any continuous permutation-invariant function (given sufficient dimensionality).
- **Group Convolution**: A fundamental tool generalizing CNNs to other transformation groups. Spherical CNNs illustrate how complicated domains ($S^2$) must factor in the relevant group ($SO(3)$).

