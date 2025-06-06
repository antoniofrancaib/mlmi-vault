Lecture by Petar 
# 1. Introduction & Motivation

Geometric Deep Learning (GDL) unifies many deep architectures by incorporating symmetry and invariance principles into their design. Traditionally, we enforce invariance (or equivariance) with respect to groups of transformations (such as translations, rotations, or permutations). However, many of the transformations we wish to model are not “nice” enough to be fully captured by groups. This calls for a more flexible language—and category theory provides exactly that.

**Key ideas:**

- **Symmetries:** Transformations that leave objects unchanged.
- **Equivariance/Invariance:** Neural network layers that react predictably to input transformations.
- **Beyond Groups:** Not every computation or transformation is invertible or composable in the group sense.
- **Category Theory (CT):** Offers a structural, abstract language to describe computation, generalising groups to categories, functors, and natural transformations.

# 2. Recap on Symmetries and the GDL Blueprint

## 2.1 What Are Symmetries?

A symmetry is any transformation that leaves an object unchanged.

**Example:** The symmetries of a triangle can be generated by 120° rotations and flips. Even though the triangle is “moved” by these transformations, its overall structure remains identical.

## 2.2 The GDL Blueprint

Many popular architectures can be seen as special cases of a general blueprint that aligns the domain of the data (denoted by $\Omega$) with an appropriate symmetry group (denoted by $\mathcal{G}$). For example:

- **CNNs:**
  - **Domain:** 2D grids
  - **Symmetry:** Translation invariance
- **Spherical CNNs:**
  - **Domain:** Sphere
  - **Symmetry:** $\text{SO}(3)$ rotations
- **Graph Neural Networks (GNNs):**
  - **Domain:** Graphs
  - **Symmetry:** Permutations of node order (the symmetric group $\Sigma$!)
- **Transformers:**
  - **Domain:** Complete graphs (attention over all pairs)
  - **Symmetry:** Permutations ($\Sigma$!)

This blueprint demonstrates that by aligning a network’s structure to the symmetries of the domain, one can re-derive many “fan-favourite” architectures.

# 3. Why Move Beyond Symmetry Groups? Enter Category Theory

## 3.1 Limitations of Group Theory

Group theory captures transformations that are invertible, associative, have an identity, and are closed under composition. However, not all transformations we care about in deep learning are invertible or composable in the same way. Thus, while groups give us a rich theory of symmetries, they are too rigid when we need to:

- Model lossy (non-invertible) operations.
- Combine operations that are only partially composable.

## 3.2 Category Theory as a General Framework

Category theory provides a language to capture the structure of computations without getting bogged down by the specifics of the data:

- **Groups → Categories:** A group is just a very special kind of category.
- **Group actions → Functors:** Instead of acting on elements directly, groups (or more general transformations) can be interpreted via mappings between categories.
- **Equivariance → Commutative Diagrams:** Equivariance constraints are naturally expressed as diagrams that must commute.

This abstract viewpoint not only recovers classical invariance/equivariance notions but also allows us to describe much more general types of computation.

# 4. Basic Concepts in Category Theory

## 4.1 Categories, Objects, and Morphisms

A category $\mathcal{C}$ consists of:

- **Objects:** Think of these as “nodes” in the universe of discourse.
- **Morphisms (Arrows):** Maps between objects. For every object $X$ there is an identity morphism $\text{id}_X$.

**Key axioms:**

- **Associativity:** For any composable $f:X \to Y$ and $g:Y \to Z$, and any $h:Z \to W$, we have
  $$h \circ (g \circ f) = (h \circ g) \circ f.$$
- **Identity:** For every $f:X \to Y$,
  $$\text{id}_Y \circ f = f \quad \text{and} \quad f \circ \text{id}_X = f.$$

## 4.2 Standard Examples

- **The Category Set:**
  - **Objects:** Sets
  - **Morphisms:** Functions between sets
- **Terminal and Initial Objects:**
  - The terminal object is one where there is exactly one morphism coming from any other object. In Set, the one-element set (unit) is terminal.
  - The initial object is one with exactly one morphism going out from it; in Set, the empty set serves this role.

## 4.3 Recovering Groups and Monoids

- **Groups as Single-Object Categories:**
  A group can be viewed as a category with one object where every morphism (transformation) is invertible.
- **Monoids as Single-Object Categories:**
  By dropping the requirement for inverses, we recover a monoid—a category with one object where all morphisms compose, but not necessarily invert.

# 5. Functors and Natural Transformations

## 5.1 Functors

A functor $F: \mathcal{C} \to \mathcal{D}$ is a structure-preserving map between categories. It assigns:

- To every object $X \in \mathcal{C}$, an object $F(X) \in \mathcal{D}$.
- To every morphism $f:X \to Y$ in $\mathcal{C}$, a morphism $F(f):F(X) \to F(Y)$ in $\mathcal{D}$.

**Preservation conditions:**

- **Identities:** $F(\text{id}_X) = \text{id}_{F(X)}$
- **Composition:** $F(g \circ f) = F(g) \circ F(f)$

If $F$ maps a category to itself, it is called an endofunctor.

## 5.2 Natural Transformations

Given two functors $F, G: \mathcal{C} \to \mathcal{D}$, a natural transformation $\tau: F \Rightarrow G$ assigns to every object $X$ in $\mathcal{C}$ a morphism $\tau_X: F(X) \to G(X)$ in $\mathcal{D}$ such that for every morphism $f:X \to Y$ in $\mathcal{C}$, the following diagram commutes:

$$
\begin{array}{ccc}
F(X) & \xrightarrow{F(f)} & F(Y) \\
\downarrow \tau_X & & \downarrow \tau_Y \\
G(X) & \xrightarrow{G(f)} & G(Y)
\end{array}
$$

This guarantees that the “way” $F$ and $G$ act on morphisms is compatible with the components of $\tau$.

# 6. Monads: Encoding Computation and Group Actions

## 6.1 What Is a Monad?

A monad in a category $\mathcal{C}$ is a tuple $(T, \eta, \mu)$ where:

- $T: \mathcal{C} \to \mathcal{C}$ is an endofunctor.
- $\eta: \text{id}_{\mathcal{C}} \Rightarrow T$ is the unit (often called “return” in functional programming).
- $\mu: T \circ T \Rightarrow T$ is the multiplication (or “join”), which “flattens” a nested computation.

These must satisfy coherence (associativity and unitality) diagrams analogous to the monoid axioms. In fact, one can say:

*A monad is just a monoid in the category of endofunctors of $\mathcal{C}$.*

## 6.2 Example: The Power Set Monad

- **Endofunctor:** $T$ maps a set $A$ to its power set $P(A)$.
- **Unit:** The unit $\eta_A: A \to P(A)$ wraps an element $a$ into a singleton $\{a\}$.
- **Join:** The multiplication $\mu_A: P(P(A)) \to P(A)$ takes a set of subsets and returns their union.

## 6.3 Group Actions as a Monad

For a group $G$, we define the group action monad on Set:

- **Endofunctor:** $G \times -$ maps a set $A$ to the product $G \times A$. For a function $f:A \to B$, $G \times f: G \times A \to G \times B$ is defined as $(g, a) \mapsto (g, f(a))$.
- **Unit ($\eta$):** The natural transformation $\eta_A: A \to G \times A$ maps $a$ to $(e, a)$, where $e$ is the identity in $G$.
- **Join ($\mu$):** The natural transformation $\mu_A: G \times (G \times A) \to G \times A$ “multiplies” the group elements: $(g, (h, a)) \mapsto (gh, a)$.

This monad neatly packages how group actions compose without “executing” the action yet.

# 7. Algebras for Monads & Equivariant Neural Network Layers

## 7.1 Monad Algebras

A monad algebra for $(T, \eta, \mu)$ is a pair $(A, \alpha)$ where:

- $A$ is a carrier object (e.g., a feature space).
- $\alpha: T A \to A$ is a structure map satisfying the following conditions:
  - **Unit law:** $\alpha \circ \eta_A = \text{id}_A$
  - **Associativity law:** $\alpha \circ T \alpha = \alpha \circ \mu_A$

These conditions ensure that the “contained computation” (as specified by the monad) is executed consistently.

## 7.2 Group Actions as Algebras

For the group action monad $G \times -$:

- A group action algebra is a pair $(A, \triangleright)$ where $A$ is the carrier (e.g., the space of pixel values) and
  $$\triangleright: G \times A \to A$$
  is the action that “executes” the transformation. For instance, in the case of image translations, the algebra might be defined on $\mathbb{R}^{H \times W}$ with the structure map performing a circular shift.

## 7.3 Monad Algebra Homomorphisms and Equivariance

A monad algebra homomorphism between algebras $(A, \alpha)$ and $(B, \beta)$ is a morphism $f: A \to B$ that makes the following diagram commute:

$$
\begin{array}{ccc}
T A & \xrightarrow{T f} & T B \\
\downarrow \alpha & & \downarrow \beta \\
A & \xrightarrow{f} & B
\end{array}
$$

In words, the map $f$ “commutes with the action” of $T$:

$$f(\alpha(x)) = \beta(T f(x)) \quad \forall x \in T A.$$

For the group action monad, this condition directly implies the familiar equivariance constraint:

$$f(g \triangleright x) = g \triangleright f(x) \quad \text{for all } g \in G, \, x \in A.$$

**Interpretation:**
An equivariant neural network layer is exactly a homomorphism between the algebras corresponding to the input and output spaces. When working in a concrete category (say, vector spaces), resolving this constraint often leads to specific weight-sharing schemes—for example, circulant matrices in translation equivariant CNNs.

# 8. Endofunctor Algebras & Coalgebras: Beyond Group-Equivariance

## 8.1 Endofunctor Algebras

Notice that in defining monad algebra homomorphisms we did not always use the full strength of the monad axioms. In fact, one may generalise to an arbitrary endofunctor $T: \mathcal{C} \to \mathcal{C}$ by considering:

- An endofunctor algebra is a pair $(A, \alpha)$ with $\alpha: T A \to A$.
- There is no requirement for “unit” or “associativity” constraints here.

**Examples:**

- **Lists:**
  Consider the endofunctor $1 + A \times -$ on Set. The familiar list type (with constructors Nil and Cons) forms an algebra for this functor.
- **Binary Trees:**
  An endofunctor $A + (- \times -)$ can describe binary trees, where each tree is either a leaf with a label or an internal node combining two subtrees.

## 8.2 Homomorphisms and Folds

A homomorphism between two endofunctor algebras (e.g., a fold over a list) must make the analogous diagram commute. For lists, this means:

- $f(\text{Nil})$ must equal a specified “base case”.
- $f(\text{Cons}(a, xs))$ must equal the operation applied to $a$ and $f(xs)$.

These conditions are analogous to the equivariance constraints in group actions but apply to structures that are not necessarily invertible (e.g., concatenation).

## 8.3 Endofunctor Coalgebras

Coalgebras are the dual notion. For an endofunctor $T$, a coalgebra is a pair $(A, \gamma)$ where:

$$\gamma: A \to T A$$

Coalgebras are well suited to represent potentially infinite computations, such as automata.

**Example: Mealy Machines**

A Mealy machine, which maps an input to an output and a new state, can be viewed as a coalgebra for the functor $I \to O \times -$ on Set:

- The structure map $\text{next}: \text{Mealy} \to (I \to O \times \text{Mealy})$ defines the machine’s evolution.
- Homomorphisms between coalgebras enforce “unfold” constraints that guarantee the preservation of the dynamic behavior.

# 9. Latest Developments: Map/Filter Invariants

Modern deep networks often discover specialized subfunctions (e.g., positional or semantic heads) on their own. To capture such “by design” behaviors:

- **Map invariance** implies that the function’s behavior is independent of the individual data elements.
- **Filter invariance** offers a form of length generalization—if some elements are removed, the function still “makes sense.”

For functions $f: [a] \to [b]$:

- **NFIs (Non-Filter Invariant functions):**
  A function is NFI if it reproduces each element in the input with a fixed proportionality (if an element appears $n$ times, it appears $k \times n$ times in the output), essentially acting as a “simplicial permutation.”

This line of work is in its early stages but promises a categorical way to encode and exploit regularities discovered by neural networks.

# 10. Beyond Layers: 2-Categories and Parameterised Architectures

## 10.1 The Limitation of Linear Layers

While our categorical treatment (via monads and endofunctor algebras) successfully recovers classical equivariant layers, it has two main limitations:

- **Linearity:** When mapping our diagrams into categories like Vect, we are limited to linear operations.
- **Parameters:** Current frameworks do not explicitly capture network parameters, which are crucial for forward/backward passes and overall architecture design.

## 10.2 Enter 2-Categories

A 2-category extends the usual notion of a category by including:

- **Objects**
- **1-Morphisms (Arrows)**
- **2-Morphisms:** Morphisms between morphisms (think “transformations of transformations”).

The classical category Cat (with categories as objects, functors as 1-morphisms, and natural transformations as 2-morphisms) is an example of a 2-category.

## 10.3 The 2-Category Para

To explicitly model parameters in neural networks, researchers introduced the 2-category Para:

- **Objects:** Sets (or spaces).
- **Morphisms:** Parametric maps—each morphism $f: A \to B$ is equipped with parameters (often represented as a function $f: A \times P \to B$).
- **2-Morphisms:** These encode reparametrisations (for instance, to model weight tying or shared parameters).

**Graphical insight:**
When drawing diagrams in Para, the parameters are shown explicitly. Composition of morphisms naturally chains both the computations and their parameters. This makes it possible to express entire architectures—including constraints such as weight tying—using commutative diagrams in a 2-categorical setting.

## 10.4 2-Categorical Diagrams and Commutativity

In 2-categories, diagrams can commute in several senses:

- **Strictly Commutative:** Equal on the nose.
- **Pseudocommutative:** Commutative up to a specified isomorphism.
- **Lax/Colax Commutative:** Commutativity holds up to a 2-morphism that need not be invertible.

Even by focusing on a subset (such as lax algebras), one can recover various weight tying strategies used in recurrent and recursive neural networks.

# 11. Concluding Remarks and Future Directions

**Unifying Perspective:**
We have seen that category theory provides a unified and abstract framework that recovers classical group equivariance and extends to more general computation and structured data types.

**From Layers to Architectures:**
While monad (and endofunctor) algebras yield linear, equivariant layers, the introduction of 2-categories (and specifically Para) promises a way to incorporate network parameters and entire architectures into the categorical framework.

**Research Frontiers:**

- **Resolving Constraints:** Translating the abstract diagrammatic constraints into concrete architectures (e.g., deriving circulant weight matrices in CNNs).
- **Beyond Invariance:** Studying map/filter invariants to understand emergent behaviors in deep networks.
- **2-Categorical Methods:** Further developing the theory to design architectures with explicit parameter sharing and reparametrization.

**Final Thought:**
The categorical deep learning picture is still evolving. It bridges disparate areas of mathematics—from group theory and algebra to 2-categories and coalgebras—allowing us to design neural networks that are “by design” robust and aligned with the symmetries (and more general invariances) of the data. For further reading, the community resource nLab is highly recommended, though be prepared to navigate some “rabbit holes”!