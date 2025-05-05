# Index
- [18-Factor-Graphs](#18-Factor-Graphs)
- [19-Applying-Message-Passing-to-TrueSkill™](#19-Applying-Message-Passing-to-TrueSkill™)
- [20-Moment-Matching-Approximation](#20-Moment-Matching-Approximation)
 
# 18-Factor-Graphs

Probabilistic ***graphical models*** provide a powerful framework for representing complex distributions and performing efficient inference. Among these models, factor graphs are particularly useful for representing the factorization of probability distributions and facilitating efficient computation of marginal and conditional probabilities through message passing algorithms.

*Definition*: A **factor graph** is a bipartite graph $G = (\mathcal{V}, \mathcal{F}, \mathcal{E})$, where:

- $\mathcal{V} = \{ X_1, \dots, X_n \}$ is a set of variable nodes.
- $\mathcal{F} = \{ f_1, \dots, f_m \}$ is a set of factor nodes.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{F}$ connects $X_i \in \mathcal{V}$ to $f_j \in \mathcal{F}$ if $X_i$ is an argument of $f_j$.

The factor graph represents a function $F(X_1, \dots, X_n)$ as:
$$F(X_1, \dots, X_n) = \prod_{f_j \in \mathcal{F}} f_j(X_{C_j}),$$
where $X_{C_j}$ is the subset of variables connected to $f_j$. For probabilistic models:
$$P(X_1, \dots, X_n) = \frac{1}{Z} \prod_{f_j \in \mathcal{F}} f_j(X_{C_j}),$$
where $Z$ is the normalizing constant.

--- 

### What are Factor Graphs?
A factor graph is a bipartite graphical model that represents the factorization of a function, typically a joint probability distribution. It consists of two types of nodes:
1. **Variable Nodes**: Represent the variables in the function.
2. **Factor Nodes**: Represent local functions (factors) that depend on a subset of variables.

Edges connect factor nodes to variable nodes if the factor depends on that variable.
#### Purpose of Factor Graphs
1. **Visualization**: Provides a clear graphical representation of the dependencies between variables and factors.
2. **Computation**: Facilitates efficient computation of marginal and conditional distributions through message passing algorithms.
3. **Generalization**: Factor graphs generalize other graphical models like Bayesian networks (directed graphs) and Markov networks (undirected graphs).
#### Example of a Factor Graph
Consider a joint probability distribution that factors as:

$$p(v,w,x,y,z)=f_1(v,w)\cdot f_2(w,x)\cdot f_3(x,y)\cdot f_4(x,z)$$

The corresponding factor graph has:
- **Variable Nodes**: $v, w, x, y, z$
- **Factor Nodes**: $f_1, f_2, f_3, f_4$
- **Edges**: Connect factors to the variables they depend on.

![text](fact-grph.png)


#### Questions We Can Answer Using Factor Graphs
1. **Marginal Distributions**: What is $p(w)$?
2. **Conditional Distributions**: What is $p(w\mid y)$?
3. **Efficient Computation**: How can we compute these distributions efficiently using the structure of the factor graph?

### Efficient Computation with Factor Graphs

#### Challenges with Naïve Computation
Computing marginals directly can be computationally expensive due to the high dimensionality and combinatorial explosion of possible variable configurations.

For example, computing $p(w)$ naively involves:

$$p(w)=\sum_v \sum_x \sum_y \sum_z f_1(v,w)f_2(w,x)f_3(x,y)f_4(x,z)$$

If each variable can take $K$ values, the computational complexity is $O(K^5)$, which becomes infeasible as $K$ and the number of variables grow.

#### Exploiting the Factor Graph Structure
The key to efficient computation lies in exploiting the distributive property of multiplication over addition and the separability of the factor graph:
1. **Distributive Property**: Allows us to rearrange sums and products to reduce computations.
2. **Tree Structure**: In tree-structured graphs (graphs without loops), each node separates the graph into disjoint subgraphs, enabling recursive computations.

### Step-by-Step Computation
1. **Group Terms**: Start by grouping terms associated with each variable or factor.
2. **Local Computations**: Compute messages locally at each node, passing summarized information to neighboring nodes.
3. **Recursive Summations**: Use the distributive property to perform sums over variables in a recursive manner, reducing the overall computational complexity.

#### Example
Compute $p(w)$ by rearranging terms:
$$p(w) = \sum_v f_1(v, w) \left( \sum_{x, y, z} f_2(w, x) \cdot f_3(x, y) \cdot f_4(x, z) \right)$$

Finally, it factorizes as:

$$p(w) = \left( \sum_v f_1(v, w) \right) \cdot \left( \sum_{x, y, z} f_2(w, x) \cdot f_3(x, y) \cdot f_4(x, z) \right)$$

The computation can be structured hierarchically, summing over $z$, then $y$, followed by $x$, and finally $v$, reflecting the factorization implied by the factor graph structure.

$$p(w)=\left(\sum_v f_1(v,w)\right)\cdot\left(\sum_x f_2(w,x)\cdot\left(\sum_y f_3(x,y)\cdot\sum_z f_4(x,z)\right)\right)$$

The complexity reduces from $O(K^5)$ to $O(K^4)$ or even $O(K^2)$ with further optimizations.

![[fact-disc.png]]

### The Sum-Product Algorithm

The sum-product algorithm is a message passing algorithm used to compute marginal distributions in factor graphs efficiently. It is also known as belief propagation or factor-graph propagation.

#### Key Steps in the Sum-Product Algorithm
1. **Initialization**: Set initial messages, typically starting with uniform distributions or prior information.
2. **Message Passing**: Iteratively compute messages from factors to variables and from variables to factors until convergence.
3. **Marginal Computation**: Once messages have stabilized, compute the marginal distributions by combining incoming messages at each variable node.

##### Messages from Factors to Variables
For a factor $f$ connected to variables $x_1, x_2, \ldots, x_n$, the message from factor $f$ to variable $x_i$ is:

$$m_{f \to t_1}(t_1) = \sum_{t_2} \sum_{t_3} \cdots \sum_{t_n} f(t_1, t_2, \ldots, t_n) \prod_{i \neq 1} m_{t_i \to f}(t_i)$$

**Interpretation**: Sum over all variables except $x_i$, multiplying the factor $f$ with messages from neighboring variables.

#### Messages from Variables to Factors
For a variable $x$ connected to factors $f_1, f_2, \ldots, f_k$, the message from variable $x$ to factor $f$ is:

$$m_{t \to f}(t) = \prod_{f_j \in F_t \setminus \{f\}} m_{f_j \to t}(t) = \frac{p(t)}{m_{f \to t}(t)}$$

**Interpretation**: Multiply all incoming messages from neighboring factors except the recipient factor $f$.

### Computing Marginals
The marginal distribution for a variable $x$ is:

$$p(x)=\prod_{f\in ne(x)} m_{f\to x}(x)$$

**Interpretation**: Multiply all incoming messages from factors connected to $x$.

---

## 19-Applying-Message-Passing-to-TrueSkill™ 
[-](#index)
### The TrueSkill™ Model
TrueSkill™ is a Bayesian rating system that models player skills and predicts match outcomes. It consists of:
1. **Player Skills ($w_i$)**: Random variables representing the skill levels of players.
2. **Performance Differences ($t_g$)**: Observed performance differences in games.
3. **Game Outcomes ($y_g$)**: Observed outcomes of games, where $y_g=+1$ if Player $I_g$ wins and $y_g=-1$ if Player $J_g$ wins.

### TrueSkill™ Factor Graph
The factor graph for TrueSkill™ includes:
1. **Prior Factors**: Representing prior beliefs about player skills.
   $$f_i(w_i)=N(w_i;\mu_0,\sigma_0^2)$$

2. **Game Factors**: Modeling the relationship between skills and performance differences.
   $$h_g(w_{I_g},w_{J_g},t_g)=N(t_g;w_{I_g}-w_{J_g},\sigma_n^2)$$

3. **Outcome Factors**: Incorporating observed game outcomes.
   $$k_g(t_g,y_g)=\delta(y_g-\text{sign}(t_g))$$

![[fact-flow.png]]

### Goals
1. **Compute Marginals**: Determine the marginal distributions of player skills $p(w_i)$.
2. **Update Beliefs**: Incorporate observed game outcomes to update beliefs about player skills.


### Addressing Challenges in TrueSkill™

#### Handling Loops in the Factor Graph
1. **Approximate Inference**: When the factor graph is not a tree, we can still apply message passing algorithms approximately.
2. **Iterative Message Passing**: Messages are passed iteratively until convergence, similar to belief propagation in loopy graphs.

#### Approximation of Non-Standard Messages
1. **Expectation Propagation (EP)**: An approximation method that replaces complex messages with approximations (e.g., Gaussian distributions) by matching moments.
2. **Moment Matching**: Adjusting the parameters of the approximating distribution so that its first and second moments match those of the true distribution.

### Expectation Propagation (EP) in TrueSkill™

EP is an iterative algorithm used to approximate complex probability distributions by simpler ones (e.g., Gaussians). It involves:

- **Approximate Factors**: Replace intractable factors with approximate ones that are tractable (e.g., Gaussian approximations).
- **Moment Matching**: Ensure that the approximate distribution matches certain moments (mean and variance) of the true distribution.
- **Iterative Updates**: Repeat the process until convergence.

#### Steps in EP for TrueSkill™
1. **Initialize Messages**: Start with initial messages, typically set to uniform or prior distributions.
2. **Update Skill Marginals**: Compute the marginal distributions for skills using incoming messages.
3. **Compute Messages from Skills to Games**: Use the current skill marginals to send messages to game factors.
4. **Compute Messages from Games to Performances**: Combine messages to compute the distribution of performance differences.
5. **Approximate Pervvformance Marginals**: Use moment matching to approximate the true distribution of performance differences with a Gaussian.
6. **Compute Messages from Performances to Games**: Update messages based on the approximate performance marginals.
7. **Compute Messages from Games to Skills**: Update skill messages based on incoming messages from performances.
8. **Iterate**: Repeat the process until messages and marginals converge.

![[fact-map.png]]
### Detailed Steps: 

##### Step 0: Initialization
**Objective**: Initialize the messages from game factors to skill variables.

###### Equations:
For each game $g$:
$$m_{h_g \to w_{I_g}}^{\tau=0}(w_{I_g}) = 1$$
$$m_{h_g \to w_{J_g}}^{\tau=0}(w_{J_g}) = 1$$

This means that initially, the messages from the game factors to the player skills are uniform distributions. We have no information from the games yet.

**Interpretation of Messages Equal to 1:**
- **Neutral Influence:** A message equal to 1 is a neutral element in multiplication. It implies that the factor sending the message is not adding any new information or influence to the variable.
- **Marginal Equals Prior:** Because the messages from the games are neutral (equal to 1), the marginal distributions of the skill variables are determined solely by their prior distributions.

---
##### Step 1: Compute Marginal Skills
**Objective**: Compute the marginal distribution for each player's skill based on the prior and incoming messages.

###### Equation:
For player $i$:
$$q^{\tau}(w_i) = f_i(w_i) \prod_g m_{h_g \to w_i}^{\tau}(w_i)$$

- $f_i(w_i)$: Prior belief about player $i$'s skill.
- $m_{h_g \to w_i}^{\tau}(w_i)$: Message from game $g$ to player $i$ at iteration $\tau$.

Since both the prior and the messages are Gaussians (or initialized as uniform, which is a special case of a Gaussian with infinite variance), the marginal $q^{\tau}(w_i)$ is also Gaussian.

**Interpretation**:  
We update our belief about each player's skill by combining the prior belief with the information received from all games involving that player.

--- 
##### Step 2: Compute Skill-to-Game Messages
**Objective**: Calculate the messages from player skills to game factors, reflecting the influence of the player's current skill estimate on the game.

###### Equations:
For each game $g$ involving player $i$:
$$m_{w_i \to h_g}^{\tau}(w_i) = \frac{q^{\tau}(w_i)}{m_{h_g \to w_i}^{\tau}(w_i)}$$

- $q^{\tau}(w_i)$: The marginal skill distribution computed in Step 1.
- $m_{h_g \to w_i}^{\tau}(w_i)$: The incoming message from game $g$ to player $i$.

**Interpretation**:  
This calculation effectively removes the influence of game $g$ from the marginal skill to isolate the impact of player $i$'s skill on that game.

**Purpose**: To compute the message to $h_g$​, we need the belief about $w_i$​ **without** considering the information from $h_g$​, to avoid double-counting.

**Isolating the Effect of Other Factors**: This division ensures that the message $m_{w_i \to h_g}^\tau(w_i)$ reflects only the influence of the prior and other games on $w_i$, excluding $h_g$.

###### Avoiding Loops in Belief Updates
- **Preventing Circular Reasoning**
If we didn't remove $m_{h_g \to w_i}^\tau(w_i)$, we'd be using information from $h_g$ to inform $h_g$ itself, which leads to inconsistencies.

**Ensuring Consistent Updates**
The division maintains the integrity of the message passing algorithm by ensuring that each update is based on independent information.


---
##### Step 3: Compute Game-to-Performance Messages
**Objective**: Compute the message from the game factor to the performance difference variable $t_g$.

###### Equation:
$$m_{h_g \to t_g}^{\tau}(t_g) = \int \int h_g(w_{I_g}, w_{J_g}, t_g) m_{w_{I_g} \to h_g}^{\tau}(w_{I_g}) m_{w_{J_g} \to h_g}^{\tau}(w_{J_g}) \, dw_{I_g} \, dw_{J_g}$$

**Explanation**:
- We integrate over the skills $w_{I_g}$ and $w_{J_g}$ of the two players in game $g$.

- $h_g(w_{I_g}, w_{J_g}, t_g)$ is the game factor represents the probabilistic relationship between the players' skills $w_{I_g}$, $w_{J_g}$, and the performance difference $t_g$. In the TrueSkill™ model, this factor is defined as: $h_g(w_{I_g}, w_{J_g}, t_g) = N(t_g; w_{I_g} - w_{J_g}, \sigma_n^2).$

- $m_{w_{I_g} \to h_g}^{\tau}(w_{I_g})$ and $m_{w_{J_g} \to h_g}^{\tau}(w_{J_g})$ are the messages from the player skills to the game factor. , representing our current beliefs about the players' skills $w_{I_g}$ and $w_{J_g}$, excluding the influence from the current game $g$. They are Gaussian distributions derived from the marginal beliefs $q^\tau(w_i)$, adjusted to avoid double-counting information from $h_g$.

**Result**:  
The resulting message $m_{h_g \to t_g}^{\tau}(t_g)$ is a Gaussian distribution over the performance difference $t_g$.

**Interpretation**:  
This message represents our belief about the performance difference $t_g$ based on the current estimates of the players' skills.

**Purpose of Integration**:
- **Aggregate Influence**: We aggregate the influence of all possible skill levels of the players on the performance difference $t_g$.
- **Compute Expected Distribution**: By integrating, we obtain the expected distribution of $t_g$ given our current knowledge.

###### What Does This Message Represent?

**Belief About $t_g$**:

The message $m_{h_g \to t_g}^\tau(t_g)$ encapsulates our updated belief about the performance difference $t_g$, considering:

1. **Current Estimates of Players' Skills**: 
   From the messages $m_{w_{I_g} \to h_g}^\tau(w_{I_g})$ and $m_{w_{J_g} \to h_g}^\tau(w_{J_g})$.

2. **Game Factor Relationship**: 
   How these skills relate to the performance difference via $h_g(w_{I_g}, w_{J_g}, t_g)$.

**Intuitive Interpretation**:

1. **Predicting Performance Difference**: 
   We are predicting the distribution of the performance difference $t_g$ based on our current knowledge of the players' skills.

2. **Combining Uncertainties**: 
   The integration accounts for the uncertainties in the players' skills and propagates this uncertainty to the performance difference.


--- 

##### Step 4: Compute Marginal Performances (Approximation)
**Objective**: Update the marginal distribution of the performance difference $t_g$ by incorporating the observed game outcome.

###### Equation:
$$q^{\tau+1}(t_g) = \text{Approx}\big(m_{h_g \to t_g}^{\tau}(t_g) \cdot k_g(t_g, y_g)\big)$$

**Explanation**:
- $m_{h_g \to t_g}^{\tau}(t_g)$: Message from the game factor to $t_g$.
- $k_g(t_g, y_g)$: The outcome factor, enforcing that the sign of $t_g$ matches the observed outcome $y_g$.

The product $m_{h_g \to t_g}^{\tau}(t_g) \cdot k_g(t_g, y_g)$ is not Gaussian because $k_g$ introduces a nonlinearity (it's a step function).

**Approximation**:  
We approximate this product with a Gaussian distribution by matching the first two moments (mean and variance). This is where EP comes into play.

**Interpretation**:  
We're updating our belief about the performance difference $t_g$ considering the actual game result, while keeping the distribution Gaussian for tractability.


---
##### Step 5: Compute Performance-to-Game Messages
**Objective**: Update the message from the performance difference $t_g$ back to the game factor $h_g$.

###### Equation:
$$m_{t_g \to h_g}^{\tau+1}(t_g) = \frac{q^{\tau+1}(t_g)}{m_{h_g \to t_g}^{\tau}(t_g)}$$

**Explanation**:
- $q^{\tau+1}(t_g)$: The approximated marginal from Step 4.
- $m_{h_g \to t_g}^{\tau}(t_g)$: The message computed in Step 3.

By dividing the updated marginal by the incoming message, we obtain the outgoing message from $t_g$ to $h_g$.

**Interpretation**:  
This message captures the updated information about the performance difference that will be sent back to the game factor.


---
##### Step 6: Compute Game-to-Skill Messages
**Objective**: Update the messages from the game factor $h_g$ back to the player skills $w_{I_g}$ and $w_{J_g}$.

###### Equations:
For Player $I_g$:
$$m_{h_g \to w_{I_g}}^{\tau+1}(w_{I_g}) = \int h_g(w_{I_g}, w_{J_g}, t_g) m_{t_g \to h_g}^{\tau+1}(t_g) m_{w_{J_g} \to h_g}^{\tau}(w_{J_g}) \, dt_g \, dw_{J_g}$$

For Player $J_g$:
$$m_{h_g \to w_{J_g}}^{\tau+1}(w_{J_g}) = \int h_g(w_{I_g}, w_{J_g}, t_g) m_{t_g \to h_g}^{\tau+1}(t_g) m_{w_{I_g} \to h_g}^{\tau}(w_{I_g}) \, dt_g \, dw_{I_g}$$

**Explanation**:  
We integrate over the performance difference $t_g$ and the opponent's skill.

**Result**:  
These integrals result in Gaussian messages back to the player skills.

**Interpretation**:  
These messages update our beliefs about each player's skill based on the updated performance difference and the opponent's skill estimate.

--- 
##### Step 7: Iterate Until Convergence
We repeat Steps 1 to 6, incrementing $\tau$ each time, until the messages and marginals stabilize. This iterative process refines our estimates of the player skills with each pass.


### Detailed Computations Using Natural Parameters

To make the computations more tractable, we can represent Gaussians using their natural parameters:

- **Precision ($r$)**: The inverse of the variance ($r = \nu^{-1}$).
- **Natural Mean ($\lambda$)**: The product of the precision and the mean ($\lambda = r \mu$).

This representation simplifies the multiplication and division of Gaussian messages.

### Computations in Terms of Natural Parameters

**Why Use Natural Parameters?**

- **Simplifies Calculations**: Operations like multiplication and division of Gaussians become addition and subtraction in natural parameter space.
- **Facilitates Message Passing**: In EP, messages often involve products and quotients of Gaussian functions. Using natural parameters makes these operations linear and thus computationally efficient.

#### Initialization:
The incoming messages $m_{h_g \to w_i}^{\tau=0}$ are initialized with zero precision and zero natural mean:
$$r_{h_g \to w_i}^{\tau=0} = 0, \; \lambda_{h_g \to w_i}^{\tau=0} = 0$$
**Interpretation**:

- **Zero Precision**: Corresponds to infinite variance (total uncertainty), making the message effectively uniform.
- **Zero Natural Mean**: Since $\lambda = r\mu$, zero precision implies $\lambda = 0$.

#### Step 1: Compute Marginal Skills
For player $i$:
$$r_i^{\tau} = r_0 + \sum_g r_{h_g \to w_i}^{\tau}$$
$$\lambda_i^{\tau} = \lambda_0 + \sum_g \lambda_{h_g \to w_i}^{\tau}$$

- $r_0$ and $\lambda_0$: Precision and natural mean from the prior $f_i(w_i)$.
- The sum is over all games involving player $i$.

**Interpretation**: 

- **Marginal Precision**: The marginal precision $r_i^\tau$ and natural mean $\lambda_i^\tau$ accumulate the information from the prior and all incoming messages from game factors.

- **Linear Update**: Due to the properties of natural parameters, the updates are additive, simplifying calculations.

#### Step 2: Compute Skill-to-Game Messages
For game $g$ involving player $i$:
$$r_{w_i \to h_g}^{\tau} = r_i^{\tau} - r_{h_g \to w_i}^{\tau}$$
$$\lambda_{w_i \to h_g}^{\tau} = \lambda_i^{\tau} - \lambda_{h_g \to w_i}^{\tau}$$

**Interpretation**:

- **Isolating Information**: By subtracting the message from $h_g$ to $w_i$, we remove the influence of game $g$ from the player's marginal belief.

- **Preventing Double-Counting**: This ensures that when computing the message to $h_g$, we're not reusing information from $h_g$ itself.


#### Step 3: Compute Game-to-Performance Messages
Compute the variance ($\nu$) and mean ($\mu$):
$$\nu_{h_g \to t_g}^{\tau} = \sigma_n^2 + \nu_{w_{I_g} \to h_g}^{\tau} + \nu_{w_{J_g} \to h_g}^{\tau}$$
$$\mu_{h_g \to t_g}^{\tau} = \mu_{w_{I_g} \to h_g}^{\tau} - \mu_{w_{J_g} \to h_g}^{\tau}$$

- $\sigma_n^2$: Performance noise variance.

- $\nu_{w_i \to h_g}^\tau$ and $\mu_{w_i \to h_g}^\tau$: Variance and mean from the messages computed in Step 2.

**Conversion from Natural to Standard Parameters**:

- **Variance**: $\nu = r^{-1}$.
- **Mean**: $\mu = \lambda / r$.

**Interpretation**:

- **Combining Uncertainties**: The variance of $t_g$​ includes the uncertainties of both players' skills and the performance noise.
- **Expected Performance Difference**: The mean of $t_g$ is the difference between the expected skills of the two players.  


### Step 4: Approximate Marginal Performances

**Objective**: Incorporate the observed game outcome $y_g$ into the marginal of $t_g$ while maintaining tractability.

**Challenge**: The exact product $m_{h_g \to t_g}^\tau(t_g) \cdot k_g(t_g, y_g)$ results in a truncated Gaussian (due to $k_g$), which is not Gaussian.

**Solution: Moment Matching**
Approximate the true distribution with a Gaussian by matching the first two moments (mean and variance).

##### Formulas

**Standardized Mean**:
$$
\hat{\mu}_g = \frac{y_g \mu_{h_g \to t_g}^\tau}{\nu_{h_g \to t_g}^\tau}
$$
**Note**: $y_g \in \{+1, -1\}$. Multiplying $\mu_{h_g \to t_g}^\tau$ by $y_g$ standardizes it to consider the positive side.

**Correction Factors**:
$$
\Psi(\hat{\mu}_g) = \frac{\phi(\hat{\mu}_g)}{\Phi(\hat{\mu}_g)}
$$
$$
\Lambda(\hat{\mu}_g) = \Psi(\hat{\mu}_g)(\Psi(\hat{\mu}_g) + \hat{\mu}_g)
$$
- $\phi(x)$: Standard normal PDF.
- $\Phi(x)$: Standard normal CDF.

**Updated Variance:**
$$
\tilde{\nu}_g^{\tau+1} = \nu_{h_g \to t_g}^\tau \left(1 - \Lambda(\hat{\mu}_g)\right)
$$

Updated Mean:
$$
\tilde{\mu}_g^{\tau+1} = \mu_{h_g \to t_g}^\tau + y_g \nu_{h_g \to t_g}^\tau \Psi(\hat{\mu}_g)
$$

**Interpretation**:

- **Mean Adjustment**: The mean shifts towards the observed outcome, accounting for the increased likelihood of $t_g$ being consistent with $y_g$.
- **Variance Reduction**: The variance decreases, reflecting increased confidence in $t_g$ due to the observed outcome.


### Step 5: Compute Performance-to-Game Messages

**Objective**: Update the message from the performance difference $t_g$ back to the game factor $h_g$.

**Equations**
Compute the natural parameters of the message:

$$
r_{t_g \to h_g}^{\tau+1} = \tilde{r}_g^{\tau+1} - r_{h_g \to t_g}^\tau
$$

$$
\lambda_{t_g \to h_g}^{\tau+1} = \tilde{\lambda}_g^{\tau+1} - \lambda_{h_g \to t_g}^\tau
$$

where:

$$
\tilde{r}_g^{\tau+1} = (\tilde{\nu}_g^{\tau+1})^{-1}
$$

$$
\tilde{\lambda}_g^{\tau+1} = \tilde{r}_g^{\tau+1} \tilde{\mu}_g^{\tau+1}
$$

$r_{h_g \to t_g}^\tau$ and $\lambda_{h_g \to t_g}^\tau$: Natural parameters from the message computed in Step 3.

**Interpretation**:

- **Adjusting for New Information**: By subtracting the previous message, we isolate the new information gained from incorporating $y_g$.
- **Ensuring Consistency**: This update maintains the correct flow of information in the factor graph.

---

### Step 6: Compute Game-to-Skill Messages

**Objective**:
Update the messages from the game factor $h_g$ back to the player skills $w_{I_g}$ and $w_{J_g}$.

**Equations**

For Player $I_g$:
$$
\nu_{h_g \to w_{I_g}}^{\tau+1} = \sigma_n^2 + \nu_{t_g \to h_g}^{\tau+1} + \nu_{w_{J_g} \to h_g}^\tau
$$

$$
\mu_{h_g \to w_{I_g}}^{\tau+1} = \mu_{w_{J_g} \to h_g}^\tau + \mu_{t_g \to h_g}^{\tau+1}
$$

For Player $J_g$:
$$
\nu_{h_g \to w_{J_g}}^{\tau+1} = \sigma_n^2 + \nu_{t_g \to h_g}^{\tau+1} + \nu_{w_{I_g} \to h_g}^\tau
$$

$$
\mu_{h_g \to w_{J_g}}^{\tau+1} = \mu_{w_{I_g} \to h_g}^\tau - \mu_{t_g \to h_g}^{\tau+1}
$$

#### Conversion to Natural Parameters:

**Precision**:
$$
r_{h_g \to w_i}^{\tau+1} = (\nu_{h_g \to w_i}^{\tau+1})^{-1}
$$

**Natural Mean**:
$$
\lambda_{h_g \to w_i}^{\tau+1} = r_{h_g \to w_i}^{\tau+1} \mu_{h_g \to w_i}^{\tau+1}
$$

**Interpretation**:

- **Incorporating Updated Performance**: The messages back to the skills include the updated information from $t_g$ and the opponent's skill estimate.
- **Asymmetry in Updates**: The messages to each player depend on both their own and the opponent's skill messages, reflecting the competitive nature of the game.


## Iterative Process:
We repeat the steps, updating messages and marginals in each iteration. The algorithm continues until the changes in the skill estimates are below a certain threshold, indicating convergence.

---

## 20-Moment-Matching-Approximation

### Why Moment Matching?
1. **Intractable Distributions**: The true distributions may involve step functions or other non-Gaussian components that are difficult to handle analytically.
2. **Gaussian Approximation**: By approximating these distributions with Gaussians, we can leverage the tractability and closed-form solutions available for Gaussian distributions.

### Handling Binary Outcomes with Gaussian Approximations

#### The Challenge

In TrueSkill™, game outcomes ($y_g$) are binary ($+1$ or $-1$), indicating which player won. However, the Bayesian framework and message passing rely on Gaussian distributions for computational efficiency. The challenge arises when attempting to incorporate binary information into a Gaussian framework.

#### Key Idea: Moment Matching

To reconcile binary outcomes with Gaussian approximations, moment matching is employed. This technique approximates the influence of the binary variable by adjusting the moments (mean and variance) of the Gaussian distribution to reflect the observed outcome.

#### Moment Matching for Truncated Gaussian Densities

##### Truncated Gaussian Density

When incorporating a binary outcome into a Gaussian framework, we effectively truncate the Gaussian distribution based on the observed outcome. For instance, if Player $I_g$ wins ($y_g = +1$), the performance difference $t_g$ must be positive, leading to a truncated Gaussian over $t_g$.

##### Defining the Truncated Gaussian

Consider the truncated Gaussian density function:

$$p(t) = \frac{1}{Z_t} \delta(y - \text{sign}(t)) N(t; \mu, \sigma^2)$$

where:

- $N(t; \mu, \sigma^2)$ is the normal distribution with mean $\mu$ and variance $\sigma^2$.
- $\delta(x)$ is the Dirac delta function (aka indicator function), ensuring $y = \text{sign}(t)$.
- $Z_t$ is the normalization constant.
- $y \in \{-1, +1\}$ represents the observed game outcome.

![[rank-truncated.png]]
##### Objective: Approximate $p(t)$ with a Gaussian $q(t)$

We aim to approximate the truncated Gaussian $p(t)$ with a Gaussian $q(t)$ that has the same first and second moments (mean and variance):

$$q(t) = N(t; \tilde{\mu}, \tilde{\sigma}^2)$$

###### First Moment (Mean)

$$E[t] = \langle t \rangle_{p(t)}$$

###### Second Central Moment (Variance)

$$V[t] = \langle t^2 \rangle_{p(t)} - \langle t \rangle_{p(t)}^2$$

##### Calculating the Moments

To determine $\tilde{\mu}$ and $\tilde{\sigma}^2$, we compute the first and second moments of $p(t)$ and set them equal to those of $q(t)$.

##### Detailed Derivation of Moments

###### Normalization Constant $Z_t$

$$Z_t = \Phi\left(\frac{y\mu}{\sigma}\right)$$

where $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution.

###### Derivation: 

**i. Case 1: $y = +1$**

When $y = +1$, $p(t)$ is non-zero only for $t > 0$. Therefore:

$$Z_t = \int_{-\infty}^{+\infty} \delta(+1 - \text{sign}(t)) \mathcal{N}(t; \mu, \sigma^2) \, dt = \int_{0}^{+\infty} \mathcal{N}(t; \mu, \sigma^2) \, dt = \int_{-\mu/\sigma}^{+\infty} \mathcal{N}(z; 0, 1) \, dz = \Phi\left(\frac{\mu}{\sigma}\right).$$

**ii. Case 2: $y = -1$**

When $y = -1$, $p(t)$ is non-zero only for $t < 0$. Therefore:

$$
Z_t = \int_{-\infty}^{+\infty} \delta(-1 - \text{sign}(t)) \mathcal{N}(t; \mu, \sigma^2) \, dt = \int_{-\infty}^{0} \mathcal{N}(t; \mu, \sigma^2) \, dt = \int_{-\infty}^{-\mu/\sigma} \mathcal{N}(z; 0, 1) \, dz = \Phi\left(-\frac{\mu}{\sigma}\right) .
$$

We can generalize this using $y \in \{-1, +1\}$:

$$
Z_t = \Phi\left(\frac{y\mu}{\sigma}\right).
$$

---
###### First Moment (Mean)

To determine the first moment (mean) of the truncated Gaussian distribution, we differentiate the normalization constant $Z_t$ with respect to the mean parameter $\mu$:

$$
\frac{\partial Z_t}{\partial \mu} = \frac{\partial}{\partial \mu} \int_{0}^{+\infty} \mathcal{N}(t; y\mu, \sigma^2) \, dt = \int_{0}^{+\infty} \frac{\partial}{\partial \mu} \mathcal{N}(t; y\mu, \sigma^2) \, dt.
$$

Evaluating the derivative inside the integral, we obtain:

$$
= \int_{0}^{+\infty} y\sigma^{-2}(t - y\mu)\mathcal{N}(t; y\mu, \sigma^2) \, dt.
$$

This expression can be rewritten by factoring out the constants $y$ and $\sigma^{-2}$:

$$
= y\sigma^{-2} \int_{0}^{+\infty} (t - y\mu)\mathcal{N}(t; y\mu, \sigma^2) \, dt.
$$

Recognizing that $p(t) = \frac{\mathcal{N}(t; y\mu, \sigma^2)}{Z_t}$ is the normalized truncated Gaussian density, the integral simplifies to:

$$
= y\sigma^{-2} Z_t \left( \langle t \rangle_{p(t)} - y\mu \right) = yZ_t\sigma^{-2} \langle t \rangle_{p(t)} - \mu Z_t\sigma^{-2}.,
$$

where $\langle t \rangle_{p(t)}$ denotes the expectation of $t$ under the distribution $p(t)$.

**Crucial Observation:** Since $y \in \{-1, +1\}$, it follows that $y^2 = 1$, effectively neutralizing the $y^2$ term in the equation.

Alternatively, the derivative of the normalization constant can also be expressed using the properties of the cumulative distribution function (CDF):

$$
\frac{\partial Z_t}{\partial \mu} = \frac{\partial}{\partial \mu} \Phi\left(\frac{y\mu}{\sigma}\right) = \frac{y}{\sigma} \mathcal{N}\left(\frac{y\mu}{\sigma}; 0, 1\right),
$$

where $\Phi$ is the CDF of the standard normal distribution, and $\mathcal{N}$ represents the probability density function (PDF) of the standard normal distribution.

By equating both expressions for $\frac{\partial Z_t}{\partial \mu}$, we solve for the expectation $\langle t \rangle_{p(t)}$:

$$\langle t \rangle_{p(t)} = y\mu + \sigma \frac{\mathcal{N}\left(\frac{y\mu}{\sigma}; 0, 1 \right)}{\Phi\left(\frac{y\mu}{\sigma}\right)} = y\mu + \sigma \Psi\left(\frac{y\mu}{\sigma}\right).$$

where we introduced the **tilting function** $\Psi(z) = \frac{\mathcal{N}(z)}{\Phi(z)}$. 

##### Second Moment (Variance)

To compute the second moment (variance) of the truncated Gaussian distribution, we differentiate the normalization constant $Z_t$ twice with respect to $\mu$:

$$
\frac{\partial^2 Z_t}{\partial \mu^2} = \frac{\partial}{\partial \mu} \int_{0}^{+\infty} y\sigma^{-2}(t - y\mu)\mathcal{N}(t; y\mu, \sigma^2) \, dt.
$$

Expanding the derivative inside the integral:

$$
= \int_{0}^{+\infty} y\sigma^{-2} \frac{\partial}{\partial \mu} \left[ (t - y\mu)\mathcal{N}(t; y\mu, \sigma^2) \right] \, dt.
$$

Carrying out the differentiation:

$$
= \int_{0}^{+\infty} y\sigma^{-2} \left[ -\sigma^{-2} + \sigma^{-4}(t - y\mu)^2 \right] \mathcal{N}(t; y\mu, \sigma^2) \, dt.
$$

Simplifying, we express this in terms of expectations under $p(t)$:

$$
= \Phi\left(\frac{y\mu}{\sigma}\right) \left\langle -\sigma^{-2} + \sigma^{-4}(t - y\mu)^2 \right\rangle_{p(t)}.
$$

Alternatively, differentiating the earlier expression for the first derivative provides another perspective:

$$
\frac{\partial^2 Z_t}{\partial \mu^2} = \frac{\partial}{\partial \mu} \left[ y \mathcal{N}\left(\frac{y\mu}{\sigma}; 0, 1\right) \right] = -\sigma^{-2} y\mu \mathcal{N}\left(\frac{y\mu}{\sigma}; 0, 1\right).
$$

By equating both expressions for $\frac{\partial^2 Z_t}{\partial \mu^2}$, we solve for the variance $V[t]$:

$$
V[t] = \sigma^2 \left(1 - \Lambda\left(\frac{y\mu}{\sigma}\right)\right),
$$

where we define:

$$
\Lambda(z) = \Psi(z) \left( \Psi(z) + z \right).
$$

**Clarification:** The term involving $y^2$ emerges during the derivation but simplifies to 1 due to $y \in \{-1, +1\}$, thereby making it redundant in the final variance expression.

<span style="color: red;"> do these calculations yourself! </span>
