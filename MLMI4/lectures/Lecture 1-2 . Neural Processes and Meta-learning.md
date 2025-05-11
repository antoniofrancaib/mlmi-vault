# --FIRST HALF--
Lecture by Richard E. Turner
***topics***: 

# --SECOND HALF--
Lecture by Jose Miguel Hernandez Lobato
***topics***: bayesian optimization 

# 1. Overview and Motivation

Bayesian Optimization (BO) is a family of methods designed to efficiently optimize expensive objective functions when we can only afford a small number of function evaluations. The objective function $f(\mathbf{x})$ is treated as a black box, meaning:

- We may not know its analytical form or structure.
- We cannot assume it is differentiable.
- Each function evaluation (experiment, simulation, or measurement) is very costly in time, money, or other resources.
- Observations of $f(\mathbf{x})$ can be noisy.

The core motivation of BO is to locate a (near) optimal point

$$
\mathbf{x}^* = \arg\min_{\mathbf{x} \in X} f(\mathbf{x}),
$$

where $X$ is usually a compact subset of $\mathbb{R}^D$, without exhaustively scanning the entire space or using many evaluations. Instead, BO methods use surrogate modeling (from machine learning) plus an acquisition function (an intelligent data-collection strategy) to balance exploration and exploitation.

## Examples of Where BO Excels

- **Hyper-parameter Tuning in Machine Learning**: E.g., tuning neural network hyper-parameters (layers, neurons per layer, learning rate, regularization constants). Evaluating each set of hyper-parameters can be expensive (long training times), and BO often finds good solutions with fewer trials than grid/random search.

- **Fast Robot Adaptation**: Optimizing a six-dimensional behavior space for damaged robots. Each experiment (attempted movement pattern) is expensive, so a method that learns and adapts quickly saves time and mechanical wear.

- **Drug and Material Discovery**: Millions (or more) of possible molecules must be tested to find high-performing drugs, OLEDs, or OPVs. A direct brute-force approach is infeasible, so an intelligent, iterative strategy is critical.

- **Optimal Design in Engineering**: Large, multi-step engineering processes are often high dimensional and expensive to evaluate. BO helps automate design decisions, finding better solutions faster than human trial-and-error.

# 2. Problem Statement and When to Use BO

We focus on solving:

$$
\min_{\mathbf{x} \in X} f(\mathbf{x}),
$$

under four key conditions:

1. $\mathbf{x}$ lies in a compact subset of $\mathbb{R}^D$ (or a similarly well-defined search space).
2. Evaluations of $f(\mathbf{x})$ are very expensive (e.g., training a big model or running a complex experiment).
3. $f$ is a black box: we have no closed-form expression or easily exploitable structure (like gradients).
4. Observations can be noisy: repeated evaluations of the same $\mathbf{x}$ may yield slightly different values.

When these conditions hold, especially the first two, Bayesian Optimization is an excellent candidate. For cheaper or simpler problems (e.g., differentiable functions with cheap evaluations), classical approaches (gradient-based, evolutionary algorithms, etc.) might be more straightforward. But in the “expensive black-box” regime, BO typically outperforms naive methods by rapidly converging with only a small number of function evaluations.

# 3. Main Idea of Bayesian Optimization

Bayesian Optimization relies on two key components:

1. **A machine learning model to predict $f(\mathbf{x})$ from observed data.**
   - This model provides a predictive distribution $p(y \mid \mathbf{x}, D_n)$ over possible function values $y = f(\mathbf{x})$.
   - Crucially, the model also provides uncertainty estimates or confidence bands, telling us how certain or uncertain we are about $f(\mathbf{x})$.

2. **An acquisition function $\alpha(\mathbf{x})$ that guides the next point to evaluate:**
   - The acquisition function is constructed to balance exploration and exploitation.
   - Formally, $\alpha(\mathbf{x})$ can be viewed as the expected utility of evaluating $f(\mathbf{x})$. In many settings,

     $$
     \alpha(\mathbf{x}) = \mathbb{E}_{y \sim p(y \mid \mathbf{x}, D_n)}[U(y \mid \mathbf{x}, D_n)],
     $$

     where $U(y \mid \mathbf{x}, D_n)$ is some notion of the utility of discovering $y$.

## Why This Works

- **Exploitation**: We use the model’s predictions to sample near points with promising low (or high) predicted values of $f(\mathbf{x})$.
- **Exploration**: We also sample in regions where the model’s uncertainty is high, searching for potentially better (unknown) solutions.

By iteratively refining the model (updating parameters and uncertainty) and choosing new points to evaluate based on the acquisition function, we make informed decisions about where to sample next. Over time, we narrow in on high-quality solutions efficiently.

# 4. Bayesian Optimization in Practice: The Algorithmic Loop

A typical Bayesian Optimization loop goes as follows:

1. **Initial Sampling**
   - Collect a small set of initial observations $\{(\mathbf{x}_i, f(\mathbf{x}_i))\}_{i=1}^n$.
   - This can be done via random design, space-filling design (e.g., Latin hypercube), or prior knowledge.

2. **Build a Predictive Model**
   - Fit (or update) a machine learning surrogate that models $f(\mathbf{x})$ based on the data $D_n$.
   - From this model, obtain a predictive distribution $p(y \mid \mathbf{x}, D_n)$ that provides both a mean prediction and an uncertainty measure.

3. **Define the Acquisition Function**
   - Design an acquisition function $\alpha(\mathbf{x})$ that translates the predictive distribution into a score for how valuable a new evaluation at $\mathbf{x}$ might be.
   - Common intuitions:
     - High potential improvement over the best observed value.
     - High uncertainty (exploration).

4. **Optimize the Acquisition Function**
   - Solve

     $$
     \mathbf{x}_{\text{next}} = \arg\max_{\mathbf{x}} \alpha(\mathbf{x}).
     $$

   - This optimization is typically much cheaper than evaluating the true objective $f(\mathbf{x})$, since $\alpha(\mathbf{x})$ is fast to compute.

5. **Evaluate the True Function**
   - Evaluate $f(\mathbf{x}_{\text{next}})$ by running the actual expensive experiment or simulation.
   - Add the new pair $(\mathbf{x}_{\text{next}}, f(\mathbf{x}_{\text{next}}))$ to the dataset $D_{n+1}$.

6. **Update the Surrogate Model**
   - Incorporate this new observation to refine model parameters and uncertainty estimates.

7. **Repeat**
   - Continue until a stopping criterion is met: budget exhausted, convergence, or performance threshold reached.

The schematic can be visualized as a cycle:
(Data) → (Model) → (Acquisition Function) → (Next Point) → (New Data) → (Updated Model) → ...

# 5. Illustrative Comparison: Bayesian Optimization vs. Uniform Exploration

A well-known practical comparison is Bayesian Optimization vs. random/grid search for hyper-parameter tuning. Empirical results often show that while random (or grid) search blindly samples points, BO learns which regions are promising and systematically improves function values faster.

For instance, in tuning Latent Dirichlet Allocation (LDA) hyper-parameters on a large text corpus (Wikipedia articles), BO reached a lower (better) perplexity or cost metric with far fewer function evaluations compared to random search. This highlights BO’s sample-efficiency—the speed at which it finds better solutions.

# 6. Exercises (and Deeper Reflection)

1. **Identify a Relevant Problem**
   - **Exercise**: Think of one problem in your own research where BO would excel.
   - Then think of another optimization problem where BO’s overhead or assumptions might make it less suitable (e.g., cheap or analytically tractable functions).

2. **Why Exploration Matters**
   - **Exercise**: Discuss why balancing exploration and exploitation is crucial in BO.
   - Specifically, what properties should the utility function $U(y \mid \mathbf{x}, D_n)$ have to ensure good exploration?

3. **Choice of Machine Learning Model**
   - **Exercise**: Which ML models (surrogates) would you choose for BO?
   - Common options might include Gaussian Processes, Random Forests, Bayesian Neural Networks, etc. Each has advantages/disadvantages regarding scalability, expressivity, and uncertainty estimation.

4. **When Pure Exploitation of the Predictive Mean Fails**
   - **Exercise**: Why might simply optimizing $\mathbb{E}[y]$ from $p(y \mid \mathbf{x}, D_n)$ fail to produce the best recommendation?
   - Hint: It ignores uncertainty and can get stuck in local minima. What are some alternative approaches (e.g., expected improvement, confidence bounds, etc.)?

# 7. Key Takeaways and Intuition

- **High-Level Principle**: BO is about active learning for optimization—each experiment is chosen to be as informative as possible.
- **Small Data, High Value**: Because we assume evaluations are expensive, we aim to learn a lot from few observations.
- **Surrogate + Acquisition**: The synergy of a predictive model (surrogate) and an acquisition function that trades off exploration vs. exploitation is the crux of BO.
- **Wide Applicability**: BO has been applied successfully in hyper-parameter tuning, robotics, materials science, engineering design, and more. Any domain with expensive black-box evaluations is a candidate.

# 8. Concluding Remarks

Bayesian Optimization is a powerful, principled approach for sequential decision-making under uncertainty. It systematically tackles the problem of “where to sample next” by leveraging a surrogate model of the objective. The combination of uncertainty modeling and intelligent data-collection underpins BO’s success in practice—particularly in scenarios where evaluations are resource-intensive.

By understanding and applying these methods, one can dramatically reduce costs and accelerate discovery in myriad fields, from machine learning (hyper-parameter tuning) and robotics (fault adaptation) to computational chemistry (drug/material design) and engineering (complex system optimization).