
### Greedy Improvement vs. Policy Gradient

- **Greedy improvement:** directly chooses actions maximizing the Q-values estimated by the critic, efficient in small state-action spaces.
- **Policy gradient improvement**: performs gradient ascent directly on policy parameters, more suited for larger spaces.

---

## Baselines and Advantage Functions

An advantage function \( A^\pi(s,a) \) describes how much better an action is compared to the average action in state \( s \):

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

This reduces variance significantly in gradient estimation:

$$
∇J(ω) ∝ \sum_s d^\pi(s) \sum_a A^\pi(s,a) \nabla_\omega \log \pi(a|s,ω)
$$

The baseline \( b(s) = V^\pi(s) \) is a natural choice as it minimizes variance without introducing bias.

---

## Key Takeaways and Intuition

- **Policy gradient methods** directly parameterize policies, allowing gradient-based optimization without explicit value computation.
- The **policy gradient theorem** provides a fundamental link between the gradient of expected returns and the policy parameters, enabling optimization using stochastic samples.
- **REINFORCE** methods rely on episodic returns, while actor-critic methods provide ongoing value estimation, resulting in more stable learning.
- **Actor-Critic methods** embody generalized policy iteration, simultaneously improving policy (actor) and evaluating value (critic), leading to a more balanced and stable training process.
- The concept of **baselines and advantage functions** plays a crucial role in reducing variance and stabilizing learning.

---

## Conclusion

Policy gradient and actor-critic methods represent fundamental tools in advanced reinforcement learning, offering efficient and powerful frameworks capable of handling large and complex state-action spaces through parameterized policies and explicit value estimation. Understanding these methods deeply provides strong foundations for advanced RL applications.