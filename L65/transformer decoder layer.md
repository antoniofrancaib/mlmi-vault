## 1. Vector-to-Scalar Projection

Current Implementation:

v_norm_flat = torch.norm(v_norm, dim=2)  # [n_nodes, d_v]
v_scalar = self.vector_projection(v_norm_flat)  # [n_nodes, d_s]
enhanced_s = s_norm + v_scalar

This computes:

- $\|v_i\| \in \mathbb{R}^{d_v}$ - Vector norms for each node
- $v_{scalar} = W_{proj}\|v_i\| \in \mathbb{R}^{d_s}$ - Linear projection to scalar space
- $s_{enhanced} = s + v_{scalar}$ - Addition of projected vector information to scalar features

## 2. Edge-Based Conditioning

Mathematical transformation:

- For node $i$, compute attention separately over two edge sets:
- Forward edges (src < dst): $\text{Attention}(q_i, K_{fwd}, V_{fwd})$ using current features
- Backward edges (src ≥ dst): $\text{Attention}(q_i, K_{bwd}, V_{bwd})$ using encoder features

- Aggregate: $\text{output}i = \frac{1}{|\mathcal{N}_i|} \sum{j \in \mathcal{N}i} \text{Attention}_j$