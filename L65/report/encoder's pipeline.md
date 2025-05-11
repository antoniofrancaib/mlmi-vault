#node-mapping: 
    $f: \mathcal{X} → \mathcal{H}$, 
    where $\mathcal{X} = \{ (s_n, v_n), (s_e, v_e), \text{graph structure} \}$, and $\mathcal{H} \subseteq \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.$

# raw-RNA-data 

├─ 
# featurizer 
- **nodes**: 15 scalars, 4 vectors → **projected to** $(64, 4)$
- **edges**: 131 scalars, 1 vector → **projected to** $(32, 1)$
- **graph**: k-NN connectivity (**k** neighbors per node)

├─ 
# initial-embedding

### **node branch** `self.W_v = torch.nn.Sequential`
- **input**:  scalars: $s_n \in \mathbb{R}^{64}$ , vectors: $v_n \in \mathbb{R}^{4 \times 3}$ 

- **layernorm**: $\text{LN}: \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3} \to \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3} \quad (s_n, v_n) \mapsto (s_n, v_n)$  

- **gvp**:  
	- **vector branch**: 
		- `self.W_v.wh`: $\mathbb{R}^{4 \times 3} \to \mathbb{R}^{16 \times 3} \quad v_n \mapsto v_{n}^h$
		- compute norm: $n(v_{n}^h) \in \mathbb{R}^{16}$  
	
	- **scalar branch**:  
		  - concatenate: $(s_n, n(v_{n}^h)) \in \mathbb{R}^{80}$  
		  -  `self.W_v.wh`: $\mathbb{R}^{80} \to \mathbb{R}^{128} \quad s_n \mapsto s_{n}^{\text{out}}$
	  
	- **vector output**:  `vector_gate=True`
		- `self.W_v.wv`: $\mathbb{R}^{16 \times 3} \to \mathbb{R}^{16 \times 3} \quad v_{n}^h \mapsto v_{n}^{\text{temp}}$
		- `self.W_v.wsv`: $\mathbb{R}^{128} \to \mathbb{R}^{16} \quad s_{n}^{\text{out}} \mapsto \text{gate}$
		- apply **sigmoid** and multiply:  
			$v_{n}^{\text{out}} = v_{n}^{\text{temp}} \odot \sigma(\text{gate})$
		
	- **output**:  $(s_{n}^{\text{out}}, v_{n}^{\text{out}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$

### **edge branch** `self.W_e = torch.nn.Sequential`
- **input**: scalars: $s_e \in \mathbb{R}^{32}$ , vectors: $v_e \in \mathbb{R}^{1 \times 3}$ 

- **layernorm**: $(s_e, v_e) \to (s_e, v_e)$  

- **gvp**:  
	- **vector branch**: 
		- `self.W_e.wh`: $\mathbb{R}^{1 \times 3} \to \mathbb{R}^{1 \times 3} \quad v_e \mapsto v_{e}^h$
		- compute norm: $n(v_{e}^h) \in \mathbb{R}^{1}$  
	
	- **scalar branch**:  
		  - concatenate: $(s_e, n(v_{e}^h)) \in \mathbb{R}^{33}$  
		  -  `self.W_e.ws`: $\mathbb{R}^{33} \to \mathbb{R}^{32} \quad s_e \mapsto s_{e}^{\text{out}}$
	  
	- **vector output**:  
		- `self.W_e.wv`: $\mathbb{R}^{1 \times 3} \to \mathbb{R}^{1 \times 3} \quad v_{e}^h \mapsto v_{e}^{\text{temp}}$
		- `self.W_e.wsv`: $\mathbb{R}^{32} \to \mathbb{R}^{1} \quad s_{e}^{\text{out}} \mapsto \text{gate}$
		- apply **sigmoid** and multiply:  
			$v_{e}^{\text{out}} = v_{e}^{\text{temp}} \odot \sigma(\text{gate})$
		
	- **output**:  $(s_{e}^{\text{out}}, v_{e}^{\text{out}}) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3}$

├─ 
# encoder layers  
(e.g. multigvpconvlayer $(l = 3)$)

**for each targe node $i$:** 
- **for each edge** $e_{ij}$:
	- **gather**:
		- **target**: $(s_{i}, v_{i}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$
		- **edge**: $(s_e, v_e) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3}$
		- **source**: $(s_{j}, v_{j}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$

	- **concatenate**:
		- vectors:  $v=[v_i; v_e; v_j] \in \mathbb{R}^{16+1+16 = 33} \text{ channels, each 3d}$ 
		- compute norm: $n(v) \in \mathbb{R}^{33}$  
		- scalars:  $s=[s_i; s_e; s_j] \in \mathbb{R}^{128+32+128 = 288}$
			- **concatenate** with the vector norm $\tilde{s} = [s; n(v)] \in \mathbb{R}^{288 + 33 = 321}$

	- **message function (via gvps)**:
		- **vector branch**:  $[W_{\text{conv}}.wh]: \mathbb{R}^{33 \times 3} \to \mathbb{R}^{33 \times 3} \quad v \mapsto v_h$ 
		- **scalar branch**:  $[W_{\text{conv}}.ws]: \mathbb{R}^{321} \to \mathbb{R}^{128} \quad \tilde{s} \mapsto s_{\text{msg}}$  
		- **vector output**:  `vector_gate=True`
			- $[W_{\text{conv}}.wv]: \mathbb{R}^{33} \to \mathbb{R}^{16} \quad v_h \mapsto v_{\text{temp}}$  
			- $[W_{\text{conv}}.wsv]: \mathbb{R}^{128} \to \mathbb{R}^{16} \quad s_{\text{msg}} \mapsto \text{gate}$ 
			- apply **sigmoid** and multiply:  
				$v_{msg} = v_{\text{temp}} \odot \sigma(\text{gate})$

		- **output message**:  
			$(s_{\text{msg}}, v_{\text{msg}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$  

	- <span style="color: red;"> second message function (via attention)</span>
	
	- **aggregate messages for each node** 
		- for each target node, aggregate incoming messages (e.g. by averaging).

	- **feedforward update:** 
		- **input**: $x_i = (s_i, v_i) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.$

		- **layer normalization:** $x_i' = \text{LN}_1(x_i) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.$

		- **feedforward mapping:** $\Delta x_i = (\Delta s_i, \Delta v_i) = \text{GVP}_{ff}(x_i') \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.$

		- **residual connection with dropout:** $x_i^{res} = x_i + \text{Dropout}(\Delta x_i)$

		- **final layer normalization:** $x_i^{final} = \text{LN}_2(x_i^{res})\in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}.$




# NEW: 
## Part 1: Self-Attention Mechanism (New)

For each target node $i$:

- Compute query, key, value projections:

- Scalar query: $q^s_i = W_q s_i \in \mathbb{R}^{128}$ (reshaped to $h$ heads)

- Scalar key/value: $k^s_j = W_k s_j, v^s_j = W_v s_j \in \mathbb{R}^{128}$ for all nodes $j$

- Vector norm query: $q^v_i = W_{qv} \|v_i\| \in \mathbb{R}^{h}$ where $\|v_i\| \in \mathbb{R}^{16}$ is the norm of each vector channel

- Vector norm key: $k^v_j = W_{kv} \|v_j\| \in \mathbb{R}^{h}$ for all nodes $j$

- Compute attention scores:

- Scalar attention: $a^s_{ij} = \frac{(q^s_i)^T k^s_j}{\sqrt{d_k}}$

- Vector attention: $a^v_{ij} = \frac{(q^v_i)^T k^v_j}{\sqrt{1}}$

- Combined attention: $a_{ij} = a^s_{ij} + a^v_{ij}$

- Mask for autoregressive property: $\tilde{a}{ij} = \begin{cases} a{ij} & \text{if } j < i \\ -\infty & \text{otherwise} \end{cases}$

- Apply softmax: $\alpha_{ij} = \frac{\exp(\tilde{a}{ij})}{\sum_k \exp(\tilde{a}{ik})}$

- Compute attention output:

- Scalar output: $s^{attn}i = \sum_j \alpha{ij} v^s_j \in \mathbb{R}^{128}$

- Apply projection: $\tilde{s}^{attn}i = W{out} s^{attn}i \in \mathbb{R}^{128}$

- Update vector features: $(s'i, v'_i) = \text{GVP}((\tilde{s}^{attn}_i, v_i))$

## Part 2: Edge-based Message Passing (Modified Original Process)

For each target node $i$:

- For each edge $e_{ij}$ where $j < i$ (autoregressive constraint):

- Gather:

- Target: $(s'{i}, v'{i}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$ (from attention stage)

- Edge: $(s_e, v_e) \in \mathbb{R}^{(32+4)} \times \mathbb{R}^{1 \times 3}$ (includes sequence info)

- Source: $(s'{j}, v'{j}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$ (from attention stage)

- Concatenate:

- Vectors: $v=[v'j; v_e] \in \mathbb{R}^{(16+1) \times 3}$ (source + edge)

- Scalars: $s=[s'j; s_e] \in \mathbb{R}^{128+(32+4)}$ (source + edge)

- Message function (via GVP):

- Process concatenated features: $(s_{msg}, v_{msg}) = \text{GVP}((s, v))$

- Aggregate messages at node $i$:

- Sum messages: $s^{edge}i = \sum{j:j<i} s_{msg}, v^{edge}i = \sum{j:j<i} v_{msg}$

- Combine attention and edge outputs:

- Final node update: $s''i = s'_i + s^{edge}_i, v''_i = v'_i + v^{edge}_i$

## Part 3: Feed-Forward Network (New)

For each node $i$:

- Apply FFN:

- $(s'''i, v'''_i) = \text{FFN}((s''_i, v''_i))$

- where $\text{FFN} = \text{GVP}2(\text{Dropout}(\text{GVP}_1((s''_i, v''_i))))$

- $\text{GVP}1: (128, 16) \rightarrow (512, 32)$

- $\text{GVP}2: (512, 32) \rightarrow (128, 16)$

- Final output with residual connection:

- $s^{final}i = s''_i + s'''_i$

- $v^{final}i = v''_i + v'''_i$

This scheme preserves the structure of your original formulation while incorporating the new attention mechanism and autoregressive constraints from the updated architecture.