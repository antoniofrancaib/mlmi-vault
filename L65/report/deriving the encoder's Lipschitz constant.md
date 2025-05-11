In short, the encoding transformation is something along these lines:
$$\text{Input} \xrightarrow{\text{LN} + \text{GVP}_{\text{embedding}}} \text{Embedded Features} \xrightarrow{\text{Encoder Layer } 1 \, (\text{GVP}_{\text{msg}} + \text{GVP}_{\text{ff}})} \cdots \xrightarrow{\text{Layer } L} \text{Output}
$$


<span style="color: blue;">create a more holistic map: </span>

$$ f_{\text{emb}} = \text{MultiGVP}_{\text{emb}} \circ \text{LN}: \{G_1, \dots, G_C\} \to \{H_1, \dots, H_C\} $$

where each $H_i \subset \left( \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} \right)^N$, and thus $$ \{H_1, \dots, H_C\} \subset \left( \left( \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} \right)^N \right)^C$$


Note in our notation $| \{G_1, \dots, G_C\}| = C$, where $C$ is the number of conformations for a given sequence. 


Layer normalization is applied per node and per conformation. In our setup, each conformation $G_i$ is a collection of $N$ nodes, with each node having raw features $(\mathbf{s}_n^{\text{raw}}, \mathbf{v}_n^{\text{raw}}) \in \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3}$. We apply LN individually to each node's features within each conformation—typically using one normalization operator on the scalar part $\mathbf{s}_n^{\text{raw}}$ and a corresponding (often modified) normalization for the vector part $\mathbf{v}_n^{\text{raw}}$ that respects its geometric structure. In notation, the LN is applied as  

$$\text{LN}: (\mathbb{R}^{64} \times \mathbb{R}^{4 \times 3})^N \to (\mathbb{R}^{64} \times \mathbb{R}^{4 \times 3})^N,$$

for each $G_i$, yielding normalized features that are then embedded by the GVP block. This ensures that each node in every conformation is normalized independently before being processed further in the encoder pipeline.




## upper bounding the encoder's Lipschitz constant (and squashing to zero):
### **embedding**
The overall encoding transformation can be written as the composition of two main parts. First, we have an initial embedding function:
$$ f_{\text{emb}} = \text{GVP}_{\text{emb}} \circ \text{LN}: \mathcal{X} \to \mathcal{H_0} \subset \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
which is implemented by applying layer normalization (assumed to be 1–Lipschitz) followed by a GVP-based embedding. 

The GVP-based embedding uses four weight matrices: $W_{\text{ws}}$, $W_{\text{wh}}$, $W_{\text{wv}}$, and $W_{\text{wsv}}$. Under standard assumptions (and ignoring the nonlinearities and layer normalization which are either 1–Lipschitz or their contribution is absorbed), one can upper-bound the Lipschitz constant of a GVP block by
$$L_{\text{GVP}_{\text{emb}}} \leq \max \left\{ \|W_{\text{ws}}\|, \|W_{\text{wh}}\|\|W_{\text{wv}}\| + \|W_{\text{ws}}\|\|W_{\text{wsv}}\| \right\}.$$
Thus, we set (using the composition rule)
$$L_{f_{\text{emb}}} \leq L_{\text{GVP}_{\text{emb}}}.$$
### **encoder layers** 
Next, we have $L$ encoder layers (in the code, $L=3$). For each layer $l=1,2,\ldots,L$, let:
$$ f_l: H_{l-1} \to H_l $$
denote the $l$th encoder layer. Each encoder layer is composed of two sub-functions: (1) a ***Message Passing Sub–Function***, and (2) a ***Feedforward Sub–Function***. 

(1) The message passing function, which is denoted by $f_{\text{msg}}^{(l)}$, is implemented using a GVP block. Its Lipschitz constant is bounded by:
$$ L_{\text{msg}}^{(l)} \leq L_{\text{GVP}_{\text{msg}}}^{(l)}. $$
meaning that the Lipschitz constant of the message function is bounded by the Lipschitz constant of the underlying GVP block -we can calculate $L_{\text{GVP}_{\text{msg}}}^{(l)}$ in a similar fashion as we did for $L_{\text{GVP}_{\text{emb}}}$.

However, before adding the message into the node update, the messages from the neighbors are aggregated using the graph’s connectivity. This aggregation can be represented by a linear operator (e.g., the adjacency matrix) $A$ with spectral norm $\|A\|$. Then, by linearity, the aggregation step has a Lipschitz constant:
$$ L_{\text{agg}}^{(l)} \leq \|A\| \cdot L_{\text{msg}}^{(l)} \leq \|A\| \cdot L_{\text{GVP}_{\text{msg}}}^{(l)}. $$
<span style="color: red;">(here, have to check if there is a residual connection) </span>

(2) The feedforward update is denoted by $f_{\text{ff}}^{(l)}$. It is implemented as another GVP block after a layer normalization, and this time it is applied in a residual manner. Thus, its Lipschitz constant is bounded by:
$$ L_{\text{ff}}^{(l)} \leq L_{\text{GVP}_{\text{ff}}}^{(l)}, $$
and then the effective Lipschitz constant for the feedforward update is:
$$ 1 + L_{\text{ff}}(l). $$
Thus, the overall Lipschitz constant of the $l$th encoder layer is bounded by the product:
$$ L(f_l) \leq \|A\| L_{\text{GVP}_{\text{msg}}}^{(l)} \cdot (1 + L_{\text{GVP}_{\text{ff}}}^{(l)}). $$
Since the full network is the composition: $f = f_L \circ f_{L-1} \circ \dots \circ f_1 \circ f_{\text{emb}},$ the overall Lipschitz constant satisfies:
$$ L(f) \leq L_{\text{GVP}_{\text{emb}}} \cdot \prod_{l=1}^{L} \|A\| L_{\text{GVP}_{\text{msg}}}^{(l)} \cdot (1 + L_{\text{GVP}_{\text{ff}}}^{(l)})$$
Suppose now that the aggregation operator is normalized so that  $\|A\| = \frac{1}{k},$ where $k$ is the number of neighbors in the $k$–regular graph. Then the bound becomes  
$$L(f) \leq L_{\text{GVP emb}} \cdot \prod_{l=1}^{L} \left( \frac{1}{k} L_{\text{GVP msg}}(l) \cdot (1 + L_{\text{GVP ff}}(l)) \right).$$
For fixed per–layer constants, as $k \to \infty$, the factor $\frac{1}{k}$ drives each term in the product toward zero. Hence, the upper bound on $L(f)$ can be made arbitrarily small via $k \to \infty$.  

recovering the notion of normalization favors oversmoothing with Lips continuity 


<span style="color: blue;">Calculating the operator norms of the weight matrices lets you derive theoretical upper bounds on the Lipschitz constant of each module. This, in turn, gives insight into how sensitive the network is to input perturbations, its robustness, and potential oversmoothing issues. In practice, these bounds can guide regularization strategies—like spectral normalization—to help control the network's stability during training. </span>???

# lower bounding the encoder's Lipschitz constant (and blowing up arbitrarily large):

Consider the encoder transformation as a composition of an embedding followed by $L$ encoder layers. The embedding is defined as  
$$\mathbf{f}_{\text{emb}} = \text{GVP}_{\text{emb}} \circ \text{LN}: \mathcal{X} \to \mathcal{H}_0,$$
and we assume that the layer normalization is 1–Lipschitz. In particular, suppose that on constant inputs the embedding satisfies  
$$\mathbf{f}_{\text{emb}}(\mathbf{0}_N) = \mathbf{0}_N \quad \text{and} \quad \mathbf{f}_{\text{emb}}(\mathbf{1}_N) \geq \lambda_{\text{emb}} \mathbf{1}_N,$$
where $\mathbf{1}_N$ is the vector (or collection of node features) with all entries equal to 1 and $\lambda_{\text{emb}} > 0$.  In other words, the embedding stage does not contract the constant signal below a known threshold (the gain $\lambda_{\text{emb}}$).

Next, consider a single encoder layer. Each such layer is composed of two sub-functions: a message passing sub–function and a feedforward sub–function applied in a residual manner. For the message passing component, assume that when applied to a constant input the corresponding GVP block has a gain of at least $\lambda_{\text{msg}}^{(l)}$; that is, it maps $\mathbf{1}_N$ to at least $\lambda_{\text{msg}}^{(l)} \mathbf{1}_N$. Before these messages are incorporated into the node update, they are aggregated using the graph’s connectivity via a linear operator $\mathbf{A}$. For a $k$–regular graph with unnormalized aggregation, we have  $\mathbf{A} \mathbf{1}_N = k \mathbf{1}_N.$  

Thus, the aggregated message is at least $\lambda_{\text{msg}}^{(l)} \cdot k \cdot \mathbf{1}_N$. Then, the feedforward sub–function (with gain at least $\lambda_{\text{ff}}^{(l)}$) acts on this aggregated message. Since the update is residual, the output of the layer is at least  $$\mathbf{1}_N + \lambda_{\text{ff}}^{(l)} (\lambda_{\text{msg}}(l) \cdot k \cdot\mathbf{1}_N) = (1 + \lambda_{\text{ff}}^{(l)} \lambda_{\text{msg}}^{(l)} \cdot  k) \mathbf{1}_N.$$
Now, propagating this behavior through all $L$ encoder layers, starting from  
$\mathbf{f}_{\text{emb}}(\mathbf{1}_N) \geq \lambda_{\text{emb}} \mathbf{1}_N,$ we obtain  
$$\mathbf{f}(\mathbf{1}_N) \geq \lambda_{\text{emb}} \prod_{l=1}^{L} (1 + \lambda_{\text{ff}}^{(l)} \lambda_{\text{msg}}^{(l)} \cdot k) \mathbf{1}_N.$$
Since $\mathbf{f}(\mathbf{0}_N) = \mathbf{0}_N$, the difference between these two outputs is at least  
$$\| \mathbf{f}(\mathbf{1}_N) - \mathbf{f}(\mathbf{0}_N) \| \geq \lambda_{\text{emb}} \prod_{l=1}^{L} (1 + \lambda_{\text{ff}}^{(l)} \lambda_{\text{msg}}^{(l)} \cdot k) \| \mathbf{1}_N \|.$$
Because the difference between the inputs is simply $\| \mathbf{1}_N - \mathbf{0}_N \| = \| \mathbf{1}_N \|$, the definition of the Lipschitz constant implies  $$L(\mathbf{f}) \geq \frac{\| \mathbf{f}(\mathbf{1}_N) - \mathbf{f}(\mathbf{0}_N) \|}{\| \mathbf{1}_N - \mathbf{0}_N \|} \geq \lambda_{\text{emb}} \prod_{l=1}^{L} (1 + \lambda_{\text{ff}}^{(l)} \lambda_{\text{msg}}^{(l)} \cdot k) \geq \lambda_{\text{emb}} (1 + c k)^L.$$where we assumed that for all the layers the product $\lambda_{\text{ff}}^{(l)} \lambda_{\text{msg}}^{(l)}$ is bounded below by some constant $c > 0$. 



Thus, as $k$ tends to infinity—corresponding to increasingly dense, unnormalized $k$–regular graphs—the lower bound diverges to infinity. This shows very formally that by controlling $k$, we can force the encoder’s Lipschitz constant to blow up arbitrarily, thereby yielding a “wider” encoding that is highly sensitive to input variations. In contrast, as described earlier, using normalized aggregation (for example, setting $\| \mathbf{A} \| = \frac{1}{k}$) will squash the Lipschitz constant toward zero, resulting in a much tighter encoding. Therefore, these design choices enable precise control over the encoding’s sensitivity.



sacar constants de otras arquitecturas  y comparaciones point set
atencion 
oversquashing - oversmoothing 
