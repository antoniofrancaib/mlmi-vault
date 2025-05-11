### 1. **Raw RNA Data & Featurization**

- **Nodes:**  
  **Raw:** 15 scalars & 4 vectors  
  **Featurized (projected):**  $$ (s_n^{\text{raw}},\mathbf{v}_n^{\text{raw}}) \in \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3} $$
- **Edges:**  
  **Raw:** 131 scalars & 1 vector  
  **Featurized (projected):**  $$ (s_e^{\text{raw}},\mathbf{v}_e^{\text{raw}}) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3} $$
- **Graph Structure:**  
  k–NN connectivity (each node is connected to **k** neighbors).

---

### **2. Initial Embedding**

#### **Node Branch:**  
$$ \underbrace{(s_n^{\text{raw}},\,\mathbf{v}_n^{\text{raw}}) \in \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3}}_{\text{Input}} \quad \xrightarrow{\text{LayerNorm}} \quad (s_n,\,\mathbf{v}_n) \in \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3} $$  
- **GVP Node Mapping:**
  - **Vector branch:** $$ \mathbf{v}_n \overset{W_{v,\text{wh}}}{\longrightarrow} \mathbf{v}_n^h \in \mathbb{R}^{16 \times 3} \quad \Rightarrow \quad n(\mathbf{v}_n^h) \in \mathbb{R}^{16} $$
  - **Scalar branch:**  $$ \text{concat}(s_n,\,n(\mathbf{v}_n^h)) \in \mathbb{R}^{80} \overset{W_{v,\text{ws}}}{\longrightarrow} s_n^{\text{out}} \in \mathbb{R}^{128} $$
  - **Vector output with gating:**  $$ \mathbf{v}_n^h \overset{W_{v,\text{wv}}}{\longrightarrow} \mathbf{v}_n^{\text{temp}} \in \mathbb{R}^{16 \times 3} \quad \text{and} \quad s_n^{\text{out}} \overset{W_{v,\text{wsv}}}{\longrightarrow} \text{gate} \in \mathbb{R}^{16} $$$$ \mathbf{v}_n^{\text{out}} = \mathbf{v}_n^{\text{temp}} \odot \sigma(\text{gate}) $$
- **Node Output Embedding:**   $$ (s_n^{\text{out}},\,\mathbf{v}_n^{\text{out}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$

#### **Edge Branch:**  
$$ \underbrace{(s_e^{\text{raw}},\,\mathbf{v}_e^{\text{raw}}) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3}}_{\text{Input}} \quad \xrightarrow{\text{LayerNorm}} \quad (s_e,\,\mathbf{v}_e) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3} $$  
- **GVP Edge Mapping:**
  - **Vector branch:**  $$ \mathbf{v}_e \overset{W_{e,\text{wh}}}{\longrightarrow} \mathbf{v}_e^h \in \mathbb{R}^{1 \times 3} \quad \Rightarrow \quad n(\mathbf{v}_e^h) \in \mathbb{R}^{1} $$
  - **Scalar branch:**  $$ \text{concat}(s_e,\,n(\mathbf{v}_e^h)) \in \mathbb{R}^{33} \overset{W_{e,\text{ws}}}{\longrightarrow} s_e^{\text{out}} \in \mathbb{R}^{32} $$
  - **Vector output with gating:**  $$ \mathbf{v}_e^h \overset{W_{e,\text{wv}}}{\longrightarrow} \mathbf{v}_e^{\text{temp}} \in \mathbb{R}^{1 \times 3} \quad \text{and} \quad s_e^{\text{out}} \overset{W_{e,\text{wsv}}}{\longrightarrow} \text{gate} \in \mathbb{R}^{1} $$$$ \mathbf{v}_e^{\text{out}} = \mathbf{v}_e^{\text{temp}} \odot \sigma(\text{gate}) $$
- **Edge Output Embedding:**  
  $$ (s_e^{\text{out}},\,\mathbf{v}_e^{\text{out}}) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3} $$

---

### **3. Encoder Layers (3 GVP Layers)**

Each encoder layer processes **node embeddings** while using the **edge embeddings** and graph structure. The node embeddings remain in the space  
$$ \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}. $$

For **each encoder layer** $l=1,2,3$:

1. **Message Passing:** For each target node $i$ and each edge $e_{ij}$:
   
   - **Gather:**
     - **Target node:**  $$ (s_i,\,\mathbf{v}_i) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
     - **Edge:**  $$ (s_e,\,\mathbf{v}_e) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3} $$
     - **Source node:**  $$ (s_j,\,\mathbf{v}_j) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
   - **Concatenation:**
     - **Vector Concatenation:**  $$ \mathbf{v} = \bigl[\mathbf{v}_i;\;\mathbf{v}_e;\;\mathbf{v}_j\bigr] \in \mathbb{R}^{33 \times 3} \quad \text{(since }16+1+16=33\text{)} $$
       Compute the vector norm:  $$ n(\mathbf{v}) \in \mathbb{R}^{33}. $$
     - **Scalar Concatenation:**  $$ s = \bigl[s_i;\;s_e;\;s_j\bigr] \in \mathbb{R}^{288} \quad (\text{since }128+32+128=288), $$
       then concatenate with $n(\mathbf{v})$ to obtain: $$ \tilde{s} = [s;\,n(\mathbf{v})] \in \mathbb{R}^{321}. $$
     
   - **Message Function (via GVP):**
     - **Vector branch:**  $$ \mathbf{v} \overset{W_{\text{conv}}.\text{wh}}{\longrightarrow} \mathbf{v}_h \in \mathbb{R}^{33 \times 3} $$
     - **Scalar branch:**  $$ \tilde{s} \overset{W_{\text{conv}}.\text{ws}}{\longrightarrow} s_{\text{msg}} \in \mathbb{R}^{128} $$
     - **Vector output with gating:**  $$ \mathbf{v}_h \overset{W_{\text{conv}}.\text{wv}}{\longrightarrow} \mathbf{v}_{\text{temp}} \quad \text{and} \quad s_{\text{msg}} \overset{W_{\text{conv}}.\text{wsv}}{\longrightarrow} \text{gate} \in \mathbb{R}^{16} $$$$ \mathbf{v}_{\text{msg}} = \mathbf{v}_{\text{temp}} \odot \sigma(\text{gate}) \quad \in \mathbb{R}^{16 \times 3}. $$
     - **Message Output:**  $$ (s_{\text{msg}},\,\mathbf{v}_{\text{msg}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
   - **Aggregation:**  
     Aggregate the messages (e.g., by averaging) from all neighbors to update node $i$.

2. **Feedforward Update:**

   - **Input:** Current node embedding:  $$ x_i = (s_i,\,\mathbf{v}_i) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
   - **LayerNorm:**  $$ x_i \overset{\text{LN}_1}{\longrightarrow} x_i' \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
   - **GVP Feedforward Mapping:**  $$ x_i' \overset{\text{GVP}_{\text{ff}}}{\longrightarrow} \Delta x_i = (\Delta s_i,\,\Delta \mathbf{v}_i) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$
   - **Residual & Dropout:**  $$ x_i^{\text{res}} = x_i + \text{Dropout}(\Delta x_i) $$
   - **Final LayerNorm:**  $$ x_i^{\text{final}} = \text{LN}_2(x_i^{\text{res}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} $$

**Output of Each Encoder Layer:**  
The node embedding remains in  $\mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}$  after every encoder layer.

---

### 4. Overall Pipeline Summary

1. **Input & Featurization:**  
   - **Nodes:** $(64,\,4)$
   - **Edges:** $(32,\,1)$

2. **Initial Embedding:**   $$ 
   \begin{array}{rcl}
   (s_n^{\text{raw}},\,\mathbf{v}_n^{\text{raw}}) \in \mathbb{R}^{64} \times \mathbb{R}^{4 \times 3} 
   &\longrightarrow& (s_n^{\text{out}},\,\mathbf{v}_n^{\text{out}}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} \\
   (s_e^{\text{raw}},\,\mathbf{v}_e^{\text{raw}}) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3} 
   &\longrightarrow& (s_e^{\text{out}},\,\mathbf{v}_e^{\text{out}}) \in \mathbb{R}^{32} \times \mathbb{R}^{1 \times 3}
   \end{array}
   $$
3. **Encoder Layers (3 Layers):**  
   For each layer $l=1,2,3$: $$ (s_i^{(l-1)},\,\mathbf{v}_i^{(l-1)}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3} 
   \quad \xrightarrow{\text{GVP Message Passing + Feedforward}} \quad
   (s_i^{(l)},\,\mathbf{v}_i^{(l)}) \in \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3}. $$
4. **Output:**  
   The final node embeddings are in  $\mathcal{H} \subseteq \mathbb{R}^{128} \times \mathbb{R}^{16 \times 3},$  which are then used for subsequent tasks (e.g., decoding, prediction, etc.).
