### **GraphAttentionLayer (Processes One Conformation)**

````python
GraphAttentionLayer(s, v):
    # Input:
    #   s: scalar features, shape = [n_nodes, d_s]
    #   v: vector features, shape = [n_nodes, d_v, 3] (or None)

    if v is not None:
        # Compute L2 norm of each vector channel for each node
        v_norm = L2_norm(v, axis=2)        # shape = [n_nodes, d_v]
        # Concatenate scalar features with vector norms
        combined_features = concat(s, v_norm)  # shape = [n_nodes, d_s + d_v]
    else:
        combined_features = s

    # Apply LayerNorm before computing attention (optional)
    if norm_first:
        combined_features = LayerNorm(combined_features)

    # Compute query (Q), key (K) from combined features
    Q = Linear_Q(combined_features)        # shape = [n_nodes, n_heads * head_dim]
    K = Linear_K(combined_features)        # shape = [n_nodes, n_heads * head_dim]
    # For values (V) use only scalar features
    V = Linear_V(s)                        # shape = [n_nodes, n_heads * d_s]

    # Reshape and permute for multi-head attention:
    Q = reshape_and_permute(Q)             # shape = [n_heads, n_nodes, head_dim]
    K = reshape_and_permute(K)             # shape = [n_heads, n_nodes, head_dim]
    V = reshape_and_permute(V)             # shape = [n_heads, n_nodes, d_s]

    # Scaled dot-product attention:
    scores = softmax((Q dot_transpose(K)) * scale)  # shape = [n_heads, n_nodes, n_nodes]
    scores = Dropout(scores)                         # Apply dropout to attention weights
    attn_output = scores dot V                       # shape = [n_heads, n_nodes, d_s]

    # Combine heads (either via averaging or concatenation then project)
    h = combine_heads(attn_output)         # shape = [n_nodes, d_s]

    # Final projection back to the scalar feature dimension
    s_out = Linear_O(h)                    # shape = [n_nodes, d_s]

    # Apply residual connection and LayerNorm (if norm_first=False)
    s_out = s + Dropout(s_out)  # Residual connection
    if not norm_first:
        s_out = LayerNorm(s_out)

    # Return updated scalars along with unchanged vector features
    return (s_out, v)

````

### **MultiAttentiveGVPLayer (Processes Multiple Conformations)**

````python
MultiAttentiveGVPLayer(x, edge_index, edge_attr):
    # Input:
    #   x = (s, V)
    #       s: scalar features, shape = [n_nodes, n_confs, d_s]
    #       V: vector features, shape = [n_nodes, n_confs, d_v, 3]
    #   edge_index, edge_attr: needed for message passing

    # Apply LayerNorm first if norm_first is True
    if norm_first:
        s, V = LayerNorm(s, V)

    # 1. GVP Branch:
    (conv_s, conv_v) = MultiGVPConvLayer(x, edge_index, edge_attr)
    # conv_s: [n_nodes, n_confs, d_s]
    # conv_v: [n_nodes, n_confs, d_v, 3]

    # 2. Attention Branch:
    Initialize an empty list: attn_outputs

    For each conformation index i in 0 to (n_confs - 1):
        # Extract features for conformation i
        s_i = s[:, i, :]          # shape = [n_nodes, d_s]
        v_i = V[:, i, :, :]       # shape = [n_nodes, d_v, 3]

        # Process using GraphAttentionLayer:
        (attn_s_i, _) = GraphAttentionLayer(s_i, v_i)
        # attn_s_i: [n_nodes, d_s]

        Append attn_s_i to attn_outputs

    End For

    # Stack the outputs along the conformation dimension:
    attn_s = stack(attn_outputs)   # shape = [n_nodes, n_confs, d_s]

    # 3. Fusion:
    combined_s = 0.5 * conv_s + 0.5 * attn_s   # shape = [n_nodes, n_confs, d_s]
    combined_v = conv_v                        # (retain the vector output from the GVP branch)

    # Apply dropout:
    combined_s, combined_v = Dropout(combined_s, combined_v)

    # Apply residual connection:
    s = s + combined_s
    V = V + combined_v

    # Apply LayerNorm after if norm_first is False
    if not norm_first:
        s, V = LayerNorm(s, V)

    # 4. Return the fused outputs
    Return (s, V)
````


### **GraphAttentionDecoderLayer**

````python
GraphAttentionDecoderLayer(s, v, edge_index, edge_attr, autoregressive=False):
    # Inputs:
    #   s: scalar node features, shape = [n_nodes, d_s]
    #   v: vector node features, shape = [n_nodes, d_v, 3] (or None)
    #   edge_index: connectivity, shape = [2, n_edges]
    #   edge_attr: edge features, shape = [n_edges, d_e]
    #   autoregressive: boolean flag for causal masking (for decoding)

    # --- Node Feature Processing ---
    if v exists:
        v_norm = L2_norm(v, axis=2)           # [n_nodes, d_v]
        node_input = concat(s, v_norm)         # [n_nodes, d_s + d_v]
    else:
        node_input = s

    # Compute node projections:
    Q = Linear_Q(node_input)                  # [n_nodes, n_heads * head_dim]
    K = Linear_K(node_input)                  # [n_nodes, n_heads * head_dim]
    V = Linear_V(s)                           # [n_nodes, n_heads * d_s]

    # --- Edge Feature Processing ---
    # Project edge features separately:
    Q_e = Linear_Q_edge(edge_attr)            # [n_edges, n_heads * head_dim_e]
    K_e = Linear_K_edge(edge_attr)            # [n_edges, n_heads * head_dim_e]
    V_e = Linear_V_edge(edge_attr)            # [n_edges, n_heads * d_e]

    # --- Reshape and Permute for Multi-Head Attention ---
    Q, K, V = reshape_and_permute(Q, K, V)      # each → [n_heads, n_nodes, head_dim]
    Q_e, K_e, V_e = reshape_and_permute(Q_e, K_e, V_e)  # each → [n_heads, n_edges, head_dim_e]

    # --- Compute Attention Scores ---
    scores_node = (Q dot_transpose(K)) * scale         # [n_heads, n_nodes, n_nodes]
    scores_edge = (Q dot_transpose(K_e)) * scale         # [n_heads, n_nodes, n_edges]

    # --- Apply Causal Masking (if autoregressive) ---
    if autoregressive:
        scores_node = apply_causal_mask(scores_node)
        scores_edge = apply_causal_mask(scores_edge)

    # --- Compute Attention Weights and Outputs ---
    node_attn = softmax(scores_node) dot V               # [n_heads, n_nodes, d_s]
    edge_attn = softmax(scores_edge) dot V_e               # [n_heads, n_nodes, d_e]

    # --- Combine Outputs ---
    combined_node = combine_heads(node_attn + edge_attn)   # [n_nodes, d_s] (after combining heads)
    updated_s = Linear_O(combined_node)                    # Project back to [n_nodes, d_s]

    # --- Residual Connection, Dropout, and LayerNorm on Node Features ---
    s_out = LayerNorm(s + Dropout(updated_s))              # [n_nodes, d_s]

    # --- Update Edge Features ---
    # For each edge, combine its original features with information from its source and destination nodes.
    new_edge_input = concat(edge_attr, s[source], s[dest]) # [n_edges, d_e + 2*d_s]
    updated_edge = Linear_E(new_edge_input)                # [n_edges, d_e]
    edge_out = LayerNorm(edge_attr + Dropout(updated_edge))  # [n_edges, d_e]

    # --- Return Updated Features ---
    return (s_out, v), edge_out

````


### **TensorMomentPooling**
````python
TensorMomentPooling(X, p, psi):
    # Input:
    #   X: tensor of shape [n_nodes, C, d]
    #        where each node i has C conformation embeddings of dimension d.
    #   p: maximum order of tensor moments (e.g., p = 2 or 3)
    #   psi: an MLP that processes the concatenated moment vector
    # Output:
    #   output: tensor of shape [n_nodes, output_dim] (set-level representation)

    Initialize an empty list: moments_list

    For order in 1 to p:
        If order == 1:
            # First moment: simply sum over conformations
            moment_1 = sum(X, axis=1)          # shape: [n_nodes, d]
            Append moment_1 to moments_list
        Else:
            # For higher orders, compute the outer product for each conformation
            # and then sum them over the set.
            moment_order = 0
            For each conformation index i from 0 to C-1:
                # Extract X_i for each node: shape [n_nodes, d]
                X_i = X[:, i, :]
                # Compute the tensor (outer) product of X_i with itself, order times.
                tensor_i = outer_product(X_i, order)  
                # tensor_i has shape [n_nodes, d, d, ..., d] (order copies)
                # Flatten tensor_i to shape [n_nodes, d^order]
                tensor_i_flat = flatten(tensor_i)   
                moment_order = moment_order + tensor_i_flat
            Append moment_order to moments_list

    # Concatenate all moments along the feature dimension:
    Phi = concatenate(moments_list, axis=-1)
         # Phi has shape [n_nodes, d + d^2 + ... + d^p]

    # Process the aggregated moments with MLP ψ:
    output = psi(Phi)      # shape: [n_nodes, output_dim]

    Return output
````

