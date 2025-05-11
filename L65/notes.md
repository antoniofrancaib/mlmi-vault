Deep Research notes: https://chatgpt.com/share/67ae362f-b7e4-8007-9b53-07f45d5bc539
https://chatgpt.com/share/67b5e56d-a138-8007-88de-45655abddc14

## teoria: 
representing RNA structures as graphs (e.g., loop-trees via adjacency matrices) and analyzing how the **encoder's Lipschitz constant**—derived from the operator norms of its weight matrices—controls the mapping from similar input structures to similar latent representations. The idea is to approximate complex activations (like ELU or SILU) with ReLUs, so that the worst-case scenario is governed by the operator norms, especially when ignoring the negative axis contributions. By examining different graph types (and their corresponding operator norms) and encoder architectures (GCNs, GATs, specialized ones like gRNAde), one can derive bounds and even a **table of Lipschitz constants**, which may suggest modifications to enhance the encoder's stability and robustness in handling various RNA graph representations.
this is good starting point: 
https://chatgpt.com/c/67b38eb9-201c-8004-bd40-148eb89ed9e8
https://chat.deepseek.com/a/chat/s/ab0a2e0e-8193-4f54-aa0f-f6141a47b1ec

check this paper and try to generalize to biomolechules geom graphs: [https://proceedings.mlr.press/v119/maron20a/maron20a.pdf](https://proceedings.mlr.press/v119/maron20a/maron20a.pdf)

## experimental:
- **different architectures** instead of GVP, more ***expressive*** (simpler or more complex?) -> ask Deep Research about overall overview of architectures 
	- IDEAS: including Higher-Order Tensors (curvature, chirality, and more complex geometric properties) with GCP layers instead of GVP, 

- attention-based pooling is nice! this might add value!
- RNA folding thermodynamics (add in the loss or maybe some RL approach). good! -- not related with the coursework so might not be the best thing ever 

- Check llms SOTA sampling techniques! and implement them - highly relevant is this!

training time takes no more than 12 hours, with the reduced dataset it should be less than (maybe 3 he said) 6?


-
- dynamical systems study -> see the gnn as a dyn sys

- add one-hot encoding to discern bt protein and rna
- add dummy variables to expand dimensions so they match the rna dimensions (or feature space)
- add a value that is obviously different to the values that rna features have so it is able to recognize the protein as sth different
- mirar en el dataset cuáles de los forbidden sequences make more sense in our context -- see what makes more sense -- physics-informed networks -- biology-informed networks -- boundary condition is the motif itself
- mirar que se cumpla en el dataset original
	- lexicographiic ordering -- intentar generar triángulos en meshes -- misma idea


introduction: 
(Context)
why inverse folding problem is relevant 

(Contributions)
itemize: 
- SOTA sampling techniques from LLMs,; min-p + beam-search ( add briefly why ), greedy decoding no maximiza la seq prob 
- library forbidden y undesignable motifs informan el decoding basado en biological plausibility 
- añadimos la proteina con un binary feature - añadir contexto 
- encoder: k es arbitrario, encoder mas general - todos nodos se pueden comunicar pero a la vez respetando la estructura de GVP - atencion entre nodos y luego al final en vez de un naive pooling -> cross attention pooling
- Lipschitz constant para motivar la arquitectura vs naive gRNAdey extreme cases 