
# Few-Shot Relation Classification with DistilBERT and CAVIA

## Complete Codebase

### Data Processing for FewRel

FewRel is a few-shot relation classification dataset of 100 relation types, each with 700 labeled instances. The relations are divided into 64 training, 16 validation, and 20 testing classes for episodic meta-learning. 

Our codebase includes a data pipeline to load and preprocess the FewRel dataset. The data is provided in JSON format (e.g., `train_wiki.json`, `val_wiki.json`) where each relation class is associated with a list of its example sentences. We parse these files and organize the data by class labels for easy sampling of few-shot tasks. The episode sampler generates N-way K-shot tasks for meta-training and evaluation. In each episode:

- **Support set**: N relation classes are chosen from the training set. For each class, K example sentences (with that relation label) are sampled to serve as labeled support data.
- **Query set**: From the same N classes, a certain number of additional examples (unseen in support) are sampled to evaluate the model's ability to classify after adaptation.

This episodic formulation follows the standard few-shot learning setting, ensuring that during meta-testing the model must classify among unseen relation types given only a few examples. The data pipeline shuffles and batches episodes for training. We ensure that tokenization (using the DistilBERT tokenizer) and necessary encoding (e.g., marking entity positions if needed) are applied consistently to both support and query instances.

### Model Architecture with CAVIA Adaptation

Our model uses DistilBERT as a sentence encoder to obtain a fixed-dimensional representation for each sentence. DistilBERT is a distilled version of BERT that is 40% smaller and 60% faster while retaining about 97% of BERT's language understanding performance ([ARXIV.ORG](https://arxiv.org/)). To reduce computational cost, we load a pre-trained `distilbert-base-uncased` model from Hugging Face Transformers and freeze its parameters. This means DistilBERT serves as a static feature extractor, and no gradients are back-propagated through it during training (except in the baseline variant where we fine-tune it). Freezing the large transformer drastically cuts down on GPU memory usage and computational overhead, enabling faster training and adaptation.

On top of the frozen encoder, we implement a lightweight classification head which will be adapted using CAVIA's context parameters. CAVIA (Fast Context Adaptation via Via Intermediate Alignment) is a meta-learning approach that introduces a small set of context parameters for fast task-specific adaptation ([PROCEEDINGS.MLR.PRESS](https://proceedings.mlr.press/)).

We partition the model’s parameters into:

- **Shared parameters ($\mathbf{θ}$)**: The DistilBERT encoder and the main weights of the classification head. These are meta-trained across tasks and are shared by all tasks.
- **Context parameters ($\mathbf{φ}$)**: A small, task-specific parameter vector that is fed as an additional input to the classifier. These are the only parameters updated during task adaptation (inner loop), providing a low-dimensional representation of the task ([PROCEEDINGS.MLR.PRESS](https://proceedings.mlr.press/)).

In our implementation, the classification head is a single fully-connected layer that takes as input the concatenation of the DistilBERT sentence embedding and the context vector $\mathbf{φ}$. For example, if DistilBERT produces a 768-dimensional [CLS] embedding, and we choose a context vector dimension ($d$) of 100, then the classifier will input a 868-dimensional vector $[\text{embedding}; \mathbf{φ}]$ and output logits for the N classes of the task. The context vector is initialized to a zero vector for each new task and treated as a learnable input to the classifier ([ARXIV.ORG](https://arxiv.org/)).

Formally, if $\mathbf{h}$ is the DistilBERT embedding and $\mathbf{φ}$ is the context vector, the classifier computes:

$$
\text{logits} = \mathbf{W} \cdot [\mathbf{h}; \mathbf{φ}] + \mathbf{b},
$$

where $\mathbf{W}$ and $\mathbf{b}$ are the shared weight matrix and bias. During meta-training, $\mathbf{W}$ and $\mathbf{b}$ (along with any DistilBERT parameters if we were fine-tuning it) are in $\mathbf{θ}$ and are not updated in the inner loop, whereas $\mathbf{φ}$ is updated for each task’s support set. This design ensures adaptation is fast and memory-efficient: the model only needs to adjust a small vector $\mathbf{φ}$ for each new task, instead of fine-tuning millions of DistilBERT parameters. As noted by Zintgraf et al., updating only a few input context parameters at test time makes the model far less prone to overfitting on small data, yet still effective at task adaptation ([PROCEEDINGS.MLR.PRESS](https://proceedings.mlr.press/)).

We implement $\mathbf{φ}$ as part of the model (e.g., `self.context = nn.Parameter(torch.zeros(context_dim))` in PyTorch) but we reset or re-initialize it for each episode's inner-loop adaptation so that each task starts with a fresh (or learned initial) context.

In this design, `self.context` represents the task-specific adaptation parameters $\mathbf{φ}$. We treat `num_classes` as the number of classes in a given episode. For meta-training, this would be N (e.g., 5-way training). At meta-test time, if we evaluate on episodes with the same N, the classifier can be reused. (If needed, our code can dynamically resize or create a new classifier for different N, but in our experiments we keep N consistent for train and test episodes for simplicity.) The code sets `requires_grad=False` for all DistilBERT parameters, ensuring they remain fixed. The forward pass concatenates the context vector to the DistilBERT [CLS] embedding before applying the linear layer. This effectively conditions the classifier on the context vector ([ARXIV.ORG](https://arxiv.org/)).

The context vector is small (e.g., 100 dims) and will be updated during the inner-loop adaptation for each task, whereas the `classifier.weight` and `classifier.bias` (and potentially the initial value of context) are updated only in the outer loop across tasks.

### Training Pipeline (Meta-Training with CAVIA)

We implement a meta-learning training pipeline inspired by Model-Agnostic Meta-Learning (MAML) but modified for CAVIA context adaptation. The training loop has two levels:

1. **Inner Loop (Adaptation)**: For each sampled task (episode), the model creates a fast adaptation to that task by updating the context vector $\mathbf{φ}$ using the support set.
2. **Outer Loop (Meta-update)**: After adaptation, the model’s performance is evaluated on the query set of that task, and the gradients from this evaluation update the shared parameters $\mathbf{θ}$ (which include the classifier weights and possibly the initial context values).

During meta-training, for each episode we perform a small number of gradient steps (e.g., 1 to 5 steps) on the support set, but only on the context parameters $\mathbf{φ}$ ($\mathbf{θ}$ remains fixed in inner loop) ([ARXIV.ORG](https://arxiv.org/)). In practice, this means we clone or copy the model’s context vector and do gradient descent on it while keeping the rest frozen. 

In our implementation, we handle this adaptation more elegantly by creating a copy of the model or context as needed. After obtaining `adapted_context` for the task, we evaluate the model on the task’s query set by injecting the adapted context (temporarily replacing the model’s context). We then compute the query loss and backpropagate it through the adaptation step to update the shared parameters. Only the outer loop updates affect θθ (the classifier weights, since DistilBERT is frozen) ([ARXIV.ORG](https://arxiv.org/)).

We optimize the outer loop using an optimizer like Adam. Each outer-loop iteration samples a batch of tasks (episodes) for efficiency. Pseudocode for one meta-training iteration:

During the backward pass, gradients flow into the classifier’s weights (and would flow into DistilBERT if it were trainable, but here it's frozen). The context gradient from the query loss influences the shared parameters through the inner-loop update (higher-order gradients) ([ARXIV.ORG](https://arxiv.org/)). We can use first-order approximation (ignoring second-order gradients) for efficiency if needed, though CAVIA in the original formulation does include higher-order gradients since the context is small.

**Efficiency considerations**: Freezing DistilBERT means we do not compute enormous gradients for it. Moreover, since DistilBERT is fixed, we can cache embeddings for the support and query sentences in each episode to avoid recomputing them multiple times during inner-loop updates. In practice, for each episode we encode all sentences with DistilBERT once, then perform inner-loop updates on φφ using those fixed embeddings. This significantly reduces computation in the inner loop (essentially turning it into a small neural network training problem). Our code provides options to precompute and reuse embeddings in episodes to further speed up training (especially beneficial when multiple inner steps are used). We train for a number of meta-epochs, each consisting of many episodes. The model with best validation performance (on validation episodes) is saved for final evaluation.

### Evaluation and Few-Shot Metrics

After meta-training, we evaluate on the test set of FewRel (20 unseen relations) by conducting few-shot episodes. During evaluation, only the context vector is adapted for each test episode, while the shared weights remain as learned. We run a small number of gradient steps on the support set of the episode to obtain φφ for that task, then evaluate on the query set and record the accuracy and F1 score. We report standard few-shot classification metrics:

- **Accuracy**: the percentage of query examples correctly classified in their episode.
    
- **Macro F1-score**: since each episode is balanced across classes, we compute the F1 per class and average, or equivalently use the overall precision/recall for the episode. This is useful to account for any imbalance and to average across episodes.
    
- **Adaptation speed**: we measure how many gradient steps are needed to achieve a certain performance on a new task. In practice, we might evaluate the model's query accuracy after 0, 1, 2, ... inner-loop updates to see how quickly it improves with more support training.
    

The codebase includes an evaluation script that can run a series of episodes (with specified N-way, K-shot) and compute average accuracy and F1. We ensure the evaluation uses deterministic adaptation (no meta-learning, just applying the learned strategy) to each task separately.

### Fine-Tuned DistilBERT Baseline

In addition to the CAVIA-based approach, we provide a baseline that uses full fine-tuning of DistilBERT for few-shot tasks. This baseline helps contrast the efficiency and performance of context-based adaptation against conventional fine-tuning. For the baseline, we follow a two-phase approach:

1. **Pre-training on base relations**: We fine-tune DistilBERT (with a classification layer) on all 64 training relations in a standard supervised way. Essentially, we treat it as a 64-way classification problem and train until convergence (or use the provided training instances). This gives the model a good initial representation for relations, analogous to how one might pre-train on base classes before few-shot adaptation ([ACLANTHOLOGY.ORG](https://aclanthology.org/)). The classification head here is a linear layer mapping DistilBERT [CLS] embedding to 64 class logits.
    
2. **Few-shot adaptation**: To evaluate on a new N-way K-shot task (with unseen classes), we initialize the model weights from the pre-trained model and fine-tune on the support set of the task. In this adaptation, all model parameters are allowed to update (DistilBERT and the classifier) using the support examples. We typically only do a few gradient steps or epochs given K is very small (to avoid overfitting). Then we evaluate on the query set.
    

We also implement a variant of the baseline where we only fine-tune the classifier layer on the support set (keeping DistilBERT frozen) to isolate the effect of large-scale fine-tuning vs. feature reuse. This acts as a “feature-based” baseline, which sometimes is more stable when K is extremely low. The baseline code leverages the same data pipeline and evaluation logic for fairness. We simply replace the adaptation procedure: instead of updating a context vector, we reload the saved DistilBERT model and run a typical training loop on the support set. We use early stopping or a fixed small number of epochs (since support set is tiny) and then evaluate on queries.

By including this baseline, the codebase allows easy comparison of:

- Accuracy and F1 of CAVIA (frozen encoder, context adaptation) vs. Full Fine-Tuning (unfrozen encoder).
    
- The number of parameters updated during adaptation (CAVIA updates a few hundred parameters vs. ~66 million in DistilBERT).
    
- The time taken for adaptation on a single task (CAVIA’s few steps on a small vector vs. backpropagating through the whole transformer).
    

### Repository Structure and Modular Design

The project is organized into a modular structure for clarity and maintainability:

- **Data Processing**: Handles loading and preprocessing of FewRel dataset.
    
- **Model Architecture**: Implements DistilBERT with CAVIA adaptation.
    
- **Training Pipeline**: Manages meta-training with CAVIA.
    
- **Evaluation**: Includes scripts for few-shot evaluation and metrics computation.
    
- **Baseline**: Provides the fine-tuning baseline for comparison.
    

### Experimentation Plan

To thoroughly evaluate the Few-Shot Relation Classification models and demonstrate the advantages of CAVIA-based adaptation, we propose the following experimentation plan:

1. **Different Few-Shot Settings**: Evaluate the models under various N-way, K-shot scenarios. For instance:
    
    - 5-way 1-shot vs 5-way 5-shot: This tests how the models perform with extremely limited data versus a slightly less limited scenario. We expect both methods to improve with more shots, but the gap between CAVIA and the baseline might change.
        
    - 10-way 5-shot (or 10-way 1-shot): A harder task with more classes. This will test the scalability of the adaptation – with more classes, the context vector and classifier have to account for a larger decision space. We can compare if the baseline fine-tuning degrades more significantly than CAVIA in this setting.
        
2. **Dataset Splits**: Use the standard train/val split for model development, but also test on the official hidden test set if available (via the FewRel evaluation server) to report final results. Additionally, as a robustness check, one could create a cross-domain evaluation using FewRel 2.0 (if available) to see how the adaptation generalizes to a different domain (though this is optional and more advanced).
    
3. **Hyperparameter Tuning**: Perform systematic tuning of key hyperparameters using the validation set:
    
    - Inner-loop learning rate (for φφ updates) – too high might overshoot given very few examples, too low might underfit.
        
    - Number of inner-loop adaptation steps – e.g., test 1 vs 5 steps. CAVIA is expected to do well with even 1 step (fast adaptation), but perhaps a couple more could fine-tune φφ better. However, too many steps might overfit the support set; we will observe the effect.
        
    - Context vector size (dimension of φφ) – this controls the capacity of the task-specific adaptation. We try small (e.g., 5 or 10) vs larger (100, 300). A larger context can encode more task info but also means more parameters to meta-learn (and potentially risk of overfitting if too large) ([ARXIV.ORG](https://arxiv.org/)). We expect an optimal intermediate size.
        
    - Meta batch size and outer-loop learning rate – to ensure stable meta-training.
        
4. **Baseline Comparisons**: We will compare the following models:
    
    - CAVIA (Ours): DistilBERT frozen, context-adaptive classifier.
        
    - Full Fine-Tune Baseline: DistilBERT fine-tuned on support (the one described).
        
    - Partial Fine-Tune Baseline: DistilBERT frozen at adaptation, only classifier fine-tuned on support (a simpler transfer learning baseline).
        
    - MAML-like fine-tuning (if time permits): For completeness, one could also try a MAML approach on DistilBERT (allowing some or all weights to adapt in inner loop) to see if CAVIA’s restriction truly helps. However, this is computationally heavy. Instead, we rely on reported findings that CAVIA outperforms MAML on classification ([PROCEEDINGS.MLR.PRESS](https://proceedings.mlr.press/)), and focus our comparisons on (1)-(3).
        
5. **Ablation Studies**:
    
    - Context Vector Size: As mentioned, test small vs large context dimension. If too small, the model may not adapt well (underfits task differences); if too large, it may overfit support or make outer-loop training harder. This ablation will confirm the sensitivity of the method to this hyperparameter.
        
    - Number of Adaptation Steps: Try 0 (i.e., no inner-loop update, just relying on initial φφ) vs 1, 5 steps. 0 steps essentially tests the quality of the meta-learned initialization of φφ ([ARXIV.ORG](https://arxiv.org/)). We expect 1 or a few steps to significantly boost performance over 0, demonstrating the benefit of fast adaptation. More than a few steps might not yield further gains if the model already fits the support set well.
        
    - Freezing vs Fine-tuning DistilBERT: Although our main method freezes DistilBERT, one ablation could be to allow DistilBERT’s last layer to update in inner loop or outer loop to see if that improves performance. This is somewhat contrary to our goal of minimal GPU usage, but it provides insight. We hypothesize that freezing does not hurt much due to DistilBERT’s strong representations, and fine-tuning it on so few examples might overfit. This ablation could confirm that.
        
    - Support set size (K): Evaluate performance as K varies (1, 5, 10). We expect both methods to improve with more support data, but observe whether CAVIA’s advantage is most pronounced in the extremely low-data regime (K=1) where full fine-tuning is most difficult.
        

For each experiment, we will use the validation set for tuning and then report final results on the test set. We will ensure multiple runs (with different random seeds) for reliability, especially since few-shot results can have high variance. We will also measure standard deviation of accuracy across episodes to gauge consistency. Additionally, to measure efficiency, we can log the time taken per training episode and per adaptation at test. CAVIA’s updates involve a small vector and should be very fast, whereas the baseline must compute gradients for the whole DistilBERT. We can report the average adaptation time per task for both approaches (e.g., in milliseconds) to highlight the computational benefit.

### Report Guidelines

When writing the report for this project, we will include the following key elements to ensure it is comprehensive and clear:

1. **Methodology Description**: A concise recap of the approach, including a diagram if possible. For instance, a figure illustrating the model architecture (DistilBERT feeding into the classifier with the context vector) and the meta-learning loop would be helpful for readers to visually grasp CAVIA’s mechanism. We will also briefly describe FewRel and the few-shot task setup so the context is clear.
    
2. **Quantitative Results**: We will present tables of results comparing the CAVIA-based model and the fine-tuning baseline:
    
    - Main table for accuracy and F1 scores on the few-shot tasks (e.g., 5-way 1-shot, 5-way 5-shot, 10-way 5-shot). Each cell can show mean ± std over multiple runs or episodes. This will directly answer how the methods stack up. For example:
        
        |Model|5-way 1-shot Acc.|5-way 5-shot Acc.|5-way 1-shot F1|5-way 5-shot F1|
        |---|---|---|---|---|
        |CAVIA (DistilBERT)|XX%|YY%|XX%|YY%|
        |Fine-tune DistilBERT|ZZ%|WW%|ZZ%|WW%|
        |Frozen BERT + classifier|UU%|VV%|UU%|VV%|
        
        (Values are placeholders – actual numbers will be filled in from experiments.)
        
    - We will highlight in the text the improvement of CAVIA over the baseline (e.g., “CAVIA yields a +5% absolute accuracy gain in 5-way 1-shot over full fine-tuning, while also being 5× faster in adaptation.”).
        
3. **Adaptation Speed Analysis**: A figure plotting query accuracy vs. number of adaptation steps for CAVIA versus baseline. For CAVIA, this might be accuracy after 0, 1, 2, 3 inner-loop updates. For baseline, we can plot accuracy after a certain number of fine-tuning gradient steps on support (or after each epoch if we treat each support fine-tune as one epoch). This plot will illustrate how quickly each approach adapts. We expect to see CAVIA start at a decent accuracy even at 0 steps (due to meta-learned φφ initialization) and reach high accuracy after 1-2 steps, whereas the baseline might start low and improve slowly or even degrade if overfitting. This addresses the “fast adaptation” claim visually.
    
4. **Ablation Study Results**: We will include a small table or bar chart for the ablation experiments:
    
    - Context dimensionality vs. performance: e.g., a bar chart of accuracy for context dim = 0 (no context, essentially no adaptation) vs 10 vs 50 vs 200. This would show the sweet spot where performance peaks.
        
    - Inner-loop steps vs performance: a line chart perhaps redundant with the adaptation speed figure, or a table showing final accuracy using 0,1,5 steps.
        
    - Perhaps a comparison of using different amounts of DistilBERT fine-tuning (frozen vs last-layer vs full) to show that freezing works well for our scenario (if we did that experiment).
        
    
    Each ablation result will be briefly discussed to provide insight (e.g., “We found that increasing context size beyond 100 did not yield further gains, suggesting diminishing returns in representational capacity for φφ.”).
    
5. **Qualitative Analysis**: Although relation classification is primarily quantitative, we will try to provide a qualitative perspective:
    
    - We can give an example of a support set and a difficult query, and discuss how the models performed. For instance, if the relation is `member_of` vs `capital_of`, and the query sentence is tricky, did the CAVIA-adapted model correctly identify the relation whereas baseline did not? This can be illustrated by showing the sentence and the predicted label by each model. It helps readers understand what errors are being made. Perhaps the baseline might confuse relations if it didn't adapt well, whereas CAVIA's adaptation helped it focus on the correct relation.
        
    - If possible, we could visualize the learned context vector (though 100-d is hard to interpret directly). Alternatively, we could project context vectors of different tasks (e.g., via PCA) to see if the meta-learning placed them in a sensible space (this is a bit advanced, but could be interesting to show how tasks relate in the context space).
        
    - We will also note any observation such as: “the baseline often overfits on the support examples – e.g., sometimes predicting the relation that appeared in support even when query sentence was of a different relation – while CAVIA, by only adjusting a small context, tended to be more restrained and generalize better.”
        
6. **Efficiency and Training Curve**: We can include a brief section or figure on efficiency. Possibly a table of training time per episode or total training time for CAVIA vs baseline. And maybe a plot of meta-training progress: e.g., meta-validation accuracy vs training episodes for CAVIA and for baseline (if baseline is also trained episodically or just final fine-tune). This could show that CAVIA converges quickly. If our meta-training loss curve is informative, we might show that as well.
    
7. **Discussion**: In the report, we’ll discuss the results, emphasizing:
    
    - CAVIA’s strong performance and quick adaptation relative to the heavy fine-tuning approach, corroborating the claim that updating only context parameters (task embeddings) at test time is effective ([PROCEEDINGS.MLR.PRESS](https://proceedings.mlr.press/)).
        
    - The scenarios where the baseline might catch up (for example, with more shots, fine-tuning might narrow the gap).
        
    - Limitations: e.g., if context is too small, tasks that are very dissimilar might not adapt perfectly; or if tasks require nuanced language understanding, a frozen encoder might miss out (though DistilBERT is quite strong).
        
    - Potential improvements or future work: using a better way to incorporate context (perhaps adding it at multiple layers, similar to FiLM conditioning ([ARXIV.ORG](https://arxiv.org/)), although we kept it simple), or trying this approach on other few-shot NLP tasks.
        
8. **Figures and Tables**: We will ensure all figures and tables are properly labeled and referenced in the text. A final figure might visually summarize the comparison: for example, a bar chart of overall performance (CAVIA vs Baseline) and a note about parameter counts or adaptation time. All tables will have captions explaining what they show. Where appropriate, we will cite relevant work (for instance, citing the FewRel paper for human performance or prior state-of-art, and citing the CAVIA paper to reaffirm why our approach was expected to work).
    

By following these guidelines, the report will clearly communicate the implementation details, experimental results, and implications of using a CAVIA-based context adaptation approach versus a traditional fine-tuning approach for few-shot relation classification. The combination of quantitative metrics, visualizations, and explanation will provide a thorough understanding of the system’s performance and characteristics. The end result will demonstrate a compute-efficient yet competitive solution for few-shot learning on the FewRel benchmark, aligning with the expectations of fast adaptation and strong results with minimal intervention.