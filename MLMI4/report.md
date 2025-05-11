## Classification

| Method                       |    **5-way accuracy**    |    **5-way accuracy**     |     |
| ---------------------------- | :----------------------: | :-----------------------: | --- |
|                              |        **1-shot**        |        **5-shot**         |     |
| **MAML (32)**                |                          |                           |     |
| **MAML (64)**                |                          |                           |     |
| **CAVIA (32)**               | **45.94% ± 0.84%** (60k) |  **59.48% ± 0.8%** (20k)  |     |
| **CAVIA (128)**              | **45.92% ± 0.86%** (10k) | **62.795% ± 0.79%** (10k) |     |
| **CAVIA (512)**              |                          |  **54.33% ± 0.76%** (1k)  |     |
| **CAVIA (512, first order)** | 48.491 (+/- 0.9) (14700) |  **49.18% ± 0.78%** (2k)  |     |
|                              |                          |                           |     |
classification - multimodal data - e.g. names to language origins 
regression - follow a polynomial function sampling the degree and polynomial 
celabe ablation studies - MAML


cavia (32) -> 60k iterations but maybe we turn to something like 30k now for the others 


> **Experimental Setup:**  
> We ran **few-shot classification** experiments on the **Mini-Imagenet** dataset to compare **MAML** and **CAVIA**. Each model was trained on 5-way tasks (1-shot or 5-shot) for 60,000 meta-iterations, using 2 inner-loop gradient steps, an inner learning rate of 0.1, and an outer learning rate of 0.001. We varied the convolutional backbone from 32 to 512 filters and, for CAVIA, included a 100-dimensional context vector to be adapted per task.

> **HPC Execution:**  
> The jobs were submitted on an HPC cluster via a Slurm script, which handled the environment setup (e.g., conda activation), logging, and the training process. After completion, we evaluated each checkpoint on 1,000 randomly sampled tasks from the test set, recording final accuracies in the table below. This setup replicates the methodology in the original CAVIA paper and provides a direct comparison between MAML and CAVIA under varying model capacities.