This repository is the official implementation for experiments on PointNet++ in [A Unified Game-Theoretic Interpretation of Adversarial Robustness](https://arxiv.org/abs/2103.07364).

### Pre-trained models and the dataset

We used the pre-trained PointNet++ on the ModelNet40 dataset, which was saved in "./model_best.t7". The dataset should be saved as .npy in 'data/modelnet40_numpy/'

### Usage
1. Generate adversarial examples on the pre-trained model

   ```python
   python gene_adv_imgs.py --root=<path_to_data> --model_path=<path_to_model>
   ```

2. Sample variable pairs and contexts for the computation of interactions

   ```python
   python gene_pairs.py --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

3. Compute the model output on masked samples

   ```python
   python m_order_interaction_logit.py --root=<path_to_data> --model_path=<path_to_model> --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

4. Compute interactions based on the model output

   ```python
   python compute_interactions.py --root=<path_to_data> --model_path=<path_to_model> --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

5. Plot the average interaction in normal samples and adversarial examples, just like Figure 3 in the paper.

   ```
   python plot.py  --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```
