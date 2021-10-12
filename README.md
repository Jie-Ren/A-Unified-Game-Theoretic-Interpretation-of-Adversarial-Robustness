This repository is the official implementation of [A Unified Game-Theoretic Interpretation of Adversarial Robustness](https://arxiv.org/abs/2103.07364). 

### Requirements

```
pip install -r requirements.txt
```

### Pre-trained Models

We used the pre-trained models on the ImageNet dataset released by https://github.com/microsoft/robust-models-transfer.

### Usage

Here, we take the untargeted PGD attack on the standard ResNet-18 as an example.

1. Generate adversarial examples on the pre-trained model

   ```python
   python gene_adv_imgs_untarget.py --root=<path_to_data> --dataset_name=imagenet --arch=resnet18 --std_model_path=<path_to_model>
   ```

2. Sample variable pairs and contexts for the computation of interactions

   ```python
   python gene_pairs.py --arch=resnet18 --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

3. Compute the model output on masked samples

   ```python
   python m_order_interaction_logit.py --root=<path_to_data> --dataset_name=imagenet --arch=resnet18 --std_model_path=<path_to_model> --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

4. Compute interactions based on the model output

   ```python
   python compute_interactions.py --root=<path_to_data> --dataset_name=imagenet --arch=resnet18 --std_model_path=<path_to_model> --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

5. Plot the average interaction in normal samples and adversarial examples, just like Figure 3 in the paper.

   ```
   python plot.py --arch=resnet --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
   ```

   