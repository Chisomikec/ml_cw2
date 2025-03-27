# Active Learning with TypiClust

This repository contains an implementation of the TypiClust (TPC(RP)) active learning algorithm, along with additional experiments that compare the original vs. modified (Gaussian-based) typicality measures. The code is designed to run on the CIFAR-10 dataset and supports:

- Fully supervised learning  
- Self-supervised learning (via a pretrained SimCLR model)  
- Semi-supervised learning with fixed-budget or iterative training  

---

## File Overview

### `al_strategies.ipynb`
This is the main script used to reproduce different active learning experiments. It includes:

- Fully supervised active learning loops  
- Self-supervised active learning using SimCLR  
- Semi-supervised fixed-budget and iterative experiments  
- A toggle for using the original or Gaussian-based typicality in TPC(RP)

### `simclr_cifar-10.pth\simclr_cifar-10.pth`
Pretrained SimCLR model used to extract embeddings during self-supervised experiments.

### `model_best.pth`
Classifier checkpoint. 

### `resnet_cifar.py`
Contains a ResNet-18 architecture adapted for CIFAR-10.

---

##  Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Chisomikec/ml_cw2.git
cd ml_cw2
```
### 2. Run the Notebook

Ensure you have installed all requirements using `pip install -r requirements.txt`
Make sure you have Jupyter and `ipykernel` installed:

```bash
pip install notebook ipykernel
```
Open `al_strategies.ipynb` in **Jupyter Notebook** or **VS Code** and run the notebook cells sequentially to reproduce the experiments.

---

## Model Checkpoints

- **SimCLR Pretrained Model**: Ensure `simclr_model/simclr_model.pth` exists before running self-supervised experiments.
- **Classifier Model**: `model_best.pth` may be used for continued training or evaluation *(optional)*.

---

## Switching Between Original and Modified TPC(RP)

To switch from the original inverse-distance typicality measure to the Gaussian-based version, update the following lines in `al_strategies.ipynb`:

```python
# Original:
typicality_vals = compute_typicality_sklearn(...)
best_idx = max(candidates, key=lambda g_idx: typicality_vals[idx2row[g_idx]])

# Modified:
scores = compute_gaussian_density(...)
best_idx = max(candidates, key=lambda g_idx: scores[idx2row[g_idx]])
```
Uncomment the version you want to use.

---

## Results
The notebook generates:

- Accuracy vs. Labeled Sample Count plots
- Semi-supervised accuracy under a fixed budget (e.g. B=10)
- Optional statistical comparisons (e.g., t-tests between methods)
- 
---

## Notes
- Hyperparameters like learning rate, k (neighbors), and α (Gaussian smoothing) may need tuning for best results.
- Update paths to `simclr_model.pth` and `model_best.pth` if you relocate these files.
- Default device is `cpu`
