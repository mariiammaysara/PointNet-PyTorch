# PointNet-PyTorch

A modular PyTorch reimplementation of **PointNet** (Qi et al., 2017), built entirely from scratch, for 3D point cloud classification on ModelNet40.

PointNet solves a deceptively hard problem: point clouds are unordered sets of 3D coordinates, so any model consuming them directly needs to be invariant to the order in which points are given, and robust to rigid transformations of the input. PointNet solves this with a simple but elegant combination of shared per-point MLPs, learned alignment networks (T-Nets), and a symmetric max-pooling operation to aggregate a global shape descriptor. I implemented it as a foundational building block for [Spatialize](https://github.com/mariiammaysara/Spatialize), my text-to-3D spatial reasoning project.

## Architecture

```
Input points (B, N, 3)
        │
        ▼
   Input T-Net (3x3) ──► align input to canonical pose
        │
        ▼
  Shared MLP (3 → 64 → 64)
        │
        ▼
  Feature T-Net (64x64) ──► align feature space
        │
        ▼
  Shared MLP (64 → 64 → 128 → 1024)
        │
        ▼
   Max Pooling (symmetric, permutation-invariant)
        │
        ▼
  Global feature (1024-dim)
        │
        ▼
  FC Classifier (1024 → 512 → 256 → num_classes)
```

- **Shared MLPs** are implemented as `Conv1d` layers applied identically to every point (equivalent to a per-point fully connected layer with shared weights).
- **T-Nets** are small sub-networks that predict a transformation matrix from the data itself, initialized to predict the identity matrix so training starts stable.
- **Max pooling** over the point dimension is the symmetric function that makes the whole network invariant to input point order.
- A regularization term (`||I - AAᵀ||²`) is added to the loss to keep the feature transform matrix close to orthogonal.

## Experiments (Ablations)

To understand why each component of PointNet matters, not just implement it:

| Variant | Test Accuracy |
|---|---|
| Full model | [FILL IN] |
| Without input T-Net | [FILL IN] |
| Without feature T-Net | [FILL IN] |
| Average pooling instead of max pooling | [FILL IN] |
| Without data augmentation | [FILL IN] |
| Without orthogonal regularization | [FILL IN] |

## Results

| Metric | Value |
|---|---|
| Test accuracy (ModelNet40, 40 classes) | [FILL IN] |
| Best-performing classes | [FILL IN] |
| Hardest classes | [FILL IN] |
| Training time | [FILL IN] (Colab T4) |
| Epochs | [FILL IN] |

Confusion matrix: `results/confusion_matrix.png`

## Project Structure

```
pointnet-pytorch/
├── data/                  # ModelNet40 (downloaded, not committed)
├── models/
│   └── pointnet.py        # T-Net + PointNetClassifier
├── dataset.py              # ModelNet40 loading, sampling, augmentation
├── train.py                 # training loop
├── evaluate.py               # evaluation + per-class accuracy + confusion matrix
├── utils.py                   # regularization loss, helpers
├── config.py                   # hyperparameters
└── tests/                       # shape/sanity tests for T-Net and model
```

## How to Run

**Setup**
```bash
git clone https://github.com/mariiammaysara/PointNet-PyTorch.git
cd PointNet-PyTorch/pointnet-from-scratch
pip install -r requirements.txt
```

**Training (recommended: Colab, GPU)**
```python
!git clone https://github.com/mariiammaysara/PointNet-PyTorch.git
%cd PointNet-PyTorch/pointnet-from-scratch
!pip install -r requirements.txt
!python train.py --epochs 100 --batch_size 32
```

**Evaluation**
```bash
python evaluate.py
```

## Implementation Notes

- **Identity initialization in T-Net**: the last FC layer's weights are zeroed and its bias set to the flattened identity matrix, so both T-Nets start out predicting "no transformation." Without this, training is noticeably less stable early on.
- **Regularization weight**: the feature transform regularization term is scaled by `0.001` per the paper — too high and it dominates the classification loss, too low and the 64×64 transform can drift.
- **Data augmentation**: random rotation around the Y-axis (up-axis) plus small Gaussian jitter, applied only at training time.
- **Bugs I hit and fixed**: [FILL IN — e.g. BatchNorm with batch size 1 during debugging, shape mismatches in batch matrix multiply, etc.]

## Reference

```
Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017).
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.
CVPR 2017. arXiv:1612.00593
https://arxiv.org/abs/1612.00593
```
---

<div align="center">

**Implemented from first principles by Mariam Maysara.**

</div>
