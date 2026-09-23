# HyperVerge Campus Placement — ResNet-18 Semi-Supervised Classifier

This repository contains a single VS Code/Jupyter notebook for the HyperVerge deep-learning internship assignment.

## Approach

- **Phase 1:** randomly initialized ResNet-18 trained only on labeled images, with balanced sampling, augmentation, label smoothing, cosine learning-rate decay, and held-out validation.
- **Phase 2:** confidence-filtered and class-balanced pseudo-labeling of the supplied unlabeled images, followed by fine-tuning of the same ResNet-18.
- No external image dataset or pretrained model weights are used.
- The notebook generates `phase1_predictions.csv` and `phase2_predictions.csv` with the required `path,predicted_label` schema.

## Running locally

1. Extract the supplied training and test archives beside the notebook using the folder layout expected in the first notebook cells.
2. Open `Campus_Assignment_2025.ipynb` in VS Code.
3. Select a Python Jupyter kernel with PyTorch, pandas, NumPy, and Pillow. CUDA-enabled PyTorch is recommended.
4. Run all cells. Interrupted phases resume from checkpoints and cached pseudo-label predictions.

The assignment datasets and model checkpoints are intentionally excluded from Git because of their size and redistribution constraints.
