# Biologically Inspired Robustness: Evaluating Recurrence, Stochasticity, and Attention in ResNet-18 Against Adversarial Attacks

**Authors:** Sanskriti Mehta, Joshin Aji  
**Course:** CS9873 — Western University  
**Department:** Computer Science

## Overview

Deep learning models achieve high accuracy on clean images but remain vulnerable to adversarial perturbations — small, imperceptible input modifications that cause incorrect predictions. This project investigates whether three mechanisms observed in the primate visual cortex can improve the adversarial robustness of ResNet-18 on CIFAR-10 **without adversarial training**:

1. **Recurrent feedback processing** — shared-weight iterative refinement after the final residual stage
2. **Neural stochasticity** — persistent Gaussian noise injection across all residual stages
3. **Selective channel attention** — Squeeze-and-Excitation (SE) blocks at every residual stage

## Key Results

| Model | Clean Acc. | FGSM Acc. | PGD Acc. |
|---|---|---|---|
| Baseline ResNet-18 | 94.95% | 19.30% | 17.34% |
| Recurrent ResNet-18 | 94.52% | 20.00% | 18.28% |
| Noisy ResNet-18 | 94.88% | 17.46% | 15.47% |
| Attention ResNet-18 | 95.01% | 20.62% | 15.62% |

- **Recurrence** is the only mechanism that improves robustness under both FGSM and PGD.
- **Attention** helps against single-step FGSM but collapses under iterative PGD.
- **Noise injection** degrades robustness across the board.
- The recurrent model at T=1 inference steps achieves **22.66% PGD accuracy** (+5.32% over baseline).

## Project Structure

```
├── Baseline Model.ipynb                  # Baseline ResNet-18 training and evaluation
├── Brain-Inspired Recurrent Mechanisms.ipynb  # Recurrent variant experiments
├── Phase2_4_Brain_Inspired.ipynb         # Noisy and Attention variant experiments
├── Adversarial_Demo.ipynb                # Visual demo of adversarial attacks across models
├── resnet18_cifar10_clean.pth            # Trained baseline weights
├── recurrent_resnet18.pth                # Trained recurrent variant weights
├── noisy_resnet18.pth                    # Trained noisy variant weights
├── attention_resnet18.pth                # Trained attention variant weights
├── cifar-10-batches-py/                  # CIFAR-10 dataset
└── README.md
```

## Setup

### Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- Adversarial Robustness Toolbox (ART)
- torchvision, matplotlib, numpy

### Install dependencies

```bash
pip install torch torchvision adversarial-robustness-toolbox matplotlib numpy
```

### Running

1. **Train / evaluate the baseline:** Open `Baseline Model.ipynb` and run all cells.
2. **Train / evaluate bio-inspired variants:** Open `Phase2_4_Brain_Inspired.ipynb` and `Brain-Inspired Recurrent Mechanisms.ipynb`.
3. **Visualise adversarial examples:** Open `Adversarial_Demo.ipynb`.

Pre-trained weights (`.pth` files) are included — you can skip training and load them directly for evaluation.

## Attack Configuration

- **FGSM:** ε = 8/255, single-step, full test set (10,000 images)
- **PGD:** ε = 8/255, α = 2/255, 10 iterations, random init, evaluated on ~640 images

## Citation

If you reference this work:

```
Mehta, S. and Aji, J. (2026). Biologically Inspired Robustness: Evaluating Recurrence,
Stochasticity, and Attention in ResNet-18 Against Adversarial Attacks. CS9873 Final Project,
Western University.
```
