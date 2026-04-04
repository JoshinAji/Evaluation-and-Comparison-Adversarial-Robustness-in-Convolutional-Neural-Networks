# Brain-Inspired Adversarial Robustness — Phase 2-4
## Current State
* Baseline ResNet-18 trained on CIFAR-10: **94.95%** clean accuracy
* Baseline adversarial results: FGSM **19.30%**, PGD **17.19%** (ε=8/255)
* Saved weights: `resnet18_cifar10_clean.pth`
* All data loading and attack infrastructure in place
## Phase 2: Three Brain-Inspired Architectures
### Model 1 — RecurrentResNet18 (Recurrent Processing)
* **Bio motivation:** Top-down/lateral recurrent connections allow iterative feature refinement; humans need this processing time to resist adversarial examples
* **Implementation:** Shared-weight recurrent refinement block (two 3×3 convolutions + BN) applied T times after layer4, with residual skip connections
* **Hyperparameter:** time_steps T=3 (also evaluated at T=1,2,4 at inference)
### Model 2 — NoisyResNet18 (Noise Injection)
* **Bio motivation:** Neural stochasticity provides natural robustness to small perturbations
* **Implementation:** GaussianNoise(σ=0.1) layers injected after each residual stage, active during both training and inference
### Model 3 — AttentionResNet18 (Selective Attention)
* **Bio motivation:** Selective attention filters salient features and suppresses adversarial noise
* **Implementation:** Squeeze-and-Excitation (SE) blocks after each residual stage (reduction=16)
## Phase 3: Training & Attack Evaluation
* Same training setup as baseline: 100 epochs, SGD (lr=0.1, cosine decay to 1e-4), weight_decay=5e-4, batch_size=128
* Same attacks: FGSM (ε=8/255, full test set) and PGD (ε=8/255, 10 iter, 5 batches)
* Each model's best weights saved to disk
## Phase 4: Comparative Analysis
* Results table: all 4 models × (clean acc, FGSM acc, PGD acc, robustness gaps)
* Bar charts: clean vs adversarial accuracy, robustness gaps, improvement over baseline
* Time-step analysis: recurrent model accuracy vs T (1-4) to test the "more processing time = more robust" hypothesis
## Deliverable
Replace `Brain-Inspired Recurrent Mechanisms.ipynb` with full Phase 2-4 implementation notebook