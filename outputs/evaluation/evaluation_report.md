# Adaptive CXR Enhancement: Benchmark Evaluation Report

**Evaluated on 96 test chest radiograph samples.**

## 1. Image Restoration Quality Comparison

| Cohort | PSNR (dB) | SSIM | MAE |
| :--- | :---: | :---: | :---: |
| **Degraded (Raw)** | 23.08 | 0.432 | 0.0771 |
| **Generic Filter (CLAHE + Bilateral)** | 14.87 | 0.357 | 0.1573 |
| **Proposed (Model 1 + Model 2 FiLM)** | **28.33** | **0.709** | **0.0298** |

## 2. Clinical Diagnostic Efficacy (Model 3 Backbones)

### Backbone: `densenet121`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 50.0% | 0.637 | 0.667 | 100.0% | 0.0% |
| **Degraded** | 50.0% | 0.507 | 0.667 | 100.0% | 0.0% |
| **Generic** | 50.0% | 0.562 | 0.667 | 100.0% | 0.0% |
| **Adaptive** | 50.0% | 0.361 | 0.667 | 100.0% | 0.0% |

### Backbone: `resnet50`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 50.0% | 0.894 | 0.667 | 100.0% | 0.0% |
| **Degraded** | 50.0% | 0.493 | 0.667 | 100.0% | 0.0% |
| **Generic** | 50.0% | 0.510 | 0.667 | 100.0% | 0.0% |
| **Adaptive** | 49.0% | 0.424 | 0.657 | 97.9% | 0.0% |

### Backbone: `efficientnet_b0`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 50.0% | 0.479 | 0.667 | 100.0% | 0.0% |
| **Degraded** | 51.0% | 0.511 | 0.671 | 100.0% | 2.1% |
| **Generic** | 52.1% | 0.521 | 0.667 | 95.8% | 8.3% |
| **Adaptive** | 46.9% | 0.366 | 0.605 | 81.2% | 12.5% |
