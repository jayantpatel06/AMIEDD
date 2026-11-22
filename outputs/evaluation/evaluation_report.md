# Adaptive CXR Enhancement: Benchmark Evaluation Report

**Evaluated on 48 test chest radiograph samples.**

## 1. Image Restoration Quality Comparison

| Cohort | PSNR (dB) | SSIM | MAE |
| :--- | :---: | :---: | :---: |
| **Degraded (Raw)** | 30.62 | 0.509 | 0.0682 |
| **Generic Filter (CLAHE + Bilateral)** | 15.50 | 0.416 | 0.1475 |
| **Proposed (Model 1 + Model 2 FiLM)** | **29.34** | **0.718** | **0.0280** |

## 2. Clinical Diagnostic Efficacy (Model 3 Backbones)

### Backbone: `densenet121`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 93.8% | 1.000 | 0.941 | 100.0% | 87.5% |
| **Degraded** | 64.6% | 0.431 | 0.514 | 37.5% | 91.7% |
| **Generic** | 39.6% | 0.306 | 0.216 | 16.7% | 62.5% |
| **Adaptive** | 60.4% | 0.818 | 0.698 | 91.7% | 29.2% |

### Backbone: `resnet50`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 50.0% | 0.658 | 0.667 | 100.0% | 0.0% |
| **Degraded** | 35.4% | 0.323 | 0.415 | 45.8% | 25.0% |
| **Generic** | 35.4% | 0.297 | 0.114 | 8.3% | 62.5% |
| **Adaptive** | 37.5% | 0.352 | 0.516 | 66.7% | 8.3% |

### Backbone: `efficientnet_b0`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 50.0% | 0.440 | 0.000 | 0.0% | 100.0% |
| **Degraded** | 70.8% | 0.672 | 0.650 | 54.2% | 87.5% |
| **Generic** | 66.7% | 0.656 | 0.636 | 58.3% | 75.0% |
| **Adaptive** | 50.0% | 0.194 | 0.000 | 0.0% | 100.0% |
