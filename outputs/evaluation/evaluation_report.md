# Adaptive CXR Enhancement: Benchmark Evaluation Report

**Evaluated on 624 unique test chest radiograph samples.**

## 1. Model 1 Degradation Detector Performance

**Overall 4D Vector MAE:** `0.0219`

| Degradation | Presence Accuracy | Vector MAE |
| :--- | :---: | :---: |
| **Noise** | 100.0% | 0.0100 |
| **Blur** | 97.9% | 0.0353 |
| **Contrast** | 99.7% | 0.0250 |
| **Compression** | 100.0% | 0.0172 |

## 2. Image Restoration Quality Comparison

| Cohort | PSNR (dB) | SSIM | MAE |
| :--- | :---: | :---: | :---: |
| **Degraded (Raw)** | 24.33 | 0.549 | 0.0855 |
| **Generic Filter (CLAHE + Bilateral)** | 17.51 | 0.522 | 0.1204 |
| **Proposed (Model 1 + Model 2 FiLM)** | **31.83** | **0.875** | **0.0205** |

## 3. Clinical Diagnostic Efficacy (Calibrated Model 3 Backbones @ 0.50)

### Backbone: `densenet121`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 85.4% | 0.906 | 0.890 | 94.1% | 70.9% |
| **Degraded** | 74.5% | 0.842 | 0.829 | 98.7% | 34.2% |
| **Generic** | 72.6% | 0.763 | 0.805 | 90.8% | 42.3% |
| **Adaptive** | 84.6% | 0.903 | 0.879 | 89.5% | 76.5% |

### Backbone: `resnet50`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 89.7% | 0.942 | 0.920 | 94.1% | 82.5% |
| **Degraded** | 76.0% | 0.820 | 0.826 | 91.5% | 50.0% |
| **Generic** | 73.1% | 0.753 | 0.801 | 86.7% | 50.4% |
| **Adaptive** | 87.0% | 0.937 | 0.892 | 86.2% | 88.5% |

### Backbone: `efficientnet_b0`
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 89.4% | 0.927 | 0.920 | 97.2% | 76.5% |
| **Degraded** | 73.6% | 0.627 | 0.824 | 98.7% | 31.6% |
| **Generic** | 74.7% | 0.632 | 0.830 | 98.7% | 34.6% |
| **Adaptive** | 89.9% | 0.941 | 0.923 | 97.2% | 77.8% |

### Consensus Ensemble (50% EffNet / 30% DenseNet / 20% ResNet)
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 89.7% | 0.942 | 0.923 | 97.7% | 76.5% |
| **Degraded** | 75.0% | 0.821 | 0.833 | 99.7% | 33.8% |
| **Generic** | 76.3% | 0.747 | 0.837 | 97.7% | 40.6% |
| **Adaptive** | 90.2% | 0.947 | 0.925 | 96.2% | 80.3% |

### Safety-First Clinical Triage Verdict (Borderline Threshold @ 0.45)
| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 88.8% | 0.942 | 0.916 | 98.5% | 72.6% |
| **Degraded** | 74.7% | 0.821 | 0.832 | 100.0% | 32.5% |
| **Generic** | 75.8% | 0.747 | 0.835 | 98.2% | 38.5% |
| **Adaptive** | 90.1% | 0.947 | 0.924 | 97.2% | 78.2% |
