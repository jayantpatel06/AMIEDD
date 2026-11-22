# Adaptive Degradation-Aware Chest X-Ray Enhancement for Disease Diagnosis

An end-to-end deep learning framework designed to diagnose medical image degradations, adaptively restore chest radiographs using Feature-wise Linear Modulation (FiLM), and evaluate downstream diagnostic efficacy across clinical AI radiologist backbones.

---

## Architecture Overview

```
                      +-----------------------------+
                      | Raw Degraded CXR Image      |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Step 1: Model 1 (ResNet18)  |
                      | "Problem Detective"         |
                      +-----------------------------+
                                     |
                          Diagnosed Severity Vector
                          [Noise, Blur, Contrast, JPEG]
                                     |
                                     v
                      +-----------------------------+
                      | Step 2: Model 2 (FiLM U-Net)|
                      | "Adaptive Image Cleaner"    |
                      +-----------------------------+
                                     |
                           Restored CXR Image
                                     |
                                     v
                      +-----------------------------+
                      | Step 3: Model 3 (Evaluator) |
                      | "AI Radiologists"           |
                      | DenseNet121 / ResNet50      |
                      +-----------------------------+
                                     |
                    Compare Accuracy, AUC-ROC, F1, PSNR
```

---

## Quickstart

### 1. Environment Setup
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Dataset Setup
To download the Kaggle Chest X-Ray (Pneumonia) dataset:
1. Ensure your Kaggle API key is at `~/.kaggle/kaggle.json`.
2. Run:
```bash
python -c "from src.data.download_kaggle import setup_kaggle_dataset; setup_kaggle_dataset('data/chest_xray')"
```
*(If no Kaggle key is found, procedural realistic CXR phantoms are generated automatically for immediate testing and verification).*

### 3. Run the Full Pipeline
```bash
# End-to-end: Train Model 1 -> Train Model 2 -> Clinical Downstream Evaluation
python scripts/run_full_pipeline.py --mode all --quick_run
```

### 4. Interactive Single-Image Demonstration
```bash
python scripts/demo_single_image.py --noise 0.6 --blur 0.5 --output outputs/visualizations/demo_single_image.png
```

### 5. Running Tests
```bash
pytest tests/ -v
```
