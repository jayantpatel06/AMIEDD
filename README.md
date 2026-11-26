# AMIEDD — Adaptive Degradation-Aware Chest X-Ray Enhancement & Diagnostic System

**AMIEDD** is an end-to-end medical imaging framework and full-stack clinical web application that:
1. **Diagnoses physical radiographic corruptions** blindly as a continuous 4D severity vector (`noise`, `blur`, `contrast`, `compression`),
2. **Adaptively restores the radiograph** via a Feature-wise Linear Modulation (**FiLM**) U-Net conditioned on the exact estimated corruptions, and
3. **Evaluates pneumonia likelihood** through a domain-adapted, Platt-calibrated **3-Radiologist Deep Ensemble** (`EfficientNet-B0`, `DenseNet-121`, `ResNet-50`) with a safety-first clinical triage protocol.

---

## Table of Contents
1. [Why AMIEDD?](#1-why-amiedd)
2. [System Architecture & Data Flow](#2-system-architecture--data-flow)
3. [Physical Degradation Pipeline](#3-physical-degradation-pipeline)
4. [Neural Models & Mathematical Design](#4-neural-models--mathematical-design)
5. [Calibrated Consensus & Clinical Triage](#5-calibrated-consensus--clinical-triage)
6. [Verified 624-Scan Benchmark Results](#6-verified-624-scan-benchmark-results)
7. [Web Application (FastAPI + React Studio)](#7-web-application-fastapi--react-studio)
8. [Installation & Quickstart](#8-installation--quickstart)
9. [Project Structure](#9-project-structure)
10. [Testing & Verification](#10-testing--verification)

---

## 1. Why AMIEDD??

Real-world chest X-rays (CXRs) frequently suffer from low-dose sensor quantum noise, respiratory motion blur, underexposure/dynamic-range loss, and lossy PACS JPEG compression. Standard enhancement techniques (like CLAHE or unsharp masking) are **blind to the corruption type**: they amplify sensor noise into pseudo-opacities, distort radiographic density (`PSNR` drops to `17.51 dB`), and leave downstream classifiers with a **59–66% false-alarm rate** on healthy lungs.

AMIEDD decouples **degradation diagnosis**, **conditioned neural restoration**, and **domain-adapted consensus classification**, improving restored image quality by **+7.50 dB PSNR** (`0.875 SSIM`) and raising downstream diagnostic accuracy on corrupted scans from **75.0% → 90.2%** (`0.947` AUC-ROC).

---

## 2. System Architecture & Data Flow

```
                      +-----------------------------------+
                      |   Input Chest X-Ray (224x224)     |
                      |   Grayscale Tensor x ∈ [0, 1]     |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      | Model 1: Degradation Detector     |
                      | ResNet-18 Multi-Task Head         |
                      |  • Presence logits (BCE)          |
                      |  • Severity ∈ [0, 1]^4 (SmoothL1) |
                      +-----------------------------------+
                                        |
                    4D Vector d = [noise, blur, contrast, comp]
                         (if max(d) < 0.01 -> skip Model 2)
                                        |
                                        v
                      +-----------------------------------+
                      | Model 2: Adaptive FiLM U-Net      |
                      | Encoder-Decoder + Skip Conns      |
                      | Decoder modulated by (1+γ(d))x+β(d)|
                      | Output: clamp(x + tanh(Δ), 0, 1)  |
                      +-----------------------------------+
                                        |
                           Restored Radiograph (+7.50 dB)
                                        |
                                        v
                      +-----------------------------------+
                      | Model 3: 3-Radiologist Ensemble   |
                      |  • EfficientNet-B0  (50% weight)  |
                      |  • DenseNet-121     (30% weight)  |
                      |  • ResNet-50        (20% weight)  |
                      | Degradation-Aware Platt Scaling   |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      | 3-Tier Clinical Triage Output     |
                      | High Risk ≥75% | Borderline ≥45%  |
                      | Low Risk <45%  | Individual Scores|
                      +-----------------------------------+
```

---

## 3. Physical Degradation Pipeline

Implemented in [`src/degradations/pipeline.py`](src/degradations/pipeline.py) and mirrored byte-for-byte in client-side Canvas synthesis inside [`StudioPage.tsx`](frontend/src/components/StudioPage.tsx), degradations are applied in realistic acquisition-to-archive physical order:

| Stage | Operator | Physical Mechanism | Parameter Range (`s ∈ [0, 1]`) |
| :--- | :--- | :--- | :--- |
| **1. Contrast** | `LowContrast` | Detector underexposure / flat dynamic range | `mean + (0.75 − 0.55·s)·(x − mean)` (`factor ∈ [0.20, 0.75]`) |
| **2. Blur** | `GaussianBlur` | Respiratory motion / optical defocus | Isotropic kernel `k ∈ [3, 15]` (odd), `σ_blur ∈ [0.5, 3.5]` |
| **3. Noise** | `GaussianNoise` | Low-dose quantum / electronics sensor noise | Additive `N(0, σ_noise²)` with `σ_noise ∈ [0.02, 0.25]` |
| **4. Compression** | `JPEGCompression` | Lossy PACS transmission & DCT block ringing | `quality = round(75 − 65·s)` (`quality ∈ [10, 75]`) |

> Additional physical variants (`PoissonNoise`, `MotionBlur`, `GammaDistortion`) are also implemented in [`src/degradations/`](src/degradations/). In the Testing Studio UI, setting PACS compression to `0%` preserves lossless PNG transport.

---

## 4. Neural Models & Mathematical Design

### Model 1 — Degradation Detector ([`degradation_detector.py`](src/models/degradation_detector.py))
- **Backbone:** ResNet-18 modified for 1-channel grayscale CXR input (ImageNet RGB weights averaged across channels for initialization), `dropout=0.2`, `512`-dim pooled feature embedding.
- **Dual Heads:**
  - `presence_head`: `Linear(512, 4)` trained with `BCEWithLogitsLoss` (active ground truth `vector > 0.05`).
  - `severity_head`: `Linear(512, 4) → Sigmoid` trained with masked `SmoothL1Loss` (`severity_weight = 2.0`) computed only over active degradations.
- **Inference Gating (`predict_vector`):**
  $$\hat{d} = \sigma(\text{logits}_{\text{pres}}) \odot \hat{s} \odot \mathbb{I}[\sigma(\text{logits}_{\text{pres}}) \ge 0.35]$$

### Model 2 — FiLM Adaptive U-Net ([`adaptive_unet.py`](src/models/adaptive_unet.py), [`film_layer.py`](src/models/film_layer.py))
- **Backbone:** 4-stage encoder (`32 → 64 → 128 → 256 → 512` bottleneck) with `BatchNorm2d` + `LeakyReLU(0.2)`, `MaxPool2d(2)` downsampling, and `ConvTranspose2d` skip-connected decoder blocks.
- **FiLM Conditioning (`FiLMGenerator`):** An MLP (`4 → 128 → 2C`) with zero-initialized final weights predicts per-channel affine scale and shift parameters from $\hat{d}$ at every decoder stage:
  $$h_c = (1 + \gamma_c(\hat{d})) \cdot \text{Norm}(\text{Conv}(x))_c + \beta_c(\hat{d})$$
- **Residual Reconstruction:** Outputs $\hat{x}_{\text{clean}} = \text{clamp}(x_{\text{deg}} + \tanh(\Delta), 0, 1)$. Zero initialization ensures clean pass-through when $\hat{d} = 0$, and `server.py` skips Model 2 altogether whenever $\max(\hat{d}) < 0.01$.
- **Composite Restoration Loss ([`losses.py`](src/training/losses.py)):**
  $$\mathcal{L}_{\text{enh}} = \mathcal{L}_1 + 0.2 \cdot (1 - \text{SSIM}) + 0.1 \cdot \mathcal{L}_{\text{Sobel}}$$

### Model 3 — Domain-Adapted Consensus Panel ([`disease_classifier.py`](src/models/disease_classifier.py), [`finetune_classifiers.py`](scripts/finetune_classifiers.py))
- **Architectures:** `EfficientNet-B0`, `DenseNet-121`, and `ResNet-50` (`1`-ch grayscale replicated to `3`-ch and normalized with ImageNet statistics).
- **Paired Clean + FiLM-Restored Domain Adaptation:** Because frozen ImageNet backbones trained solely on clean scans mistook U-Net smoothing/noise residuals for pulmonary consolidations, all three classifiers were fine-tuned (top semantic blocks `features.5–8`, `denseblock4+norm5`, `layer4` + heads) on balanced `(clean, FiLM-restored)` image pairs (`320` Normal + `320` Pneumonia train, `80 + 80` val) using Cross-Entropy (`label_smoothing=0.05`) plus consistency regularization:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \text{KL}(P_{\text{rest}} \,\|\, P_{\text{clean}}) + 0.1 \cdot \text{SmoothL1}(\Delta z_{\text{rest}}, \Delta z_{\text{clean}})$$
  This aligns predictions across clean and heavily corrupted scans so healthy lungs remain stable under degradation.

---

## 5. Calibrated Consensus & Clinical Triage

Implemented once in [`src/models/consensus.py`](src/models/consensus.py) and loaded from [`configs/calibration.json`](configs/calibration.json), both the FastAPI server and evaluation scripts share identical scoring logic:

1. **Degradation-Aware Platt Calibration** (fitted on the held-out Kaggle `val/` split):
   $$P_k = \sigma\!\left(\frac{(z_{k,1} - z_{k,0}) - (b_{0,k} + b_{1,k} \cdot \bar{d})}{T_k}\right)$$
   where $\bar{d} = \frac{1}{4}\sum_{i=1}^4 \hat{d}_i$.

2. **Weighted Soft-Voting Consensus:**
   $$P_{\text{consensus}} = 0.50 \cdot P_{\text{EfficientNet-B0}} + 0.30 \cdot P_{\text{DenseNet-121}} + 0.20 \cdot P_{\text{ResNet-50}}$$

   | Backbone | Weight | Restored Accuracy | Restored AUC | Role |
   | :--- | :---: | :---: | :---: | :--- |
   | **EfficientNet-B0** | **50%** | 89.9% | 0.941 | Primary spatially invariant classifier |
   | **DenseNet-121** | **30%** | 84.6% | 0.903 | Dense multi-scale feature reuse |
   | **ResNet-50** | **20%** | 87.0% | 0.937 | High-specificity (`88.5%`) tie-breaker |

3. **Three-Tier Clinical Triage:**

   | Risk Tier | Consensus Score | `has_pneumonia` | Clinical Action |
   | :--- | :---: | :---: | :--- |
   | **High Risk** | $P \ge 75\%$ | `true` | Confirmed radiopaque pulmonary consolidation |
   | **Indeterminate / Borderline** | $45\% \le P < 75\%$ | `true` | Safety-first flag for urgent human radiologist review |
   | **Low Risk (Normal)** | $P < 45\%$ | `false` | Clear lung fields confirmed by consensus |

---

## 6. Verified 624-Scan Benchmark Results

Evaluated on all **624 unique Kaggle `test/` scans** (`234` Normal, `390` Pneumonia) via [`scripts/run_step3_eval.py`](scripts/run_step3_eval.py). Full outputs are saved in [`outputs/evaluation/evaluation_report.md`](outputs/evaluation/evaluation_report.md) and [`evaluation_report.json`](outputs/evaluation/evaluation_report.json).

### Model 1 Degradation Estimator
- **Overall 4D Vector MAE:** `0.0219`
- **Noise:** `100.0%` presence accuracy (`0.0100` MAE) | **Blur:** `97.9%` (`0.0353`) | **Contrast:** `99.7%` (`0.0250`) | **Compression:** `100.0%` (`0.0172`)

### Model 2 Restoration Quality
| Cohort | PSNR (dB) | SSIM | MAE |
| :--- | :---: | :---: | :---: |
| **Degraded (Raw)** | 24.33 | 0.549 | 0.0855 |
| **Generic Filter (CLAHE + Bilateral)** | 17.51 | 0.522 | 0.1204 |
| **Proposed (Model 1 + Model 2 FiLM)** | **31.83 (+7.50 dB)** | **0.875 (+0.326)** | **0.0205** |

### Model 3 Clinical Diagnostic Consensus
| Cohort | Accuracy (@0.50) | AUC-ROC | F1-Score | Sensitivity (@0.50 / @0.45) | Specificity (@0.50 / @0.45) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 89.7% | 0.942 | 0.923 | 97.7% / 98.5% | 76.5% / 72.6% |
| **Degraded (Raw)** | 75.0% | 0.821 | 0.833 | 99.7% / 100.0% | 33.8% / 32.5% |
| **Generic CLAHE + Bilateral** | 76.3% | 0.747 | 0.837 | 97.7% / 98.2% | 40.6% / 38.5% |
| **AMIEDD Restored (`Adaptive`)** | **90.2%** | **0.947** | **0.925** | **96.2% / 97.2%** | **80.3% / 78.2%** |

---

## 7. Web Application (FastAPI + React Studio)

- **Backend ([`server.py`](server.py)):**
  - `GET /api/health` — reports device (`cpu`/`cuda`), loaded checkpoints, and active consensus backbones.
  - `POST /api/pipeline/degrade` — server-side physical degradation simulator using `DegradationPipeline`.
  - `POST /api/pipeline/predict` — decodes input via `cv2.INTER_AREA`, runs Model 1 → Model 2 (bypassed if clean) → Model 3 calibrated consensus, and returns restored base64 PNG + per-model breakdown + triage verdict.
  - Mounts `frontend/dist/` directly as static assets on port `8000`.
- **Frontend ([`frontend/`](frontend/)):** React 19 + Vite + TypeScript + TailwindCSS v3 (`#FAF8F5` / `#F5F2EB` light medical theme):
  1. **Overview** ([`OverviewPage.tsx`](frontend/src/components/OverviewPage.tsx)) — project overview, 3-step pipeline explanation, and clinical metrics.
  2. **Testing Studio** ([`StudioPage.tsx`](frontend/src/components/StudioPage.tsx)) — preset scans (`normal_patient`, `bacterial_pneumonia`, `viral_pneumonia`) or custom upload, real-time degradation sliders, downloadable corrupted scan, **Interactive FiLM Restoration Slider** alongside controls, and smooth auto-scroll down to the side-by-side **Model 1 Degradation & Error** and **Model 3 Consensus Panel** results.
  3. **Architecture** ([`ArchitecturePage.tsx`](frontend/src/components/ArchitecturePage.tsx)) — pipeline flowchart, FiLM & loss math, clinical thresholds, and 624-scan benchmark matrix.

---

## 8. Installation & Quickstart

### 1. Python Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -e .
```

### 2. Dataset Setup
Place the Kaggle Chest X-Ray (Pneumonia) dataset at `data/chest_xray/` (`train/`, `val/`, `test/` with `NORMAL/` and `PNEUMONIA/`), or configure `~/.kaggle/kaggle.json` and run:
```bash
python -c "from src.data.download_kaggle import setup_kaggle_dataset; setup_kaggle_dataset('data/chest_xray')"
```
*(If no Kaggle key is present, [`download_kaggle.py`](src/data/download_kaggle.py) generates procedural CXR phantoms for testing.)*

### 3. Build Frontend & Start Web App
```bash
cd frontend
npm install
npm run build
cd ..

python server.py
```
Open **http://localhost:8000** in your browser. *(For frontend hot-reload dev mode, run `npm run dev` in `frontend/`; `vite.config.ts` proxies `/api` to `localhost:8000`.)*

### 4. Training, Fine-Tuning & Evaluation Scripts
```bash
# Full pipeline (Model 1 -> Model 2 -> Step 3 eval)
python scripts/run_full_pipeline.py --mode all

# Paired clean+restored domain adaptation & validation Platt calibration for Model 3
python scripts/finetune_classifiers.py

# Standalone 624-scan benchmark evaluation (outputs/evaluation/evaluation_report.{md,json})
python scripts/run_step3_eval.py

# Single-image 5-panel visualization demo
python scripts/demo_single_image.py --noise 0.6 --blur 0.5 --contrast 0.3 --compression 0.4
```

---

## 9. Project Structure

```
AMIEDD/
├── server.py                         # FastAPI inference & static-hosting server (port 8000)
├── pyproject.toml                    # Dependencies and pytest configuration
├── configs/
│   ├── default_config.yaml           # Training hyperparameters, dataset & checkpoint paths
│   └── calibration.json              # Validation-fitted Platt scaling & triage thresholds
├── src/
│   ├── models/
│   │   ├── degradation_detector.py   # Model 1: ResNet-18 dual presence/severity estimator
│   │   ├── film_layer.py             # FiLMGenerator MLP & affine modulation block
│   │   ├── adaptive_unet.py          # Model 2: Residual U-Net with FiLM decoder blocks
│   │   ├── disease_classifier.py     # Model 3: DenseNet-121 / EfficientNet-B0 / ResNet-50
│   │   └── consensus.py              # Shared Platt calibration, consensus vote & clinical triage
│   ├── degradations/
│   │   ├── base.py                   # Abstract BaseDegradation interface
│   │   ├── noise.py                  # GaussianNoise & PoissonNoise
│   │   ├── blur.py                   # GaussianBlur & MotionBlur
│   │   ├── contrast.py               # LowContrast & GammaDistortion
│   │   ├── compression.py            # JPEGCompression (PACS DCT)
│   │   └── pipeline.py               # DegradationPipeline (Contrast->Blur->Noise->JPEG)
│   ├── training/
│   │   ├── losses.py                 # MultiTaskDetectorLoss, SSIMLoss, SobelEdgeLoss
│   │   ├── train_detector.py         # Model 1 training loop
│   │   └── train_enhancer.py         # Model 2 FiLM U-Net training loop
│   ├── evaluation/
│   │   ├── compare_enhancement.py    # Clean vs Degraded vs Generic vs Adaptive benchmark
│   │   ├── metrics.py                # PSNR, SSIM, MAE, ROC-AUC, Sensitivity, Specificity
│   │   └── visualize.py              # 5-panel sample_comparison renderer
│   ├── data/
│   │   ├── cxr_dataset.py            # CXRDataset (Windows-deduplicated) + on-the-fly corruptions
│   │   └── download_kaggle.py        # Kaggle downloader + procedural CXR phantom generator
│   └── utils/helpers.py              # Seed, device, thread, and YAML config utilities
├── scripts/
│   ├── run_full_pipeline.py          # Full Model 1 -> Model 2 -> Model 3 orchestrator
│   ├── finetune_classifiers.py       # Paired clean+restored domain adaptation + calibration fit
│   ├── run_step3_eval.py             # Full test-set benchmark runner
│   └── demo_single_image.py          # Single-image visual comparison generator
├── tests/                            # 15 pytest unit & end-to-end integration tests
├── frontend/                         # React 19 + Vite + TypeScript + TailwindCSS v3 app
│   ├── public/samples/               # Real Kaggle CXR presets (Normal, Bacterial, Viral)
│   └── src/components/               # OverviewPage, StudioPage, ArchitecturePage, Navbar, Slider
├── outputs/
│   ├── evaluation/                   # evaluation_report.md & evaluation_report.json
│   └── visualizations/               # demo_single_image.png & sample_comparison.png
├── checkpoints/                      # Trained weights (gitignored)
└── data/chest_xray/                  # 5,856 Kaggle CXR scans (gitignored)
```

---

## 10. Testing & Verification

```bash
# Run Python unit & end-to-end tests (15/15 passing)
pytest tests/ -v

# Build and lint the frontend
cd frontend
npm run build
npm run lint
```
