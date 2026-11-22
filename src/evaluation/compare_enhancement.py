"""Comprehensive benchmark comparing Clean, Degraded, Generic, and Adaptive Enhancement."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.degradation_detector import DegradationDetector
from ..models.adaptive_unet import AdaptiveUNet
from ..models.disease_classifier import DiseaseClassifier
from .metrics import compute_image_metrics, compute_diagnostic_metrics


def apply_generic_enhancer(img_np: np.ndarray) -> np.ndarray:
    """
    Standard generic clinical image enhancement baseline:
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    followed by a bilateral smoothing filter.
    """
    uint8_img = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(uint8_img)

    # 2. Bilateral filter (edge-preserving smoothing)
    filtered = cv2.bilateralFilter(equalized, d=5, sigmaColor=35, sigmaSpace=35)

    return (filtered.astype(np.float32) / 255.0)


def evaluate_enhancement_pipeline(
    test_loader: DataLoader,
    detector: DegradationDetector,
    enhancer: AdaptiveUNet,
    disease_models: Dict[str, DiseaseClassifier],
    device: torch.device,
    output_dir: str = "outputs/evaluation",
) -> Dict[str, Any]:
    """
    Run complete evaluation across Clean, Degraded, Generic, and Adaptive cohorts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    detector = detector.to(device)
    enhancer = enhancer.to(device)
    detector.eval()
    enhancer.eval()

    for model in disease_models.values():
        model.to(device)
        model.eval()

    # Collectors for image metrics
    img_metrics = {
        "degraded": {"psnr": [], "ssim": [], "mae": []},
        "generic": {"psnr": [], "ssim": [], "mae": []},
        "adaptive": {"psnr": [], "ssim": [], "mae": []},
    }

    # Collectors for clinical predictions
    labels_all: List[int] = []
    preds = {
        name: {
            "clean": [],
            "degraded": [],
            "generic": [],
            "adaptive": [],
        }
        for name in disease_models.keys()
    }

    # Iterate through test dataset
    pbar = tqdm(test_loader, desc="Evaluating Pipeline", leave=False)
    for degraded, clean, vector, labels in pbar:
        batch_size = degraded.size(0)
        labels_all.extend(labels.tolist())

        degraded_dev = degraded.to(device)
        clean_dev = clean.to(device)

        with torch.no_grad():
            # Step 1: Diagnose with Model 1
            pred_vectors = detector.predict_vector(degraded_dev)

            # Step 2: Enhance with Model 2 (Adaptive FiLM U-Net)
            adaptive_restored = enhancer(degraded_dev, pred_vectors)

        # Generate generic baseline on CPU
        generic_restored = []
        for i in range(batch_size):
            deg_np = degraded[i, 0].cpu().numpy()
            clean_np = clean[i, 0].cpu().numpy()
            adapt_np = adaptive_restored[i, 0].cpu().numpy()

            gen_np = apply_generic_enhancer(deg_np)
            generic_restored.append(torch.from_numpy(gen_np).unsqueeze(0))

            # Image quality comparisons
            m_deg = compute_image_metrics(deg_np, clean_np)
            m_gen = compute_image_metrics(gen_np, clean_np)
            m_adp = compute_image_metrics(adapt_np, clean_np)

            for k in ["psnr", "ssim", "mae"]:
                img_metrics["degraded"][k].append(m_deg[k])
                img_metrics["generic"][k].append(m_gen[k])
                img_metrics["adaptive"][k].append(m_adp[k])

        generic_dev = torch.stack(generic_restored).to(device)

        # Step 3: Run Model 3 disease classifiers
        for name, d_model in disease_models.items():
            with torch.no_grad():
                prob_clean = d_model.predict_proba(clean_dev)[:, 1].cpu().tolist()
                prob_deg = d_model.predict_proba(degraded_dev)[:, 1].cpu().tolist()
                prob_gen = d_model.predict_proba(generic_dev)[:, 1].cpu().tolist()
                prob_adp = d_model.predict_proba(adaptive_restored)[:, 1].cpu().tolist()

                preds[name]["clean"].extend(prob_clean)
                preds[name]["degraded"].extend(prob_deg)
                preds[name]["generic"].extend(prob_gen)
                preds[name]["adaptive"].extend(prob_adp)

    # Aggregate Image Metrics
    summary_image_metrics = {
        cohort: {k: float(np.mean(vals)) for k, vals in metrics.items()}
        for cohort, metrics in img_metrics.items()
    }

    # Aggregate Clinical Metrics
    summary_clinical_metrics = {}
    for name in disease_models.keys():
        summary_clinical_metrics[name] = {
            cohort: compute_diagnostic_metrics(labels_all, preds[name][cohort])
            for cohort in ["clean", "degraded", "generic", "adaptive"]
        }

    full_report = {
        "num_test_samples": len(labels_all),
        "image_restoration_quality": summary_image_metrics,
        "clinical_diagnosis_performance": summary_clinical_metrics,
    }

    # Save JSON report
    with open(out_path / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Generate Markdown Table
    md_table = _generate_markdown_report(full_report)
    with open(out_path / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(md_table)

    return full_report


def _generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate Markdown summary table of clinical and restoration metrics."""
    img_q = report["image_restoration_quality"]
    clin = report["clinical_diagnosis_performance"]

    lines = [
        "# Adaptive CXR Enhancement: Benchmark Evaluation Report\n",
        f"**Evaluated on {report['num_test_samples']} test chest radiograph samples.**\n",
        "## 1. Image Restoration Quality Comparison\n",
        "| Cohort | PSNR (dB) | SSIM | MAE |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Degraded (Raw)** | {img_q['degraded']['psnr']:.2f} | {img_q['degraded']['ssim']:.3f} | {img_q['degraded']['mae']:.4f} |",
        f"| **Generic Filter (CLAHE + Bilateral)** | {img_q['generic']['psnr']:.2f} | {img_q['generic']['ssim']:.3f} | {img_q['generic']['mae']:.4f} |",
        f"| **Proposed (Model 1 + Model 2 FiLM)** | **{img_q['adaptive']['psnr']:.2f}** | **{img_q['adaptive']['ssim']:.3f}** | **{img_q['adaptive']['mae']:.4f}** |\n",
        "## 2. Clinical Diagnostic Efficacy (Model 3 Backbones)\n",
    ]

    for model_name, cohorts in clin.items():
        lines.append(f"### Backbone: `{model_name}`")
        lines.append("| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for cohort_name in ["clean", "degraded", "generic", "adaptive"]:
            m = cohorts[cohort_name]
            label = f"**{cohort_name.capitalize()}**"
            lines.append(f"| {label} | {m['accuracy']*100:.1f}% | {m['auc_roc']:.3f} | {m['f1']:.3f} | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% |")
        lines.append("")

    return "\n".join(lines)
