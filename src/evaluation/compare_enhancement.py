"""Comprehensive benchmark comparing Clean, Degraded, Generic, and Adaptive Enhancement."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..degradations.pipeline import DegradationPipeline
from ..models.degradation_detector import DegradationDetector
from ..models.adaptive_unet import AdaptiveUNet
from ..models.disease_classifier import DiseaseClassifier
from ..models.consensus import load_calibration_config, predict_consensus_batch
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

    # Collectors for detector accuracy and image quality metrics
    gt_vectors: List[np.ndarray] = []
    pred_vectors_all: List[np.ndarray] = []
    img_metrics = {
        "degraded": {"psnr": [], "ssim": [], "mae": []},
        "generic": {"psnr": [], "ssim": [], "mae": []},
        "adaptive": {"psnr": [], "ssim": [], "mae": []},
    }

    # Collectors for clinical predictions (calibrated individual backbones + weighted consensus)
    calibration_cfg, ensemble_weights, triage_thresholds = load_calibration_config()
    labels_all: List[int] = []
    preds = {
        name: {
            "clean": [],
            "degraded": [],
            "generic": [],
            "adaptive": [],
        }
        for name in list(disease_models.keys()) + ["consensus_ensemble"]
    }

    # Iterate through test dataset
    pbar = tqdm(test_loader, desc="Evaluating Pipeline", leave=False)
    for degraded, clean, vector, labels in pbar:
        batch_size = degraded.size(0)
        labels_all.extend(labels.tolist())
        gt_vectors.append(vector.cpu().numpy())

        degraded_dev = degraded.to(device)
        clean_dev = clean.to(device)

        with torch.no_grad():
            # Step 1: Diagnose with Model 1
            pred_vectors = detector.predict_vector(degraded_dev)
            pred_vectors_all.append(pred_vectors.cpu().numpy())

            # Step 2: Enhance with Model 2 (Adaptive FiLM U-Net)
            adaptive_restored = torch.clamp(enhancer(degraded_dev, pred_vectors), 0.0, 1.0)

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

        # Step 3: Run Model 3 calibrated consensus panel across cohorts
        clean_zero_vec = torch.zeros_like(pred_vectors)
        cohort_inputs = [
            ("clean", clean_dev, clean_zero_vec),
            ("degraded", degraded_dev, pred_vectors),
            ("generic", generic_dev, pred_vectors),
            ("adaptive", adaptive_restored, pred_vectors),
        ]
        with torch.no_grad():
            for cohort_name, cohort_tensor, deg_vec in cohort_inputs:
                ens_probs, per_arch = predict_consensus_batch(
                    disease_models,
                    cohort_tensor,
                    degradation_vector=deg_vec,
                    weights=ensemble_weights,
                    calibration=calibration_cfg,
                )
                preds["consensus_ensemble"][cohort_name].extend(ens_probs.cpu().tolist())
                for arch_name, arch_probs in per_arch.items():
                    preds[arch_name][cohort_name].extend(arch_probs.cpu().tolist())

    # Aggregate Detector Metrics
    gt_v_arr = np.concatenate(gt_vectors, axis=0) if gt_vectors else np.zeros((0, 4))
    pr_v_arr = np.concatenate(pred_vectors_all, axis=0) if pred_vectors_all else np.zeros((0, 4))
    detector_metrics: Dict[str, Any] = {
        "overall_mae": float(np.mean(np.abs(pr_v_arr - gt_v_arr))) if len(gt_v_arr) else 0.0,
        "per_degradation": {},
    }
    for idx, deg_name in enumerate(DegradationPipeline.DEGRADATION_NAMES):
        if len(gt_v_arr):
            gt_pres = gt_v_arr[:, idx] > 0.05
            pr_pres = pr_v_arr[:, idx] > 0.05
            pres_acc = float(np.mean(gt_pres == pr_pres))
            mae_val = float(np.mean(np.abs(pr_v_arr[:, idx] - gt_v_arr[:, idx])))
        else:
            pres_acc, mae_val = 0.0, 0.0
        detector_metrics["per_degradation"][deg_name] = {
            "presence_accuracy": pres_acc,
            "mae": mae_val,
        }

    # Aggregate Image Metrics
    summary_image_metrics = {
        cohort: {k: float(np.mean(vals)) for k, vals in metrics.items()}
        for cohort, metrics in img_metrics.items()
    }

    # Aggregate Clinical Metrics (evaluated at standard 0.50 cutoff + safety triage 0.45 for consensus)
    summary_clinical_metrics = {}
    for name in list(disease_models.keys()) + ["consensus_ensemble"]:
        summary_clinical_metrics[name] = {
            cohort: compute_diagnostic_metrics(labels_all, preds[name][cohort], threshold=0.50)
            for cohort in ["clean", "degraded", "generic", "adaptive"]
        }

    borderline_th = float(triage_thresholds.get("borderline", 0.45))
    triage_metrics = {
        cohort: compute_diagnostic_metrics(labels_all, preds["consensus_ensemble"][cohort], threshold=borderline_th)
        for cohort in ["clean", "degraded", "generic", "adaptive"]
    }

    full_report = {
        "num_test_samples": len(labels_all),
        "detector_performance": detector_metrics,
        "image_restoration_quality": summary_image_metrics,
        "clinical_diagnosis_performance": summary_clinical_metrics,
        "consensus_triage_045": triage_metrics,
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
    """Generate Markdown summary table of detector, restoration, and calibrated clinical metrics."""
    det_q = report.get("detector_performance", {})
    img_q = report["image_restoration_quality"]
    clin = report["clinical_diagnosis_performance"]
    triage = report.get("consensus_triage_045", {})

    lines = [
        "# Adaptive CXR Enhancement: Benchmark Evaluation Report\n",
        f"**Evaluated on {report['num_test_samples']} unique test chest radiograph samples.**\n",
    ]

    if det_q:
        lines.extend([
            "## 1. Model 1 Degradation Detector Performance\n",
            f"**Overall 4D Vector MAE:** `{det_q.get('overall_mae', 0.0):.4f}`\n",
            "| Degradation | Presence Accuracy | Vector MAE |",
            "| :--- | :---: | :---: |",
        ])
        for deg_name, vals in det_q.get("per_degradation", {}).items():
            lines.append(f"| **{deg_name.capitalize()}** | {vals['presence_accuracy']*100:.1f}% | {vals['mae']:.4f} |")
        lines.append("")

    lines.extend([
        "## 2. Image Restoration Quality Comparison\n",
        "| Cohort | PSNR (dB) | SSIM | MAE |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Degraded (Raw)** | {img_q['degraded']['psnr']:.2f} | {img_q['degraded']['ssim']:.3f} | {img_q['degraded']['mae']:.4f} |",
        f"| **Generic Filter (CLAHE + Bilateral)** | {img_q['generic']['psnr']:.2f} | {img_q['generic']['ssim']:.3f} | {img_q['generic']['mae']:.4f} |",
        f"| **Proposed (Model 1 + Model 2 FiLM)** | **{img_q['adaptive']['psnr']:.2f}** | **{img_q['adaptive']['ssim']:.3f}** | **{img_q['adaptive']['mae']:.4f}** |\n",
        "## 3. Clinical Diagnostic Efficacy (Calibrated Model 3 Backbones @ 0.50)\n",
    ])

    for model_name, cohorts in clin.items():
        title = "Consensus Ensemble (50% EffNet / 30% DenseNet / 20% ResNet)" if model_name == "consensus_ensemble" else f"Backbone: `{model_name}`"
        lines.append(f"### {title}")
        lines.append("| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for cohort_name in ["clean", "degraded", "generic", "adaptive"]:
            m = cohorts[cohort_name]
            label = f"**{cohort_name.capitalize()}**"
            lines.append(f"| {label} | {m['accuracy']*100:.1f}% | {m['auc_roc']:.3f} | {m['f1']:.3f} | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% |")
        lines.append("")

    if triage:
        lines.extend([
            "### Safety-First Clinical Triage Verdict (Borderline Threshold @ 0.45)",
            "| Cohort | Accuracy | AUC-ROC | F1-Score | Sensitivity | Specificity |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cohort_name in ["clean", "degraded", "generic", "adaptive"]:
            m = triage[cohort_name]
            label = f"**{cohort_name.capitalize()}**"
            lines.append(f"| {label} | {m['accuracy']*100:.1f}% | {m['auc_roc']:.3f} | {m['f1']:.3f} | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% |")
        lines.append("")

    return "\n".join(lines)
