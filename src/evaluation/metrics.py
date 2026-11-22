"""Evaluation metrics for image restoration quality and clinical diagnostic performance."""
from typing import Dict, Any, List, Union
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_image_metrics(pred_np: np.ndarray, target_np: np.ndarray) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, and MAE between two normalized images [0, 1].
    Args:
        pred_np: (H, W) or (H, W, C) float32 in [0, 1]
        target_np: (H, W) or (H, W, C) float32 in [0, 1]
    """
    pred_np = np.clip(pred_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)

    # Peak Signal-to-Noise Ratio (handle zero-error identically)
    mse = float(np.mean((pred_np - target_np) ** 2))
    if mse <= 1e-10:
        psnr = 100.0
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))

    # Structural Similarity Index
    ssim = float(structural_similarity(target_np, pred_np, data_range=1.0))

    # Mean Absolute Error
    mae = float(np.mean(np.abs(pred_np - target_np)))

    return {"psnr": psnr, "ssim": ssim, "mae": mae}


def compute_roc_auc_numpy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute exact AUC-ROC via Wilcoxon-Mann-Whitney U rank statistic."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count pairs where pos > neg + 0.5 * (pos == neg)
    diffs = pos[:, None] - neg[None, :]
    u = np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)
    return float(u / (n_pos * n_neg))


def compute_diagnostic_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute clinical disease detection metrics:
      Accuracy, AUC-ROC, Macro F1, Sensitivity (Recall), Specificity, Precision.
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = len(y_true)

    acc = float((tp + tn) / max(total, 1))
    auc = compute_roc_auc_numpy(y_true, y_prob)

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))  # Sensitivity
    specificity = float(tn / max(tn + fp, 1))

    if precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    # Macro F1 (average F1 of class 0 and class 1)
    neg_prec = float(tn / max(tn + fn, 1))
    neg_rec = specificity
    neg_f1 = float(2 * neg_prec * neg_rec / max(neg_prec + neg_rec, 1e-8))
    f1_macro = float((f1 + neg_f1) / 2.0)

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "f1": f1,
        "f1_macro": f1_macro,
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
    }
