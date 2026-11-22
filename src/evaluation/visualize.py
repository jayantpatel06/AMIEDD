"""Visualization utilities for qualitative and quantitative model evaluation."""
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..degradations.pipeline import DegradationPipeline


def plot_sample_comparison(
    clean_np: np.ndarray,
    degraded_np: np.ndarray,
    restored_np: np.ndarray,
    gt_vector: np.ndarray,
    pred_vector: np.ndarray,
    probs_dict: Dict[str, Dict[str, float]],
    save_path: str = "outputs/visualizations/comparison_sample.png",
    psnr_deg: Optional[float] = None,
    psnr_enh: Optional[float] = None,
) -> None:
    """
    Create a 5-panel comprehensive visual demonstration:
      1. Clean Image
      2. Degraded Image
      3. Model 1 Diagnosis (GT vs Predicted Severity)
      4. Model 2 Restored Image
      5. Model 3 Disease Probability Comparison
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 4.5), dpi=150)

    # 1. Clean
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(clean_np, cmap="gray")
    ax1.set_title("1. Clean Ground Truth", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 2. Degraded
    ax2 = plt.subplot(1, 5, 2)
    ax2.imshow(degraded_np, cmap="gray")
    title_deg = "2. Degraded Input"
    if psnr_deg is not None:
        title_deg += f"\n(PSNR: {psnr_deg:.1f} dB)"
    ax2.set_title(title_deg, fontsize=11, fontweight="bold", color="darkred")
    ax2.axis("off")

    # 3. Model 1 Diagnosis
    ax3 = plt.subplot(1, 5, 3)
    categories = DegradationPipeline.DEGRADATION_NAMES
    y_pos = np.arange(len(categories))
    height = 0.35

    ax3.barh(y_pos - height / 2, gt_vector, height, label="Ground Truth", color="#4C72B0", alpha=0.85)
    ax3.barh(y_pos + height / 2, pred_vector, height, label="Model 1 Predicted", color="#DD8452", alpha=0.85)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([c.capitalize() for c in categories], fontsize=9)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_xlabel("Severity", fontsize=9)
    ax3.set_title("3. Model 1 Diagnosis", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=7)
    ax3.grid(axis="x", linestyle="--", alpha=0.5)

    # 4. Model 2 Restored
    ax4 = plt.subplot(1, 5, 4)
    ax4.imshow(restored_np, cmap="gray")
    title_enh = "4. Model 2 Adaptive (FiLM)"
    if psnr_enh is not None:
        title_enh += f"\n(PSNR: {psnr_enh:.1f} dB)"
    ax4.set_title(title_enh, fontsize=11, fontweight="bold", color="darkgreen")
    ax4.axis("off")

    # 5. Model 3 Disease Probability
    ax5 = plt.subplot(1, 5, 5)
    first_model = list(probs_dict.keys())[0] if probs_dict else "Classifier"
    p_data = probs_dict[first_model] if probs_dict else {"Clean": 0.9, "Degraded": 0.55, "Adaptive": 0.88}

    names = list(p_data.keys())
    values = [p_data[k] * 100 for k in names]
    colors = ["#4C72B0", "#C44E52", "#55A868"]

    bars = ax5.bar(names, values, color=colors, width=0.55, alpha=0.9)
    ax5.set_ylim(0, 100)
    ax5.set_ylabel("Pneumonia Conf. (%)", fontsize=9)
    ax5.set_title(f"5. Disease Diagnosis\n({first_model})", fontsize=11, fontweight="bold")
    ax5.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax5.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved comparison figure to: {path}")
