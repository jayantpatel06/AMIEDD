"""Shared calibration, multi-radiologist consensus voting, and clinical triage rules."""
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
import torch

from .disease_classifier import DiseaseClassifier

DEFAULT_ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "efficientnet_b0": 0.50,
    "densenet121": 0.30,
    "resnet50": 0.20,
}

# Degradation-aware Platt scaling defaults fitted on Kaggle val split after paired domain adaptation
DEFAULT_CALIBRATION: Dict[str, Dict[str, float]] = {
    "efficientnet_b0": {"bias_base": 1.037, "bias_deg_slope": 0.339, "temperature": 0.8},
    "densenet121": {"bias_base": 0.987, "bias_deg_slope": 0.533, "temperature": 0.825},
    "resnet50": {"bias_base": 1.796, "bias_deg_slope": 0.0, "temperature": 0.8},
}

DEFAULT_TRIAGE_THRESHOLDS: Dict[str, float] = {
    "high_risk": 0.75,
    "borderline": 0.45,
}


def load_calibration_config(
    config_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """Load calibration, weights, and triage thresholds from JSON with embedded fallbacks."""
    path = config_path or (Path(__file__).resolve().parents[2] / "configs" / "calibration.json")
    calibration = {k: dict(v) for k, v in DEFAULT_CALIBRATION.items()}
    weights = dict(DEFAULT_ENSEMBLE_WEIGHTS)
    thresholds = dict(DEFAULT_TRIAGE_THRESHOLDS)

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for arch, params in data.get("classifiers", {}).items():
            calibration[arch] = {
                "bias_base": float(params["bias_base"]),
                "bias_deg_slope": float(params.get("bias_deg_slope", 0.0)),
                "temperature": float(params["temperature"]),
            }
        if "ensemble_weights" in data:
            weights = {k: float(v) for k, v in data["ensemble_weights"].items()}
        if "triage_thresholds" in data:
            thresholds = {k: float(v) for k, v in data["triage_thresholds"].items()}

    return calibration, weights, thresholds


def calibrated_classifier_proba(
    clf: DiseaseClassifier,
    x: torch.Tensor,
    arch: str,
    degradation_vector: Optional[torch.Tensor] = None,
    calibration: Optional[Dict[str, Dict[str, float]]] = None,
) -> torch.Tensor:
    """
    Compute calibrated pneumonia probability P(pneumonia | x, d) for a single classifier.

    Args:
        clf: Loaded DiseaseClassifier module.
        x: Input tensor of shape (B, 1, H, W).
        arch: Architecture key ('efficientnet_b0', 'densenet121', 'resnet50').
        degradation_vector: Optional 4D tensor (B, 4) from DegradationDetector.
        calibration: Optional calibration dict mapping arch -> Platt parameters.
    Returns:
        1D tensor of calibrated pneumonia probabilities of shape (B,).
    """
    cal_map = calibration if calibration is not None else DEFAULT_CALIBRATION
    logits = clf(x)
    params = cal_map.get(arch)
    if params is None:
        return torch.softmax(logits, dim=1)[:, 1]

    diff = logits[:, 1] - logits[:, 0]
    bias = torch.full_like(diff, float(params["bias_base"]))
    slope = float(params.get("bias_deg_slope", 0.0))
    if degradation_vector is not None and slope != 0.0:
        deg_mag = degradation_vector.mean(dim=1).to(device=diff.device, dtype=diff.dtype)
        bias = bias + slope * deg_mag

    temp = max(float(params["temperature"]), 1e-3)
    return torch.sigmoid((diff - bias) / temp)


def predict_consensus_batch(
    classifiers: Mapping[str, DiseaseClassifier],
    x: torch.Tensor,
    degradation_vector: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    calibration: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Evaluate all available classifiers and combine them via normalized soft-voting.

    Returns:
        consensus: (B,) calibrated ensemble probability tensor.
        per_model: mapping arch -> (B,) calibrated probability tensor.
    """
    weight_map = weights if weights is not None else DEFAULT_ENSEMBLE_WEIGHTS
    active_weights = {
        arch: float(weight_map.get(arch, 1.0))
        for arch in classifiers.keys()
    }
    total_weight = sum(active_weights.values())
    if total_weight <= 0.0:
        total_weight = float(len(classifiers))
        active_weights = {arch: 1.0 for arch in classifiers.keys()}

    per_model: Dict[str, torch.Tensor] = {}
    consensus = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)

    for arch, clf in classifiers.items():
        clf.eval()
        prob = calibrated_classifier_proba(
            clf, x, arch, degradation_vector=degradation_vector, calibration=calibration
        )
        per_model[arch] = prob
        consensus = consensus + (active_weights[arch] / total_weight) * prob

    return consensus, per_model


def classify_triage(
    p_pneumonia: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str, str]:
    """
    Map consensus probability to the 3-tier safety-first clinical triage verdict.
    Returns (has_pneumonia, clinical_risk, recommendation).
    """
    th = thresholds if thresholds is not None else DEFAULT_TRIAGE_THRESHOLDS
    high_risk = float(th.get("high_risk", 0.75))
    borderline = float(th.get("borderline", 0.45))

    if p_pneumonia >= high_risk:
        return (
            True,
            "High Risk (Pneumonia Confirmed by Consensus)",
            "High consensus of radiopaque pulmonary consolidation. Confirmed pneumonia indicated.",
        )
    if p_pneumonia >= borderline:
        return (
            True,  # Safety-first: flag borderline cases for radiologist review
            "Indeterminate / Borderline (Review Required)",
            "Equivocal findings across radiologist models. Flagged for urgent human radiologist review.",
        )
    return (
        False,
        "Low Risk (Normal)",
        "Clear lung fields with high multi-model consensus.",
    )
