"""Domain-adaptive paired fine-tuning of Model 3 classifiers on Clean + FiLM-Restored CXR scans."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cxr_dataset import CXRDataset
from src.degradations.pipeline import DegradationPipeline
from src.evaluation.metrics import compute_diagnostic_metrics
from src.models.adaptive_unet import AdaptiveUNet
from src.models.degradation_detector import DegradationDetector
from src.models.disease_classifier import DiseaseClassifier
from src.utils.helpers import load_config, set_seed


def _sample_diverse_degradation(pipeline: DegradationPipeline, clean_np: np.ndarray) -> np.ndarray:
    """Sample diverse corruptions including isolated extremes and heavy mixes."""
    r = random.random()
    if r < 0.30:
        # Isolated single strong degradation (0.35 - 0.95)
        vec = np.zeros(4, dtype=np.float32)
        idx = random.randrange(4)
        vec[idx] = float(np.random.uniform(0.35, 0.95))
        deg, _ = pipeline.apply_vector(clean_np, vec)
        return deg
    if r < 0.55:
        # Heavy / extreme multi-degradation
        deg, _ = pipeline.apply_random(clean_np, mode="any", min_severity=0.45, max_severity=0.95)
        return deg
    # Standard mixed degradation
    deg, _ = pipeline.apply_random(clean_np, mode="any", min_severity=0.20, max_severity=0.85)
    return deg


def _build_paired_split(
    ds: CXRDataset,
    indices: List[int],
    detector: DegradationDetector,
    enhancer: AdaptiveUNet,
    pipeline: DegradationPipeline,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-compute (clean, restored, pred_vec, label) tensors for fast CPU epoch loops."""
    cleans: List[torch.Tensor] = []
    degradeds: List[torch.Tensor] = []
    labels: List[int] = []

    for idx in indices:
        file_path, label = ds.samples[idx]
        clean_np = ds._load_image(file_path)
        deg_np = _sample_diverse_degradation(pipeline, clean_np)
        cleans.append(torch.from_numpy(clean_np).unsqueeze(0).float())
        degradeds.append(torch.from_numpy(deg_np).unsqueeze(0).float())
        labels.append(label)

    clean_t = torch.stack(cleans)
    deg_t = torch.stack(degradeds)
    label_t = torch.tensor(labels, dtype=torch.long)

    restored_chunks: List[torch.Tensor] = []
    pv_chunks: List[torch.Tensor] = []
    detector.eval()
    enhancer.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_deg = deg_t[start : start + batch_size]
            pv = detector.predict_vector(batch_deg)
            rest = torch.clamp(enhancer(batch_deg, pv), 0.0, 1.0)
            restored_chunks.append(rest)
            pv_chunks.append(pv)

    return clean_t, torch.cat(restored_chunks, dim=0), torch.cat(pv_chunks, dim=0), label_t


def _configure_trainable_layers(model: DiseaseClassifier, arch: str) -> None:
    """Unfreeze upper semantic blocks + classifier head for fast, stable adaptation."""
    for name, param in model.model.named_parameters():
        if arch == "efficientnet_b0":
            param.requires_grad = any(k in name for k in ["features.5", "features.6", "features.7", "features.8", "classifier"])
        elif arch == "densenet121":
            param.requires_grad = any(k in name for k in ["denseblock4", "norm5", "classifier"])
        elif arch == "resnet50":
            param.requires_grad = any(k in name for k in ["layer4", "fc"])
        else:
            param.requires_grad = True


def _finetune_single_model(
    arch: str,
    ckpt_path: Path,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
) -> DiseaseClassifier:
    """Train one DiseaseClassifier with paired clean+restored consistency loss."""
    model = DiseaseClassifier(architecture=arch, num_classes=2, pretrained=not ckpt_path.exists())
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    _configure_trainable_layers(model, arch)

    clean_tr, rest_tr, _, y_tr = train_data
    clean_va, rest_va, _, y_va = val_data

    tr_loader = DataLoader(TensorDataset(clean_tr, rest_tr, y_tr), batch_size=16, shuffle=True)
    va_loader = DataLoader(TensorDataset(clean_va, rest_va, y_va), batch_size=32, shuffle=False)

    head_params, body_params = [], []
    for name, p in model.model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name or "fc" in name:
            head_params.append(p)
        else:
            body_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": lr},
            {"params": head_params, "lr": lr * 4.0},
        ],
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_score = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        for b_clean, b_rest, b_y in tr_loader:
            optimizer.zero_grad()
            # Concatenate clean + restored pairs into one efficient batch pass
            x_cat = torch.cat([b_clean, b_rest], dim=0)
            y_cat = torch.cat([b_y, b_y], dim=0)
            logits_cat = model(x_cat)
            n = b_clean.size(0)
            l_clean, l_rest = logits_cat[:n], logits_cat[n:]

            loss_ce = criterion(logits_cat, y_cat)
            # Force restored predictions to remain consistent with clean lung predictions
            p_clean = F.softmax(l_clean.detach(), dim=1)
            loss_cons = F.kl_div(F.log_softmax(l_rest, dim=1), p_clean, reduction="batchmean")
            # Penalize raw logit margin drift between clean and restored
            diff_clean = (l_clean[:, 1] - l_clean[:, 0]).detach()
            diff_rest = l_rest[:, 1] - l_rest[:, 0]
            loss_margin = F.smooth_l1_loss(diff_rest, diff_clean)

            loss = loss_ce + 0.5 * loss_cons + 0.1 * loss_margin
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * n

        scheduler.step()

        # Validation evaluation on both clean and restored cohorts
        model.eval()
        p_c_all, p_r_all, y_all = [], [], []
        with torch.no_grad():
            for b_clean, b_rest, b_y in va_loader:
                p_c_all.extend(model.predict_proba(b_clean)[:, 1].tolist())
                p_r_all.extend(model.predict_proba(b_rest)[:, 1].tolist())
                y_all.extend(b_y.tolist())

        m_clean = compute_diagnostic_metrics(y_all, p_c_all, 0.50)
        m_rest = compute_diagnostic_metrics(y_all, p_r_all, 0.50)
        score = 0.35 * m_clean["accuracy"] + 0.65 * m_rest["accuracy"]
        print(
            f"[{arch:15s}] Ep {ep}/{epochs} ({time.time()-t_ep:.1f}s) | "
            f"Loss={running_loss/len(y_tr):.4f} | "
            f"Val Clean: Acc={m_clean['accuracy']*100:.1f}% AUC={m_clean['auc_roc']:.3f} | "
            f"Val Restored: Acc={m_rest['accuracy']*100:.1f}% AUC={m_rest['auc_roc']:.3f} "
            f"(Sens={m_rest['sensitivity']*100:.1f}%, Spec={m_rest['specificity']*100:.1f}%)",
            flush=True,
        )
        if score >= best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    torch.save(model.state_dict(), ckpt_path)
    print(f"[{arch}] Saved best domain-adapted weights to {ckpt_path}", flush=True)
    return model


def _fit_calibration_on_val(
    classifiers: Dict[str, DiseaseClassifier],
    val_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    out_path: Path,
) -> None:
    """Fit clean + degradation-aware Platt calibration strictly on held-out val tensors."""
    clean_va, rest_va, pv_va, y_va = val_data
    x_all = torch.cat([clean_va, rest_va], dim=0)
    vmag = torch.cat([torch.zeros(pv_va.size(0)), pv_va.mean(dim=1)], dim=0).numpy()
    y = torch.cat([y_va, y_va], dim=0).numpy().astype(np.float64)

    weights = {"efficientnet_b0": 0.50, "densenet121": 0.30, "resnet50": 0.20}
    calibrated: Dict[str, Dict[str, float]] = {}
    ens_probs = np.zeros_like(y)

    with torch.no_grad():
        for arch, clf in classifiers.items():
            clf.eval()
            logits = clf(x_all)
            d = (logits[:, 1] - logits[:, 0]).numpy().astype(np.float64)

            def loss_fn(params):
                b0, b1, log_t = params
                t = np.exp(log_t)
                z = np.clip((d - (b0 + b1 * vmag)) / t, -30.0, 30.0)
                pr = 1.0 / (1.0 + np.exp(-z))
                eps = 1e-7
                nll = -np.mean(1.15 * y * np.log(pr + eps) + (1.0 - y) * np.log(1.0 - pr + eps))
                reg = 1e-3 * (b0**2 + 0.2 * b1**2 + (log_t - np.log(1.5)) ** 2)
                return nll + reg

            res = minimize(loss_fn, x0=[0.0, 0.0, np.log(1.5)], method="BFGS")
            b0 = float(res.x[0])
            b1 = float(res.x[1])
            temp = float(np.clip(np.exp(res.x[2]), 0.8, 5.0))
            calibrated[arch] = {
                "bias_base": round(b0, 3),
                "bias_deg_slope": round(b1, 3),
                "temperature": round(temp, 3),
            }
            pr = 1.0 / (1.0 + np.exp(-np.clip((d - (b0 + b1 * vmag)) / temp, -30.0, 30.0)))
            ens_probs += weights[arch] * pr

    half = len(y_va)
    m_c = compute_diagnostic_metrics(y[:half], ens_probs[:half], 0.50)
    m_r = compute_diagnostic_metrics(y[half:], ens_probs[half:], 0.50)
    print(
        f"\n[Calibration] Val Clean: Acc={m_c['accuracy']*100:.1f}% Sens={m_c['sensitivity']*100:.1f}% Spec={m_c['specificity']*100:.1f}% | "
        f"Val Restored: Acc={m_r['accuracy']*100:.1f}% Sens={m_r['sensitivity']*100:.1f}% Spec={m_r['specificity']*100:.1f}%",
        flush=True,
    )

    payload = {
        "description": "Platt calibration parameters fitted on held-out train-val split after paired clean+restored domain adaptation",
        "ensemble_weights": weights,
        "triage_thresholds": {"high_risk": 0.75, "borderline": 0.45},
        "classifiers": calibrated,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[Calibration] Wrote {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Domain-adaptive paired fine-tuning for Model 3")
    parser.add_argument("--train_per_class", type=int, default=320, help="Balanced train images per class")
    parser.add_argument("--val_per_class", type=int, default=80, help="Balanced val images per class")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1.2e-4, help="Base backbone learning rate")
    args = parser.parse_args()

    t0 = time.time()
    torch.set_num_threads(8)
    cfg = load_config("configs/default_config.yaml")
    set_seed(cfg["project"]["seed"])

    pipeline = DegradationPipeline()
    train_ds = CXRDataset(cfg["data"]["dataset_dir"], split="train", image_size=(224, 224), pipeline=pipeline)

    norm_idxs = [i for i, (_, lbl) in enumerate(train_ds.samples) if lbl == 0]
    pneu_idxs = [i for i, (_, lbl) in enumerate(train_ds.samples) if lbl == 1]
    random.shuffle(norm_idxs)
    random.shuffle(pneu_idxs)

    tr_n = norm_idxs[: args.train_per_class]
    va_n = norm_idxs[args.train_per_class : args.train_per_class + args.val_per_class]
    tr_p = pneu_idxs[: args.train_per_class]
    va_p = pneu_idxs[args.train_per_class : args.train_per_class + args.val_per_class]

    tr_indices = tr_n + tr_p
    va_indices = va_n + va_p
    random.shuffle(tr_indices)

    det = DegradationDetector(1, 4, pretrained=False)
    det.load_state_dict(torch.load(Path(cfg["detector"]["checkpoint_dir"]) / "best_detector.pth", map_location="cpu")["model_state_dict"])
    enh = AdaptiveUNet(1, 1, 32, 4, residual=True)
    enh.load_state_dict(torch.load(Path(cfg["enhancer"]["checkpoint_dir"]) / "best_enhancer.pth", map_location="cpu")["model_state_dict"])

    print(f"[Setup] Caching {len(tr_indices)} train pairs + {len(va_indices)} val pairs...", flush=True)
    tc = time.time()
    train_data = _build_paired_split(train_ds, tr_indices, det, enh, pipeline)
    val_data = _build_paired_split(train_ds, va_indices, det, enh, pipeline)
    print(f"[Setup] Cached all pairs in {time.time()-tc:.1f}s", flush=True)

    disease_dir = Path(cfg["disease_models"]["checkpoint_dir"])
    trained_models: Dict[str, DiseaseClassifier] = {}
    for arch in ["efficientnet_b0", "densenet121", "resnet50"]:
        epochs = args.epochs + 1 if arch == "efficientnet_b0" else args.epochs
        trained_models[arch] = _finetune_single_model(
            arch=arch,
            ckpt_path=disease_dir / f"{arch}.pth",
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            lr=args.lr,
        )

    _fit_calibration_on_val(trained_models, val_data, Path("configs/calibration.json"))
    print(f"\n[Done] Total domain adaptation finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
