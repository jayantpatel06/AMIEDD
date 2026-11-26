"""Dedicated runner to execute Step 3: Clinical Downstream Evaluation directly."""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import set_seed, get_device, load_config
from src.data.cxr_dataset import CXRDataset
from src.models.degradation_detector import DegradationDetector
from src.models.adaptive_unet import AdaptiveUNet
from src.models.disease_classifier import DiseaseClassifier
from src.models.consensus import load_calibration_config, predict_consensus_batch
from src.evaluation.compare_enhancement import evaluate_enhancement_pipeline
from src.evaluation.visualize import plot_sample_comparison


def main():
    print("=" * 60)
    print("  STEP 3: CLINICAL DOWNSTREAM EVALUATION (OFFLINE MODE)  ")
    print("=" * 60)

    cfg = load_config("configs/default_config.yaml")
    set_seed(cfg["project"]["seed"])
    device = get_device(preferred="cpu", num_threads=cfg["project"]["num_threads"])
    print(f"[Step 3] Device: {device}")

    # Load test dataset
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    image_size = tuple(cfg["data"]["image_size"])
    test_ds = CXRDataset(dataset_dir, split="test", image_size=image_size, deterministic=True)
    num_workers = 0 if sys.platform.startswith("win") else 2
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)
    print(f"[Step 3] Loaded {len(test_ds)} unique test images from {dataset_dir / 'test'}")

    # Load Model 1 (Problem Detective)
    det_ckpt = Path(cfg["detector"]["checkpoint_dir"]) / "best_detector.pth"
    detector = DegradationDetector(in_channels=1, num_degradations=4, pretrained=False)
    if det_ckpt.exists():
        ckpt = torch.load(det_ckpt, map_location=device)
        detector.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"[Step 3] Loaded Model 1 (Detector) from {det_ckpt}")
    else:
        print("[Step 3] WARNING: Model 1 checkpoint not found. Using initialized weights.")
    detector.eval()

    # Load Model 2 (Adaptive FiLM Enhancer)
    enh_ckpt = Path(cfg["enhancer"]["checkpoint_dir"]) / "best_enhancer.pth"
    enhancer = AdaptiveUNet(in_channels=1, out_channels=1, base_channels=32, cond_dim=4, residual=True)
    if enh_ckpt.exists():
        ckpt = torch.load(enh_ckpt, map_location=device)
        enhancer.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"[Step 3] Loaded Model 2 (Enhancer) from {enh_ckpt}")
    else:
        print("[Step 3] WARNING: Model 2 checkpoint not found. Using initialized weights.")
    enhancer.eval()

    # Load Model 3 (Disease Classifiers)
    disease_models = {}
    disease_ckpt_dir = Path(cfg["disease_models"]["checkpoint_dir"])
    for arch in cfg["disease_models"]["architectures"]:
        ckpt_path = disease_ckpt_dir / f"{arch}.pth"
        model = DiseaseClassifier(architecture=arch, num_classes=2, pretrained=False)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[Step 3] Loaded Model 3 ({arch}) from {ckpt_path}")
        else:
            print(f"[Step 3] WARNING: {arch} checkpoint not found. Using initialized weights.")
        disease_models[arch] = model

    # Execute Evaluation
    print("\n[Step 3] Running downstream benchmark on Clean, Degraded, Generic, and Adaptive cohorts...")
    report = evaluate_enhancement_pipeline(
        test_loader=test_loader,
        detector=detector,
        enhancer=enhancer,
        disease_models=disease_models,
        device=device,
        output_dir=cfg["evaluation"]["results_dir"],
    )

    # Print Report
    report_md_path = Path(cfg["evaluation"]["results_dir"]) / "evaluation_report.md"
    if report_md_path.exists():
        print("\n" + report_md_path.read_text())

    # Generate visual comparison figure
    print("[Step 3] Generating 5-panel sample comparison figure...")
    sample_batch = next(iter(test_loader))
    deg_s, clean_s, vec_s, label_s = sample_batch

    calibration_cfg, ensemble_weights, _ = load_calibration_config()
    with torch.no_grad():
        pred_v = detector.predict_vector(deg_s[:1].to(device))
        restored = torch.clamp(enhancer(deg_s[:1].to(device), pred_v), 0.0, 1.0)
        zero_v = torch.zeros_like(pred_v)

        p_clean_ens, per_clean = predict_consensus_batch(disease_models, clean_s[:1].to(device), zero_v, ensemble_weights, calibration_cfg)
        p_deg_ens, per_deg = predict_consensus_batch(disease_models, deg_s[:1].to(device), pred_v, ensemble_weights, calibration_cfg)
        p_enh_ens, per_enh = predict_consensus_batch(disease_models, restored, pred_v, ensemble_weights, calibration_cfg)

        probs_dict = {
            "Consensus Ensemble": {
                "Clean": float(p_clean_ens[0].item()),
                "Degraded": float(p_deg_ens[0].item()),
                "Adaptive": float(p_enh_ens[0].item()),
            }
        }
        for name in disease_models.keys():
            probs_dict[name] = {
                "Clean": float(per_clean[name][0].item()),
                "Degraded": float(per_deg[name][0].item()),
                "Adaptive": float(per_enh[name][0].item()),
            }

    plot_sample_comparison(
        clean_np=clean_s[0, 0].numpy(),
        degraded_np=deg_s[0, 0].numpy(),
        restored_np=restored[0, 0].cpu().numpy(),
        gt_vector=vec_s[0].numpy(),
        pred_vector=pred_v[0].cpu().numpy(),
        probs_dict=probs_dict,
        save_path=f"{cfg['evaluation']['visualizations_dir']}/sample_comparison.png",
    )

    print(f"\n[Step 3] Done! Report saved to {cfg['evaluation']['results_dir']}/evaluation_report.md")
    print(f"[Step 3] Visualization saved to {cfg['evaluation']['visualizations_dir']}/sample_comparison.png")


if __name__ == "__main__":
    main()
