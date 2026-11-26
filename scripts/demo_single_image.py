"""Interactive single-image demonstration script."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, get_device
from src.degradations.pipeline import DegradationPipeline
from src.models.degradation_detector import DegradationDetector
from src.models.adaptive_unet import AdaptiveUNet
from src.models.disease_classifier import DiseaseClassifier
from src.evaluation.visualize import plot_sample_comparison
from src.evaluation.metrics import compute_image_metrics


def main():
    parser = argparse.ArgumentParser(description="Single-Image Interactive Demonstration")
    parser.add_argument("--image", type=str, default=None, help="Path to clean input X-Ray (optional)")
    parser.add_argument("--noise", type=float, default=0.6, help="Noise severity in [0, 1]")
    parser.add_argument("--blur", type=float, default=0.5, help="Blur severity in [0, 1]")
    parser.add_argument("--contrast", type=float, default=0.0, help="Contrast loss severity in [0, 1]")
    parser.add_argument("--compression", type=float, default=0.4, help="JPEG compression severity in [0, 1]")
    parser.add_argument("--output", type=str, default="outputs/visualizations/demo_single_image.png")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(preferred="cpu")
    pipeline = DegradationPipeline()

    # Load or generate clean image
    if args.image and Path(args.image).exists():
        raw = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        clean = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    else:
        # Search in dataset directory
        dataset_dir = Path(cfg["data"]["dataset_dir"])
        pneumonia_samples = list((dataset_dir / "test" / "PNEUMONIA").glob("*.jpeg"))
        if pneumonia_samples:
            raw = cv2.imread(str(pneumonia_samples[0]), cv2.IMREAD_GRAYSCALE)
            clean = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        else:
            from src.data.download_kaggle import _create_procedural_cxr
            clean = _create_procedural_cxr(is_pneumonia=True, seed=42)

    # Apply degradation vector
    gt_vector = np.array([args.noise, args.blur, args.contrast, args.compression], dtype=np.float32)
    degraded, _ = pipeline.apply_vector(clean, gt_vector)

    # Load Model 1 (Detector)
    detector = DegradationDetector(in_channels=1, num_degradations=4, pretrained=False)
    det_ckpt = Path(cfg["detector"]["checkpoint_dir"]) / "best_detector.pth"
    if det_ckpt.exists():
        ckpt = torch.load(det_ckpt, map_location=device)
        detector.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    detector.eval()

    # Load Model 2 (Enhancer)
    enhancer = AdaptiveUNet(in_channels=1, out_channels=1, base_channels=32, cond_dim=4, residual=True)
    enh_ckpt = Path(cfg["enhancer"]["checkpoint_dir"]) / "best_enhancer.pth"
    if enh_ckpt.exists():
        ckpt = torch.load(enh_ckpt, map_location=device)
        enhancer.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    enhancer.eval()

    # Load Model 3 (Multi-Radiologist Panel: DenseNet121, EfficientNet-B0, ResNet50)
    disease_models = {}
    disease_ckpt_dir = Path(cfg["disease_models"]["checkpoint_dir"])
    for arch in ["densenet121", "efficientnet_b0", "resnet50"]:
        d_ckpt = disease_ckpt_dir / f"{arch}.pth"
        d_model = DiseaseClassifier(architecture=arch, num_classes=2, pretrained=not d_ckpt.exists())
        if d_ckpt.exists():
            d_model.load_state_dict(torch.load(d_ckpt, map_location=device))
        d_model.eval()
        disease_models[arch] = d_model

    # Inference
    deg_t = torch.from_numpy(degraded).unsqueeze(0).unsqueeze(0).float()
    clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        # Step 1: Diagnose
        pred_vector_t = detector.predict_vector(deg_t)
        pred_vector = pred_vector_t[0].numpy()

        # Step 2: Restore
        restored_t = enhancer(deg_t, pred_vector_t)
        restored = restored_t[0, 0].numpy()

        # Step 3: Disease probability across panel
        probs_dict = {}
        for arch, m in disease_models.items():
            p_clean = m.predict_proba(clean_t)[0, 1].item()
            p_deg = m.predict_proba(deg_t)[0, 1].item()
            p_enh = m.predict_proba(restored_t)[0, 1].item()
            probs_dict[arch] = {"Clean": p_clean, "Degraded": p_deg, "Adaptive": p_enh}

    # Compute PSNR
    m_deg = compute_image_metrics(degraded, clean)
    m_enh = compute_image_metrics(restored, clean)

    plot_sample_comparison(
        clean_np=clean,
        degraded_np=degraded,
        restored_np=restored,
        gt_vector=gt_vector,
        pred_vector=pred_vector,
        probs_dict=probs_dict,
        save_path=args.output,
        psnr_deg=m_deg["psnr"],
        psnr_enh=m_enh["psnr"],
    )

    print(f"\n[Demo Results]")
    print(f"Ground Truth Degradation: {pipeline.vector_to_dict(gt_vector)}")
    print(f"Model 1 Diagnosis:        {pipeline.vector_to_dict(pred_vector)}")
    print(f"Image Quality: Degraded PSNR = {m_deg['psnr']:.2f} dB -> Restored PSNR = {m_enh['psnr']:.2f} dB (+{m_enh['psnr']-m_deg['psnr']:.2f} dB)")
    print(f"Disease Probability: Degraded = {p_deg*100:.1f}% -> Restored = {p_enh*100:.1f}% (Clean Baseline = {p_clean*100:.1f}%)")


if __name__ == "__main__":
    main()
