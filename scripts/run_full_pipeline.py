"""End-to-end orchestration runner for the 3-step pipeline."""
import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import set_seed, get_device, load_config
from src.degradations.pipeline import DegradationPipeline
from src.data.cxr_dataset import CXRDataset
from src.data.download_kaggle import setup_kaggle_dataset, generate_synthetic_samples
from src.models.degradation_detector import DegradationDetector
from src.models.adaptive_unet import AdaptiveUNet
from src.models.disease_classifier import DiseaseClassifier
from src.training.train_detector import train_detector
from src.training.train_enhancer import train_enhancer
from src.evaluation.compare_enhancement import evaluate_enhancement_pipeline
from src.evaluation.visualize import plot_sample_comparison


def main():
    parser = argparse.ArgumentParser(description="Adaptive Degradation-Aware CXR Enhancement Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train_detector", "train_enhancer", "eval"], help="Pipeline mode")
    parser.add_argument("--step", type=str, default=None, choices=["1", "2", "3", "eval"], help="Convenience alias to run a specific step (1, 2, or 3)")
    parser.add_argument("--epochs_detector", type=int, default=None, help="Override detector epochs")
    parser.add_argument("--epochs_enhancer", type=int, default=None, help="Override enhancer epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--quick_run", action="store_true", help="Run fast verification on mini-dataset")
    args = parser.parse_args()

    # Map step alias if provided
    if args.step == "1":
        args.mode = "train_detector"
    elif args.step == "2":
        args.mode = "train_enhancer"
    elif args.step in ["3", "eval"]:
        args.mode = "eval"

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(preferred=cfg["project"]["device"], num_threads=cfg["project"]["num_threads"])
    print(f"[Pipeline] Running on device: {device}")

    # Prepare Dataset
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not (dataset_dir / "train" / "NORMAL").exists() or not (dataset_dir / "train" / "PNEUMONIA").exists():
        print("[Pipeline] Setting up dataset...")
        setup_kaggle_dataset(str(dataset_dir))
        # If still empty (e.g. no Kaggle credentials), provide procedural CXR phantoms for verification
        if not (dataset_dir / "train" / "NORMAL").exists():
            print("[Pipeline] Initializing procedural CXR benchmark samples...")
            generate_synthetic_samples(str(dataset_dir), num_train=60 if args.quick_run else 120, num_test=24)

    image_size = tuple(cfg["data"]["image_size"])
    batch_size = args.batch_size or (8 if args.quick_run else cfg["data"]["batch_size"])

    train_ds = CXRDataset(dataset_dir, split="train", image_size=image_size)
    val_ds = CXRDataset(dataset_dir, split="val", image_size=image_size, deterministic=True)
    test_ds = CXRDataset(dataset_dir, split="test", image_size=image_size, deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[Pipeline] Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Instantiate Models
    # Model 1: Degradation Detector
    det_ckpt_dir = Path(cfg["detector"]["checkpoint_dir"])
    best_det_path = det_ckpt_dir / "best_detector.pth"
    # Only use pretrained weights if no trained checkpoint exists yet
    detector_pretrained = cfg["detector"]["pretrained"] and not best_det_path.exists()
    detector = DegradationDetector(
        in_channels=cfg["detector"]["in_channels"],
        num_degradations=cfg["detector"]["num_degradations"],
        pretrained=detector_pretrained,
    )

    # Model 2: Adaptive FiLM U-Net
    enhancer = AdaptiveUNet(
        in_channels=cfg["enhancer"]["in_channels"],
        out_channels=cfg["enhancer"]["out_channels"],
        base_channels=cfg["enhancer"]["base_channels"],
        cond_dim=cfg["enhancer"]["cond_dim"],
        residual=True,
    )

    enh_ckpt_dir = Path(cfg["enhancer"]["checkpoint_dir"])

    # Phase 1: Train Model 1 (Problem Detective)
    if args.mode in ["all", "train_detector"]:
        epochs_det = args.epochs_detector or (2 if args.quick_run else cfg["detector"]["epochs"])
        print(f"\n{'='*25} STEP 1: TRAINING MODEL 1 (DETECTOR) ({epochs_det} EPOCHS) {'='*25}")
        train_detector(
            model=detector,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs_det,
            learning_rate=cfg["detector"]["learning_rate"],
            checkpoint_dir=str(det_ckpt_dir),
        )

    # Load best detector checkpoint if exists
    best_det_path = det_ckpt_dir / "best_detector.pth"
    if best_det_path.exists():
        ckpt = torch.load(best_det_path, map_location=device)
        detector.load_state_dict(ckpt["model_state_dict"])
        print(f"[Pipeline] Loaded Model 1 weights from {best_det_path}")

    # Phase 2: Train Model 2 (Adaptive FiLM Enhancer)
    if args.mode in ["all", "train_enhancer"]:
        epochs_enh = args.epochs_enhancer or (3 if args.quick_run else cfg["enhancer"]["epochs"])
        print(f"\n{'='*25} STEP 2: TRAINING MODEL 2 (ADAPTIVE ENHANCER) ({epochs_enh} EPOCHS) {'='*25}")
        train_enhancer(
            model=enhancer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            detector=detector,
            epochs=epochs_enh,
            learning_rate=cfg["enhancer"]["learning_rate"],
            checkpoint_dir=str(enh_ckpt_dir),
        )

    # Load best enhancer checkpoint if exists
    best_enh_path = enh_ckpt_dir / "best_enhancer.pth"
    if best_enh_path.exists():
        ckpt = torch.load(best_enh_path, map_location=device)
        enhancer.load_state_dict(ckpt["model_state_dict"])
        print(f"[Pipeline] Loaded Model 2 weights from {best_enh_path}")

    # Phase 3: Downstream Evaluation (Model 3 AI Radiologists)
    if args.mode in ["all", "eval"]:
        print(f"\n{'='*25} STEP 3: CLINICAL DOWNSTREAM EVALUATION {'='*25}")
        disease_models = {}
        disease_ckpt_dir = Path(cfg["disease_models"]["checkpoint_dir"])
        disease_ckpt_dir.mkdir(parents=True, exist_ok=True)

        for arch in cfg["disease_models"]["architectures"]:
            ckpt_path = disease_ckpt_dir / f"{arch}.pth"
            if ckpt_path.exists():
                print(f"[Pipeline] Loaded existing trained {arch} disease model from {ckpt_path} (no download needed).")
                model = DiseaseClassifier(architecture=arch, num_classes=2, pretrained=False)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                print(f"[Pipeline] Calibrating {arch} linear head on clean baseline images (4 epochs)...")
                model = DiseaseClassifier(architecture=arch, num_classes=2, pretrained=True)
                model.fit_classifier_head(train_loader, device=device, epochs=4, learning_rate=1e-3)
                torch.save(model.state_dict(), ckpt_path)

            disease_models[arch] = model

        report = evaluate_enhancement_pipeline(
            test_loader=test_loader,
            detector=detector,
            enhancer=enhancer,
            disease_models=disease_models,
            device=device,
            output_dir=cfg["evaluation"]["results_dir"],
        )

        # Print summary
        report_md_path = Path(cfg["evaluation"]["results_dir"]) / "evaluation_report.md"
        if report_md_path.exists():
            print("\n" + report_md_path.read_text())

        # Generate sample visualization
        print("[Pipeline] Generating sample visual comparison...")
        sample_batch = next(iter(test_loader))
        deg_sample, clean_sample, vec_sample, label_sample = sample_batch

        detector.eval()
        enhancer.eval()
        with torch.no_grad():
            pred_v = detector.predict_vector(deg_sample[:1].to(device))
            restored = enhancer(deg_sample[:1].to(device), pred_v)

            probs_dict = {}
            for name, m in disease_models.items():
                m.eval()
                p_clean = m.predict_proba(clean_sample[:1].to(device))[0, 1].item()
                p_deg = m.predict_proba(deg_sample[:1].to(device))[0, 1].item()
                p_enh = m.predict_proba(restored)[0, 1].item()
                probs_dict[name] = {"Clean": p_clean, "Degraded": p_deg, "Adaptive": p_enh}

        plot_sample_comparison(
            clean_np=clean_sample[0, 0].numpy(),
            degraded_np=deg_sample[0, 0].numpy(),
            restored_np=restored[0, 0].cpu().numpy(),
            gt_vector=vec_sample[0].numpy(),
            pred_vector=pred_v[0].cpu().numpy(),
            probs_dict=probs_dict,
            save_path=f"{cfg['evaluation']['visualizations_dir']}/sample_comparison.png",
        )

    print("\n[Pipeline] Completed successfully!")


if __name__ == "__main__":
    main()
