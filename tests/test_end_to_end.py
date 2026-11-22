"""Integration test for mini end-to-end training and evaluation cycle."""
import pytest
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.download_kaggle import generate_synthetic_samples
from src.data.cxr_dataset import CXRDataset
from src.models.degradation_detector import DegradationDetector
from src.models.adaptive_unet import AdaptiveUNet
from src.models.disease_classifier import DiseaseClassifier
from src.training.train_detector import train_detector
from src.training.train_enhancer import train_enhancer
from src.evaluation.compare_enhancement import evaluate_enhancement_pipeline


def test_mini_end_to_end(tmp_path):
    device = torch.device("cpu")
    data_dir = tmp_path / "data"

    # 1. Generate mini procedural dataset
    generate_synthetic_samples(str(data_dir), num_train=12, num_test=6)

    train_ds = CXRDataset(data_dir, split="train", image_size=(128, 128))
    val_ds = CXRDataset(data_dir, split="val", image_size=(128, 128), deterministic=True)
    test_ds = CXRDataset(data_dir, split="test", image_size=(128, 128), deterministic=True)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # 2. Train Model 1 (1 epoch)
    detector = DegradationDetector(in_channels=1, num_degradations=4, pretrained=False)
    det_ckpt = tmp_path / "det_ckpts"
    history_det = train_detector(
        model=detector,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=1,
        checkpoint_dir=str(det_ckpt),
    )
    assert len(history_det["train_loss"]) == 1
    assert (det_ckpt / "last_detector.pth").exists()

    # 3. Train Model 2 (1 epoch)
    enhancer = AdaptiveUNet(in_channels=1, out_channels=1, base_channels=16, cond_dim=4, residual=True)
    enh_ckpt = tmp_path / "enh_ckpts"
    history_enh = train_enhancer(
        model=enhancer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        detector=detector,
        epochs=1,
        checkpoint_dir=str(enh_ckpt),
    )
    assert len(history_enh["train_loss"]) == 1
    assert (enh_ckpt / "last_enhancer.pth").exists()

    # 4. Evaluate with Model 3
    disease_model = DiseaseClassifier(architecture="densenet121", num_classes=2, pretrained=False)
    report = evaluate_enhancement_pipeline(
        test_loader=test_loader,
        detector=detector,
        enhancer=enhancer,
        disease_models={"densenet121": disease_model},
        device=device,
        output_dir=str(tmp_path / "eval"),
    )

    assert "image_restoration_quality" in report
    assert "clinical_diagnosis_performance" in report
    assert (tmp_path / "eval" / "evaluation_report.json").exists()
    assert (tmp_path / "eval" / "evaluation_report.md").exists()
