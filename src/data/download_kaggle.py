"""Utility to download, verify, and organize the Kaggle Chest X-Ray (Pneumonia) dataset."""
import os
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional


KAGGLE_DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
DIRECT_MIRROR_URL = "https://github.com/ieee8023/covid-chestxray-dataset/archive/refs/heads/master.zip"  # reference fallback


def setup_kaggle_dataset(target_dir: str = "data/chest_xray") -> Path:
    """
    Ensure the Chest X-Ray dataset is present in target_dir.
    Checks directory structure:
      target_dir/
        train/NORMAL, train/PNEUMONIA
        val/NORMAL, val/PNEUMONIA
        test/NORMAL, test/PNEUMONIA
    """
    dest = Path(target_dir).resolve()
    if (dest / "train" / "NORMAL").exists() and (dest / "train" / "PNEUMONIA").exists():
        train_normal = len(list((dest / "train" / "NORMAL").glob("*.*")))
        train_pneu = len(list((dest / "train" / "PNEUMONIA").glob("*.*")))
        if train_normal > 0 and train_pneu > 0:
            print(f"[Dataset] Found existing dataset at {dest} ({train_normal} normal, {train_pneu} pneumonia in train).")
            return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Check if kaggle CLI is installed and configured
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print(f"[Dataset] Downloading via Kaggle API: {KAGGLE_DATASET_SLUG}...")
        try:
            import subprocess
            cmd = f"kaggle datasets download -d {KAGGLE_DATASET_SLUG} -p {dest.parent} --unzip"
            subprocess.run(cmd, shell=True, check=True)
            # Kaggle unzips to chest_xray or similar
            possible_unzipped = dest.parent / "chest_xray"
            if possible_unzipped.exists() and possible_unzipped != dest:
                for item in possible_unzipped.iterdir():
                    shutil.move(str(item), str(dest))
            return dest
        except Exception as e:
            print(f"[Dataset] Kaggle CLI download encountered an issue: {e}")

    print(f"[Dataset] Kaggle credentials not found at {kaggle_json}.")
    print("[Dataset] Please either configure ~/.kaggle/kaggle.json or place chest_xray folder at data/chest_xray/")
    return dest


def generate_synthetic_samples(target_dir: str = "data/chest_xray", num_train: int = 100, num_test: int = 30) -> Path:
    """
    Generates realistic procedural chest X-ray phantoms for quick testing and development.
    Simulates thoracic cage, lung parenchyma, and radiopaque infiltrates for pneumonia.
    """
    import cv2
    import numpy as np

    dest = Path(target_dir).resolve()
    splits = {"train": num_train, "val": max(num_test // 2, 10), "test": num_test}

    for split_name, count in splits.items():
        for label_name in ["NORMAL", "PNEUMONIA"]:
            folder = dest / split_name / label_name
            folder.mkdir(parents=True, exist_ok=True)

            existing = list(folder.glob("*.jpeg"))
            if len(existing) >= count:
                continue

            for i in range(count):
                img = _create_procedural_cxr(is_pneumonia=(label_name == "PNEUMONIA"), seed=i + (1000 if label_name == "PNEUMONIA" else 0))
                cv2.imwrite(str(folder / f"sample_{i:04d}.jpeg"), (img * 255).astype(np.uint8))

    print(f"[Dataset] Created procedural benchmark dataset at {dest} ({num_train} train per class).")
    return dest


def _create_procedural_cxr(is_pneumonia: bool = False, seed: int = 0) -> "np.ndarray":
    """Procedurally synthesize a realistic chest radiograph phantom."""
    import numpy as np
    import cv2

    np.random.seed(seed)
    h, w = 224, 224
    img = np.zeros((h, w), dtype=np.float32)

    # Mediastinum and soft tissue outline (elliptical background)
    cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 10, h // 2 - 5), 0, 0, 360, 0.25, -1)

    # Lung fields (two low-density dark regions)
    # Left lung
    cv2.ellipse(img, (w // 2 - 45, h // 2 + 5), (32, 65), -5, 0, 360, 0.08, -1)
    # Right lung
    cv2.ellipse(img, (w // 2 + 45, h // 2 + 5), (32, 65), 5, 0, 360, 0.08, -1)

    # Cardiac silhouette (dense radiopaque heart region towards left lung base)
    cv2.ellipse(img, (w // 2 - 15, h // 2 + 30), (28, 38), -25, 0, 360, 0.65, -1)

    # Spine column
    cv2.line(img, (w // 2, 20), (w // 2, h - 20), 0.55, 12)

    # Rib cage shadows (bilateral arcs)
    for y_offset in range(40, h - 40, 22):
        cv2.ellipse(img, (w // 2, y_offset), (w // 2 - 25, 20), 0, 0, 180, 0.35, 3)

    # Clavicles (collar bones)
    cv2.line(img, (w // 2 - 70, 45), (w // 2 - 10, 52), 0.60, 4)
    cv2.line(img, (w // 2 + 70, 45), (w // 2 + 10, 52), 0.60, 4)

    # If pneumonia, add patchy alveolar consolidations / opacities in lower/middle lung fields
    if is_pneumonia:
        side = -1 if np.random.rand() > 0.5 else 1
        cx = w // 2 + side * np.random.randint(25, 55)
        cy = h // 2 + np.random.randint(10, 45)
        for _ in range(5):
            ox = cx + np.random.randint(-15, 15)
            oy = cy + np.random.randint(-15, 15)
            r = np.random.randint(12, 24)
            intensity = np.random.uniform(0.35, 0.65)
            cv2.circle(img, (ox, oy), r, intensity, -1)

    # Smooth anatomical transitions with Gaussian filter
    img = cv2.GaussianBlur(img, (9, 9), 3.0)

    # Add anatomical texture
    texture = np.random.normal(0.0, 0.02, (h, w)).astype(np.float32)
    img = np.clip(img + texture, 0.0, 1.0)
    return img
