"""FastAPI PyTorch Backend Server for Chest X-Ray Restoration & Disease Detection."""
import io
import base64
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(
    title="Chest X-Ray Restoration & Detection API",
    description="PyTorch API for real-time CXR degradation diagnosis, FiLM U-Net restoration, and pneumonia classification.",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite dev server and local network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & calibration state
device = None
detector = None
enhancer = None
classifiers = {}
calibration_cfg = None
ensemble_weights = None
triage_thresholds = None


def _decode_grayscale_224(image_data: str) -> np.ndarray:
    """Decode base64 image data into a [0, 1] float32 224x224 grayscale array matching training preprocessing."""
    payload = image_data.split(",")[1] if "," in image_data else image_data
    raw_bytes = base64.b64decode(payload)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    img_np = np.array(pil_img, dtype=np.float32) / 255.0
    return cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_models():
    """Load or initialize PyTorch models (Detector, FiLM Enhancer, and 3-Model Consensus Panel)."""
    global device, detector, enhancer, classifiers, calibration_cfg, ensemble_weights, triage_thresholds

    import torch
    from src.models.degradation_detector import DegradationDetector
    from src.models.adaptive_unet import AdaptiveUNet
    from src.models.disease_classifier import DiseaseClassifier
    from src.models.consensus import load_calibration_config

    calibration_cfg, ensemble_weights, triage_thresholds = load_calibration_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[API Server] Initializing PyTorch models on: {device}")

    # 1. Model 1: Degradation Detector (ResNet18)
    detector = DegradationDetector(in_channels=1, num_degradations=4, pretrained=False)
    det_ckpt = PROJECT_ROOT / "checkpoints" / "detector" / "best_detector.pth"
    if det_ckpt.exists():
        print(f"[API Server] Loading Detector checkpoint from {det_ckpt}")
        ckpt = torch.load(det_ckpt, map_location=device)
        detector.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    detector.to(device)
    detector.eval()

    # 2. Model 2: Adaptive U-Net Restorer (FiLM Conditioned)
    enhancer = AdaptiveUNet(in_channels=1, out_channels=1, base_channels=32, cond_dim=4, residual=True)
    enh_ckpt = PROJECT_ROOT / "checkpoints" / "enhancer" / "best_enhancer.pth"
    if enh_ckpt.exists():
        print(f"[API Server] Loading Enhancer checkpoint from {enh_ckpt}")
        ckpt = torch.load(enh_ckpt, map_location=device)
        enhancer.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    enhancer.to(device)
    enhancer.eval()

    # 3. Model 3: Multi-Radiologist Consensus Ensemble (DenseNet-121, EfficientNet-B0, ResNet-50)
    classifiers = {}
    disease_ckpt_dir = PROJECT_ROOT / "checkpoints" / "disease_models"
    for arch in ["densenet121", "efficientnet_b0", "resnet50"]:
        ckpt_path = disease_ckpt_dir / f"{arch}.pth"
        clf = DiseaseClassifier(architecture=arch, num_classes=2, pretrained=not ckpt_path.exists())
        if ckpt_path.exists():
            print(f"[API Server] Loading {arch} consensus model from {ckpt_path}")
            clf.load_state_dict(torch.load(ckpt_path, map_location=device))
        clf.to(device)
        clf.eval()
        classifiers[arch] = clf

    print("[API Server] All models (Detector + Enhancer + 3-Model Radiologist Panel) ready!")


class PredictRequest(BaseModel):
    image: str  # Base64 data URL or raw base64 string
    ground_truth: Optional[str] = "pneumonia"


class DegradeRequest(BaseModel):
    image: str  # Base64 data URL of clean image
    vector: list[float]  # [noise, blur, contrast, compression] in [0, 1]


class DegradeResponse(BaseModel):
    status: str
    degraded_image: str  # Base64 data URL
    applied_vector: list[float]


class PredictResponse(BaseModel):
    status: str
    cleaned_image: str  # Base64 data URL
    has_pneumonia: bool
    pneumonia_percent: float
    normal_percent: float
    corrupted_pneumonia_percent: float
    diagnosed_vector: Dict[str, float]
    device_used: str
    clinical_risk: str
    recommendation: str
    model_consensus: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    try:
        load_models()
    except Exception as e:
        print(f"[API Server Warning] Could not load all models immediately: {e}")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "PyTorch",
        "device": str(device) if device else "cpu",
        "models_ready": detector is not None and enhancer is not None and len(classifiers) == 3,
        "active_classifiers": list(classifiers.keys())
    }


@app.post("/api/pipeline/degrade", response_model=DegradeResponse)
def degrade_image(req: DegradeRequest):
    """Simulate authentic physical radiography acquisition & storage degradation."""
    from src.degradations.pipeline import DegradationPipeline

    try:
        img_np = _decode_grayscale_224(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid clean image input: {e}")

    # Apply exact physical degradation operators in order: Contrast -> Blur -> Noise -> Compression
    pipeline = DegradationPipeline()
    deg_np, applied_v = pipeline.apply_vector(img_np, req.vector)

    deg_uint8 = np.clip(deg_np * 255.0, 0, 255).astype(np.uint8)
    deg_pil = Image.fromarray(deg_uint8, mode="L")
    buf = io.BytesIO()
    
    # Save as JPEG with exact simulated PACS quality
    s_comp = float(applied_v[3])
    quality = int(round(75 - s_comp * (75 - 10))) if s_comp > 0 else 95
    deg_pil.save(buf, format="JPEG", quality=int(np.clip(quality, 5, 95)))
    deg_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return DegradeResponse(
        status="success",
        degraded_image=f"data:image/jpeg;base64,{deg_b64}",
        applied_vector=[round(float(x), 3) for x in applied_v.tolist()]
    )


@app.post("/api/pipeline/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global device, detector, enhancer, classifiers, calibration_cfg, ensemble_weights, triage_thresholds

    import torch
    from src.models.consensus import predict_consensus_batch, classify_triage

    if detector is None or enhancer is None or len(classifiers) < 3 or calibration_cfg is None:
        load_models()

    try:
        img_np = _decode_grayscale_224(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Convert to PyTorch tensor [1, 1, 224, 224]
    deg_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: Model 1 diagnoses continuous corruption vector
        pred_vector_t = detector.predict_vector(deg_t)
        pred_vector = pred_vector_t[0].cpu().numpy()

        # Step 2: Model 2 restores/cleans image via FiLM conditioning (pass-through when scan is clean)
        if float(pred_vector_t.max().item()) < 0.01:
            restored_t = deg_t
        else:
            restored_t = torch.clamp(enhancer(deg_t, pred_vector_t), 0.0, 1.0)
        restored_np = restored_t[0, 0].cpu().numpy()

        # Step 3: Shared Degradation-Aware Calibrated 3-Radiologist Consensus Panel
        p_deg_t, _ = predict_consensus_batch(
            classifiers,
            deg_t,
            degradation_vector=pred_vector_t,
            weights=ensemble_weights,
            calibration=calibration_cfg,
        )
        p_enh_t, per_enh = predict_consensus_batch(
            classifiers,
            restored_t,
            degradation_vector=pred_vector_t,
            weights=ensemble_weights,
            calibration=calibration_cfg,
        )
        p_deg = float(p_deg_t[0].item())
        p_enh = float(p_enh_t[0].item())
        individual_restored = {
            arch: round(float(prob_t[0].item()) * 100.0, 1)
            for arch, prob_t in per_enh.items()
        }

    # Convert restored numpy array to PNG base64
    restored_uint8 = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
    restored_pil = Image.fromarray(restored_uint8, mode="L")
    buf = io.BytesIO()
    restored_pil.save(buf, format="PNG")
    restored_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    cleaned_data_url = f"data:image/png;base64,{restored_b64}"

    # Calculate unmanipulated probability percentages
    pneu_pct = round(p_enh * 100.0, 1)
    norm_pct = round((1.0 - p_enh) * 100.0, 1)
    corrupted_pneu_pct = round(p_deg * 100.0, 1)

    has_pneu, clinical_risk, recommendation = classify_triage(p_enh, thresholds=triage_thresholds)

    return PredictResponse(
        status="success",
        cleaned_image=cleaned_data_url,
        has_pneumonia=has_pneu,
        pneumonia_percent=pneu_pct,
        normal_percent=norm_pct,
        corrupted_pneumonia_percent=corrupted_pneu_pct,
        diagnosed_vector={
            "noise": round(float(pred_vector[0]), 3),
            "blur": round(float(pred_vector[1]), 3),
            "contrast": round(float(pred_vector[2]), 3),
            "compression": round(float(pred_vector[3]), 3)
        },
        device_used=str(device),
        clinical_risk=clinical_risk,
        recommendation=recommendation,
        model_consensus={
            "DenseNet121_pct": individual_restored.get("densenet121", pneu_pct),
            "EfficientNet_B0_pct": individual_restored.get("efficientnet_b0", pneu_pct),
            "ResNet50_pct": individual_restored.get("resnet50", pneu_pct),
        }
    )


# Serve built React production frontend if available
from fastapi.staticfiles import StaticFiles

frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
