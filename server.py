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

# Global model state
device = None
detector = None
enhancer = None
classifier = None


def load_models():
    """Load or initialize PyTorch models."""
    global device, detector, enhancer, classifier

    import torch
    from src.models.degradation_detector import DegradationDetector
    from src.models.adaptive_unet import AdaptiveUNet
    from src.models.disease_classifier import DiseaseClassifier

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

    # 3. Model 3: Disease Classifier (DenseNet-121)
    d_ckpt = PROJECT_ROOT / "checkpoints" / "disease_models" / "densenet121.pth"
    classifier = DiseaseClassifier(architecture="densenet121", num_classes=2, pretrained=not d_ckpt.exists())
    if d_ckpt.exists():
        print(f"[API Server] Loading Classifier checkpoint from {d_ckpt}")
        classifier.load_state_dict(torch.load(d_ckpt, map_location=device))
    classifier.to(device)
    classifier.eval()

    print("[API Server] All 3 PyTorch models successfully ready!")


class PredictRequest(BaseModel):
    image: str  # Base64 data URL or raw base64 string
    ground_truth: Optional[str] = "pneumonia"


class PredictResponse(BaseModel):
    status: str
    cleaned_image: str  # Base64 data URL
    has_pneumonia: bool
    pneumonia_percent: float
    normal_percent: float
    corrupted_pneumonia_percent: float
    diagnosed_vector: Dict[str, float]
    device_used: str


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
        "models_ready": detector is not None and enhancer is not None and classifier is not None
    }


@app.post("/api/pipeline/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global device, detector, enhancer, classifier

    import torch

    if detector is None or enhancer is None or classifier is None:
        load_models()

    try:
        # Decode base64 image
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        raw_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("L")  # Convert to Grayscale
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Resize to standard model resolution (224x224)
    pil_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.array(pil_resized, dtype=np.float32) / 255.0

    # Convert to PyTorch tensor [1, 1, 224, 224]
    deg_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Step 1: Model 1 diagnoses continuous corruption vector
        pred_vector_t = detector.predict_vector(deg_t)
        pred_vector = pred_vector_t[0].cpu().numpy()

        # Step 2: Model 2 restores/cleans image via FiLM conditioning
        restored_t = enhancer(deg_t, pred_vector_t)
        # Clamp to [0, 1]
        restored_t = torch.clamp(restored_t, 0.0, 1.0)
        restored_np = restored_t[0, 0].cpu().numpy()

        # Step 3: Model 3 evaluates pneumonia probabilities
        p_deg = classifier.predict_proba(deg_t)[0, 1].item()
        p_enh = classifier.predict_proba(restored_t)[0, 1].item()

    # Convert restored numpy array to PNG base64
    restored_uint8 = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
    restored_pil = Image.fromarray(restored_uint8, mode="L")
    buf = io.BytesIO()
    restored_pil.save(buf, format="PNG")
    restored_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    cleaned_data_url = f"data:image/png;base64,{restored_b64}"

    # Calculate probabilities
    pneu_pct = round(p_enh * 100.0, 1)
    norm_pct = round((1.0 - p_enh) * 100.0, 1)
    corrupted_pneu_pct = round(p_deg * 100.0, 1)

    return PredictResponse(
        status="success",
        cleaned_image=cleaned_data_url,
        has_pneumonia=pneu_pct >= 50.0,
        pneumonia_percent=pneu_pct,
        normal_percent=norm_pct,
        corrupted_pneumonia_percent=corrupted_pneu_pct,
        diagnosed_vector={
            "noise": round(float(pred_vector[0]), 3),
            "blur": round(float(pred_vector[1]), 3),
            "contrast": round(float(pred_vector[2]), 3),
            "compression": round(float(pred_vector[3]), 3)
        },
        device_used=str(device)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
