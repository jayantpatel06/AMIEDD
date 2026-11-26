export interface PredictRequest {
  image: string; // Base64 data URL
  ground_truth?: string;
}

export interface PredictResponse {
  status: string;
  cleaned_image: string; // Base64 data URL
  has_pneumonia: boolean;
  pneumonia_percent: number;
  normal_percent: number;
  corrupted_pneumonia_percent: number;
  diagnosed_vector: {
    noise: number;
    blur: number;
    contrast: number;
    compression: number;
  };
  device_used: string;
  clinical_risk: string;
  recommendation: string;
  model_consensus: {
    DenseNet121_pct: number;
    EfficientNet_B0_pct: number;
    ResNet50_pct: number;
  };
}

export interface HealthResponse {
  status: string;
  backend: string;
  device: string;
  models_ready: boolean;
  active_classifiers: string[];
}

export interface DegradationSettings {
  noise: number;       // 0 to 100
  blur: number;        // 0 to 10
  contrast: number;    // -50 to 50
  compression: number; // 0 (lossless) to 100 (heavy)
}
