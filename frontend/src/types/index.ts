/**
 * RadFiLM Studio - Shared TypeScript Definitions
 * Precision Clinical Chest X-Ray Adaptive Restoration & Diagnostic AI Benchmark
 */

export interface DegradationVector {
  noise: number;       // [0, 1] - Quantum Poisson / Gaussian sensor noise
  blur: number;        // [0, 1] - Patient respiratory/cardiac motion blur
  contrast: number;    // [0, 1] - Underexposure / low dynamic range contrast loss
  compression: number; // [0, 1] - Telemedicine JPEG / DCT block artifacting
}

export interface ImageMetrics {
  psnr: number; // Peak Signal-to-Noise Ratio (dB)
  ssim: number; // Structural Similarity Index [0-1]
  mae: number;  // Mean Absolute Error [0-1]
}

export interface DiagnosticCohortMetrics {
  accuracy: number;
  auc_roc: number;
  f1: number;
  f1_macro?: number;
  precision: number;
  sensitivity: number; // True Positive Rate (Recall)
  specificity: number; // True Negative Rate
}

export type ModelArchitecture = 'densenet121' | 'resnet50' | 'efficientnet_b0';
export type CohortState = 'clean' | 'degraded' | 'generic' | 'adaptive';

export interface ModelPrediction {
  normalProb: number;
  pneumoniaProb: number;
  predictedLabel: 'Normal' | 'Pneumonia';
  confidence: number;
}

export interface DiagnosticPredictions {
  densenet121: Record<CohortState, ModelPrediction>;
  resnet50: Record<CohortState, ModelPrediction>;
  efficientnet_b0: Record<CohortState, ModelPrediction>;
}

export interface BenchmarkCohortData {
  num_test_samples: number;
  image_restoration_quality: {
    degraded: ImageMetrics;
    generic: ImageMetrics;
    adaptive: ImageMetrics;
  };
  clinical_diagnosis_performance: {
    densenet121: Record<CohortState, DiagnosticCohortMetrics>;
    resnet50: Record<CohortState, DiagnosticCohortMetrics>;
    efficientnet_b0: Record<CohortState, DiagnosticCohortMetrics>;
  };
}

export interface CaseStudyItem {
  id: string;
  patientId: string;
  age: number;
  gender: 'M' | 'F' | 'Other';
  studyDate: string;
  projection: 'PA' | 'AP';
  groundTruthLabel: 'Normal' | 'Pneumonia';
  pathologySubtype?: string;
  severity: 'None' | 'Mild' | 'Moderate' | 'Severe';
  clinicalHistory: string;
  radiologistImpression: string;
  exposureVector: DegradationVector;
  recommendedWindowLevel?: {
    window: number;
    level: number;
    presetName: string;
  };
}

export interface WindowLevelPreset {
  id: string;
  name: string;
  window: number;
  level: number;
  description: string;
}

export interface FilmModulationLayer {
  layerName: string;
  depth: number;
  gammaMean: number;
  betaMean: number;
  activeChannels: number;
}

export interface ViewportDisplayMode {
  splitMode: 'side-by-side' | 'slider-split' | 'difference-heatmap' | 'overlay';
  activeCohort: CohortState;
  showLoupe: boolean;
  loupeZoom: number;
  window: number;
  level: number;
}
