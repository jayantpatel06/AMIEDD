import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Upload,
  Download,
  Play,
  RotateCcw,
  Dice5,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  Cpu,
  Layers,
  Sparkles,
  Info
} from 'lucide-react';
import type { PredictResponse, DegradationSettings } from '../types';
import { ImageComparisonSlider } from './ImageComparisonSlider';

interface SamplePreset {
  id: string;
  name: string;
  type: string;
  path: string;
}

const PRESETS: SamplePreset[] = [
  {
    id: 'normal',
    name: 'Normal Healthy Patient',
    type: 'Clear lung fields',
    path: '/samples/normal_patient.jpeg'
  },
  {
    id: 'bacterial',
    name: 'Bacterial Consolidation',
    type: 'Right lower lobe opacities',
    path: '/samples/bacterial_pneumonia.jpeg'
  },
  {
    id: 'viral',
    name: 'Viral Pneumonia',
    type: 'Diffuse interstitial infiltrates',
    path: '/samples/viral_pneumonia.jpeg'
  }
];

// Exact 1D Separable Gaussian Convolution with Clamped Edges
function applySeparableGaussianBlur(
  pixels: Float32Array,
  width: number,
  height: number,
  sigma: number,
  kernelSize: number
) {
  const half = Math.floor(kernelSize / 2);
  const kernel = new Float32Array(kernelSize);
  let kSum = 0;
  for (let i = 0; i < kernelSize; i++) {
    const x = i - half;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    kSum += kernel[i];
  }
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= kSum;
  }

  const temp = new Float32Array(pixels.length);

  // Horizontal convolution pass
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -half; k <= half; k++) {
        const px = Math.min(width - 1, Math.max(0, x + k));
        sum += pixels[rowOffset + px] * kernel[k + half];
      }
      temp[rowOffset + x] = sum;
    }
  }

  // Vertical convolution pass
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let sum = 0;
      for (let k = -half; k <= half; k++) {
        const py = Math.min(height - 1, Math.max(0, y + k));
        sum += temp[py * width + x] * kernel[k + half];
      }
      pixels[y * width + x] = sum;
    }
  }
}

export const StudioPage: React.FC = () => {
  // Input image state
  const [selectedImageSrc, setSelectedImageSrc] = useState<string>('/samples/bacterial_pneumonia.jpeg');
  const [degradedDataUrl, setDegradedDataUrl] = useState<string>('');

  // Sliders state
  const [degradation, setDegradation] = useState<DegradationSettings>({
    noise: 40,
    blur: 2.5,
    contrast: -15,
    compression: 65,
  });

  // Pipeline status & result state
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  // Canvas ref for real-time client-side degradation synthesis
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseImgRef = useRef<HTMLImageElement | null>(null);

  // Exact Physical Degradation Synthesis (matches Python training pipeline 1:1)
  const applyDegradations = useCallback(() => {
    const canvas = canvasRef.current;
    const img = baseImgRef.current;
    if (!canvas || !img || !img.complete || img.naturalWidth === 0) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // Use native 224x224 model resolution to prevent downsampling low-pass artifacts
    const size = 224;
    canvas.width = size;
    canvas.height = size;

    // Draw base radiograph
    ctx.drawImage(img, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size);
    const data = imgData.data;
    const N = size * size;
    const pixels = new Float32Array(N);

    // Convert to grayscale luminance [0, 255]
    for (let i = 0; i < N; i++) {
      const idx = i * 4;
      pixels[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
    }

    // --- STEP 1: PHYSICAL CONTRAST REDUCTION (Sensor dynamic range loss) ---
    const s_contrast = Math.min(1.0, Math.max(0.0, Math.abs(degradation.contrast) / 50.0));
    if (s_contrast > 0.0) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += pixels[i];
      const mean = sum / N;
      const factor = 0.75 - s_contrast * 0.55;
      for (let i = 0; i < N; i++) {
        pixels[i] = mean + factor * (pixels[i] - mean);
      }
    }

    // --- STEP 2: PHYSICAL MOTION / DEFOCUS BLUR ---
    const s_blur = Math.min(1.0, Math.max(0.0, degradation.blur / 8.0));
    if (s_blur > 0.0) {
      const sigma = 0.5 + s_blur * (3.5 - 0.5);
      const raw_k = Math.round(3 + s_blur * (15 - 3));
      const k = raw_k % 2 === 1 ? raw_k : raw_k + 1;
      applySeparableGaussianBlur(pixels, size, size, sigma, k);
    }

    // --- STEP 3: PHYSICAL QUANTUM & ELECTRONIC SENSOR NOISE ---
    const s_noise = Math.min(1.0, Math.max(0.0, degradation.noise / 100.0));
    if (s_noise > 0.0) {
      const sigma = (0.02 + s_noise * (0.25 - 0.02)) * 255.0;
      for (let i = 0; i < N; i++) {
        const u1 = Math.random() || 0.0001;
        const u2 = Math.random() || 0.0001;
        const randG = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        pixels[i] += randG * sigma;
      }
    }

    // Write back to canvas buffer with clamping [0, 255]
    for (let i = 0; i < N; i++) {
      const idx = i * 4;
      const val = Math.max(0, Math.min(255, Math.round(pixels[i])));
      data[idx] = val;
      data[idx + 1] = val;
      data[idx + 2] = val;
      data[idx + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);

    // --- STEP 4: PHYSICAL JPEG PACS TRANSMISSION COMPRESSION ---
    const s_comp = Math.min(1.0, Math.max(0.0, degradation.compression / 100.0));
    const qualityInt = s_comp > 0 ? Math.round(75 - s_comp * 65) : 95;
    const jpegQuality = Math.max(0.08, Math.min(0.95, qualityInt / 100));
    const finalDataUrl = canvas.toDataURL('image/jpeg', jpegQuality);
    setDegradedDataUrl(finalDataUrl);
  }, [degradation]);

  // Load new base image whenever selectedImageSrc changes
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      baseImgRef.current = img;
      applyDegradations();
    };
    img.src = selectedImageSrc;
  }, [selectedImageSrc, applyDegradations]);

  // Update canvas whenever sliders change
  useEffect(() => {
    applyDegradations();
  }, [degradation, applyDegradations]);

  // Handle local file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        setSelectedImageSrc(event.target.result as string);
        setPrediction(null);
      }
    };
    reader.readAsDataURL(file);
  };

  // Reset sliders
  const handleReset = () => {
    setDegradation({
      noise: 0,
      blur: 0,
      contrast: 0,
      compression: 0,
    });
  };

  // Randomize realistic hospital noise
  const handleRandomize = () => {
    setDegradation({
      noise: Math.floor(20 + Math.random() * 50),
      blur: Number((1.0 + Math.random() * 3.5).toFixed(1)),
      contrast: Math.floor(-35 + Math.random() * 25),
      compression: Math.floor(40 + Math.random() * 50),
    });
  };

  // Download corrupted image
  const handleDownloadDegraded = () => {
    if (!degradedDataUrl) return;
    const a = document.createElement('a');
    a.href = degradedDataUrl;
    a.download = `corrupted_cxr_amiedd_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Send image to API pipeline
  const handleRunPipeline = async () => {
    if (!degradedDataUrl) return;
    setIsProcessing(true);
    setErrorMsg(null);

    try {
      const response = await fetch('/api/pipeline/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: degradedDataUrl,
          ground_truth: 'unknown',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      const data: PredictResponse = await response.json();
      setPrediction(data);
    } catch (err: unknown) {
      console.error('Pipeline error:', err);
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(`Connection error: ${msg}. Make sure server.py is running on port 8000.`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-8 py-2">
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Header bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-[#EBE5D8]">
        <div>
          <h2 className="text-2xl font-bold text-[#1F2421] tracking-tight">
            Interactive Degradation & Diagnosis Studio
          </h2>
          <p className="text-xs sm:text-sm text-[#5F6862]">
            Inject realistic PACS transmission corruptions and run the full 3-stage neural pipeline.
          </p>
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Input Selection & Degradation Sliders (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Preset Clinical Scans */}
          <div className="p-5 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-3">
            <label className="text-xs font-bold text-[#1F2421] uppercase tracking-wider block">
              1. Select Clinical CXR Scan
            </label>
            <div className="grid grid-cols-3 gap-2">
              {PRESETS.map((preset) => {
                const isSelected = selectedImageSrc === preset.path;
                return (
                  <button
                    key={preset.id}
                    onClick={() => {
                      setSelectedImageSrc(preset.path);
                      setPrediction(null);
                    }}
                    className={`p-2.5 rounded-2xl border text-left transition-all ${
                      isSelected
                        ? 'bg-[#E6ECE5] border-[#4F7757] text-[#1F2421] shadow-sm'
                        : 'bg-[#FAF8F5] border-[#EBE5D8] text-[#5F6862] hover:bg-white'
                    }`}
                  >
                    <p className="text-[11px] font-semibold leading-tight line-clamp-1">
                      {preset.name}
                    </p>
                    <p className="text-[10px] text-[#5F6862] mt-0.5 line-clamp-1">{preset.type}</p>
                  </button>
                );
              })}
            </div>

            {/* Upload Custom CXR */}
            <div className="pt-2">
              <label className="flex items-center justify-center gap-2 w-full p-2.5 rounded-2xl border-2 border-dashed border-[#DDD4C0] hover:border-[#3D3627] bg-[#FAF8F5] cursor-pointer text-xs font-medium text-[#5F6862] hover:text-[#1F2421] transition-all">
                <Upload className="w-3.5 h-3.5" />
                <span>Upload Custom Radiograph</span>
                <input
                  type="file"
                  accept="image/png, image/jpeg, image/jpg"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Degradation Simulator Controls */}
          <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-[#1F2421] uppercase tracking-wider">
                2. Simulate Hardware Degradations
              </label>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleRandomize}
                  title="Randomize realistic hospital corruptions"
                  className="p-1.5 rounded-xl bg-[#F5F2EB] hover:bg-[#EBE5D8] text-[#3D3627] text-xs font-medium flex items-center gap-1 transition-all"
                >
                  <Dice5 className="w-3.5 h-3.5" />
                  <span className="text-[11px]">Randomize</span>
                </button>
                <button
                  onClick={handleReset}
                  title="Reset to clean scan"
                  className="p-1.5 rounded-xl bg-[#F5F2EB] hover:bg-[#EBE5D8] text-[#5F6862] hover:text-[#1F2421] transition-all"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Slider 1: Gaussian Quantum Noise */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-[#1F2421]">Sensor Quantum Noise</span>
                <span className="font-mono text-[#5F6862]">{degradation.noise}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={degradation.noise}
                onChange={(e) =>
                  setDegradation({ ...degradation, noise: parseInt(e.target.value) })
                }
              />
              <span className="text-[10px] text-[#5F6862] block">
                Simulates low radiation dose detector noise
              </span>
            </div>

            {/* Slider 2: Patient Motion Blur */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-[#1F2421]">Respiratory Motion Blur</span>
                <span className="font-mono text-[#5F6862]">{degradation.blur} px</span>
              </div>
              <input
                type="range"
                min="0"
                max="8"
                step="0.5"
                value={degradation.blur}
                onChange={(e) =>
                  setDegradation({ ...degradation, blur: parseFloat(e.target.value) })
                }
              />
              <span className="text-[10px] text-[#5F6862] block">
                Simulates breathing motion during acquisition
              </span>
            </div>

            {/* Slider 3: Contrast Miscalibration */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-[#1F2421]">Exposure / Contrast Shift</span>
                <span className="font-mono text-[#5F6862]">
                  {degradation.contrast > 0 ? `+${degradation.contrast}` : degradation.contrast}%
                </span>
              </div>
              <input
                type="range"
                min="-50"
                max="50"
                value={degradation.contrast}
                onChange={(e) =>
                  setDegradation({ ...degradation, contrast: parseInt(e.target.value) })
                }
              />
              <span className="text-[10px] text-[#5F6862] block">
                Negative is underexposed / washed out, positive is overexposed
              </span>
            </div>

            {/* Slider 4: Lossy JPEG Compression */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-[#1F2421]">PACS Lossy Compression</span>
                <span className="font-mono text-[#5F6862]">{degradation.compression}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={degradation.compression}
                onChange={(e) =>
                  setDegradation({ ...degradation, compression: parseInt(e.target.value) })
                }
              />
              <span className="text-[10px] text-[#5F6862] block">
                Simulates network bandwidth compression blockiness
              </span>
            </div>

            {/* Action Buttons */}
            <div className="pt-2 flex flex-col sm:flex-row gap-3">
              <button
                onClick={handleDownloadDegraded}
                disabled={!degradedDataUrl}
                className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-3 rounded-2xl border border-[#DDD4C0] bg-[#FAF8F5] hover:bg-white text-xs font-semibold text-[#1F2421] shadow-sm transition-all active:scale-[0.98]"
              >
                <Download className="w-3.5 h-3.5" />
                Download Corrupted
              </button>

              <button
                onClick={handleRunPipeline}
                disabled={isProcessing || !degradedDataUrl}
                className="flex-[1.4] inline-flex items-center justify-center gap-2 px-4 py-3 rounded-2xl bg-[#3D3627] hover:bg-[#1F2421] text-xs font-semibold text-white shadow-md transition-all active:scale-[0.98] disabled:opacity-50"
              >
                {isProcessing ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Processing Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Send to AI Pipeline</span>
                  </>
                )}
              </button>
            </div>

            {errorMsg && (
              <div className="p-3 rounded-2xl bg-[#FAF4F2] border border-[#E8CFC7] text-xs text-[#944432] flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-[#C8644E]" />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Live Corrupted Preview OR Pipeline Results (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {!prediction ? (
            /* Standalone Live Degradation Preview prior to running pipeline */
            <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-[#1F2421]">Degraded Scan Preview</h3>
                  <p className="text-xs text-[#5F6862]">
                    Move sliders on the left to observe live changes before dispatching to pipeline
                  </p>
                </div>
                <span className="text-[11px] font-medium px-2.5 py-1 rounded-xl bg-[#F5F2EB] text-[#5F6862]">
                  Live Client View
                </span>
              </div>

              <div className="flex justify-center p-4 bg-[#FAF8F5] rounded-3xl border border-[#EBE5D8]">
                {degradedDataUrl ? (
                  <img
                    src={degradedDataUrl}
                    alt="Degraded Preview"
                    className="max-w-[360px] w-full aspect-square rounded-2xl object-contain shadow-sm border border-[#EBE5D8]"
                  />
                ) : (
                  <div className="w-[360px] aspect-square rounded-2xl bg-[#F5F2EB] flex items-center justify-center text-xs text-[#5F6862]">
                    Loading scan...
                  </div>
                )}
              </div>

              <div className="p-4 rounded-2xl bg-[#F5F2EB]/70 border border-[#EBE5D8] flex items-start gap-3 text-xs text-[#5F6862]">
                <Info className="w-4 h-4 shrink-0 text-[#A89775] mt-0.5" />
                <p>
                  Click <strong className="text-[#1F2421]">Send to AI Pipeline</strong> to run Model 1 (corruption estimator), Model 2 (FiLM restorer), and Model 3 (3-model consensus diagnostic panel).
                </p>
              </div>
            </div>
          ) : (
            /* Results View after running pipeline */
            <div className="space-y-6">
              {/* Interactive Comparison Slider */}
              <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-[#4F7757]" />
                    <h3 className="text-base font-bold text-[#1F2421]">
                      Interactive FiLM Restoration Slider
                    </h3>
                  </div>
                  <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#E6ECE5] text-[#38563E]">
                    Restoration Complete
                  </span>
                </div>

                <ImageComparisonSlider
                  originalImage={degradedDataUrl}
                  cleanedImage={prediction.cleaned_image}
                  originalLabel="Corrupted Input"
                  cleanedLabel="AMIEDD FiLM Restored"
                />
              </div>

              {/* Model 1: Continuous Physical Corruption Vector & Estimation Error */}
              <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-[#1F2421] uppercase tracking-wider flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-[#5F6862]" />
                    Model 1: Injected vs. Detected Degradation & Error
                  </h4>
                  <span className="text-[11px] font-mono text-[#5F6862]">
                    Normalized Scale [0.00 – 1.00]
                  </span>
                </div>

                <p className="text-xs text-[#5F6862] leading-relaxed">
                  The model operates on a normalized continuous scale from <span className="font-mono text-[#1F2421]">0.00</span> (clean) to <span className="font-mono text-[#1F2421]">1.00</span> (maximum corruption). Below is the comparison between your slider inputs and what Model 1 detected blindly:
                </p>

                {/* Detailed Comparison Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-[#EBE5D8] text-[10px] uppercase text-[#5F6862] tracking-wider">
                        <th className="py-2 px-3">Degradation Type</th>
                        <th className="py-2 px-3">Slider Injected</th>
                        <th className="py-2 px-3">Model 1 Detected</th>
                        <th className="py-2 px-3">Estimation Error (&Delta;)</th>
                        <th className="py-2 px-3">Detector Accuracy</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#F5F2EB] font-mono text-xs">
                      {/* Noise */}
                      {(() => {
                        const inputVal = degradation.noise / 100;
                        const detectedVal = prediction.diagnosed_vector.noise;
                        const err = detectedVal - inputVal;
                        const absErr = Math.abs(err);
                        const acc = Math.max(0, 100 - absErr * 100).toFixed(1);
                        return (
                          <tr className="hover:bg-[#FAF8F5]">
                            <td className="py-2.5 px-3 font-sans font-semibold text-[#1F2421]">
                              Quantum Sensor Noise
                            </td>
                            <td className="py-2.5 px-3 text-[#5F6862]">
                              {inputVal.toFixed(3)} <span className="font-sans text-[11px]">({degradation.noise}%)</span>
                            </td>
                            <td className="py-2.5 px-3 font-bold text-[#1F2421]">
                              {detectedVal.toFixed(3)} <span className="font-sans text-[11px]">({(detectedVal * 100).toFixed(1)}%)</span>
                            </td>
                            <td className={`py-2.5 px-3 font-semibold ${absErr < 0.1 ? 'text-[#38563E]' : 'text-[#D9822B]'}`}>
                              {err >= 0 ? `+${err.toFixed(3)}` : err.toFixed(3)}
                            </td>
                            <td className="py-2.5 px-3 text-[#38563E] font-bold font-sans">
                              {acc}%
                            </td>
                          </tr>
                        );
                      })()}

                      {/* Blur */}
                      {(() => {
                        const inputVal = degradation.blur / 8.0;
                        const detectedVal = prediction.diagnosed_vector.blur;
                        const err = detectedVal - inputVal;
                        const absErr = Math.abs(err);
                        const acc = Math.max(0, 100 - absErr * 100).toFixed(1);
                        return (
                          <tr className="hover:bg-[#FAF8F5]">
                            <td className="py-2.5 px-3 font-sans font-semibold text-[#1F2421]">
                              Respiratory Motion Blur
                            </td>
                            <td className="py-2.5 px-3 text-[#5F6862]">
                              {inputVal.toFixed(3)} <span className="font-sans text-[11px]">({degradation.blur}px / 8px)</span>
                            </td>
                            <td className="py-2.5 px-3 font-bold text-[#1F2421]">
                              {detectedVal.toFixed(3)} <span className="font-sans text-[11px]">({(detectedVal * 8.0).toFixed(1)}px eq)</span>
                            </td>
                            <td className={`py-2.5 px-3 font-semibold ${absErr < 0.1 ? 'text-[#38563E]' : 'text-[#D9822B]'}`}>
                              {err >= 0 ? `+${err.toFixed(3)}` : err.toFixed(3)}
                            </td>
                            <td className="py-2.5 px-3 text-[#38563E] font-bold font-sans">
                              {acc}%
                            </td>
                          </tr>
                        );
                      })()}

                      {/* Contrast */}
                      {(() => {
                        const inputVal = Math.abs(degradation.contrast) / 50.0;
                        const detectedVal = prediction.diagnosed_vector.contrast;
                        const err = detectedVal - inputVal;
                        const absErr = Math.abs(err);
                        const acc = Math.max(0, 100 - absErr * 100).toFixed(1);
                        return (
                          <tr className="hover:bg-[#FAF8F5]">
                            <td className="py-2.5 px-3 font-sans font-semibold text-[#1F2421]">
                              Contrast Deviation
                            </td>
                            <td className="py-2.5 px-3 text-[#5F6862]">
                              {inputVal.toFixed(3)} <span className="font-sans text-[11px]">({degradation.contrast}%)</span>
                            </td>
                            <td className="py-2.5 px-3 font-bold text-[#1F2421]">
                              {detectedVal.toFixed(3)} <span className="font-sans text-[11px]">({(detectedVal * 100).toFixed(1)}%)</span>
                            </td>
                            <td className={`py-2.5 px-3 font-semibold ${absErr < 0.1 ? 'text-[#38563E]' : 'text-[#D9822B]'}`}>
                              {err >= 0 ? `+${err.toFixed(3)}` : err.toFixed(3)}
                            </td>
                            <td className="py-2.5 px-3 text-[#38563E] font-bold font-sans">
                              {acc}%
                            </td>
                          </tr>
                        );
                      })()}

                      {/* Compression */}
                      {(() => {
                        const inputVal = degradation.compression / 100;
                        const detectedVal = prediction.diagnosed_vector.compression;
                        const err = detectedVal - inputVal;
                        const absErr = Math.abs(err);
                        const acc = Math.max(0, 100 - absErr * 100).toFixed(1);
                        return (
                          <tr className="hover:bg-[#FAF8F5]">
                            <td className="py-2.5 px-3 font-sans font-semibold text-[#1F2421]">
                              PACS JPEG Compression
                            </td>
                            <td className="py-2.5 px-3 text-[#5F6862]">
                              {inputVal.toFixed(3)} <span className="font-sans text-[11px]">({degradation.compression}%)</span>
                            </td>
                            <td className="py-2.5 px-3 font-bold text-[#1F2421]">
                              {detectedVal.toFixed(3)} <span className="font-sans text-[11px]">({(detectedVal * 100).toFixed(1)}%)</span>
                            </td>
                            <td className={`py-2.5 px-3 font-semibold ${absErr < 0.1 ? 'text-[#38563E]' : 'text-[#D9822B]'}`}>
                              {err >= 0 ? `+${err.toFixed(3)}` : err.toFixed(3)}
                            </td>
                            <td className="py-2.5 px-3 text-[#38563E] font-bold font-sans">
                              {acc}%
                            </td>
                          </tr>
                        );
                      })()}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Model 3: Radiologist Consensus & Clinical Triage Panel */}
              <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-5">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-[#1F2421] uppercase tracking-wider">
                    Model 3: Clinical Diagnostic Consensus Panel
                  </h4>
                  <div className="flex items-center gap-1.5 text-xs text-[#5F6862]">
                    <Cpu className="w-3.5 h-3.5" />
                    <span>{prediction.device_used.toUpperCase()}</span>
                  </div>
                </div>

                {/* 3-Tier Clinical Triage Badge */}
                <div
                  className={`p-4 rounded-2xl border flex items-start gap-3 ${
                    prediction.pneumonia_percent >= 75
                      ? 'bg-[#FAF4F2] border-[#E8CFC7] text-[#944432]'
                      : prediction.pneumonia_percent >= 45
                      ? 'bg-[#FDF9F0] border-[#EBDCB8] text-[#8C6D1F]'
                      : 'bg-[#F4F7F4] border-[#C9D7C8] text-[#38563E]'
                  }`}
                >
                  {prediction.pneumonia_percent >= 75 ? (
                    <AlertOctagon className="w-5 h-5 shrink-0 mt-0.5 text-[#C8644E]" />
                  ) : prediction.pneumonia_percent >= 45 ? (
                    <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5 text-[#D9822B]" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5 text-[#4F7757]" />
                  )}

                  <div className="space-y-1">
                    <p className="text-sm font-bold tracking-tight">
                      {prediction.clinical_risk}
                    </p>
                    <p className="text-xs leading-relaxed opacity-90">
                      {prediction.recommendation}
                    </p>
                  </div>
                </div>

                {/* Consensus Probability Distribution Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-semibold text-[#1F2421]">
                      Consensus Confidence: Pneumonia ({prediction.pneumonia_percent}%) vs Normal ({prediction.normal_percent}%)
                    </span>
                    <span className="text-[11px] text-[#5F6862]">
                      (Corrupted was: {prediction.corrupted_pneumonia_percent}%)
                    </span>
                  </div>

                  <div className="w-full h-3.5 bg-[#EBE5D8] rounded-full overflow-hidden flex shadow-inner">
                    <div
                      className="h-full bg-[#4F7757] transition-all duration-500"
                      style={{ width: `${prediction.normal_percent}%` }}
                      title={`Normal: ${prediction.normal_percent}%`}
                    />
                    <div
                      className="h-full bg-[#C8644E] transition-all duration-500"
                      style={{ width: `${prediction.pneumonia_percent}%` }}
                      title={`Pneumonia: ${prediction.pneumonia_percent}%`}
                    />
                  </div>
                  <div className="flex justify-between text-[11px] text-[#5F6862]">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#4F7757]" /> Normal
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#C8644E]" /> Pneumonia
                    </span>
                  </div>
                </div>

                {/* Individual Radiologist Breakdown */}
                <div className="pt-2 border-t border-[#EBE5D8] space-y-2">
                  <p className="text-xs font-semibold text-[#1F2421]">
                    Individual Model Opinion Breakdown (3-Radiologist Consensus)
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8]">
                      <div className="flex items-center justify-between text-[11px] text-[#5F6862]">
                        <span>EfficientNet-B0</span>
                        <span className="font-semibold text-[#1F2421]">50% wt</span>
                      </div>
                      <p className="text-base font-bold text-[#1F2421] mt-1">
                        {prediction.model_consensus.EfficientNet_B0_pct}%
                      </p>
                      <p className="text-[10px] text-[#5F6862]">Pneumonia score</p>
                    </div>

                    <div className="p-3 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8]">
                      <div className="flex items-center justify-between text-[11px] text-[#5F6862]">
                        <span>DenseNet-121</span>
                        <span className="font-semibold text-[#1F2421]">30% wt</span>
                      </div>
                      <p className="text-base font-bold text-[#1F2421] mt-1">
                        {prediction.model_consensus.DenseNet121_pct}%
                      </p>
                      <p className="text-[10px] text-[#5F6862]">Pneumonia score</p>
                    </div>

                    <div className="p-3 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8]">
                      <div className="flex items-center justify-between text-[11px] text-[#5F6862]">
                        <span>ResNet-50</span>
                        <span className="font-semibold text-[#1F2421]">20% wt</span>
                      </div>
                      <p className="text-base font-bold text-[#1F2421] mt-1">
                        {prediction.model_consensus.ResNet50_pct}%
                      </p>
                      <p className="text-[10px] text-[#5F6862]">Pneumonia score</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
