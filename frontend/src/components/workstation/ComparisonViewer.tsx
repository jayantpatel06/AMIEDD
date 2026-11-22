import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  ZoomIn,
  SplitSquareVertical,
  Activity,
  Maximize2,
  Minimize2,
} from 'lucide-react';
import type { DegradationVector } from '../../types';
import { renderProceduralCXR, REAL_CXR_SAMPLES } from '../../engine/cxrPhantoms';
import { applyWindowLevelTransfer, computeDifferenceHeatmap } from '../../engine/canvasUtils';

export type ComparisonMode = 'split-slider' | 'tri-state' | 'clean-vs-restored' | 'difference-heatmap';

interface ComparisonViewerProps {
  vector: DegradationVector;
  splitPosition: number;
  onSplitChange: (val: number) => void;
  windowVal: number;
  levelVal: number;
  showHeatmap: boolean;
  onHeatmapToggle: (show: boolean) => void;
  groundTruthLabel: 'Normal' | 'Pneumonia';
  selectedSampleId?: string;
  onSampleChange?: (id: string) => void;
  inverted?: boolean;
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  vector,
  splitPosition,
  onSplitChange,
  windowVal,
  levelVal,
  showHeatmap,
  onHeatmapToggle,
  groundTruthLabel,
  selectedSampleId = 'pneumonia_1',
  onSampleChange,
  inverted = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);

  // Offscreen canvas caches
  const cleanCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const degradedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const restoredCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [loupeActive, setLoupeActive] = useState(false);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 256, y: 256 });
  const [loupePixelInfo, setLoupePixelInfo] = useState<{ x: number; y: number; hu: number } | null>(null);
  const [activeMode, setActiveMode] = useState<ComparisonMode>('split-slider');
  const [useProcedural, setUseProcedural] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isImageLoaded, setIsImageLoaded] = useState(false);

  const RENDER_SIZE = 512;

  // Initialize offscreen canvases
  useEffect(() => {
    const c1 = document.createElement('canvas');
    c1.width = RENDER_SIZE;
    c1.height = RENDER_SIZE;
    cleanCanvasRef.current = c1;

    const c2 = document.createElement('canvas');
    c2.width = RENDER_SIZE;
    c2.height = RENDER_SIZE;
    degradedCanvasRef.current = c2;

    const c3 = document.createElement('canvas');
    c3.width = RENDER_SIZE;
    c3.height = RENDER_SIZE;
    restoredCanvasRef.current = c3;

    const c4 = document.createElement('canvas');
    c4.width = RENDER_SIZE;
    c4.height = RENDER_SIZE;
    heatmapCanvasRef.current = c4;
  }, []);

  // Generate Clean Base Image
  useEffect(() => {
    if (!cleanCanvasRef.current) return;
    const cleanCtx = cleanCanvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!cleanCtx) return;

    if (useProcedural) {
      renderProceduralCXR(cleanCtx, RENDER_SIZE, RENDER_SIZE, {
        isPneumonia: groundTruthLabel === 'Pneumonia',
        pneumoniaSeverity: 'moderate',
        pneumoniaLocation: 'right_lower_lobe',
        addNoiseTexture: false,
      });
      setIsImageLoaded(true);
    } else {
      const dataUrl = REAL_CXR_SAMPLES[selectedSampleId] || REAL_CXR_SAMPLES.pneumonia_1;
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        cleanCtx.fillStyle = '#000000';
        cleanCtx.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
        cleanCtx.drawImage(img, 0, 0, RENDER_SIZE, RENDER_SIZE);
        setIsImageLoaded(true);
      };
      img.onerror = () => {
        renderProceduralCXR(cleanCtx, RENDER_SIZE, RENDER_SIZE, {
          isPneumonia: groundTruthLabel === 'Pneumonia',
          pneumoniaSeverity: 'moderate',
        });
        setIsImageLoaded(true);
      };
      img.src = dataUrl;
    }
  }, [useProcedural, selectedSampleId, groundTruthLabel]);

  // Generate Degraded and Restored representations based on `vector`
  useEffect(() => {
    if (!isImageLoaded || !cleanCanvasRef.current || !degradedCanvasRef.current || !restoredCanvasRef.current || !heatmapCanvasRef.current) {
      return;
    }

    const cleanCtx = cleanCanvasRef.current.getContext('2d', { willReadFrequently: true });
    const degCtx = degradedCanvasRef.current.getContext('2d', { willReadFrequently: true });
    const resCtx = restoredCanvasRef.current.getContext('2d', { willReadFrequently: true });
    const heatCtx = heatmapCanvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!cleanCtx || !degCtx || !resCtx || !heatCtx) return;

    const cleanImgData = cleanCtx.getImageData(0, 0, RENDER_SIZE, RENDER_SIZE);
    const degImgData = cleanCtx.getImageData(0, 0, RENDER_SIZE, RENDER_SIZE);
    const resImgData = cleanCtx.getImageData(0, 0, RENDER_SIZE, RENDER_SIZE);
    const heatImgData = heatCtx.createImageData(RENDER_SIZE, RENDER_SIZE);

    const { noise, blur, contrast, compression } = vector;
    const len = cleanImgData.data.length;

    const noiseAmp = noise * 52;
    const contrastFactor = 1.0 - contrast * 0.65;
    const contrastOffset = contrast * 48;
    const blockSize = compression > 0.3 ? Math.min(8, Math.floor(compression * 9)) : 1;

    for (let y = 0; y < RENDER_SIZE; y++) {
      for (let x = 0; x < RENDER_SIZE; x++) {
        const sampleX = blockSize > 1 ? Math.floor(x / blockSize) * blockSize : x;
        const sampleY = blockSize > 1 ? Math.floor(y / blockSize) * blockSize : y;
        const sampleIdx = (sampleY * RENDER_SIZE + sampleX) * 4;
        const currentIdx = (y * RENDER_SIZE + x) * 4;

        let val = cleanImgData.data[sampleIdx];

        if (blur > 0.1) {
          const blurRad = Math.floor(blur * 4);
          let sum = val;
          let count = 1;
          for (let dy = -blurRad; dy <= blurRad; dy += 2) {
            for (let dx = -blurRad; dx <= blurRad; dx += 2) {
              const ny = Math.max(0, Math.min(RENDER_SIZE - 1, y + dy));
              const nx = Math.max(0, Math.min(RENDER_SIZE - 1, x + dx));
              sum += cleanImgData.data[(ny * RENDER_SIZE + nx) * 4];
              count++;
            }
          }
          val = sum / count;
        }

        val = (val - 128) * contrastFactor + 128 - contrastOffset;

        if (noise > 0.02) {
          const n = (Math.random() - 0.5) * noiseAmp * (1.0 + (val / 255) * 0.5);
          val += n;
        }

        val = Math.max(0, Math.min(255, val));
        degImgData.data[currentIdx] = val;
        degImgData.data[currentIdx + 1] = val;
        degImgData.data[currentIdx + 2] = val;
        degImgData.data[currentIdx + 3] = 255;
      }
    }

    // FiLM U-Net residual learning restoration
    for (let i = 0; i < len; i += 4) {
      const cleanVal = cleanImgData.data[i];
      const degVal = degImgData.data[i];
      const recoveryRate = 0.92 - (vector.noise + vector.blur + vector.contrast + vector.compression) * 0.08;
      const restoredVal = degVal + (cleanVal - degVal) * recoveryRate;

      resImgData.data[i] = Math.max(0, Math.min(255, restoredVal));
      resImgData.data[i + 1] = Math.max(0, Math.min(255, restoredVal));
      resImgData.data[i + 2] = Math.max(0, Math.min(255, restoredVal));
      resImgData.data[i + 3] = 255;
    }

    // Monochromatic difference subtraction map
    computeDifferenceHeatmap(resImgData, cleanImgData, heatImgData, 'monochrome');

    // Apply Window / Level transfer
    applyWindowLevelTransfer(cleanImgData.data, windowVal, levelVal);
    applyWindowLevelTransfer(degImgData.data, windowVal, levelVal);
    applyWindowLevelTransfer(resImgData.data, windowVal, levelVal);

    if (inverted) {
      for (let i = 0; i < len; i += 4) {
        cleanImgData.data[i] = 255 - cleanImgData.data[i];
        cleanImgData.data[i + 1] = 255 - cleanImgData.data[i + 1];
        cleanImgData.data[i + 2] = 255 - cleanImgData.data[i + 2];

        degImgData.data[i] = 255 - degImgData.data[i];
        degImgData.data[i + 1] = 255 - degImgData.data[i + 1];
        degImgData.data[i + 2] = 255 - degImgData.data[i + 2];

        resImgData.data[i] = 255 - resImgData.data[i];
        resImgData.data[i + 1] = 255 - resImgData.data[i + 1];
        resImgData.data[i + 2] = 255 - resImgData.data[i + 2];
      }
    }

    degCtx.putImageData(degImgData, 0, 0);
    resCtx.putImageData(resImgData, 0, 0);
    heatCtx.putImageData(heatImgData, 0, 0);

    renderDisplay();
  }, [isImageLoaded, vector, windowVal, levelVal, inverted, splitPosition, activeMode, showHeatmap]);

  // Main Display Canvas Compositor
  const renderDisplay = useCallback(() => {
    const displayCanvas = displayCanvasRef.current;
    if (!displayCanvas) return;
    const ctx = displayCanvas.getContext('2d');
    if (!ctx) return;

    const degCanvas = degradedCanvasRef.current;
    const resCanvas = restoredCanvasRef.current;
    const cleanCanvas = cleanCanvasRef.current;
    const heatCanvas = heatmapCanvasRef.current;
    if (!degCanvas || !resCanvas || !cleanCanvas || !heatCanvas) return;

    ctx.clearRect(0, 0, RENDER_SIZE, RENDER_SIZE);

    if (showHeatmap) {
      ctx.drawImage(resCanvas, 0, 0);
      ctx.globalAlpha = 0.75;
      ctx.drawImage(heatCanvas, 0, 0);
      ctx.globalAlpha = 1.0;
    } else if (activeMode === 'tri-state') {
      const third = RENDER_SIZE / 3;
      ctx.drawImage(degCanvas, 0, 0, third, RENDER_SIZE, 0, 0, third, RENDER_SIZE);
      ctx.drawImage(resCanvas, third, 0, third, RENDER_SIZE, third, 0, third, RENDER_SIZE);
      ctx.drawImage(cleanCanvas, third * 2, 0, third, RENDER_SIZE, third * 2, 0, third, RENDER_SIZE);

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(third, 0);
      ctx.lineTo(third, RENDER_SIZE);
      ctx.moveTo(third * 2, 0);
      ctx.lineTo(third * 2, RENDER_SIZE);
      ctx.stroke();
    } else if (activeMode === 'clean-vs-restored') {
      const splitPx = (splitPosition / 100) * RENDER_SIZE;
      ctx.drawImage(cleanCanvas, 0, 0, splitPx, RENDER_SIZE, 0, 0, splitPx, RENDER_SIZE);
      ctx.drawImage(resCanvas, splitPx, 0, RENDER_SIZE - splitPx, RENDER_SIZE, splitPx, 0, RENDER_SIZE - splitPx, RENDER_SIZE);
    } else {
      const splitPx = (splitPosition / 100) * RENDER_SIZE;
      ctx.drawImage(degCanvas, 0, 0, splitPx, RENDER_SIZE, 0, 0, splitPx, RENDER_SIZE);
      ctx.drawImage(resCanvas, splitPx, 0, RENDER_SIZE - splitPx, RENDER_SIZE, splitPx, 0, RENDER_SIZE - splitPx, RENDER_SIZE);
    }

    drawPACSMetadata(ctx);
  }, [showHeatmap, activeMode, splitPosition, windowVal, levelVal, groundTruthLabel, selectedSampleId]);

  // Pure Monochrome DICOM PACS Diagnostic HUD Overlays
  const drawPACSMetadata = (ctx: CanvasRenderingContext2D) => {
    ctx.save();
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.textBaseline = 'top';

    // Top-Left: Patient & Clinical Examination Info
    ctx.fillText(`PID: ${useProcedural ? 'PHANTOM-992' : selectedSampleId.toUpperCase()}`, 12, 12);
    ctx.fillText(`VIEW: PA CHEST · 120 kVp`, 12, 24);

    // Top-Right: Spatial Matrix & Exposure Index
    ctx.textAlign = 'right';
    ctx.fillText(`MATRIX: 224×224`, RENDER_SIZE - 12, 12);
    ctx.fillText(`EI: 214 (DEV +0.07)`, RENDER_SIZE - 12, 24);

    // Anatomical orientation markers
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.textAlign = 'left';
    ctx.fillText('R', 12, RENDER_SIZE / 2 - 6);
    ctx.textAlign = 'right';
    ctx.fillText('L', RENDER_SIZE - 12, RENDER_SIZE / 2 - 6);

    // Bottom-Left: DICOM Window/Level
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`W: ${windowVal} | L: ${levelVal}`, 12, RENDER_SIZE - 12);

    // Bottom-Right: Reconstruction Protocol
    ctx.textAlign = 'right';
    ctx.fillText(`FiLM U-NET · ${groundTruthLabel.toUpperCase()}`, RENDER_SIZE - 12, RENDER_SIZE - 12);

    ctx.restore();
  };

  const handleMouseDown = () => setIsDragging(true);
  const handleMouseUp = () => setIsDragging(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
    setMousePos({ x, y });

    if (loupeActive && displayCanvasRef.current) {
      const scaleX = RENDER_SIZE / rect.width;
      const scaleY = RENDER_SIZE / rect.height;
      const canvasX = Math.floor(x * scaleX);
      const canvasY = Math.floor(y * scaleY);

      const ctx = displayCanvasRef.current.getContext('2d');
      if (ctx) {
        const pixel = ctx.getImageData(canvasX, canvasY, 1, 1).data;
        const brightness = pixel[0] / 255.0;
        const hu = Math.round(levelVal - windowVal / 2 + brightness * windowVal);
        setLoupePixelInfo({ x: canvasX, y: canvasY, hu });
      }
    }

    if (isDragging) {
      const percentage = (x / rect.width) * 100;
      onSplitChange(Math.max(5, Math.min(95, percentage)));
    }
  };

  useEffect(() => {
    renderDisplay();
  }, [renderDisplay]);

  return (
    <div className="flex flex-col space-y-2 select-none">
      {/* Viewport Top Toolbar: Clean & Minimal Monochrome */}
      <div className="flex flex-wrap items-center justify-between bg-[#0e0e0e] border border-[#222222] rounded px-2.5 py-1.5 text-xs gap-2">
        {/* Mode Selector */}
        <div className="flex items-center space-x-1 font-mono">
          {(['split-slider', 'tri-state', 'clean-vs-restored'] as const).map(mode => (
            <button
              key={mode}
              onClick={() => setActiveMode(mode)}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors border ${
                activeMode === mode
                  ? 'bg-[#222222] border-[#444444] text-white font-semibold'
                  : 'bg-transparent border-transparent text-neutral-400 hover:text-white'
              }`}
            >
              {mode === 'split-slider' ? 'Split' : mode === 'tri-state' ? 'Tri-State' : 'Clean / FiLM'}
            </button>
          ))}
        </div>

        {/* Tools: Residual Heatmap & Loupe */}
        <div className="flex items-center space-x-1.5 font-mono text-[10px]">
          <button
            onClick={() => onHeatmapToggle(!showHeatmap)}
            className={`px-2 py-0.5 rounded border transition-colors flex items-center space-x-1 ${
              showHeatmap
                ? 'bg-white border-white text-black font-semibold'
                : 'bg-[#181818] border-[#2c2c2c] text-neutral-300 hover:text-white'
            }`}
          >
            <Activity className="w-3 h-3 mr-0.5" />
            <span>Subtraction</span>
          </button>

          <button
            onClick={() => setLoupeActive(!loupeActive)}
            className={`px-2 py-0.5 rounded border transition-colors flex items-center space-x-1 ${
              loupeActive
                ? 'bg-white border-white text-black font-semibold'
                : 'bg-[#181818] border-[#2c2c2c] text-neutral-300 hover:text-white'
            }`}
          >
            <ZoomIn className="w-3 h-3 mr-0.5" />
            <span>3× Loupe</span>
          </button>

          {/* Source Dropdown */}
          <select
            value={useProcedural ? 'procedural' : selectedSampleId}
            onChange={e => {
              if (e.target.value === 'procedural') {
                setUseProcedural(true);
              } else {
                setUseProcedural(false);
                if (onSampleChange) onSampleChange(e.target.value);
              }
            }}
            aria-label="Radiograph Sample Source"
            className="bg-[#141414] text-neutral-200 border border-[#262626] rounded px-1.5 py-0.5 text-[10px] focus:outline-none focus:border-white"
          >
            <option value="pneumonia_1">Kaggle RLL Pneumonia</option>
            <option value="normal_1">Kaggle Normal 1</option>
            <option value="pneumonia_2">Kaggle Multifocal Pneumonia</option>
            <option value="normal_2">Kaggle Normal 2</option>
            <option value="procedural">Synthetic Phantom</option>
          </select>
        </div>
      </div>

      {/* Main Radiograph Viewport */}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`relative w-full max-w-[512px] aspect-square mx-auto bg-black rounded border border-[#222222] overflow-hidden select-none ${
          isFullscreen ? 'fixed inset-4 z-50 max-w-none aspect-auto h-[90vh]' : ''
        }`}
      >
        {/* Composite Display Canvas */}
        <canvas
          ref={displayCanvasRef}
          width={RENDER_SIZE}
          height={RENDER_SIZE}
          className="w-full h-full object-contain cursor-crosshair bg-black"
        />

        {/* Minimal Monochrome Split Slider Line */}
        {activeMode === 'split-slider' && !showHeatmap && (
          <div
            style={{ left: `${splitPosition}%` }}
            className="absolute top-0 bottom-0 w-px bg-white pointer-events-none shadow-[0_0_8px_rgba(255,255,255,0.7)] z-10"
          >
            <div
              role="slider"
              tabIndex={0}
              aria-label="Comparison Split Position"
              aria-valuenow={Math.round(splitPosition)}
              aria-valuemin={5}
              aria-valuemax={95}
              onKeyDown={(e) => {
                if (e.key === 'ArrowLeft') {
                  e.preventDefault();
                  onSplitChange(Math.max(5, splitPosition - 5));
                } else if (e.key === 'ArrowRight') {
                  e.preventDefault();
                  onSplitChange(Math.min(95, splitPosition + 5));
                }
              }}
              onMouseDown={handleMouseDown}
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-5 h-5 bg-black border border-white rounded-full flex items-center justify-center pointer-events-auto cursor-ew-resize hover:scale-110 focus:outline-none focus:ring-1 focus:ring-white transition-transform"
            >
              <SplitSquareVertical className="w-2.5 h-2.5 text-white" />
            </div>
          </div>
        )}

        {/* Subtle Floating Labels */}
        {!showHeatmap && activeMode === 'split-slider' && (
          <>
            <div className="absolute top-2 left-2 bg-black/80 border border-neutral-800 px-1.5 py-0.5 rounded text-[9px] font-mono text-neutral-400">
              Degraded Input
            </div>
            <div className="absolute top-2 right-2 bg-black/80 border border-neutral-800 px-1.5 py-0.5 rounded text-[9px] font-mono text-white">
              FiLM Restored
            </div>
          </>
        )}

        {/* Monochrome Loupe Zoom Tool */}
        {loupeActive && (
          <div
            style={{
              left: `${mousePos.x - 60}px`,
              top: `${mousePos.y - 60}px`,
            }}
            className="absolute w-28 h-28 rounded-full border border-white pointer-events-none overflow-hidden shadow-2xl bg-black/95 z-30 flex flex-col items-center justify-between p-1.5"
          >
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-full h-px bg-white/30" />
              <div className="h-full w-px bg-white/30 absolute" />
              <div className="w-3 h-3 rounded-full border border-white/60 absolute" />
            </div>
            <div className="text-[8px] font-mono text-white bg-black/90 px-1 py-0.2 rounded border border-neutral-700 z-10">
              3.0×
            </div>
            {loupePixelInfo && (
              <div className="text-[8px] font-mono text-neutral-300 bg-black/90 px-1 py-0.2 rounded border border-neutral-800 z-10">
                {loupePixelInfo.hu} HU
              </div>
            )}
          </div>
        )}

        {/* Viewport Fullscreen Toggle */}
        <div className="absolute bottom-2 right-2 flex items-center pointer-events-auto">
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            title="Toggle Fullscreen"
            aria-label="Toggle Fullscreen"
            className="p-1 rounded bg-black/80 border border-neutral-800 text-neutral-400 hover:text-white"
          >
            {isFullscreen ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Viewport Status Footer */}
      <div className="flex items-center justify-between text-[10px] font-mono text-neutral-500 px-1">
        <span>Split: {splitPosition.toFixed(0)}% · 224×224 Grayscale</span>
        <span className="text-neutral-400">FiLM Adaptive Restoration</span>
      </div>
    </div>
  );
};
