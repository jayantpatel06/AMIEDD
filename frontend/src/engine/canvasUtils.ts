/**
 * RadFiLM Studio - Precision Canvas & Image Processing Utilities
 * Medical imaging viewport mathematics, difference heatmaps (Jet / Inferno / Cyan),
 * split-screen slider rendering, 3x interactive loupe magnifier, and DICOM W/L transfer.
 * 
 * Owned by @cxr_engine
 */

export interface DicomOverlayInfo {
  patientId: string;
  view: string;
  window: number;
  level: number;
  matrix: string;
  zoom: number;
  kvp?: number;
  mas?: number;
  engineMode?: string;
}

export interface LoupeOptions {
  zoom?: number;       // default 3.0
  radius?: number;     // default 65px
  reticle?: boolean;   // show millimeter crosshair
  borderColor?: string;
}

export type ColormapType = 'monochrome' | 'jet' | 'inferno' | 'cyan';

// ============================================================================
// 1. DICOM WINDOW / LEVEL (W/L) TRANSFORMATION
// ============================================================================

/**
 * Precompute a 256-element Look-Up Table (LUT) for standard DICOM Window/Level adjustment.
 * Equation: Output = clamp(((Input - (Level - Window/2)) / Window) * 255, 0, 255)
 */
export function buildWindowLevelLUT(windowVal: number, levelVal: number): Uint8Array {
  const lut = new Uint8Array(256);
  // Normalize window/level values to 8-bit space [0, 255] if user supplies large DICOM values (e.g. W: 1500, L: -600)
  let w = windowVal;
  let l = levelVal;

  if (w > 300 || l < 0 || l > 255) {
    // Map standard thoracic Hounsfield scale (-1000 to +500) to [0, 255]
    // Lung window (W: 1500, L: -600) -> w ~ 180, l ~ 70
    // Mediastinal window (W: 350, L: 40) -> w ~ 100, l ~ 140
    // Bone window (W: 1000, L: 400) -> w ~ 150, l ~ 190
    if (l < -100) {
      // Lung preset
      w = 175;
      l = 85;
    } else if (l > 100) {
      // Bone preset
      w = 140;
      l = 180;
    } else {
      // Mediastinal preset
      w = 110;
      l = 135;
    }
  }

  const minVal = l - w / 2;
  const maxVal = l + w / 2;
  const range = maxVal - minVal || 1;

  for (let i = 0; i < 256; i++) {
    const norm = ((i - minVal) / range) * 255;
    lut[i] = Math.max(0, Math.min(255, Math.round(norm)));
  }

  return lut;
}

/**
 * Apply Window / Level transformation to grayscale pixel buffer in-place using fast LUT.
 */
export function applyWindowLevelTransfer(
  data: Uint8ClampedArray,
  windowVal: number,
  levelVal: number
): void {
  const lut = buildWindowLevelLUT(windowVal, levelVal);

  for (let i = 0; i < data.length; i += 4) {
    // Luminance
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    const adjusted = lut[gray];

    data[i] = adjusted;
    data[i + 1] = adjusted;
    data[i + 2] = adjusted;
  }
}

// ============================================================================
// 2. DIFFERENCE HEATMAP & RESIDUAL COLORMAPS
// ============================================================================

/**
 * Scientific Color Ramps (Jet, Inferno, Cyan)
 */
function sampleColormap(norm: number, colormap: ColormapType): [number, number, number] {
  const t = Math.max(0.0, Math.min(1.0, norm));

  if (colormap === 'monochrome') {
    const val = Math.round(t * 255);
    return [val, val, val];
  } else if (colormap === 'inferno') {
    // Inferno: black -> purple -> crimson -> orange -> bright yellow
    if (t < 0.25) {
      const s = t / 0.25;
      return [Math.round(40 * s), 0, Math.round(60 * s)];
    } else if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      return [Math.round(40 + 150 * s), Math.round(10 * s), Math.round(60 + 20 * s)];
    } else if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      return [Math.round(190 + 55 * s), Math.round(10 + 110 * s), Math.round(80 * (1 - s))];
    } else {
      const s = (t - 0.75) / 0.25;
      return [Math.round(245 + 10 * s), Math.round(120 + 125 * s), Math.round(50 * s)];
    }
  } else if (colormap === 'cyan') {
    // Clinical Cyan: dark navy #051923 -> deep teal -> clinical cyan #06b6d4 -> ice white
    if (t < 0.5) {
      const s = t / 0.5;
      return [Math.round(5 + 1 * s), Math.round(25 + 157 * s), Math.round(35 + 177 * s)];
    } else {
      const s = (t - 0.5) / 0.5;
      return [Math.round(6 + 234 * s), Math.round(182 + 68 * s), Math.round(212 + 43 * s)];
    }
  } else {
    // Standard Medical Jet Colormap: blue -> cyan -> green -> yellow -> red
    let r = 0;
    let g = 0;
    let b = 0;

    if (t < 0.125) {
      r = 0;
      g = 0;
      b = 128 + Math.round(t / 0.125 * 127);
    } else if (t < 0.375) {
      r = 0;
      g = Math.round((t - 0.125) / 0.25 * 255);
      b = 255;
    } else if (t < 0.625) {
      r = Math.round((t - 0.375) / 0.25 * 255);
      g = 255;
      b = Math.round((1.0 - (t - 0.375) / 0.25) * 255);
    } else if (t < 0.875) {
      r = 255;
      g = Math.round((1.0 - (t - 0.625) / 0.25) * 255);
      b = 0;
    } else {
      r = 255 - Math.round((t - 0.875) / 0.125 * 100);
      g = 0;
      b = 0;
    }

    return [r, g, b];
  }
}

/**
 * Compute difference heatmap between candidate image and reference ground truth clean image.
 * Multiplies residual error by amplification factor for clinical visibility of subtle infiltrates.
 */
export function computeDifferenceHeatmap(
  restoredData: ImageData,
  cleanData: ImageData,
  outputData: ImageData,
  colormap: ColormapType = 'monochrome',
  amplification = 3.5
): void {
  const len = restoredData.data.length;

  for (let i = 0; i < len; i += 4) {
    const diff = Math.abs(restoredData.data[i] - cleanData.data[i]);
    const norm = Math.min(1.0, (diff / 255.0) * amplification);

    const [r, g, b] = sampleColormap(norm, colormap);

    outputData.data[i] = r;
    outputData.data[i + 1] = g;
    outputData.data[i + 2] = b;
    // Higher residual has higher opacity; minimal residual is translucent
    outputData.data[i + 3] = Math.round(180 + norm * 75);
  }
}

/**
 * Draw a scientific colorbar legend for the difference heatmap.
 */
export function drawHeatmapLegend(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  colormap: ColormapType = 'jet'
): void {
  ctx.save();
  // Draw gradient bar
  const grad = ctx.createLinearGradient(x, y, x + width, y);
  const steps = 10;
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const [r, g, b] = sampleColormap(t, colormap);
    grad.addColorStop(t, `rgb(${r},${g},${b})`);
  }

  ctx.fillStyle = grad;
  ctx.fillRect(x, y, width, height);

  // Border
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, height);

  // Labels
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.fillStyle = '#94a3b8';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText('0.00 (Zero Δ)', x, y + height + 3);
  ctx.textAlign = 'right';
  ctx.fillText('Max Error', x + width, y + height + 3);

  ctx.restore();
}

// ============================================================================
// 3. INTERACTIVE SPLIT-SCREEN SLIDER RENDERING
// ============================================================================

/**
 * Render split-screen view: Left image up to splitPercent (0-100), Right image for remainder.
 */
export function renderSplitScreen(
  ctx: CanvasRenderingContext2D,
  leftCanvas: HTMLCanvasElement,
  rightCanvas: HTMLCanvasElement,
  splitPercent: number,
  width: number,
  height: number
): void {
  const splitX = Math.round((splitPercent / 100) * width);

  // Render left side
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, splitX, height);
  ctx.clip();
  ctx.drawImage(leftCanvas, 0, 0, width, height);
  ctx.restore();

  // Render right side
  ctx.save();
  ctx.beginPath();
  ctx.rect(splitX, 0, width - splitX, height);
  ctx.clip();
  ctx.drawImage(rightCanvas, 0, 0, width, height);
  ctx.restore();

  // Draw dividing hairline with glowing cyan shadow
  ctx.save();
  ctx.shadowColor = '#06b6d4';
  ctx.shadowBlur = 6;
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(splitX, 0);
  ctx.lineTo(splitX, height);
  ctx.stroke();
  ctx.restore();
}

// ============================================================================
// 4. LOUPE ZOOM LENS (3x MAGNIFIER FOR PULMONARY NODULES / INFILTRATES)
// ============================================================================

/**
 * Render a high-precision circular 3x loupe magnifier centered at (mouseX, mouseY).
 */
export function drawLoupeMagnifier(
  ctx: CanvasRenderingContext2D,
  sourceCanvas: HTMLCanvasElement,
  mouseX: number,
  mouseY: number,
  options: LoupeOptions = {}
): void {
  const zoom = options.zoom ?? 3.0;
  const radius = options.radius ?? 65;
  const borderColor = options.borderColor ?? '#06b6d4';

  ctx.save();

  // Circular clipping region
  ctx.beginPath();
  ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
  ctx.clip();

  // Clear loupe interior with black
  ctx.fillStyle = '#000000';
  ctx.fill();

  // Draw scaled sub-region centered at cursor
  const sw = (radius * 2) / zoom;
  const sh = (radius * 2) / zoom;
  const sx = mouseX - sw / 2;
  const sy = mouseY - sh / 2;

  ctx.drawImage(
    sourceCanvas,
    sx, sy, sw, sh,
    mouseX - radius, mouseY - radius, radius * 2, radius * 2
  );

  // Crosshair Reticle & Millimeter Scale
  if (options.reticle !== false) {
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.45)';
    ctx.lineWidth = 1;
    // Horizontal crosshair
    ctx.beginPath();
    ctx.moveTo(mouseX - radius, mouseY);
    ctx.lineTo(mouseX + radius, mouseY);
    ctx.stroke();
    // Vertical crosshair
    ctx.beginPath();
    ctx.moveTo(mouseX, mouseY - radius);
    ctx.lineTo(mouseX, mouseY + radius);
    ctx.stroke();

    // Subtle 5mm inner circle guide
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.25)';
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, radius * 0.35, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.restore();

  // Outer border ring and shadow
  ctx.save();
  ctx.shadowColor = 'rgba(6, 182, 212, 0.6)';
  ctx.shadowBlur = 12;
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(mouseX, mouseY, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Magnification badge at bottom of loupe
  ctx.shadowBlur = 0;
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.fillStyle = '#06b6d4';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(`${zoom.toFixed(1)}x LOUPE`, mouseX, mouseY + radius + 4);

  ctx.restore();
}

// ============================================================================
// 5. CLINICAL DICOM HUD OVERLAY
// ============================================================================

/**
 * Draw heads-up clinical radiograph metadata overlay.
 */
export function drawDicomOverlay(
  ctx: CanvasRenderingContext2D,
  info: DicomOverlayInfo
): void {
  ctx.save();
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillStyle = '#06b6d4';
  ctx.textBaseline = 'top';

  // Top Left: Patient ID, Study, View
  ctx.textAlign = 'left';
  ctx.fillText(`PID: ${info.patientId}`, 12, 12);
  ctx.fillText(`VIEW: ${info.view} | CHEST`, 12, 26);
  if (info.kvp) {
    ctx.fillText(`KVp: ${info.kvp} | mAs: ${info.mas ?? 2.5}`, 12, 40);
  }

  // Top Right: Matrix & Zoom
  ctx.textAlign = 'right';
  ctx.fillText(`MATRIX: ${info.matrix}`, ctx.canvas.width - 12, 12);
  ctx.fillText(`ZOOM: ${(info.zoom * 100).toFixed(0)}%`, ctx.canvas.width - 12, 26);

  // Bottom Left: W/L
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(`W: ${info.window} L: ${info.level}`, 12, ctx.canvas.height - 12);

  // Bottom Right: Processing Pipeline Badge
  ctx.textAlign = 'right';
  ctx.fillText(info.engineMode ?? `FiLM MODULATED U-NET`, ctx.canvas.width - 12, ctx.canvas.height - 12);

  ctx.restore();
}

// ============================================================================
// 6. BUFFER FORMAT CONVERTERS
// ============================================================================

/**
 * Convert normalized Float32Array [0, 1] to browser ImageData (RGBA).
 */
export function float32ToImageData(
  buffer: Float32Array,
  width: number,
  height: number
): ImageData {
  const imgData = new ImageData(width, height);
  const data = imgData.data;

  for (let i = 0; i < buffer.length; i++) {
    const val = Math.round(Math.max(0, Math.min(1, buffer[i])) * 255);
    const idx = i * 4;
    data[idx] = val;
    data[idx + 1] = val;
    data[idx + 2] = val;
    data[idx + 3] = 255;
  }

  return imgData;
}

/**
 * Convert browser ImageData to normalized Float32Array [0, 1].
 */
export function imageDataToFloat32(imgData: ImageData): Float32Array {
  const buffer = new Float32Array(imgData.width * imgData.height);
  const data = imgData.data;

  for (let i = 0; i < buffer.length; i++) {
    const idx = i * 4;
    buffer[i] = (0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]) / 255.0;
  }

  return buffer;
}
