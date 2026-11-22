/**
 * RadFiLM Studio - High-Fidelity Procedural Chest X-Ray (CXR) Phantom Generator
 * Precision anatomical simulation of posteroanterior (PA) chest radiographs.
 * Simulates thoracic cage, lung parenchyma, tracheobronchial tree, cardiac silhouette,
 * hemidiaphragms, ribs, clavicles, and pathological consolidations (pneumonia).
 * 
 * Owned by @cxr_engine
 */

import sampleImagesData from './sampleImages.json';

export interface CXRGenerationOptions {
  seed?: number;
  isPneumonia?: boolean;
  pneumoniaSeverity?: 'mild' | 'moderate' | 'severe';
  pneumoniaLocation?: 'right_lower_lobe' | 'right_middle_lobe' | 'left_lower_lobe' | 'bilateral';
  pneumoniaPattern?: 'consolidation' | 'interstitial' | 'bronchopneumonia';
  patientGender?: 'M' | 'F';
  exposureQuality?: number; // [0.5 - 1.5]
  addNoiseTexture?: boolean;
}

export interface RealSampleMetadata {
  id: string;
  label: 'Normal' | 'Pneumonia';
  description: string;
  source: string;
}

export const REAL_CXR_SAMPLES: Record<string, string> = {
  normal_1: sampleImagesData.normal_1,
  pneumonia_1: sampleImagesData.pneumonia_1,
  normal_2: sampleImagesData.normal_2,
  pneumonia_2: sampleImagesData.pneumonia_2,
};

export const REAL_SAMPLE_CATALOG: RealSampleMetadata[] = [
  {
    id: 'normal_1',
    label: 'Normal',
    description: 'Bilateral clear lung fields, normal cardiothoracic ratio (<0.50), sharp costophrenic angles.',
    source: 'Kaggle CXR Test Cohort (sample_0000.jpeg)',
  },
  {
    id: 'pneumonia_1',
    label: 'Pneumonia',
    description: 'Dense right lower lobe (RLL) alveolar consolidation with focal air bronchograms.',
    source: 'Kaggle CXR Test Cohort (sample_0000.jpeg)',
  },
  {
    id: 'normal_2',
    label: 'Normal',
    description: 'Clear pulmonary parenchyma, intact mediastinal borders, normal thoracic skeletal cage.',
    source: 'Kaggle CXR Test Cohort (sample_0001.jpeg)',
  },
  {
    id: 'pneumonia_2',
    label: 'Pneumonia',
    description: 'Multifocal bilateral patchy infiltrates with retrocardiac consolidation.',
    source: 'Kaggle CXR Test Cohort (sample_0001.jpeg)',
  },
];

// Simple deterministic pseudo-random number generator (Mulberry32)
function createPRNG(seed: number) {
  let s = Math.floor(seed) || 123456789;
  return function () {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Procedurally render a high-fidelity Chest X-Ray phantom directly onto an HTML5 Canvas 2D context.
 */
export function renderProceduralCXR(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  options: CXRGenerationOptions = {}
): void {
  const seed = options.seed ?? 42;
  const isPneumonia = options.isPneumonia ?? false;
  const severity = options.pneumoniaSeverity ?? 'moderate';
  const location = options.pneumoniaLocation ?? 'right_lower_lobe';
  const rand = createPRNG(seed);

  // Clear background (unattenuated X-ray detector background: deep dark slate/black)
  ctx.save();
  ctx.fillStyle = '#06080d';
  ctx.fillRect(0, 0, width, height);

  // Soft tissue mantle (chest wall and neck attenuation)
  const cx = width / 2;
  const cy = height / 2;

  // Background body contour
  const bodyGrad = ctx.createRadialGradient(cx, cy, width * 0.1, cx, cy, width * 0.55);
  bodyGrad.addColorStop(0, '#2a3547');
  bodyGrad.addColorStop(0.65, '#1e2636');
  bodyGrad.addColorStop(1, '#080c14');
  ctx.fillStyle = bodyGrad;

  ctx.beginPath();
  ctx.ellipse(cx, cy * 1.05, width * 0.44, height * 0.48, 0, 0, Math.PI * 2);
  ctx.fill();

  // Neck and supraclavicular soft tissue
  ctx.fillStyle = '#222d3d';
  ctx.fillRect(cx - width * 0.16, 0, width * 0.32, height * 0.2);

  // Hemidiaphragm baseline curves (dense subdiaphragmatic abdomen / liver / spleen)
  // Right hemidiaphragm is anatomically higher than left by ~15-20% height
  ctx.fillStyle = '#3a485c';
  // Right subdiaphragm (Liver dome)
  ctx.beginPath();
  ctx.moveTo(0, height);
  ctx.lineTo(0, height * 0.72);
  ctx.bezierCurveTo(
    width * 0.2, height * 0.65,
    width * 0.4, height * 0.67,
    cx, height * 0.76
  );
  ctx.lineTo(cx, height);
  ctx.closePath();
  ctx.fill();

  // Left subdiaphragm (Stomach / Spleen)
  ctx.beginPath();
  ctx.moveTo(cx, height);
  ctx.lineTo(cx, height * 0.76);
  ctx.bezierCurveTo(
    width * 0.6, height * 0.69,
    width * 0.8, height * 0.70,
    width, height * 0.75
  );
  ctx.lineTo(width, height);
  ctx.closePath();
  ctx.fill();

  // Gastric air bubble (Magenblase) in left upper quadrant under left hemidiaphragm
  const stomachBubbleGrad = ctx.createRadialGradient(
    width * 0.62, height * 0.76, 2,
    width * 0.62, height * 0.76, width * 0.07
  );
  stomachBubbleGrad.addColorStop(0, '#0c121b');
  stomachBubbleGrad.addColorStop(0.7, '#141d2c');
  stomachBubbleGrad.addColorStop(1, '#2c384a');
  ctx.fillStyle = stomachBubbleGrad;
  ctx.beginPath();
  ctx.ellipse(width * 0.62, height * 0.76, width * 0.065, height * 0.038, 0, 0, Math.PI * 2);
  ctx.fill();

  // Low-density Dark Lung Fields (Parenchyma)
  // Right lung (on viewer's left)
  const rLungGrad = ctx.createRadialGradient(
    width * 0.28, height * 0.44, 10,
    width * 0.28, height * 0.44, width * 0.24
  );
  rLungGrad.addColorStop(0, '#0a0f18');
  rLungGrad.addColorStop(0.6, '#101724');
  rLungGrad.addColorStop(1, '#1b2434');

  ctx.fillStyle = rLungGrad;
  ctx.beginPath();
  // Apical dome
  ctx.moveTo(cx - width * 0.08, height * 0.16);
  ctx.bezierCurveTo(
    cx - width * 0.18, height * 0.13,
    width * 0.12, height * 0.22,
    width * 0.11, height * 0.40
  );
  // Lateral chest wall border
  ctx.bezierCurveTo(
    width * 0.10, height * 0.55,
    width * 0.13, height * 0.65,
    width * 0.14, height * 0.70 // Costophrenic angle
  );
  // Diaphragmatic contour
  ctx.bezierCurveTo(
    width * 0.24, height * 0.66,
    width * 0.34, height * 0.68,
    cx - width * 0.06, height * 0.74 // Cardiophrenic angle
  );
  // Medial mediastinal border
  ctx.bezierCurveTo(
    cx - width * 0.06, height * 0.55,
    cx - width * 0.07, height * 0.32,
    cx - width * 0.08, height * 0.16
  );
  ctx.closePath();
  ctx.fill();

  // Left lung (on viewer's right)
  const lLungGrad = ctx.createRadialGradient(
    width * 0.70, height * 0.46, 10,
    width * 0.70, height * 0.46, width * 0.24
  );
  lLungGrad.addColorStop(0, '#0b1019');
  lLungGrad.addColorStop(0.6, '#111826');
  lLungGrad.addColorStop(1, '#1c2536');

  ctx.fillStyle = lLungGrad;
  ctx.beginPath();
  // Apical dome
  ctx.moveTo(cx + width * 0.08, height * 0.16);
  ctx.bezierCurveTo(
    cx + width * 0.18, height * 0.13,
    width * 0.88, height * 0.22,
    width * 0.89, height * 0.40
  );
  // Lateral chest wall border
  ctx.bezierCurveTo(
    width * 0.90, height * 0.55,
    width * 0.87, height * 0.67,
    width * 0.86, height * 0.73 // Costophrenic angle
  );
  // Diaphragmatic contour
  ctx.bezierCurveTo(
    width * 0.76, height * 0.70,
    width * 0.64, height * 0.72,
    cx + width * 0.12, height * 0.75 // Cardiophrenic angle
  );
  // Cardiac notch / left heart border
  ctx.bezierCurveTo(
    cx + width * 0.16, height * 0.64,
    cx + width * 0.10, height * 0.42,
    cx + width * 0.08, height * 0.16
  );
  ctx.closePath();
  ctx.fill();

  // Thoracic Spine Column & Intervertebral Pedicles
  ctx.fillStyle = '#42516b';
  ctx.fillRect(cx - width * 0.035, height * 0.1, width * 0.07, height * 0.7);

  // Pedicle shadows (bilateral small ovals along spine)
  ctx.fillStyle = '#556685';
  for (let s = 0; s < 10; s++) {
    const sy = height * 0.15 + s * (height * 0.055);
    // Left and right pedicles
    ctx.beginPath();
    ctx.ellipse(cx - width * 0.022, sy, width * 0.007, height * 0.012, 0, 0, Math.PI * 2);
    ctx.ellipse(cx + width * 0.022, sy, width * 0.007, height * 0.012, 0, 0, Math.PI * 2);
    ctx.fill();
    // Intervertebral disc space (subtle radiolucent band)
    ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
    ctx.fillRect(cx - width * 0.032, sy + height * 0.022, width * 0.064, height * 0.005);
    ctx.fillStyle = '#556685';
  }

  // Trachea (radiolucent central air column) & Carina Bifurcation
  ctx.fillStyle = '#0f172a';
  ctx.beginPath();
  ctx.rect(cx - width * 0.013, height * 0.06, width * 0.026, height * 0.22);
  ctx.fill();

  // Mainstem bronchi (right and left bifurcations)
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = width * 0.018;
  ctx.beginPath();
  // Right bronchus (steeper angle)
  ctx.moveTo(cx, height * 0.28);
  ctx.lineTo(cx - width * 0.055, height * 0.35);
  // Left bronchus (more horizontal)
  ctx.moveTo(cx, height * 0.28);
  ctx.lineTo(cx + width * 0.065, height * 0.36);
  ctx.stroke();

  // Mediastinum & Cardiac Silhouette
  // Aortic knob (arch) on left of upper spine
  ctx.fillStyle = '#50617e';
  ctx.beginPath();
  ctx.ellipse(cx + width * 0.035, height * 0.29, width * 0.045, height * 0.035, 0.2, 0, Math.PI * 2);
  ctx.fill();

  // Left cardiac border (apex extending towards 5th left intercostal space)
  ctx.fillStyle = '#475773';
  ctx.beginPath();
  ctx.moveTo(cx, height * 0.32);
  // Right heart border (right atrium)
  ctx.bezierCurveTo(
    cx - width * 0.06, height * 0.45,
    cx - width * 0.05, height * 0.60,
    cx - width * 0.02, height * 0.72
  );
  // Inferior cardiac border
  ctx.lineTo(cx + width * 0.14, height * 0.72);
  // Left cardiac border (left ventricle)
  ctx.bezierCurveTo(
    cx + width * 0.22, height * 0.64,
    cx + width * 0.16, height * 0.48,
    cx + width * 0.06, height * 0.34
  );
  ctx.closePath();
  ctx.fill();

  // Pulmonary Hilum & Vascular Tree Markings (Bronchovascular arborization)
  ctx.lineWidth = 1.2;
  const drawVascularBranch = (
    startX: number,
    startY: number,
    length: number,
    angle: number,
    depth: number
  ) => {
    if (depth <= 0) return;
    const endX = startX + Math.cos(angle) * length;
    const endY = startY + Math.sin(angle) * length;

    ctx.strokeStyle = `rgba(148, 163, 184, ${0.12 + depth * 0.08})`;
    ctx.lineWidth = Math.max(0.6, depth * 0.8);
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Bifurcate
    const dAngle = 0.28 + rand() * 0.18;
    drawVascularBranch(endX, endY, length * 0.72, angle - dAngle, depth - 1);
    drawVascularBranch(endX, endY, length * 0.75, angle + dAngle, depth - 1);
  };

  // Right hilum vessels
  for (let b = 0; b < 6; b++) {
    const angle = Math.PI - 0.3 + (b - 2.5) * 0.35 + (rand() - 0.5) * 0.2;
    drawVascularBranch(cx - width * 0.07, height * 0.38 + b * 5, width * 0.09, angle, 3);
  }

  // Left hilum vessels
  for (let b = 0; b < 6; b++) {
    const angle = 0.3 + (b - 2.5) * 0.35 + (rand() - 0.5) * 0.2;
    drawVascularBranch(cx + width * 0.08, height * 0.39 + b * 5, width * 0.09, angle, 3);
  }

  // Rib Cage (Posterior horizontal ribs & anterior angled ribs)
  // Posterior ribs: horizontally curved bands from spine to lateral thorax
  for (let r = 1; r <= 9; r++) {
    const yNorm = 0.16 + r * 0.06;
    const yCenter = height * yNorm;
    const rx = width * (0.28 + r * 0.015);
    const ry = height * 0.045;

    // Bone cortical margins
    ctx.strokeStyle = `rgba(186, 198, 218, ${0.28 - r * 0.015})`;
    ctx.lineWidth = width * 0.014;

    // Right posterior rib
    ctx.beginPath();
    ctx.ellipse(cx - width * 0.19, yCenter, rx * 0.55, ry, 0.05 * r, Math.PI * 0.05, Math.PI * 0.95);
    ctx.stroke();

    // Left posterior rib
    ctx.beginPath();
    ctx.ellipse(cx + width * 0.19, yCenter, rx * 0.55, ry, -0.05 * r, Math.PI * 0.05, Math.PI * 0.95);
    ctx.stroke();

    // Anterior ribs (angling downwards and medially towards sternum)
    ctx.strokeStyle = `rgba(148, 163, 184, ${0.16 - r * 0.01})`;
    ctx.lineWidth = width * 0.01;
    ctx.beginPath();
    ctx.moveTo(cx - width * 0.38, yCenter - height * 0.02);
    ctx.quadraticCurveTo(
      cx - width * 0.25, yCenter + height * 0.04,
      cx - width * 0.08, yCenter + height * 0.08
    );
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx + width * 0.38, yCenter - height * 0.02);
    ctx.quadraticCurveTo(
      cx + width * 0.25, yCenter + height * 0.04,
      cx + width * 0.08, yCenter + height * 0.08
    );
    ctx.stroke();
  }

  // Bilateral Clavicles (collar bones)
  ctx.strokeStyle = 'rgba(215, 225, 240, 0.65)';
  ctx.lineWidth = width * 0.022;
  ctx.lineCap = 'round';

  // Right clavicle
  ctx.beginPath();
  ctx.moveTo(cx - width * 0.04, height * 0.17);
  ctx.bezierCurveTo(
    cx - width * 0.14, height * 0.15,
    cx - width * 0.28, height * 0.17,
    cx - width * 0.38, height * 0.15
  );
  ctx.stroke();

  // Left clavicle
  ctx.beginPath();
  ctx.moveTo(cx + width * 0.04, height * 0.17);
  ctx.bezierCurveTo(
    cx + width * 0.14, height * 0.15,
    cx + width * 0.28, height * 0.17,
    cx + width * 0.38, height * 0.15
  );
  ctx.stroke();

  // Lateral Scapular shadows
  ctx.strokeStyle = 'rgba(100, 116, 139, 0.35)';
  ctx.lineWidth = width * 0.012;
  // Right scapula
  ctx.beginPath();
  ctx.moveTo(width * 0.11, height * 0.25);
  ctx.lineTo(width * 0.08, height * 0.45);
  ctx.stroke();
  // Left scapula
  ctx.beginPath();
  ctx.moveTo(width * 0.89, height * 0.25);
  ctx.lineTo(width * 0.92, height * 0.45);
  ctx.stroke();

  // PATHOLOGY: Pneumonia Consolidation / Infiltrates
  if (isPneumonia) {
    const sevMultiplier = severity === 'severe' ? 1.4 : severity === 'mild' ? 0.65 : 1.0;

    let patchCenters: Array<{ x: number; y: number; radius: number; intensity: number }> = [];

    if (location === 'right_lower_lobe') {
      // Right base opacity above costophrenic sulcus
      patchCenters = [
        { x: cx - width * 0.22, y: height * 0.60, radius: width * 0.14 * sevMultiplier, intensity: 0.68 },
        { x: cx - width * 0.16, y: height * 0.63, radius: width * 0.11 * sevMultiplier, intensity: 0.55 },
        { x: cx - width * 0.26, y: height * 0.56, radius: width * 0.09 * sevMultiplier, intensity: 0.48 },
      ];
    } else if (location === 'right_middle_lobe') {
      // Right mid-zone obscuring right cardiac margin (classic silhouette sign)
      patchCenters = [
        { x: cx - width * 0.15, y: height * 0.48, radius: width * 0.12 * sevMultiplier, intensity: 0.64 },
        { x: cx - width * 0.22, y: height * 0.46, radius: width * 0.10 * sevMultiplier, intensity: 0.52 },
      ];
    } else if (location === 'left_lower_lobe') {
      // Left retrocardiac or base consolidation
      patchCenters = [
        { x: cx + width * 0.20, y: height * 0.62, radius: width * 0.13 * sevMultiplier, intensity: 0.65 },
        { x: cx + width * 0.14, y: height * 0.65, radius: width * 0.10 * sevMultiplier, intensity: 0.50 },
      ];
    } else {
      // Bilateral multifocal pneumonia
      patchCenters = [
        { x: cx - width * 0.20, y: height * 0.58, radius: width * 0.11 * sevMultiplier, intensity: 0.58 },
        { x: cx + width * 0.20, y: height * 0.60, radius: width * 0.11 * sevMultiplier, intensity: 0.54 },
        { x: cx - width * 0.16, y: height * 0.46, radius: width * 0.08 * sevMultiplier, intensity: 0.42 },
      ];
    }

    // Render fluffy alveolar opacities with ragged borders
    patchCenters.forEach(patch => {
      const pGrad = ctx.createRadialGradient(
        patch.x, patch.y, patch.radius * 0.08,
        patch.x, patch.y, patch.radius
      );
      pGrad.addColorStop(0, `rgba(235, 242, 250, ${patch.intensity * 0.85})`);
      pGrad.addColorStop(0.4, `rgba(180, 195, 215, ${patch.intensity * 0.55})`);
      pGrad.addColorStop(0.8, `rgba(100, 118, 145, ${patch.intensity * 0.22})`);
      pGrad.addColorStop(1, 'rgba(15, 23, 42, 0)');

      ctx.fillStyle = pGrad;
      ctx.beginPath();
      ctx.arc(patch.x, patch.y, patch.radius, 0, Math.PI * 2);
      ctx.fill();

      // Add air bronchograms: dark radiolucent branching tubes inside the consolidation
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.65)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(patch.x - patch.radius * 0.4, patch.y - patch.radius * 0.3);
      ctx.lineTo(patch.x, patch.y);
      ctx.lineTo(patch.x + patch.radius * 0.3, patch.y + patch.radius * 0.4);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(patch.x, patch.y);
      ctx.lineTo(patch.x - patch.radius * 0.2, patch.y + patch.radius * 0.35);
      ctx.stroke();
    });
  }

  // Anatomical orientation markers 'R' and 'L' (standard radiograph projection markers)
  ctx.font = `bold ${Math.max(12, Math.floor(width * 0.038))}px monospace`;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('R', width * 0.06, height * 0.12);
  ctx.fillText('L', width * 0.94, height * 0.12);

  ctx.restore();
}

/**
 * Generate a 224x224 grayscale Float32Array phantom buffer in range [0, 1].
 */
export function generateProceduralCXRBuffer(
  width = 224,
  height = 224,
  options: CXRGenerationOptions = {}
): Float32Array {
  // Use offscreen canvas for pixel extraction
  const offscreen = document.createElement('canvas');
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Canvas 2D context unavailable');
  }

  renderProceduralCXR(ctx, width, height, options);
  const imgData = ctx.getImageData(0, 0, width, height);
  const buffer = new Float32Array(width * height);

  for (let i = 0; i < buffer.length; i++) {
    const idx = i * 4;
    // Standard NTSC grayscale luminance formula
    const gray = (0.299 * imgData.data[idx] + 0.587 * imgData.data[idx + 1] + 0.114 * imgData.data[idx + 2]) / 255.0;
    buffer[i] = Math.max(0.0, Math.min(1.0, gray));
  }

  return buffer;
}

/**
 * Generate an ImageData instance directly from procedural CXR generator.
 */
export function generateProceduralCXRImageData(
  width = 224,
  height = 224,
  options: CXRGenerationOptions = {}
): ImageData {
  const offscreen = document.createElement('canvas');
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Canvas 2D context unavailable');
  }

  renderProceduralCXR(ctx, width, height, options);
  return ctx.getImageData(0, 0, width, height);
}

/**
 * Asynchronously load a real sample radiograph as an HTMLImageElement
 */
export function loadRealSampleImage(id: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const dataUrl = REAL_CXR_SAMPLES[id] || REAL_CXR_SAMPLES.normal_1;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = err => reject(err);
    img.src = dataUrl;
  });
}

/**
 * Load a real sample radiograph as an ImageData object at specified dimensions.
 */
export async function loadRealSampleImageData(
  id: string,
  targetWidth = 224,
  targetHeight = 224
): Promise<ImageData> {
  const img = await loadRealSampleImage(id);
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Canvas 2D context unavailable');
  }

  ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
  return ctx.getImageData(0, 0, targetWidth, targetHeight);
}
