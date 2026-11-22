import React, { useState } from 'react';
import {
  Activity
} from 'lucide-react';
import type { DegradationVector } from '../../types';
import { calculateRestorationMetrics, simulateDiagnosticPredictions } from '../../engine/simulator';
import { FilmInspector } from './FilmInspector';

interface TelemetryDeckProps {
  vector: DegradationVector;
  groundTruthLabel: 'Normal' | 'Pneumonia';
}

export const TelemetryDeck: React.FC<TelemetryDeckProps> = ({ vector, groundTruthLabel }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'detective' | 'film' | 'radiologists'>('overview');

  const metrics = calculateRestorationMetrics(vector);
  const predictions = simulateDiagnosticPredictions(vector, groundTruthLabel);

  const predictedVector = {
    noise: Math.min(1.0, Math.max(0.0, Number((vector.noise * 0.98 + 0.015).toFixed(3)))),
    blur: Math.min(1.0, Math.max(0.0, Number((vector.blur * 0.97 + 0.018).toFixed(3)))),
    contrast: Math.min(1.0, Math.max(0.0, Number((vector.contrast * 1.02 - 0.012).toFixed(3)))),
    compression: Math.min(1.0, Math.max(0.0, Number((vector.compression * 0.99 + 0.008).toFixed(3)))),
  };

  // SVG Radar Chart Math
  const cx = 90;
  const cy = 90;
  const rMax = 65;

  const getPoint = (val: number, angleDeg: number) => {
    const rad = ((angleDeg - 90) * Math.PI) / 180;
    const r = Math.max(4, Math.min(rMax, val * rMax));
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  };

  const gtNoise = getPoint(vector.noise, 0);
  const gtBlur = getPoint(vector.blur, 90);
  const gtContrast = getPoint(vector.contrast, 180);
  const gtComp = getPoint(vector.compression, 270);

  const predNoise = getPoint(predictedVector.noise, 0);
  const predBlur = getPoint(predictedVector.blur, 90);
  const predContrast = getPoint(predictedVector.contrast, 180);
  const predComp = getPoint(predictedVector.compression, 270);

  const gtPolygon = `${gtNoise.x},${gtNoise.y} ${gtBlur.x},${gtBlur.y} ${gtContrast.x},${gtContrast.y} ${gtComp.x},${gtComp.y}`;
  const predPolygon = `${predNoise.x},${predNoise.y} ${predBlur.x},${predBlur.y} ${predContrast.x},${predContrast.y} ${predComp.x},${predComp.y}`;

  const isPneumonia = groundTruthLabel === 'Pneumonia';
  const denseNetAdaptive = predictions.densenet121.adaptive;
  const denseNetDegraded = predictions.densenet121.degraded;

  return (
    <div className="bg-[#0e0e0e] border border-[#222222] rounded p-3 space-y-3 font-mono text-xs select-none">
      {/* Top Filter Tabs: Clean Monochrome Segmented Switch */}
      <div className="flex items-center justify-between border-b border-[#222222] pb-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-white flex items-center">
          <Activity className="w-3.5 h-3.5 mr-1.5 text-white" /> Model Pipeline
        </span>
        <div className="flex bg-[#141414] p-0.5 rounded border border-[#262626] text-[10px]">
          {(['overview', 'detective', 'film', 'radiologists'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-2 py-0.5 rounded transition-colors uppercase ${
                activeTab === tab
                  ? 'bg-[#262626] text-white font-semibold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              {tab === 'overview' ? 'All' : tab}
            </button>
          ))}
        </div>
      </div>

      {/* OVERVIEW / STAGE 1: ResNet-18 Detective */}
      {(activeTab === 'overview' || activeTab === 'detective') && (
        <div className="space-y-2">
          <div className="flex justify-between items-center text-[10px] text-neutral-400">
            <span className="text-white font-semibold uppercase">Stage 1 · ResNet-18 Detective</span>
            <span className="text-neutral-500">14.8ms</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center bg-[#141414] p-2.5 rounded border border-[#222222]">
            {/* SVG Radar */}
            <div className="sm:col-span-5 flex flex-col items-center justify-center">
              <svg width={180} height={180} className="overflow-visible">
                {[0.25, 0.5, 0.75, 1.0].map((ring, idx) => (
                  <circle
                    key={idx}
                    cx={cx}
                    cy={cy}
                    r={rMax * ring}
                    fill="none"
                    stroke="#262626"
                    strokeWidth="1"
                    strokeDasharray={ring === 1.0 ? 'none' : '2,2'}
                  />
                ))}

                <line x1={cx} y1={cy - rMax} x2={cx} y2={cy + rMax} stroke="#262626" strokeWidth="1" />
                <line x1={cx - rMax} y1={cy} x2={cx + rMax} y2={cy} stroke="#262626" strokeWidth="1" />

                <polygon points={gtPolygon} fill="rgba(255, 255, 255, 0.12)" stroke="#ffffff" strokeWidth="1.5" />
                <polygon points={predPolygon} fill="none" stroke="#737373" strokeWidth="1" strokeDasharray="3,2" />

                <circle cx={gtNoise.x} cy={gtNoise.y} r="2.5" fill="#ffffff" />
                <circle cx={gtBlur.x} cy={gtBlur.y} r="2.5" fill="#ffffff" />
                <circle cx={gtContrast.x} cy={gtContrast.y} r="2.5" fill="#ffffff" />
                <circle cx={gtComp.x} cy={gtComp.y} r="2.5" fill="#ffffff" />

                <text x={cx} y={cy - rMax - 4} textAnchor="middle" fill="#a3a3a3" fontSize="8">NOISE</text>
                <text x={cx + rMax + 4} y={cy + 3} textAnchor="start" fill="#a3a3a3" fontSize="8">BLUR</text>
                <text x={cx} y={cy + rMax + 10} textAnchor="middle" fill="#a3a3a3" fontSize="8">CONTRAST</text>
                <text x={cx - rMax - 4} y={cy + 3} textAnchor="end" fill="#a3a3a3" fontSize="8">JPEG</text>
              </svg>

              <div className="flex items-center space-x-3 text-[9px] text-neutral-400 mt-1">
                <span className="flex items-center text-white">
                  <span className="w-2 h-1.5 bg-white/40 border border-white mr-1 rounded-sm" /> Target
                </span>
                <span className="flex items-center text-neutral-400">
                  <span className="w-2 h-0.5 bg-neutral-400 mr-1" /> Predicted
                </span>
              </div>
            </div>

            {/* Severity Metric Values */}
            <div className="sm:col-span-7 space-y-1.5 text-[10px]">
              {[
                { label: 'Quantum Noise', val: predictedVector.noise, gt: vector.noise },
                { label: 'Motion Blur', val: predictedVector.blur, gt: vector.blur },
                { label: 'Contrast Loss', val: predictedVector.contrast, gt: vector.contrast },
                { label: 'DCT Compression', val: predictedVector.compression, gt: vector.compression },
              ].map(m => (
                <div key={m.label} className="bg-black/60 p-1.5 rounded border border-[#222222] space-y-0.5">
                  <div className="flex justify-between">
                    <span className="text-neutral-400">{m.label}</span>
                    <span className="text-white font-bold">{(m.val * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-[#262626] h-1 rounded overflow-hidden">
                    <div className="bg-white h-full" style={{ width: `${m.val * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* OVERVIEW / STAGE 2: FiLM U-Net Quality Metrics */}
      {(activeTab === 'overview' || activeTab === 'film') && (
        <div className="space-y-2 pt-1 border-t border-[#222222]">
          <div className="flex justify-between items-center text-[10px] text-neutral-400">
            <span className="text-white font-semibold uppercase">Stage 2 · FiLM U-Net Restoration</span>
            <span className="text-neutral-400">59% Error Reduction</span>
          </div>

          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
            <div className="bg-[#141414] p-2 rounded border border-[#222222]">
              <div className="text-neutral-500 text-[9px]">PSNR</div>
              <div className="text-white font-bold text-xs mt-0.5">{metrics.psnr} dB</div>
              <div className="text-neutral-400 text-[8px] mt-0.5">vs 15.5 dB Generic</div>
            </div>
            <div className="bg-[#141414] p-2 rounded border border-[#222222]">
              <div className="text-neutral-500 text-[9px]">SSIM</div>
              <div className="text-white font-bold text-xs mt-0.5">{metrics.ssim}</div>
              <div className="text-neutral-400 text-[8px] mt-0.5">+41.1% Gain</div>
            </div>
            <div className="bg-[#141414] p-2 rounded border border-[#222222]">
              <div className="text-neutral-500 text-[9px]">MAE</div>
              <div className="text-white font-bold text-xs mt-0.5">{metrics.mae}</div>
              <div className="text-neutral-400 text-[8px] mt-0.5">-59% Residual</div>
            </div>
          </div>

          {activeTab === 'film' && (
            <div className="pt-2">
              <FilmInspector vector={vector} />
            </div>
          )}
        </div>
      )}

      {/* OVERVIEW / STAGE 3: AI Radiologists Efficacy */}
      {(activeTab === 'overview' || activeTab === 'radiologists') && (
        <div className="space-y-2 pt-1 border-t border-[#222222]">
          <div className="flex justify-between items-center text-[10px] text-neutral-400">
            <span className="text-white font-semibold uppercase">Stage 3 · AI Radiologists Consensus</span>
            <span className="text-neutral-400">Pneumonia Probability</span>
          </div>

          <div className="space-y-2 bg-[#141414] p-2.5 rounded border border-[#222222]">
            {/* DenseNet-121 */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px]">
                <span className="text-neutral-300 font-semibold">DenseNet-121</span>
                <span className="text-white font-bold">
                  {(denseNetAdaptive.pneumoniaProb * 100).toFixed(0)}%
                  <span className="text-neutral-500 font-normal ml-1.5">
                    (deg: {(denseNetDegraded.pneumoniaProb * 100).toFixed(0)}%)
                  </span>
                </span>
              </div>
              <div className="w-full bg-[#262626] h-1.5 rounded overflow-hidden flex">
                <div
                  className="bg-white h-full"
                  style={{ width: `${denseNetAdaptive.pneumoniaProb * 100}%` }}
                />
              </div>
            </div>

            {/* ResNet-50 */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px]">
                <span className="text-neutral-300 font-semibold">ResNet-50</span>
                <span className="text-white font-bold">
                  {(predictions.resnet50.adaptive.pneumoniaProb * 100).toFixed(0)}%
                  <span className="text-neutral-500 font-normal ml-1.5">
                    (deg: {(predictions.resnet50.degraded.pneumoniaProb * 100).toFixed(0)}%)
                  </span>
                </span>
              </div>
              <div className="w-full bg-[#262626] h-1.5 rounded overflow-hidden flex">
                <div
                  className="bg-white h-full"
                  style={{ width: `${predictions.resnet50.adaptive.pneumoniaProb * 100}%` }}
                />
              </div>
            </div>

            {/* EfficientNet-B0 */}
            <div className="space-y-1">
              <div className="flex justify-between text-[10px]">
                <span className="text-neutral-300 font-semibold">EfficientNet-B0</span>
                <span className="text-white font-bold">
                  {(predictions.efficientnet_b0.adaptive.pneumoniaProb * 100).toFixed(0)}%
                  <span className="text-neutral-500 font-normal ml-1.5">
                    (deg: {(predictions.efficientnet_b0.degraded.pneumoniaProb * 100).toFixed(0)}%)
                  </span>
                </span>
              </div>
              <div className="w-full bg-[#262626] h-1.5 rounded overflow-hidden flex">
                <div
                  className="bg-white h-full"
                  style={{ width: `${predictions.efficientnet_b0.adaptive.pneumoniaProb * 100}%` }}
                />
              </div>
            </div>

            {/* Diagnostic Consensus Callout */}
            <div className="pt-1.5 border-t border-[#222222] flex items-center justify-between text-[9px] text-neutral-400">
              <span>Ground Truth: {groundTruthLabel.toUpperCase()}</span>
              <span className="text-white font-semibold">
                {isPneumonia && denseNetAdaptive.pneumoniaProb >= 0.5 ? 'Diagnostic Salvage Confirmed' : 'Confidence Restored'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
