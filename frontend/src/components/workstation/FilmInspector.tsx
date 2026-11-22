import React, { useState, useMemo } from 'react';
import { Layers, HelpCircle } from 'lucide-react';
import type { DegradationVector } from '../../types';

interface FilmInspectorProps {
  vector: DegradationVector;
}

interface LayerModulationData {
  id: string;
  name: string;
  depth: number;
  channels: number;
  resolution: string;
  description: string;
  gammaMean: number;
  gammaStd: number;
  betaMean: number;
  betaStd: number;
  channelGammas: number[];
  channelBetas: number[];
  energyNorm: number;
}

export const FilmInspector: React.FC<FilmInspectorProps> = ({ vector }) => {
  const [selectedLayerId, setSelectedLayerId] = useState<string>('dec4');
  const [hoveredChannel, setHoveredChannel] = useState<{ index: number; gamma: number; beta: number } | null>(null);

  const layersData: LayerModulationData[] = useMemo(() => {
    const { noise, blur, contrast, compression } = vector;

    const specs = [
      {
        id: 'dec4',
        name: 'Decoder 4 (Bottleneck Bridge)',
        depth: 4,
        channels: 256,
        resolution: '14 × 14',
        description: 'Global semantic feature recalibration & severe artifact suppression',
        weights: { noise: 0.15, blur: 0.35, contrast: 0.50, comp: 0.30 },
      },
      {
        id: 'dec3',
        name: 'Decoder 3 (Deep Structural)',
        depth: 3,
        channels: 128,
        resolution: '28 × 28',
        description: 'Anatomical landmark alignment & DCT block boundary de-ringing',
        weights: { noise: 0.25, blur: 0.45, contrast: 0.40, comp: 0.55 },
      },
      {
        id: 'dec2',
        name: 'Decoder 2 (Mid-Frequency)',
        depth: 2,
        channels: 64,
        resolution: '56 × 56',
        description: 'Bronchovascular markings & rib cortex edge recovery',
        weights: { noise: 0.55, blur: 0.40, contrast: 0.30, comp: 0.40 },
      },
      {
        id: 'dec1',
        name: 'Decoder 1 (Fine Radiographic)',
        depth: 1,
        channels: 32,
        resolution: '112 × 112',
        description: 'Quantum mottle Poisson noise suppression & fine alveolar texture',
        weights: { noise: 0.85, blur: 0.20, contrast: 0.25, comp: 0.25 },
      },
    ];

    return specs.map(spec => {
      const condStrength =
        spec.weights.noise * noise +
        spec.weights.blur * blur +
        spec.weights.contrast * contrast +
        spec.weights.comp * compression;

      const gammaMean = 1.0 + (condStrength - 0.5) * 0.42;
      const betaMean = (contrast * 0.3 - 0.15) + (noise * -0.1);

      const numChannelsToDisplay = 24;
      const channelGammas: number[] = [];
      const channelBetas: number[] = [];

      let sumSqGamma = 0;
      for (let c = 0; c < numChannelsToDisplay; c++) {
        const hash = Math.sin((spec.depth * 100 + c) * 12.9898) * 43758.5453;
        const channelVariance = (hash - Math.floor(hash)) - 0.5;

        const g = Math.max(0.4, Math.min(1.8, gammaMean + channelVariance * (0.2 + condStrength * 0.3)));
        const b = Math.max(-0.4, Math.min(0.4, betaMean + channelVariance * (0.1 + condStrength * 0.2)));

        channelGammas.push(Number(g.toFixed(3)));
        channelBetas.push(Number(b.toFixed(3)));
        sumSqGamma += g * g;
      }

      const energyNorm = Number((Math.sqrt(sumSqGamma) / Math.sqrt(numChannelsToDisplay)).toFixed(3));

      return {
        id: spec.id,
        name: spec.name,
        depth: spec.depth,
        channels: spec.channels,
        resolution: spec.resolution,
        description: spec.description,
        gammaMean: Number(gammaMean.toFixed(3)),
        gammaStd: Number((0.08 + condStrength * 0.06).toFixed(3)),
        betaMean: Number(betaMean.toFixed(3)),
        betaStd: Number((0.04 + condStrength * 0.05).toFixed(3)),
        channelGammas,
        channelBetas,
        energyNorm,
      };
    });
  }, [vector]);

  const activeLayer = layersData.find(l => l.id === selectedLayerId) || layersData[0];

  return (
    <div className="bg-[#0e0e0e] border border-[#222222] rounded p-3 space-y-3 font-mono text-xs select-none">
      {/* Layer Depth Selector */}
      <div className="grid grid-cols-4 gap-1">
        {layersData.map(layer => {
          const isSelected = selectedLayerId === layer.id;
          return (
            <button
              key={layer.id}
              onClick={() => setSelectedLayerId(layer.id)}
              className={`p-1.5 rounded text-left border transition-colors ${
                isSelected
                  ? 'bg-[#262626] border-white text-white font-semibold'
                  : 'bg-[#141414] border-[#262626] text-neutral-400 hover:text-white'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold">{layer.id.toUpperCase()}</span>
                <span className="text-[8px] text-neutral-500">D{layer.depth}</span>
              </div>
              <div className="text-[9px] text-neutral-500 truncate mt-0.5">
                {layer.channels}ch
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Layer Info */}
      <div className="bg-[#141414] p-2.5 rounded border border-[#222222] space-y-2">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-[11px] font-semibold text-white flex items-center space-x-1.5">
              <Layers className="w-3 h-3 text-white" />
              <span>{activeLayer.name}</span>
            </div>
            <div className="text-[9px] text-neutral-400 mt-0.5 leading-snug">
              {activeLayer.description}
            </div>
          </div>
          <div className="text-right text-[9px] text-neutral-400">
            Norm: <strong className="text-white">{activeLayer.energyNorm}</strong>
          </div>
        </div>

        {/* Global Modulation Parameters */}
        <div className="grid grid-cols-2 gap-2 pt-1 text-[10px]">
          <div className="bg-black/60 p-1.5 rounded border border-[#222222] space-y-1">
            <div className="flex justify-between items-center text-neutral-400">
              <span>Scale (1 + γ)</span>
              <span className="text-white font-bold">{activeLayer.gammaMean}</span>
            </div>
            <div className="w-full bg-[#262626] h-1 rounded overflow-hidden">
              <div
                className="bg-white h-full"
                style={{ width: `${Math.min(100, Math.max(10, ((activeLayer.gammaMean - 0.4) / 1.4) * 100))}%` }}
              />
            </div>
          </div>

          <div className="bg-black/60 p-1.5 rounded border border-[#222222] space-y-1">
            <div className="flex justify-between items-center text-neutral-400">
              <span>Bias (β)</span>
              <span className="text-white font-bold">
                {activeLayer.betaMean >= 0 ? `+${activeLayer.betaMean}` : activeLayer.betaMean}
              </span>
            </div>
            <div className="w-full bg-[#262626] h-1 rounded overflow-hidden">
              <div
                className="bg-white h-full"
                style={{ width: `${Math.min(100, Math.abs(activeLayer.betaMean) * 200)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* 24-Channel Grayscale Modulation Blocks */}
      <div className="space-y-1">
        <div className="flex items-center justify-between text-[10px] text-neutral-400">
          <span>Channel Modulation Heatmap (24 / {activeLayer.channels} ch)</span>
          {hoveredChannel && (
            <span className="text-white font-bold">
              Ch {hoveredChannel.index}: γ={hoveredChannel.gamma} β={hoveredChannel.beta}
            </span>
          )}
        </div>

        <div className="grid grid-cols-12 sm:grid-cols-24 gap-1 p-1.5 bg-black rounded border border-[#222222]">
          {activeLayer.channelGammas.map((gamma, idx) => {
            const beta = activeLayer.channelBetas[idx];
            const intensity = Math.min(1, Math.max(0, (gamma - 0.5) / 1.0));
            const lightness = Math.round(30 + intensity * 200);

            return (
              <div
                key={idx}
                onMouseEnter={() => setHoveredChannel({ index: idx + 1, gamma, beta })}
                onMouseLeave={() => setHoveredChannel(null)}
                style={{ backgroundColor: `rgb(${lightness}, ${lightness}, ${lightness})` }}
                className="h-6 rounded-sm cursor-pointer flex items-center justify-center border border-black/40"
              >
                <span className={`text-[7px] font-bold ${lightness > 130 ? 'text-black' : 'text-white'}`}>
                  {idx + 1}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* FiLM Formula */}
      <div className="bg-black/60 p-1.5 rounded border border-[#222222] flex items-center justify-between text-[9px] text-neutral-500">
        <div className="flex items-center space-x-1.5 text-neutral-300">
          <HelpCircle className="w-3 h-3 text-neutral-500" />
          <span>Affine: h&apos; = (1 + γ) · Norm(Conv(h)) + β</span>
        </div>
        <span>s ∈ [0, 1]⁴</span>
      </div>
    </div>
  );
};
