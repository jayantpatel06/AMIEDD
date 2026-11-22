import React from 'react';
import { SunMedium, Contrast, RotateCcw, ArrowUpDown } from 'lucide-react';
import type { WindowLevelPreset } from '../../types';
import { WINDOW_LEVEL_PRESETS } from '../../data/benchmarkData';

interface WindowLevelControlsProps {
  windowVal: number;
  levelVal: number;
  onWindowChange: (val: number) => void;
  onLevelChange: (val: number) => void;
  activePresetId?: string;
  onPresetSelect: (preset: WindowLevelPreset) => void;
  inverted?: boolean;
  onInvertToggle?: () => void;
}

export const WindowLevelControls: React.FC<WindowLevelControlsProps> = ({
  windowVal,
  levelVal,
  onWindowChange,
  onLevelChange,
  activePresetId,
  onPresetSelect,
  inverted = false,
  onInvertToggle,
}) => {
  const minVal = Math.round(levelVal - windowVal / 2);
  const maxVal = Math.round(levelVal + windowVal / 2);

  const svgWidth = 220;
  const svgHeight = 24;
  const huToX = (hu: number) => {
    const clamped = Math.max(-1000, Math.min(1000, hu));
    return ((clamped + 1000) / 2000) * svgWidth;
  };
  const x1 = huToX(minVal);
  const x2 = huToX(maxVal);

  return (
    <div className="bg-[#0e0e0e] border border-[#222222] rounded p-3 space-y-2.5 select-none font-mono">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#222222] pb-1.5">
        <div className="flex items-center space-x-1.5 text-xs font-semibold text-neutral-200">
          <Contrast className="w-3.5 h-3.5 text-white" />
          <span className="uppercase tracking-wider text-[11px]">Window / Level (HU)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          {onInvertToggle && (
            <button
              onClick={onInvertToggle}
              title="Toggle Radiograph Polarity"
              className={`px-1.5 py-0.5 rounded text-[10px] border transition-colors flex items-center space-x-1 ${
                inverted
                  ? 'bg-white border-white text-black font-semibold'
                  : 'bg-[#181818] border-[#2c2c2c] text-neutral-400 hover:text-white'
              }`}
            >
              <ArrowUpDown className="w-2.5 h-2.5 mr-0.5" />
              <span>{inverted ? 'Inverted' : 'Standard'}</span>
            </button>
          )}
          <button
            onClick={() => {
              const defaultPreset = WINDOW_LEVEL_PRESETS.find(p => p.id === 'lung') || WINDOW_LEVEL_PRESETS[0];
              onPresetSelect(defaultPreset);
            }}
            title="Reset to Lung Window"
            aria-label="Reset to Lung Window"
            className="p-1 rounded bg-[#181818] border border-[#2c2c2c] text-neutral-400 hover:text-white transition-colors"
          >
            <RotateCcw className="w-2.5 h-2.5" />
          </button>
        </div>
      </div>

      {/* Preset Buttons */}
      <div className="grid grid-cols-4 gap-1">
        {WINDOW_LEVEL_PRESETS.map(preset => {
          const isSelected = activePresetId === preset.id || (windowVal === preset.window && levelVal === preset.level);
          return (
            <button
              key={preset.id}
              onClick={() => onPresetSelect(preset)}
              className={`py-1 px-1.5 rounded text-center border transition-colors ${
                isSelected
                  ? 'bg-[#262626] border-white text-white font-semibold'
                  : 'bg-[#141414] border-[#262626] text-neutral-400 hover:border-neutral-700 hover:text-neutral-200'
              }`}
            >
              <div className="text-[10px] truncate leading-tight">{preset.name}</div>
              <div className="text-[9px] text-neutral-500 mt-0.5">
                {preset.window}/{preset.level}
              </div>
            </button>
          );
        })}
      </div>

      {/* Dual Sliders: Width & Center */}
      <div className="space-y-2 pt-1 text-[11px]">
        {/* Window Width */}
        <div className="space-y-1">
          <div className="flex justify-between items-center text-neutral-400">
            <span className="flex items-center text-[10px]">
              <SunMedium className="w-3 h-3 mr-1 text-neutral-300" /> Width (W)
            </span>
            <div className="flex items-center space-x-1">
              <input
                type="number"
                min="50"
                max="3000"
                step="25"
                value={windowVal}
                onChange={e => onWindowChange(Math.max(50, Math.min(3000, Number(e.target.value))))}
                className="w-14 bg-[#141414] border border-[#262626] rounded px-1 py-0.2 text-right text-white text-[10px] font-bold focus:outline-none focus:border-white"
              />
              <span className="text-[9px] text-neutral-500">HU</span>
            </div>
          </div>
          <input
            type="range"
            min="50"
            max="3000"
            step="10"
            aria-label="Window Width"
            value={windowVal}
            onChange={e => onWindowChange(Number(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Window Level */}
        <div className="space-y-1">
          <div className="flex justify-between items-center text-neutral-400">
            <span className="flex items-center text-[10px]">
              <Contrast className="w-3 h-3 mr-1 text-neutral-300" /> Center (L)
            </span>
            <div className="flex items-center space-x-1">
              <input
                type="number"
                min="-1000"
                max="1000"
                step="25"
                value={levelVal}
                onChange={e => onLevelChange(Math.max(-1000, Math.min(1000, Number(e.target.value))))}
                className="w-14 bg-[#141414] border border-[#262626] rounded px-1 py-0.2 text-right text-white text-[10px] font-bold focus:outline-none focus:border-white"
              />
              <span className="text-[9px] text-neutral-500">HU</span>
            </div>
          </div>
          <input
            type="range"
            min="-1000"
            max="1000"
            step="10"
            aria-label="Window Level"
            value={levelVal}
            onChange={e => onLevelChange(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Transfer Curve Ramp */}
      <div className="pt-1 flex items-center justify-between text-[9px] text-neutral-500">
        <span>Ramp: [{minVal} to {maxVal} HU]</span>
        <svg width={svgWidth} height={svgHeight} className="overflow-visible bg-black rounded border border-[#262626]">
          <defs>
            <linearGradient id="monoHuRamp" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#000000" />
              <stop offset={`${(x1 / svgWidth) * 100}%`} stopColor="#000000" />
              <stop offset={`${(x2 / svgWidth) * 100}%`} stopColor="#ffffff" />
              <stop offset="100%" stopColor="#ffffff" />
            </linearGradient>
          </defs>
          <rect x={0} y={svgHeight - 6} width={svgWidth} height={6} fill="url(#monoHuRamp)" />
          <path
            d={`M 0 ${svgHeight - 6} L ${Math.max(0, x1)} ${svgHeight - 6} L ${Math.min(svgWidth, x2)} 2 L ${svgWidth} 2`}
            fill="none"
            stroke="#ffffff"
            strokeWidth="1.5"
          />
        </svg>
      </div>
    </div>
  );
};
