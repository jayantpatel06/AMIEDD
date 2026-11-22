import React, { useState } from 'react';
import { 
  Activity, 
  Sliders, 
  Info, 
  CheckSquare, 
  Square, 
  Sparkles
} from 'lucide-react';
import { BENCHMARK_REPORT, ROC_CURVES } from '../../data/benchmarkData';
import type { ModelArchitecture, CohortState } from '../../types';

export const RocComparison: React.FC = () => {
  const [selectedArch, setSelectedArch] = useState<ModelArchitecture>('densenet121');
  const [visibleCohorts, setVisibleCohorts] = useState<Record<CohortState, boolean>>({
    clean: true,
    adaptive: true,
    degraded: true,
    generic: true
  });
  const [hoveredPoint, setHoveredPoint] = useState<{
    cohort: CohortState;
    fpr: number;
    tpr: number;
    threshold: number;
    x: number;
    y: number;
  } | null>(null);
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0.50);

  const toggleCohort = (c: CohortState) => {
    setVisibleCohorts(prev => ({ ...prev, [c]: !prev[c] }));
  };

  const curveData = ROC_CURVES[selectedArch];
  const cohortsData = BENCHMARK_REPORT.clinical_diagnosis_performance[selectedArch];

  // Colors for each cohort (Monochrome scientific scheme)
  const cohortConfig: Record<CohortState, { label: string; color: string; stroke: string; bg: string; border: string }> = {
    clean: { 
      label: 'Ground Truth Clean', 
      color: '#ffffff', 
      stroke: '#ffffff', 
      bg: 'bg-neutral-800', 
      border: 'border-white' 
    },
    adaptive: { 
      label: 'Proposed FiLM U-Net', 
      color: '#d4d4d4', 
      stroke: '#d4d4d4', 
      bg: 'bg-neutral-800', 
      border: 'border-neutral-400' 
    },
    degraded: { 
      label: 'Simulated Degraded', 
      color: '#525252', 
      stroke: '#525252', 
      bg: 'bg-neutral-900', 
      border: 'border-neutral-700' 
    },
    generic: { 
      label: 'Generic Filter (CLAHE+Bilateral)', 
      color: '#8c8c8c', 
      stroke: '#8c8c8c', 
      bg: 'bg-neutral-900', 
      border: 'border-neutral-600' 
    }
  };

  // SVG coordinate transformation
  // Box: x in [50, 360], y in [30, 340]
  const plotLeft = 50;
  const plotTop = 30;
  const plotWidth = 310;
  const plotHeight = 310;
  const plotBottom = plotTop + plotHeight;
  const plotRight = plotLeft + plotWidth;

  const toSvgX = (fpr: number) => plotLeft + fpr * plotWidth;
  const toSvgY = (tpr: number) => plotBottom - tpr * plotHeight;

  // Build SVG path string from points
  const buildSvgPath = (points: { fpr: number; tpr: number }[]) => {
    if (points.length === 0) return '';
    return points
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toSvgX(p.fpr).toFixed(1)} ${toSvgY(p.tpr).toFixed(1)}`)
      .join(' ');
  };

  // Calculate operating point for adaptive cohort based on decisionThreshold
  const adaptivePoints = curveData.adaptive.points;
  let closestAdaptivePoint = adaptivePoints[0];
  let minDiff = 999;
  for (const pt of adaptivePoints) {
    const diff = Math.abs(pt.threshold - decisionThreshold);
    if (diff < minDiff) {
      minDiff = diff;
      closestAdaptivePoint = pt;
    }
  }

  const operatingSensitivity = closestAdaptivePoint.tpr;
  const operatingSpecificity = 1 - closestAdaptivePoint.fpr;
  const youdenJ = operatingSensitivity + operatingSpecificity - 1;

  return (
    <div className="bg-[#0f172a] border border-[#1e293b] rounded-lg p-5 space-y-5 font-mono text-xs">
      {/* Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded bg-cyan-950/80 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Receiver Operating Characteristic (ROC-AUC) Analysis
            </h3>
            <p className="text-[11px] text-slate-400 font-sans">
              Clinical diagnostic sensitivity vs false positive rate trajectory across cohorts
            </p>
          </div>
        </div>

        {/* Architecture Selector */}
        <div className="flex items-center space-x-2">
          <span className="text-[11px] text-slate-400 uppercase hidden sm:inline">Backbone:</span>
          <div className="flex items-center bg-[#131d31] p-1 rounded border border-[#1e293b]">
            {(['densenet121', 'resnet50', 'efficientnet_b0'] as ModelArchitecture[]).map(arch => (
              <button
                key={arch}
                onClick={() => setSelectedArch(arch)}
                className={`px-3 py-1 rounded transition-colors text-xs font-semibold ${
                  selectedArch === arch
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {arch === 'densenet121' ? 'DenseNet-121' : arch === 'resnet50' ? 'ResNet-50' : 'EfficientNet-B0'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Cohort Visibility Toggles */}
      <div className="flex flex-wrap items-center gap-3 bg-[#090d16] p-3 rounded-lg border border-[#1e293b]">
        <span className="text-[11px] text-slate-400 uppercase font-semibold mr-1">Filter Curves:</span>
        {(['clean', 'adaptive', 'degraded', 'generic'] as CohortState[]).map(cohort => {
          const cfg = cohortConfig[cohort];
          const isVisible = visibleCohorts[cohort];
          const aucVal = curveData[cohort].auc;

          return (
            <button
              key={cohort}
              onClick={() => toggleCohort(cohort)}
              className={`flex items-center space-x-2 px-2.5 py-1 rounded border transition-all text-xs ${
                isVisible ? `${cfg.bg} ${cfg.border} text-slate-200` : 'bg-transparent border-[#1e293b] text-slate-500'
              }`}
            >
              {isVisible ? (
                <CheckSquare className="w-3.5 h-3.5" style={{ color: cfg.color }} />
              ) : (
                <Square className="w-3.5 h-3.5 text-slate-600" />
              )}
              <span className="font-semibold">{cfg.label}</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded font-bold" style={{ backgroundColor: `${cfg.color}22`, color: cfg.color }}>
                AUC {aucVal.toFixed(3)}
              </span>
            </button>
          );
        })}
      </div>

      {/* Main ROC Plot Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* SVG ROC Plot Container */}
        <div className="lg:col-span-7 flex flex-col items-center">
          <div className="relative w-full max-w-[440px] aspect-square bg-[#05080f] rounded-xl border border-[#1e293b] p-2 shadow-inner">
            <svg
              viewBox="0 0 400 400"
              className="w-full h-full font-mono text-[10px] select-none"
              onMouseLeave={() => setHoveredPoint(null)}
            >
              {/* Plot Background Area */}
              <rect
                x={plotLeft}
                y={plotTop}
                width={plotWidth}
                height={plotHeight}
                fill="#090d16"
                stroke="#1e293b"
                strokeWidth="1"
              />

              {/* Gridlines & Ticks */}
              {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map(v => {
                const yPos = toSvgY(v);
                const xPos = toSvgX(v);
                return (
                  <g key={v}>
                    {/* Horizontal Grid */}
                    <line
                      x1={plotLeft}
                      y1={yPos}
                      x2={plotRight}
                      y2={yPos}
                      stroke="#1e293b"
                      strokeDasharray="2 3"
                    />
                    {/* Vertical Grid */}
                    <line
                      x1={xPos}
                      y1={plotTop}
                      x2={xPos}
                      y2={plotBottom}
                      stroke="#1e293b"
                      strokeDasharray="2 3"
                    />
                    {/* Y-axis labels (TPR) */}
                    <text x={plotLeft - 8} y={yPos + 3} fill="#64748b" textAnchor="end">
                      {v.toFixed(1)}
                    </text>
                    {/* X-axis labels (FPR) */}
                    <text x={xPos} y={plotBottom + 16} fill="#64748b" textAnchor="middle">
                      {v.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Random Classifier Baseline (AUC = 0.500) */}
              <line
                x1={plotLeft}
                y1={plotBottom}
                x2={plotRight}
                y2={plotTop}
                stroke="#334155"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />
              <text x={235} y={185} fill="#475569" transform="rotate(-45 235 185)" textAnchor="middle" className="text-[9px]">
                Chance Line (AUC 0.500)
              </text>

              {/* Render Cohort Curves */}
              {(['generic', 'degraded', 'adaptive', 'clean'] as CohortState[]).map(cohort => {
                if (!visibleCohorts[cohort]) return null;
                const cData = curveData[cohort];
                const cfg = cohortConfig[cohort];
                const pathStr = buildSvgPath(cData.points);

                return (
                  <g key={cohort} className="transition-opacity duration-200">
                    {/* Stroke Glow for Adaptive FiLM */}
                    {cohort === 'adaptive' && (
                      <path
                        d={pathStr}
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="7"
                        strokeOpacity="0.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    )}

                    {/* Main Line */}
                    <path
                      d={pathStr}
                      fill="none"
                      stroke={cfg.stroke}
                      strokeWidth={cohort === 'adaptive' ? '2.5' : '2'}
                      strokeDasharray={cohort === 'generic' ? '4 3' : 'none'}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />

                    {/* Data Points */}
                    {cData.points.map((pt, pIdx) => {
                      const cx = toSvgX(pt.fpr);
                      const cy = toSvgY(pt.tpr);
                      return (
                        <circle
                          key={pIdx}
                          cx={cx}
                          cy={cy}
                          r={cohort === 'adaptive' ? 4 : 3}
                          fill={cfg.color}
                          stroke="#090d16"
                          strokeWidth="1.5"
                          className="cursor-pointer hover:r-6 transition-all"
                          onMouseEnter={() =>
                            setHoveredPoint({
                              cohort,
                              fpr: pt.fpr,
                              tpr: pt.tpr,
                              threshold: pt.threshold,
                              x: cx,
                              y: cy
                            })
                          }
                        />
                      );
                    })}
                  </g>
                );
              })}

              {/* Active Operating Point Crosshair on Adaptive Curve */}
              {visibleCohorts.adaptive && (
                <g>
                  <circle
                    cx={toSvgX(closestAdaptivePoint.fpr)}
                    cy={toSvgY(closestAdaptivePoint.tpr)}
                    r="7"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2"
                    className="animate-ping"
                  />
                  <circle
                    cx={toSvgX(closestAdaptivePoint.fpr)}
                    cy={toSvgY(closestAdaptivePoint.tpr)}
                    r="5"
                    fill="#10b981"
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                </g>
              )}

              {/* Hover Crosshairs & Tooltip */}
              {hoveredPoint && (
                <g>
                  <line
                    x1={plotLeft}
                    y1={hoveredPoint.y}
                    x2={plotRight}
                    y2={hoveredPoint.y}
                    stroke="#94a3b8"
                    strokeWidth="1"
                    strokeDasharray="2 2"
                  />
                  <line
                    x1={hoveredPoint.x}
                    y1={plotTop}
                    x2={hoveredPoint.x}
                    y2={plotBottom}
                    stroke="#94a3b8"
                    strokeWidth="1"
                    strokeDasharray="2 2"
                  />
                </g>
              )}

              {/* Axis Titles */}
              <text x={plotLeft + plotWidth / 2} y={plotBottom + 35} fill="#94a3b8" textAnchor="middle" className="font-semibold text-[11px]">
                False Positive Rate (1 - Specificity)
              </text>
              <text
                x={-(plotTop + plotHeight / 2)}
                y={18}
                fill="#94a3b8"
                textAnchor="middle"
                transform="rotate(-90)"
                className="font-semibold text-[11px]"
              >
                True Positive Rate (Sensitivity)
              </text>
            </svg>

            {/* Hover Tooltip Overlay */}
            {hoveredPoint && (
              <div 
                className="absolute z-10 bg-[#0f172a]/95 border border-cyan-500/50 p-2.5 rounded shadow-xl text-[11px] pointer-events-none transform -translate-x-1/2 -translate-y-full"
                style={{
                  left: `${(hoveredPoint.x / 400) * 100}%`,
                  top: `${(hoveredPoint.y / 400) * 100 - 4}%`
                }}
              >
                <div className="font-bold text-slate-100 flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cohortConfig[hoveredPoint.cohort].color }}></span>
                  <span>{cohortConfig[hoveredPoint.cohort].label}</span>
                </div>
                <div className="text-slate-300 mt-1 space-y-0.5">
                  <div>Sensitivity (TPR): <span className="text-emerald-400 font-bold">{(hoveredPoint.tpr * 100).toFixed(1)}%</span></div>
                  <div>Specificity: <span className="text-cyan-400 font-bold">{((1 - hoveredPoint.fpr) * 100).toFixed(1)}%</span></div>
                  <div>FPR: <span className="text-slate-400">{(hoveredPoint.fpr * 100).toFixed(1)}%</span></div>
                  <div>Decision Cutoff: <span className="text-amber-400 font-mono">τ = {hoveredPoint.threshold.toFixed(2)}</span></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Efficacy Summary & Operating Point Tuning */}
        <div className="lg:col-span-5 space-y-4">
          {/* Clinical Operating Point Slider */}
          <div className="bg-[#090d16] p-4 rounded-lg border border-[#1e293b] space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-1.5 text-slate-200 font-semibold">
                <Sliders className="w-4 h-4 text-emerald-400" />
                <span>Decision Cutoff Threshold (τ)</span>
              </div>
              <span className="text-emerald-400 font-bold font-mono bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60">
                τ = {decisionThreshold.toFixed(2)}
              </span>
            </div>

            <input
              type="range"
              min="0.10"
              max="0.95"
              step="0.01"
              value={decisionThreshold}
              onChange={(e) => setDecisionThreshold(parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-[#1e293b] rounded-lg"
              aria-label="Clinical Decision Cutoff Slider"
            />

            <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
              <div className="bg-[#131d31] p-2 rounded border border-[#1e293b]">
                <span className="text-slate-400 block uppercase">Sensitivity</span>
                <span className="text-emerald-400 font-bold text-xs">{(operatingSensitivity * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-[#131d31] p-2 rounded border border-[#1e293b]">
                <span className="text-slate-400 block uppercase">Specificity</span>
                <span className="text-cyan-400 font-bold text-xs">{(operatingSpecificity * 100).toFixed(1)}%</span>
              </div>
              <div className="bg-[#131d31] p-2 rounded border border-[#1e293b]">
                <span className="text-slate-400 block uppercase">Youden Index</span>
                <span className="text-amber-300 font-bold text-xs">{youdenJ.toFixed(3)}</span>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 font-sans leading-tight">
              Moving cutoff τ adjusts sensitivity vs false alarm rate for high-stakes intensive care triage.
            </p>
          </div>

          {/* Model Cohort AUC Stat Cards */}
          <div className="space-y-2">
            <div className="p-3 rounded-lg bg-[#090d16] border border-cyan-800/50 flex justify-between items-center">
              <div>
                <span className="text-cyan-300 font-bold block">Ground Truth Clean Baseline</span>
                <span className="text-[10px] text-slate-400">Pristine detector resolution (100% sensitivity)</span>
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-cyan-400 font-mono">AUC {cohortsData.clean.auc_roc.toFixed(3)}</span>
                <span className="block text-[10px] text-slate-400 font-mono">Sens: {(cohortsData.clean.sensitivity * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-700/60 flex justify-between items-center">
              <div>
                <div className="flex items-center space-x-1.5 text-emerald-300 font-bold">
                  <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Proposed Adaptive FiLM U-Net</span>
                </div>
                <span className="text-[10px] text-slate-400">+0.387 AUC improvement over degraded input</span>
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-emerald-400 font-mono">AUC {cohortsData.adaptive.auc_roc.toFixed(3)}</span>
                <span className="block text-[10px] text-emerald-300 font-mono">Sens: {(cohortsData.adaptive.sensitivity * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-[#090d16] border border-red-900/50 flex justify-between items-center">
              <div>
                <span className="text-red-400 font-bold block">Simulated Degraded Radiographs</span>
                <span className="text-[10px] text-slate-400">Diagnostic collapse: high false-negative rate</span>
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-red-400 font-mono">AUC {cohortsData.degraded.auc_roc.toFixed(3)}</span>
                <span className="block text-[10px] text-red-400 font-mono">Sens: {(cohortsData.degraded.sensitivity * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-[#090d16] border border-amber-900/50 flex justify-between items-center">
              <div>
                <span className="text-amber-300 font-bold block">Generic Filter (CLAHE + Bilateral)</span>
                <span className="text-[10px] text-slate-400">Border smearing severely impairs downstream AI</span>
              </div>
              <div className="text-right">
                <span className="text-xs font-bold text-amber-400 font-mono">AUC {cohortsData.generic.auc_roc.toFixed(3)}</span>
                <span className="block text-[10px] text-amber-300 font-mono">Sens: {(cohortsData.generic.sensitivity * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Physics Insight Note */}
      <div className="bg-[#090d16] p-4 rounded-lg border border-[#1e293b] flex items-start space-x-3 font-sans text-xs">
        <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-mono font-semibold text-slate-200 uppercase text-[11px] block">
            Clinical Radiological Significance of ROC Separation:
          </span>
          <p className="text-slate-400 leading-relaxed">
            Generic histogram equalizations (CLAHE) and unconditioned bilateral filters amplify sensor noise into pseudo-consolidation patterns or over-smooth delicate peribronchial markings, reducing DenseNet-121 ROC-AUC to 0.306 (worse than random chance). By modulating intermediate U-Net feature maps using affine parameters <span className="font-mono text-cyan-400">(γ, β)</span> conditioned on the estimated degradation vector, the FiLM architecture restores critical high-frequency trabecular and alveolar contours, achieving <span className="font-mono text-emerald-400">AUC 0.818</span> and recovering sensitivity to <span className="font-mono text-emerald-400">91.7%</span>.
          </p>
        </div>
      </div>
    </div>
  );
};
