import React, { useState } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  CheckCircle2, 
  ShieldAlert, 
  FileSpreadsheet, 
  FileCode, 
  FileCheck2,
  Filter, 
  ArrowUpDown, 
  Sparkles,
  PieChart,
  ShieldCheck
} from 'lucide-react';
import { 
  BENCHMARK_REPORT, 
  generateBenchmarkCsv 
} from '../../data/benchmarkData';
import { ClinicalReportModal } from './ClinicalReportModal';
import type { ModelArchitecture, CohortState, DiagnosticCohortMetrics } from '../../types';

type SortColumn = 'architecture' | 'cohort' | 'accuracy' | 'auc_roc' | 'f1' | 'sensitivity' | 'specificity';
type SortDirection = 'asc' | 'desc';

interface TableRowData {
  architecture: ModelArchitecture;
  cohort: CohortState;
  metrics: DiagnosticCohortMetrics;
}

export const BenchmarkDashboard: React.FC = () => {
  const [selectedArch, setSelectedArch] = useState<ModelArchitecture>('densenet121');
  const [activeChartView, setActiveChartView] = useState<'bars' | 'radar' | 'resilience'>('bars');
  const [selectedMetric, setSelectedMetric] = useState<'auc_roc' | 'accuracy' | 'f1' | 'sensitivity'>('auc_roc');
  
  // Table filters & sorting
  const [tableSearch, setTableSearch] = useState<string>('');
  const [tableArchFilter, setTableArchFilter] = useState<'all' | ModelArchitecture>('all');
  const [tableCohortFilters, setTableCohortFilters] = useState<Record<CohortState, boolean>>({
    clean: true,
    degraded: true,
    generic: true,
    adaptive: true
  });
  const [sortCol, setSortCol] = useState<SortColumn>('auc_roc');
  const [sortDir, setSortDir] = useState<SortDirection>('desc');

  // Modal
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);

  // Export JSON
  const exportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(BENCHMARK_REPORT, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', 'evaluation_report.json');
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Export CSV
  const exportCSV = () => {
    const csvContent = generateBenchmarkCsv();
    const dataStr = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', 'radfilm_benchmark_evaluation.csv');
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const currentArchCohorts = BENCHMARK_REPORT.clinical_diagnosis_performance[selectedArch];

  // Prepare table data
  const architectures: ModelArchitecture[] = ['densenet121', 'resnet50', 'efficientnet_b0'];
  const cohortsList: CohortState[] = ['clean', 'degraded', 'generic', 'adaptive'];

  const allTableRows: TableRowData[] = [];
  for (const arch of architectures) {
    for (const c of cohortsList) {
      allTableRows.push({
        architecture: arch,
        cohort: c,
        metrics: BENCHMARK_REPORT.clinical_diagnosis_performance[arch][c]
      });
    }
  }

  // Filter & Sort table rows
  const filteredRows = allTableRows.filter(row => {
    if (tableArchFilter !== 'all' && row.architecture !== tableArchFilter) return false;
    if (!tableCohortFilters[row.cohort]) return false;
    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase();
      const matchArch = row.architecture.toLowerCase().includes(q);
      const matchCohort = row.cohort.toLowerCase().includes(q);
      if (!matchArch && !matchCohort) return false;
    }
    return true;
  });

  filteredRows.sort((a, b) => {
    let aVal: number | string = 0;
    let bVal: number | string = 0;

    if (sortCol === 'architecture') {
      aVal = a.architecture;
      bVal = b.architecture;
      return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    } else if (sortCol === 'cohort') {
      aVal = a.cohort;
      bVal = b.cohort;
      return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    } else {
      aVal = a.metrics[sortCol] ?? 0;
      bVal = b.metrics[sortCol] ?? 0;
      return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    }
  });

  const handleSort = (col: SortColumn) => {
    if (sortCol === col) {
      setSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortCol(col);
      setSortDir('desc');
    }
  };

  // Radar chart mathematical coordinates for 5 axes:
  // Accuracy, AUC-ROC, F1, Sensitivity, Specificity
  const radarAxes = [
    { key: 'accuracy', label: 'Accuracy' },
    { key: 'auc_roc', label: 'AUC-ROC' },
    { key: 'f1', label: 'F1 Score' },
    { key: 'sensitivity', label: 'Sensitivity' },
    { key: 'specificity', label: 'Specificity' }
  ];

  const radarRadius = 110;
  const radarCenter = { x: 160, y: 150 };

  const getRadarPoint = (axisIdx: number, val: number) => {
    const angle = (Math.PI * 2 / radarAxes.length) * axisIdx - Math.PI / 2;
    const r = Math.max(0, Math.min(1, val)) * radarRadius;
    return {
      x: radarCenter.x + r * Math.cos(angle),
      y: radarCenter.y + r * Math.sin(angle)
    };
  };

  const buildRadarPolygon = (cohort: CohortState) => {
    const data = currentArchCohorts[cohort];
    const points = radarAxes.map((axis, idx) => {
      const val = data[axis.key as keyof DiagnosticCohortMetrics] ?? 0;
      const pt = getRadarPoint(idx, val);
      return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
    });
    return points.join(' ');
  };

  return (
    <div className="space-y-6 font-mono text-xs">
      {/* 4 Interactive Cohort Metric Delta Callouts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: SSIM Recovery */}
        <div className="bg-[#0f172a] border border-[#1e293b] hover:border-emerald-500/40 rounded-xl p-5 transition-all shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">SSIM Structural Gain</span>
            <div className="w-7 h-7 rounded bg-emerald-950/80 border border-emerald-800/60 flex items-center justify-center text-emerald-400">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-3xl font-extrabold text-emerald-400">+41.1%</span>
            <span className="text-xs text-slate-400 font-mono">0.718 vs 0.509</span>
          </div>
          <div className="mt-1 flex items-center space-x-1.5 text-[11px] text-emerald-300">
            <span>+72.7% vs generic filter (0.416)</span>
          </div>
          <p className="mt-2 text-[11px] text-slate-400 font-sans leading-relaxed">
            FiLM U-Net preserves structural bronchial walls & lung margins without generic lowpass blurring.
          </p>
        </div>

        {/* Card 2: DenseNet-121 AUC Recovery */}
        <div className="bg-[#0f172a] border border-[#1e293b] hover:border-cyan-500/40 rounded-xl p-5 transition-all shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">DenseNet AUC Recovery</span>
            <div className="w-7 h-7 rounded bg-cyan-950/80 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
              <CheckCircle2 className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-3xl font-extrabold text-cyan-400">+0.387 AUC</span>
            <span className="text-xs text-slate-400 font-mono">0.818 vs 0.431</span>
          </div>
          <div className="mt-1 flex items-center space-x-1.5 text-[11px] text-cyan-300">
            <span>Sens: 37.5% → 91.7% (+54.2% gain)</span>
          </div>
          <p className="mt-2 text-[11px] text-slate-400 font-sans leading-relaxed">
            Rescues downstream clinical diagnosis from sub-chance diagnostic collapse under severe mottle.
          </p>
        </div>

        {/* Card 3: Reconstruction Error Reduction */}
        <div className="bg-[#0f172a] border border-[#1e293b] hover:border-emerald-500/40 rounded-xl p-5 transition-all shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">MAE Reconstruction Error</span>
            <div className="w-7 h-7 rounded bg-emerald-950/80 border border-emerald-800/60 flex items-center justify-center text-emerald-400">
              <ShieldAlert className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-3xl font-extrabold text-emerald-400">-59.0%</span>
            <span className="text-xs text-slate-400 font-mono">0.0280 vs 0.0682</span>
          </div>
          <div className="mt-1 flex items-center space-x-1.5 text-[11px] text-emerald-300">
            <span>-81.0% error vs generic (0.1475)</span>
          </div>
          <p className="mt-2 text-[11px] text-slate-400 font-sans leading-relaxed">
            Quantitatively minimizes pixel-level distortion across all 48 benchmark radiograph cohorts.
          </p>
        </div>

        {/* Card 4: PSNR Preservation */}
        <div className="bg-[#0f172a] border border-[#1e293b] hover:border-cyan-500/40 rounded-xl p-5 transition-all shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">PSNR vs Filter Baseline</span>
            <div className="w-7 h-7 rounded bg-cyan-950/80 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
              <Sparkles className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-3xl font-extrabold text-cyan-400">+13.84 dB</span>
            <span className="text-xs text-slate-400 font-mono">29.34 vs 15.50</span>
          </div>
          <div className="mt-1 flex items-center space-x-1.5 text-[11px] text-cyan-300">
            <span>Peak Signal-to-Noise Ratio</span>
          </div>
          <p className="mt-2 text-[11px] text-slate-400 font-sans leading-relaxed">
            Eliminates destructive border-clipping artifacts produced by standard CLAHE bilateral passes.
          </p>
        </div>
      </div>

      {/* Interactive Visualizations: Multi-Architecture Bar Chart & Radar Chart */}
      <div className="bg-[#0f172a] border border-[#1e293b] rounded-xl p-5 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded bg-cyan-950 border border-cyan-800/50 flex items-center justify-center text-cyan-400">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
                Degradation Resilience Across Diagnostic Architectures
              </h2>
              <p className="text-[11px] text-slate-400 font-sans">
                Comparative analysis of DenseNet-121, ResNet-50, and EfficientNet-B0 under adaptive restoration
              </p>
            </div>
          </div>

          {/* Visualization Controls */}
          <div className="flex flex-wrap items-center gap-2">
            {/* View Switcher */}
            <div className="flex items-center bg-[#131d31] p-1 rounded-lg border border-[#1e293b] text-xs">
              <button
                onClick={() => setActiveChartView('bars')}
                className={`px-3 py-1 rounded transition-colors flex items-center space-x-1.5 ${
                  activeChartView === 'bars'
                    ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Cohort Bars</span>
              </button>

              <button
                onClick={() => setActiveChartView('radar')}
                className={`px-3 py-1 rounded transition-colors flex items-center space-x-1.5 ${
                  activeChartView === 'radar'
                    ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <PieChart className="w-3.5 h-3.5" />
                <span>5-Axis Radar</span>
              </button>

              <button
                onClick={() => setActiveChartView('resilience')}
                className={`px-3 py-1 rounded transition-colors flex items-center space-x-1.5 ${
                  activeChartView === 'resilience'
                    ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Resilience Matrix</span>
              </button>
            </div>

            {/* Architecture Selector for Radar */}
            {activeChartView === 'radar' && (
              <div className="flex items-center bg-[#131d31] p-1 rounded-lg border border-[#1e293b] text-xs">
                {(['densenet121', 'resnet50', 'efficientnet_b0'] as ModelArchitecture[]).map(arch => (
                  <button
                    key={arch}
                    onClick={() => setSelectedArch(arch)}
                    className={`px-2.5 py-1 rounded transition-colors ${
                      selectedArch === arch
                        ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {arch === 'densenet121' ? 'DenseNet' : arch === 'resnet50' ? 'ResNet' : 'EfficientNet'}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Chart View 1: Cohort Bar Comparison */}
        {activeChartView === 'bars' && (
          <div className="space-y-4">
            {/* Metric Selector for Bars */}
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
              <span className="text-slate-400">Select Target Efficacy Metric:</span>
              <div className="flex items-center bg-[#131d31] p-1 rounded border border-[#1e293b]">
                {[
                  { key: 'auc_roc', label: 'AUC-ROC' },
                  { key: 'accuracy', label: 'Accuracy' },
                  { key: 'f1', label: 'F1 Score' },
                  { key: 'sensitivity', label: 'Sensitivity (Recall)' }
                ].map(m => (
                  <button
                    key={m.key}
                    onClick={() => setSelectedMetric(m.key as any)}
                    className={`px-3 py-1 rounded transition-colors ${
                      selectedMetric === m.key
                        ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Multi-Architecture Grouped Bars */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              {(['densenet121', 'resnet50', 'efficientnet_b0'] as ModelArchitecture[]).map(arch => {
                const perf = BENCHMARK_REPORT.clinical_diagnosis_performance[arch];
                const archName = arch === 'densenet121' ? 'DenseNet-121' : arch === 'resnet50' ? 'ResNet-50' : 'EfficientNet-B0';

                return (
                  <div key={arch} className="bg-[#090d16] p-4 rounded-xl border border-[#1e293b] space-y-3">
                    <div className="flex justify-between items-center border-b border-[#1e293b] pb-2">
                      <span className="font-bold text-slate-200">{archName}</span>
                      <span className="text-[10px] text-cyan-400 uppercase font-mono">
                        {selectedMetric.toUpperCase()}
                      </span>
                    </div>

                    <div className="space-y-2.5 pt-1">
                      {[
                        { cohort: 'clean', label: 'Clean Baseline', val: perf.clean[selectedMetric] ?? 0, color: '#06b6d4', bg: 'bg-cyan-500' },
                        { cohort: 'adaptive', label: 'Proposed FiLM', val: perf.adaptive[selectedMetric] ?? 0, color: '#10b981', bg: 'bg-emerald-500' },
                        { cohort: 'degraded', label: 'Degraded Input', val: perf.degraded[selectedMetric] ?? 0, color: '#ef4444', bg: 'bg-red-500' },
                        { cohort: 'generic', label: 'Generic Filter', val: perf.generic[selectedMetric] ?? 0, color: '#f59e0b', bg: 'bg-amber-500' }
                      ].map(bar => (
                        <div key={bar.cohort} className="space-y-1">
                          <div className="flex justify-between items-center text-[11px]">
                            <span className="text-slate-400">{bar.label}</span>
                            <span className="font-mono font-bold" style={{ color: bar.color }}>
                              {selectedMetric === 'auc_roc' ? bar.val.toFixed(3) : `${(bar.val * 100).toFixed(1)}%`}
                            </span>
                          </div>
                          <div className="w-full bg-[#1e293b] h-2 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${bar.bg}`}
                              style={{ width: `${Math.max(2, Math.min(100, bar.val * 100))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Architecture Delta Badge */}
                    <div className="pt-2 border-t border-[#1e293b] text-[10px] flex justify-between items-center text-slate-400">
                      <span>Adaptive vs Degraded:</span>
                      <span className={`font-bold font-mono ${
                        (perf.adaptive[selectedMetric] ?? 0) >= (perf.degraded[selectedMetric] ?? 0)
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}>
                        {((perf.adaptive[selectedMetric] ?? 0) - (perf.degraded[selectedMetric] ?? 0) >= 0 ? '+' : '')}
                        {selectedMetric === 'auc_roc'
                          ? ((perf.adaptive[selectedMetric] ?? 0) - (perf.degraded[selectedMetric] ?? 0)).toFixed(3)
                          : `${(((perf.adaptive[selectedMetric] ?? 0) - (perf.degraded[selectedMetric] ?? 0)) * 100).toFixed(1)}%`}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Chart View 2: 5-Axis Clinical Radar Chart */}
        {activeChartView === 'radar' && (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
            <div className="md:col-span-7 flex justify-center">
              <svg viewBox="0 0 320 300" className="w-full max-w-[340px] aspect-square bg-[#05080f] rounded-xl border border-[#1e293b] p-3 font-mono text-[9px]">
                {/* Concentric Web Grid (0.2, 0.4, 0.6, 0.8, 1.0) */}
                {[0.2, 0.4, 0.6, 0.8, 1.0].map(level => {
                  const pts = radarAxes.map((_, idx) => {
                    const pt = getRadarPoint(idx, level);
                    return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
                  }).join(' ');

                  return (
                    <g key={level}>
                      <polygon
                        points={pts}
                        fill="none"
                        stroke="#1e293b"
                        strokeWidth="1"
                        strokeDasharray={level === 1.0 ? 'none' : '2 2'}
                      />
                      <text x={radarCenter.x + 3} y={radarCenter.y - level * radarRadius - 2} fill="#475569" className="text-[8px]">
                        {level.toFixed(1)}
                      </text>
                    </g>
                  );
                })}

                {/* Spokes from Center */}
                {radarAxes.map((axis, idx) => {
                  const endPt = getRadarPoint(idx, 1.0);
                  const labelPt = getRadarPoint(idx, 1.18);
                  return (
                    <g key={axis.key}>
                      <line
                        x1={radarCenter.x}
                        y1={radarCenter.y}
                        x2={endPt.x}
                        y2={endPt.y}
                        stroke="#1e293b"
                        strokeWidth="1"
                      />
                      <text
                        x={labelPt.x}
                        y={labelPt.y + 3}
                        fill="#94a3b8"
                        textAnchor="middle"
                        className="font-semibold text-[9px]"
                      >
                        {axis.label}
                      </text>
                    </g>
                  );
                })}

                {/* Clean Cohort Polygon */}
                <polygon
                  points={buildRadarPolygon('clean')}
                  fill="#06b6d4"
                  fillOpacity="0.10"
                  stroke="#06b6d4"
                  strokeWidth="1.5"
                />

                {/* Generic Filter Polygon */}
                <polygon
                  points={buildRadarPolygon('generic')}
                  fill="#f59e0b"
                  fillOpacity="0.08"
                  stroke="#f59e0b"
                  strokeWidth="1.5"
                  strokeDasharray="3 2"
                />

                {/* Degraded Polygon */}
                <polygon
                  points={buildRadarPolygon('degraded')}
                  fill="#ef4444"
                  fillOpacity="0.08"
                  stroke="#ef4444"
                  strokeWidth="1.5"
                />

                {/* Adaptive FiLM Polygon */}
                <polygon
                  points={buildRadarPolygon('adaptive')}
                  fill="#10b981"
                  fillOpacity="0.25"
                  stroke="#10b981"
                  strokeWidth="2.5"
                />
              </svg>
            </div>

            {/* Radar Legend & Clinical Interpretation */}
            <div className="md:col-span-5 space-y-3">
              <div className="p-3 rounded-lg bg-[#090d16] border border-cyan-800/40">
                <div className="flex items-center justify-between text-cyan-300 font-bold">
                  <span>Ground Truth Clean</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Pristine baseline with balanced sensitivity and high specificity.</p>
              </div>

              <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-700/50">
                <div className="flex items-center justify-between text-emerald-300 font-bold">
                  <span>Proposed Adaptive FiLM U-Net</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Wide sensitivity envelope preserving high-risk pneumonia cases (91.7%).</p>
              </div>

              <div className="p-3 rounded-lg bg-[#090d16] border border-red-900/40">
                <div className="flex items-center justify-between text-red-400 font-bold">
                  <span>Simulated Degraded</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Severe shrinkage along sensitivity axis (false negative disaster).</p>
              </div>

              <div className="p-3 rounded-lg bg-[#090d16] border border-amber-900/40">
                <div className="flex items-center justify-between text-amber-300 font-bold">
                  <span>Generic Filter (CLAHE+Bilateral)</span>
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Severe collapse across AUC, F1, and Sensitivity.</p>
              </div>
            </div>
          </div>
        )}

        {/* Chart View 3: Resilience Matrix */}
        {activeChartView === 'resilience' && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#090d16] text-slate-400 uppercase text-[11px] border-b border-[#1e293b]">
                <tr>
                  <th className="py-2.5 px-3">Architecture</th>
                  <th className="py-2.5 px-3">Clean AUC</th>
                  <th className="py-2.5 px-3">Degraded AUC</th>
                  <th className="py-2.5 px-3">Generic AUC</th>
                  <th className="py-2.5 px-3">Adaptive AUC</th>
                  <th className="py-2.5 px-3">AUC Net Recovery</th>
                  <th className="py-2.5 px-3">Sensitivity Gain</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1e293b]/60 bg-[#090d16]">
                <tr className="hover:bg-[#131d31]">
                  <td className="py-3 px-3 font-bold text-cyan-300">DenseNet-121</td>
                  <td className="py-3 px-3">1.000</td>
                  <td className="py-3 px-3 text-red-400">0.431</td>
                  <td className="py-3 px-3 text-amber-400">0.306</td>
                  <td className="py-3 px-3 font-bold text-emerald-400">0.818</td>
                  <td className="py-3 px-3 font-bold text-emerald-300">+0.387 AUC</td>
                  <td className="py-3 px-3 font-bold text-emerald-300">+54.2% (37.5% → 91.7%)</td>
                </tr>
                <tr className="hover:bg-[#131d31]">
                  <td className="py-3 px-3 font-bold text-slate-300">ResNet-50</td>
                  <td className="py-3 px-3">0.658</td>
                  <td className="py-3 px-3 text-red-400">0.323</td>
                  <td className="py-3 px-3 text-amber-400">0.297</td>
                  <td className="py-3 px-3 font-bold text-slate-200">0.352</td>
                  <td className="py-3 px-3 font-bold text-slate-300">+0.029 AUC</td>
                  <td className="py-3 px-3 font-bold text-slate-300">+20.8% (45.8% → 66.7%)</td>
                </tr>
                <tr className="hover:bg-[#131d31]">
                  <td className="py-3 px-3 font-bold text-slate-300">EfficientNet-B0</td>
                  <td className="py-3 px-3">0.440</td>
                  <td className="py-3 px-3 text-emerald-400">0.672</td>
                  <td className="py-3 px-3 text-emerald-400">0.656</td>
                  <td className="py-3 px-3 text-amber-400">0.194</td>
                  <td className="py-3 px-3 text-red-400">-0.478 AUC</td>
                  <td className="py-3 px-3 text-red-400">-54.2% (Feature shift)</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Filterable Benchmark Table with CSV/JSON Export */}
      <div className="bg-[#0f172a] border border-[#1e293b] rounded-xl p-5 space-y-4">
        {/* Table Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Complete Clinical Benchmark Cohort Database (N=48)
            </h3>
          </div>

          {/* Action Buttons: Export CSV, Export JSON, Generate Report */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setIsReportModalOpen(true)}
              className="px-3 py-1.5 rounded bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 flex items-center space-x-1.5 transition-colors font-semibold"
            >
              <FileCheck2 className="w-3.5 h-3.5 text-cyan-400" />
              <span>Generate Clinical Report</span>
            </button>

            <button
              onClick={exportCSV}
              className="px-3 py-1.5 rounded bg-[#131d31] hover:bg-[#1a253e] text-slate-300 border border-[#1e293b] flex items-center space-x-1.5 transition-colors"
            >
              <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-400" />
              <span>Export CSV</span>
            </button>

            <button
              onClick={exportJSON}
              className="px-3 py-1.5 rounded bg-[#131d31] hover:bg-[#1a253e] text-slate-300 border border-[#1e293b] flex items-center space-x-1.5 transition-colors"
            >
              <FileCode className="w-3.5 h-3.5 text-cyan-400" />
              <span>Export JSON</span>
            </button>
          </div>
        </div>

        {/* Filters Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center">
          {/* Search */}
          <div className="sm:col-span-4">
            <input
              type="text"
              placeholder="Search architecture or cohort..."
              value={tableSearch}
              onChange={(e) => setTableSearch(e.target.value)}
              className="w-full bg-[#090d16] border border-[#1e293b] rounded-lg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>

          {/* Architecture Filter */}
          <div className="sm:col-span-4 flex items-center space-x-1 bg-[#090d16] p-1 rounded-lg border border-[#1e293b]">
            <span className="text-[10px] text-slate-500 px-2 uppercase">Arch:</span>
            {(['all', 'densenet121', 'resnet50', 'efficientnet_b0'] as const).map(a => (
              <button
                key={a}
                onClick={() => setTableArchFilter(a)}
                className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                  tableArchFilter === a
                    ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {a === 'all' ? 'All' : a === 'densenet121' ? 'DenseNet' : a === 'resnet50' ? 'ResNet' : 'EfficientNet'}
              </button>
            ))}
          </div>

          {/* Cohort Checkboxes */}
          <div className="sm:col-span-4 flex items-center justify-end space-x-2 text-[11px]">
            {(['clean', 'adaptive', 'degraded', 'generic'] as CohortState[]).map(c => (
              <button
                key={c}
                onClick={() => setTableCohortFilters(prev => ({ ...prev, [c]: !prev[c] }))}
                className={`px-2 py-0.5 rounded border capitalize transition-colors ${
                  tableCohortFilters[c]
                    ? 'bg-[#131d31] border-cyan-500/40 text-cyan-300 font-semibold'
                    : 'bg-[#090d16] border-[#1e293b] text-slate-600'
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        {/* Table Container */}
        <div className="overflow-x-auto rounded-lg border border-[#1e293b]">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#090d16] text-slate-400 uppercase text-[11px] border-b border-[#1e293b]">
              <tr>
                <th 
                  onClick={() => handleSort('architecture')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>Architecture</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('cohort')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>Cohort State</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('accuracy')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>Accuracy</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('auc_roc')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>AUC-ROC</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('f1')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>F1 Score</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('sensitivity')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>Sensitivity (TPR)</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('specificity')}
                  className="py-2.5 px-3 cursor-pointer hover:text-slate-200"
                >
                  <div className="flex items-center space-x-1">
                    <span>Specificity (TNR)</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e293b]/60 bg-[#0b101c]">
              {filteredRows.map((row, idx) => {
                const isAdaptive = row.cohort === 'adaptive';
                const isClean = row.cohort === 'clean';
                const isDegraded = row.cohort === 'degraded';

                return (
                  <tr 
                    key={idx}
                    className={`hover:bg-[#131d31] transition-colors ${
                      isAdaptive ? 'bg-emerald-950/15' : ''
                    }`}
                  >
                    <td className="py-2.5 px-3 font-semibold text-slate-300">
                      {row.architecture === 'densenet121' ? 'DenseNet-121' : row.architecture === 'resnet50' ? 'ResNet-50' : 'EfficientNet-B0'}
                    </td>
                    <td className="py-2.5 px-3 font-medium flex items-center space-x-2">
                      <span className={`w-2 h-2 rounded-full ${
                        isClean ? 'bg-cyan-400' : isAdaptive ? 'bg-emerald-400 animate-pulse' : isDegraded ? 'bg-red-400' : 'bg-amber-400'
                      }`} />
                      <span className={
                        isClean ? 'text-cyan-300' : isAdaptive ? 'text-emerald-300 font-bold' : isDegraded ? 'text-red-400' : 'text-amber-400'
                      }>
                        {row.cohort.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono">{(row.metrics.accuracy * 100).toFixed(1)}%</td>
                    <td className={`py-2.5 px-3 font-mono font-bold ${
                      isAdaptive ? 'text-emerald-300' : isClean ? 'text-cyan-300' : isDegraded ? 'text-red-400' : 'text-amber-400'
                    }`}>
                      {row.metrics.auc_roc.toFixed(3)}
                    </td>
                    <td className="py-2.5 px-3 font-mono">{row.metrics.f1.toFixed(3)}</td>
                    <td className={`py-2.5 px-3 font-mono font-semibold ${
                      row.metrics.sensitivity > 0.85 ? 'text-emerald-400' : 'text-slate-300'
                    }`}>
                      {(row.metrics.sensitivity * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 font-mono">{(row.metrics.specificity * 100).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Clinical Report Modal */}
      <ClinicalReportModal
        isOpen={isReportModalOpen}
        onClose={() => setIsReportModalOpen(false)}
      />
    </div>
  );
};
