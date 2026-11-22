import React, { useState } from 'react';
import { 
  Layers, 
  BarChart3, 
  FileText, 
  Crosshair
} from 'lucide-react';
import { WorkstationView } from './components/workstation/WorkstationView';
import { BenchmarkDashboard } from './components/benchmarks/BenchmarkDashboard';
import { RocComparison } from './components/benchmarks/RocComparison';
import { CaseStudies } from './components/benchmarks/CaseStudies';
import type { CaseStudyItem } from './types';
import { SAMPLE_CASE_STUDIES } from './data/benchmarkData';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'workstation' | 'benchmarks' | 'casestudies'>('workstation');
  const [selectedCase, setSelectedCase] = useState<CaseStudyItem>(SAMPLE_CASE_STUDIES[0]);

  const handleSelectCase = (caseItem: CaseStudyItem) => {
    setSelectedCase(caseItem);
    setActiveTab('workstation');
  };

  return (
    <div className="min-h-screen bg-black text-neutral-200 flex flex-col font-mono selection:bg-neutral-800 selection:text-white">
      {/* Single Ultra-Clean Monochrome Header */}
      <header className="bg-[#0a0a0a] border-b border-[#222222] sticky top-0 z-50 px-4 py-2">
        <div className="max-w-[1800px] mx-auto flex items-center justify-between gap-4">
          {/* Logo & Product Title */}
          <div className="flex items-center space-x-2.5">
            <div className="w-6 h-6 rounded bg-[#181818] border border-[#333333] flex items-center justify-center text-white">
              <Crosshair className="w-3.5 h-3.5" />
            </div>
            <div className="flex items-baseline space-x-1.5">
              <span className="text-xs font-bold tracking-widest text-white uppercase">
                RADFILM
              </span>
              <span className="text-[10px] text-neutral-500 font-mono">
                v2.4
              </span>
            </div>
          </div>

          {/* Clean Segmented Tab Switcher */}
          <nav className="flex items-center bg-[#141414] p-0.5 rounded border border-[#262626] text-xs">
            <button
              onClick={() => setActiveTab('workstation')}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded transition-colors text-[11px] ${
                activeTab === 'workstation'
                  ? 'bg-[#262626] text-white font-semibold'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              <Layers className="w-3 h-3" />
              <span>Workstation</span>
            </button>

            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded transition-colors text-[11px] ${
                activeTab === 'benchmarks'
                  ? 'bg-[#262626] text-white font-semibold'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              <BarChart3 className="w-3 h-3" />
              <span>Benchmarks & ROC</span>
            </button>

            <button
              onClick={() => setActiveTab('casestudies')}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded transition-colors text-[11px] ${
                activeTab === 'casestudies'
                  ? 'bg-[#262626] text-white font-semibold'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              <FileText className="w-3 h-3" />
              <span>Case Studies</span>
            </button>
          </nav>

          {/* Active Case Quick Info */}
          <div className="hidden md:flex items-center space-x-2 text-[11px] text-neutral-400">
            <span className="text-neutral-500">Case:</span>
            <span className="text-neutral-200 font-semibold">{selectedCase.id}</span>
            <span className="text-neutral-500">({selectedCase.groundTruthLabel})</span>
          </div>
        </div>
      </header>

      {/* Main Workspace Body */}
      <main className="flex-1 max-w-[1800px] w-full mx-auto">
        {activeTab === 'workstation' && (
          <WorkstationView
            activeCase={selectedCase}
            onSelectCase={setSelectedCase}
          />
        )}

        {activeTab === 'benchmarks' && (
          <div className="p-4 lg:p-6 space-y-6">
            <BenchmarkDashboard />
            <RocComparison />
          </div>
        )}

        {activeTab === 'casestudies' && (
          <div className="p-4 lg:p-6">
            <CaseStudies onSelectCase={handleSelectCase} />
          </div>
        )}
      </main>

      {/* Minimal Footer */}
      <footer className="bg-[#0a0a0a] border-t border-[#1c1c1c] px-4 py-2 text-[10px] text-neutral-500 font-mono">
        <div className="max-w-[1800px] mx-auto flex items-center justify-between">
          <span>RadFiLM Adaptive U-Net CXR Restoration · NIH Test Cohort (N=48)</span>
          <span className="text-neutral-400">14.8ms Inference · DenseNet121 · ResNet50</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
