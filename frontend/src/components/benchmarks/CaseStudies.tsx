import React, { useState } from 'react';
import { 
  FileText, 
  ArrowRight, 
  ShieldCheck, 
  Search, 
  FileCheck2, 
  Check
} from 'lucide-react';
import { SAMPLE_CASE_STUDIES } from '../../data/benchmarkData';
import { ClinicalReportModal } from './ClinicalReportModal';
import type { CaseStudyItem } from '../../types';
import type { EnrichedCaseStudyItem } from '../../data/benchmarkData';

interface CaseStudiesProps {
  onSelectCase?: (caseItem: CaseStudyItem) => void;
}

export const CaseStudies: React.FC<CaseStudiesProps> = ({ onSelectCase }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPathology, setSelectedPathology] = useState<'All' | 'Pneumonia' | 'Normal'>('All');
  const [selectedSeverity, setSelectedSeverity] = useState<'All' | 'Mild' | 'Moderate' | 'Severe' | 'None'>('All');
  const [activeReportCase, setActiveReportCase] = useState<EnrichedCaseStudyItem | null>(null);
  const [loadedCaseId, setLoadedCaseId] = useState<string | null>(null);

  // Filter cases
  const filteredCases = SAMPLE_CASE_STUDIES.filter(item => {
    if (selectedPathology !== 'All' && item.groundTruthLabel !== selectedPathology) return false;
    if (selectedSeverity !== 'All' && item.severity !== selectedSeverity) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchId = item.id.toLowerCase().includes(q);
      const matchPatient = item.patientId.toLowerCase().includes(q);
      const matchSubtype = item.pathologySubtype ? item.pathologySubtype.toLowerCase().includes(q) : false;
      const matchHistory = item.clinicalHistory.toLowerCase().includes(q);
      if (!matchId && !matchPatient && !matchSubtype && !matchHistory) return false;
    }
    return true;
  });

  const handleLoadCase = (item: EnrichedCaseStudyItem) => {
    setLoadedCaseId(item.id);
    if (onSelectCase) {
      onSelectCase(item);
    }
    setTimeout(() => setLoadedCaseId(null), 3000);
  };

  return (
    <div className="space-y-6 font-mono text-xs">
      {/* Header & Clinical Context */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1e293b] pb-4">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded bg-cyan-950 border border-cyan-800/50 flex items-center justify-center text-cyan-400">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
              Curated Clinical Radiograph Cohort Case Studies
            </h2>
            <p className="text-[11px] text-slate-400 font-sans">
              Real-world degradation challenges: quantum noise, motion blur, pediatric underexposure & telemedicine DCT compression
            </p>
          </div>
        </div>

        <span className="text-xs text-slate-400 font-mono bg-[#131d31] px-3 py-1.5 rounded border border-[#1e293b]">
          Showing {filteredCases.length} of {SAMPLE_CASE_STUDIES.length} Curated Cases
        </span>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-[#0f172a] p-4 rounded-xl border border-[#1e293b] grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
        {/* Search */}
        <div className="md:col-span-5 relative">
          <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by Case ID, Patient, Subtype, or History..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#090d16] border border-[#1e293b] rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
          />
        </div>

        {/* Pathology Filter */}
        <div className="md:col-span-4 flex items-center space-x-1 bg-[#090d16] p-1 rounded-lg border border-[#1e293b]">
          <span className="text-[10px] text-slate-500 px-2 uppercase">Pathology:</span>
          {(['All', 'Pneumonia', 'Normal'] as const).map(p => (
            <button
              key={p}
              onClick={() => setSelectedPathology(p)}
              className={`px-2.5 py-0.5 rounded text-[11px] transition-colors ${
                selectedPathology === p
                  ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Severity Filter */}
        <div className="md:col-span-3 flex items-center space-x-1 bg-[#090d16] p-1 rounded-lg border border-[#1e293b]">
          <span className="text-[10px] text-slate-500 px-1.5 uppercase">Severity:</span>
          {(['All', 'Mild', 'Moderate', 'Severe', 'None'] as const).map(s => (
            <button
              key={s}
              onClick={() => setSelectedSeverity(s)}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                selectedSeverity === s
                  ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Case Study Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {filteredCases.map(item => {
          const isLoaded = loadedCaseId === item.id;

          return (
            <div
              key={item.id}
              className="bg-[#0f172a] border border-[#1e293b] hover:border-cyan-500/40 rounded-xl p-5 transition-all flex flex-col justify-between space-y-4 shadow-sm"
            >
              <div className="space-y-3.5">
                {/* Top Demographics Bar */}
                <div className="flex items-center justify-between border-b border-[#1e293b] pb-3">
                  <div className="flex items-center space-x-2.5">
                    <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 font-bold border border-cyan-800">
                      {item.id}
                    </span>
                    <span className="text-slate-200 font-bold">{item.patientId}</span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-400">{item.age}y {item.gender}</span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-400">{item.projection} View</span>
                  </div>

                  <span
                    className={`px-2.5 py-0.5 rounded text-[11px] font-bold ${
                      item.groundTruthLabel === 'Pneumonia'
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}
                  >
                    {item.groundTruthLabel} ({item.severity})
                  </span>
                </div>

                {/* Pathology & History */}
                <div>
                  <h3 className="font-bold text-slate-100 text-sm leading-snug">
                    {item.pathologySubtype}
                  </h3>
                  <p className="text-slate-400 text-xs mt-1.5 leading-relaxed font-sans">
                    {item.clinicalHistory}
                  </p>
                </div>

                {/* Radiologist Narrative Impression */}
                <div className="bg-[#131d31] p-3 rounded-lg border border-[#1e293b] space-y-1.5 font-sans">
                  <div className="flex items-center text-[11px] font-mono text-cyan-400 font-bold">
                    <ShieldCheck className="w-3.5 h-3.5 mr-1.5" />
                    <span>Radiologist Impression:</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-normal italic">
                    "{item.radiologistImpression}"
                  </p>
                </div>

                {/* Degradation Vector Breakdown (4 Bars) */}
                <div className="bg-[#090d16] p-3 rounded-lg border border-[#1e293b] space-y-2">
                  <div className="flex justify-between items-center text-[10px] text-slate-400 uppercase font-bold">
                    <span>ResNet-18 Detected Exposure Impairment</span>
                    <span className="text-cyan-400">R^4 Vector</span>
                  </div>

                  <div className="grid grid-cols-4 gap-2 text-center text-[10px]">
                    <div className="bg-[#131d31] p-1.5 rounded border border-[#1e293b]">
                      <span className="text-slate-500 block text-[9px]">NOISE</span>
                      <span className="font-bold text-slate-200">{(item.exposureVector.noise * 100).toFixed(0)}%</span>
                      <div className="w-full bg-[#1e293b] h-1 rounded mt-1 overflow-hidden">
                        <div className="bg-cyan-500 h-full" style={{ width: `${item.exposureVector.noise * 100}%` }} />
                      </div>
                    </div>

                    <div className="bg-[#131d31] p-1.5 rounded border border-[#1e293b]">
                      <span className="text-slate-500 block text-[9px]">BLUR</span>
                      <span className="font-bold text-slate-200">{(item.exposureVector.blur * 100).toFixed(0)}%</span>
                      <div className="w-full bg-[#1e293b] h-1 rounded mt-1 overflow-hidden">
                        <div className="bg-cyan-500 h-full" style={{ width: `${item.exposureVector.blur * 100}%` }} />
                      </div>
                    </div>

                    <div className="bg-[#131d31] p-1.5 rounded border border-[#1e293b]">
                      <span className="text-slate-500 block text-[9px]">CONTRAST</span>
                      <span className="font-bold text-slate-200">{(item.exposureVector.contrast * 100).toFixed(0)}%</span>
                      <div className="w-full bg-[#1e293b] h-1 rounded mt-1 overflow-hidden">
                        <div className="bg-cyan-500 h-full" style={{ width: `${item.exposureVector.contrast * 100}%` }} />
                      </div>
                    </div>

                    <div className="bg-[#131d31] p-1.5 rounded border border-[#1e293b]">
                      <span className="text-slate-500 block text-[9px]">JPEG</span>
                      <span className="font-bold text-slate-200">{(item.exposureVector.compression * 100).toFixed(0)}%</span>
                      <div className="w-full bg-[#1e293b] h-1 rounded mt-1 overflow-hidden">
                        <div className="bg-cyan-500 h-full" style={{ width: `${item.exposureVector.compression * 100}%` }} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* AI Radiologist Diagnostic Predictions Readout */}
                <div className="bg-[#090d16] p-3 rounded-lg border border-[#1e293b] space-y-2">
                  <div className="flex justify-between items-center text-[10px] text-slate-400 uppercase font-bold">
                    <span>DenseNet-121 Diagnostic Probability Trajectory</span>
                    <span className="text-emerald-400 font-mono">Consensus: {item.predictions.diagnosticConsensus}</span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                    <div className="bg-[#131d31] p-1.5 rounded border border-[#1e293b]">
                      <span className="text-slate-500 block text-[9px]">DEGRADED</span>
                      <span className="font-bold text-red-400">
                        {(item.predictions.densenet121.degraded * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="bg-[#131d31] p-1.5 rounded border border-[#1e293b]">
                      <span className="text-slate-500 block text-[9px]">GENERIC FILTER</span>
                      <span className="font-bold text-amber-400">
                        {(item.predictions.densenet121.generic * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="bg-emerald-950/40 p-1.5 rounded border border-emerald-700/50">
                      <span className="text-emerald-400 block text-[9px] font-bold">FiLM RESTORED</span>
                      <span className="font-bold text-emerald-300">
                        {(item.predictions.densenet121.adaptive * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[#1e293b]">
                <button
                  onClick={() => setActiveReportCase(item)}
                  className="py-2 px-3 rounded bg-[#131d31] hover:bg-[#1a253e] text-slate-300 border border-[#1e293b] hover:border-slate-600 transition-colors flex items-center justify-center space-x-1.5"
                >
                  <FileCheck2 className="w-3.5 h-3.5 text-cyan-400" />
                  <span>View Clinical Report</span>
                </button>

                <button
                  onClick={() => handleLoadCase(item)}
                  className={`py-2 px-3 rounded border transition-all flex items-center justify-center space-x-2 font-semibold ${
                    isLoaded
                      ? 'bg-emerald-600 text-white border-emerald-500 shadow-md'
                      : 'bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border-cyan-500/40 hover:border-cyan-400'
                  }`}
                >
                  {isLoaded ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-white" />
                      <span>Loaded into Workstation!</span>
                    </>
                  ) : (
                    <>
                      <span>Load into Workstation</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Case Study Specific Clinical Report Modal */}
      <ClinicalReportModal
        isOpen={activeReportCase !== null}
        onClose={() => setActiveReportCase(null)}
        caseItem={activeReportCase}
      />
    </div>
  );
};
