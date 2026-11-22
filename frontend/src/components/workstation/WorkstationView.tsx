import React, { useState, useEffect } from 'react';
import {
  Sliders,
  RotateCcw,
  User
} from 'lucide-react';
import { ComparisonViewer } from './ComparisonViewer';
import { TelemetryDeck } from './TelemetryDeck';
import { WindowLevelControls } from './WindowLevelControls';
import type { DegradationVector, WindowLevelPreset, CaseStudyItem } from '../../types';
import { DEFAULT_DEGRADATION_VECTOR, SAMPLE_CASE_STUDIES } from '../../data/benchmarkData';

interface WorkstationViewProps {
  activeCase?: CaseStudyItem | null;
  onSelectCase?: (caseItem: CaseStudyItem) => void;
}

export const WorkstationView: React.FC<WorkstationViewProps> = ({
  activeCase,
  onSelectCase,
}) => {
  const [vector, setVector] = useState<DegradationVector>(DEFAULT_DEGRADATION_VECTOR);
  const [activePresetId, setActivePresetId] = useState<string>('lung');
  const [splitPosition, setSplitPosition] = useState<number>(50);
  const [activeWindow, setActiveWindow] = useState<number>(1500);
  const [activeLevel, setActiveLevel] = useState<number>(-600);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [groundTruthLabel, setGroundTruthLabel] = useState<'Normal' | 'Pneumonia'>('Pneumonia');
  const [selectedCaseId, setSelectedCaseId] = useState<string>('CS-001');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('pneumonia_1');
  const [inverted, setInverted] = useState<boolean>(false);

  const CLINICAL_SCENARIOS = [
    { id: 'default', name: 'Benchmark Mean', vector: { noise: 0.45, blur: 0.35, contrast: 0.40, compression: 0.50 } },
    { id: 'quantum_noise', name: 'Quantum Noise', vector: { noise: 0.85, blur: 0.15, contrast: 0.25, compression: 0.15 } },
    { id: 'motion_blur', name: 'Motion Blur', vector: { noise: 0.20, blur: 0.85, contrast: 0.30, compression: 0.20 } },
    { id: 'underexposure', name: 'Contrast Deficit', vector: { noise: 0.30, blur: 0.15, contrast: 0.85, compression: 0.15 } },
    { id: 'telehealth', name: 'Telehealth DCT', vector: { noise: 0.25, blur: 0.20, contrast: 0.20, compression: 0.90 } },
    { id: 'pristine', name: 'Pristine Reference', vector: { noise: 0.00, blur: 0.00, contrast: 0.00, compression: 0.00 } },
  ];

  useEffect(() => {
    if (activeCase && activeCase.id !== selectedCaseId) {
      applyCaseSelection(activeCase.id);
    }
  }, [activeCase]);

  const applyCaseSelection = (caseId: string) => {
    setSelectedCaseId(caseId);
    const cs = SAMPLE_CASE_STUDIES.find(c => c.id === caseId);
    if (cs) {
      setGroundTruthLabel(cs.groundTruthLabel);
      setVector(cs.exposureVector);
      if (cs.recommendedWindowLevel) {
        setActiveWindow(cs.recommendedWindowLevel.window);
        setActiveLevel(cs.recommendedWindowLevel.level);
      }
      if (cs.groundTruthLabel === 'Pneumonia') {
        setSelectedSampleId(caseId === 'CS-004' ? 'pneumonia_2' : 'pneumonia_1');
      } else {
        setSelectedSampleId(caseId === 'CS-003' ? 'normal_2' : 'normal_1');
      }
      if (onSelectCase) {
        onSelectCase(cs);
      }
    }
  };

  const handleSliderChange = (key: keyof DegradationVector, val: number) => {
    setVector(prev => ({ ...prev, [key]: val }));
  };

  const handlePresetSelect = (preset: WindowLevelPreset) => {
    setActivePresetId(preset.id);
    setActiveWindow(preset.window);
    setActiveLevel(preset.level);
  };

  const resetVector = () => {
    setVector(DEFAULT_DEGRADATION_VECTOR);
  };

  const currentCase = SAMPLE_CASE_STUDIES.find(c => c.id === selectedCaseId) || SAMPLE_CASE_STUDIES[0];
  const compositeSeverity = ((vector.noise + vector.blur + vector.contrast + vector.compression) / 4).toFixed(3);

  return (
    <div className="p-3 lg:p-4 bg-black text-neutral-200 min-h-[calc(100vh-4rem)] font-mono">
      {/* 3-Column Workstation Grid: Direct, Clean, No Duplicate Header */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 items-start">
        
        {/* ========================================================================= */}
        {/* LEFT COLUMN: Controls & Study Context (3 cols on xl, 4 cols on lg)        */}
        {/* ========================================================================= */}
        <div className="lg:col-span-5 xl:col-span-3 flex flex-col space-y-3">
          
          {/* Case & Simulation Profile Selector */}
          <div className="bg-[#0e0e0e] border border-[#222222] rounded p-3 space-y-2.5">
            <div className="flex justify-between items-center text-xs">
              <span className="text-white font-semibold flex items-center">
                <User className="w-3.5 h-3.5 mr-1 text-white" />
                Case Study
              </span>
              <div className="flex space-x-1">
                <button
                  onClick={() => setGroundTruthLabel('Normal')}
                  className={`px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
                    groundTruthLabel === 'Normal'
                      ? 'bg-white border-white text-black font-bold'
                      : 'border-[#262626] text-neutral-400 hover:text-white'
                  }`}
                >
                  Normal
                </button>
                <button
                  onClick={() => setGroundTruthLabel('Pneumonia')}
                  className={`px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
                    groundTruthLabel === 'Pneumonia'
                      ? 'bg-white border-white text-black font-bold'
                      : 'border-[#262626] text-neutral-400 hover:text-white'
                  }`}
                >
                  Pneumonia
                </button>
              </div>
            </div>

            <select
              value={selectedCaseId}
              onChange={e => applyCaseSelection(e.target.value)}
              aria-label="Select Clinical Case Study"
              className="w-full bg-[#141414] text-white border border-[#262626] rounded px-2 py-1 text-[11px] focus:outline-none focus:border-white"
            >
              {SAMPLE_CASE_STUDIES.map(cs => (
                <option key={cs.id} value={cs.id}>
                  {cs.id}: {cs.pathologySubtype || cs.groundTruthLabel} [{cs.age}y {cs.gender}]
                </option>
              ))}
            </select>

            {/* Quick Profile Dropdown */}
            <div className="flex items-center justify-between text-[10px] pt-1 border-t border-[#222222]">
              <span className="text-neutral-400">Simulation:</span>
              <select
                onChange={e => {
                  const sc = CLINICAL_SCENARIOS.find(s => s.id === e.target.value);
                  if (sc) setVector(sc.vector);
                }}
                aria-label="Simulation Scenario Preset"
                className="bg-[#141414] text-neutral-200 border border-[#262626] rounded px-1.5 py-0.5 text-[10px] focus:outline-none focus:border-white"
              >
                {CLINICAL_SCENARIOS.map(sc => (
                  <option key={sc.id} value={sc.id}>{sc.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Degradation Vector Synthesis */}
          <div className="bg-[#0e0e0e] border border-[#222222] rounded p-3 space-y-2.5">
            <div className="flex items-center justify-between border-b border-[#222222] pb-1.5">
              <span className="text-xs font-semibold uppercase tracking-wider text-white flex items-center">
                <Sliders className="w-3.5 h-3.5 mr-1 text-white" /> Degradation
              </span>
              <div className="flex items-center space-x-2 text-[10px]">
                <span className="text-neutral-400">Mean: <strong className="text-white">{compositeSeverity}</strong></span>
                <button
                  onClick={resetVector}
                  title="Reset Degradation"
                  aria-label="Reset Degradation"
                  className="p-0.5 rounded bg-[#181818] border border-[#2c2c2c] text-neutral-400 hover:text-white"
                >
                  <RotateCcw className="w-2.5 h-2.5" />
                </button>
              </div>
            </div>

            <div className="space-y-2 text-[11px]">
              {[
                { key: 'noise', label: '1. Quantum Noise', val: vector.noise },
                { key: 'blur', label: '2. Motion Blur', val: vector.blur },
                { key: 'contrast', label: '3. Contrast Deficit', val: vector.contrast },
                { key: 'compression', label: '4. DCT Compression', val: vector.compression },
              ].map(item => (
                <div key={item.key} className="space-y-0.5">
                  <div className="flex justify-between items-center text-[10px]">
                    <span className="text-neutral-400">{item.label}</span>
                    <span className="text-white font-bold">{(item.val * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    aria-label={item.label}
                    value={item.val}
                    onChange={e => handleSliderChange(item.key as keyof DegradationVector, parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Window / Level Controls */}
          <WindowLevelControls
            windowVal={activeWindow}
            levelVal={activeLevel}
            onWindowChange={setActiveWindow}
            onLevelChange={setActiveLevel}
            activePresetId={activePresetId}
            onPresetSelect={handlePresetSelect}
            inverted={inverted}
            onInvertToggle={() => setInverted(!inverted)}
          />

          {/* Clinical Record Snippet */}
          <div className="bg-[#0e0e0e] border border-[#222222] rounded p-2.5 text-[10px] space-y-1 text-neutral-400">
            <div className="flex justify-between text-neutral-300">
              <span className="font-semibold">{currentCase.patientId}</span>
              <span>{currentCase.studyDate}</span>
            </div>
            <div className="leading-relaxed line-clamp-2">
              {currentCase.clinicalHistory}
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* CENTER COLUMN: Primary Viewport (5 cols on xl, 7 cols on lg)              */}
        {/* ========================================================================= */}
        <div className="lg:col-span-7 xl:col-span-5 flex flex-col space-y-2 sticky top-14">
          <ComparisonViewer
            vector={vector}
            splitPosition={splitPosition}
            onSplitChange={setSplitPosition}
            windowVal={activeWindow}
            levelVal={activeLevel}
            showHeatmap={showHeatmap}
            onHeatmapToggle={setShowHeatmap}
            groundTruthLabel={groundTruthLabel}
            selectedSampleId={selectedSampleId}
            onSampleChange={setSelectedSampleId}
            inverted={inverted}
          />
        </div>

        {/* ========================================================================= */}
        {/* RIGHT COLUMN: 3-Stage Telemetry Deck (4 cols on xl, 12 cols on lg)        */}
        {/* ========================================================================= */}
        <div className="lg:col-span-12 xl:col-span-4 flex flex-col space-y-3">
          <TelemetryDeck vector={vector} groundTruthLabel={groundTruthLabel} />
        </div>
      </div>
    </div>
  );
};
