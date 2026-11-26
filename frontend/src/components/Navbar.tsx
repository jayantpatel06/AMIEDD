import React from 'react';
import { Activity, Layers, Sliders, CheckCircle2, AlertCircle } from 'lucide-react';
import type { HealthResponse } from '../types';

interface NavbarProps {
  activeTab: 'overview' | 'studio' | 'architecture';
  setActiveTab: (tab: 'overview' | 'studio' | 'architecture') => void;
  health: HealthResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  return (
    <header className="sticky top-0 z-50 bg-[#FAF8F5]/90 backdrop-blur-md border-b border-[#EBE5D8] px-4 sm:px-8 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        {/* Brand identity */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-[#E6ECE5] border border-[#C9D7C8] flex items-center justify-center text-[#38563E] shadow-sm">
            <Activity className="w-5 h-5 stroke-[2.2]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-lg tracking-tight text-[#1F2421]">AMIEDD</span>
              <span className="px-2 py-0.5 text-[11px] font-medium tracking-wide uppercase bg-[#EBE5D8] text-[#5F6862] rounded-full">
                Clinical AI
              </span>
            </div>
            <p className="text-xs text-[#5F6862]">
              Adaptive Degradation-Aware Chest X-Ray Enhancement
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center bg-[#F5F2EB] p-1.5 rounded-2xl border border-[#EBE5D8] self-start md:self-auto shadow-inner">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
              activeTab === 'overview'
                ? 'bg-white text-[#1F2421] shadow-sm font-semibold'
                : 'text-[#5F6862] hover:text-[#1F2421] hover:bg-white/50'
            }`}
          >
            <Layers className="w-4 h-4" />
            Overview
          </button>

          <button
            onClick={() => setActiveTab('studio')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
              activeTab === 'studio'
                ? 'bg-white text-[#1F2421] shadow-sm font-semibold'
                : 'text-[#5F6862] hover:text-[#1F2421] hover:bg-white/50'
            }`}
          >
            <Sliders className="w-4 h-4" />
            Testing Studio
          </button>

          <button
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
              activeTab === 'architecture'
                ? 'bg-white text-[#1F2421] shadow-sm font-semibold'
                : 'text-[#5F6862] hover:text-[#1F2421] hover:bg-white/50'
            }`}
          >
            <Activity className="w-4 h-4" />
            Architecture
          </button>
        </nav>

        {/* Backend health status badge */}
        <div className="hidden lg:flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-xl border bg-white/70 border-[#EBE5D8]">
          {health?.models_ready ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-[#4F7757]" />
              <span className="text-[#38563E]">
                Models Online ({health.active_classifiers.length}-Radiologist Consensus)
              </span>
            </>
          ) : (
            <>
              <AlertCircle className="w-4 h-4 text-[#C8644E]" />
              <span className="text-[#944432]">
                {health ? 'Models Initializing' : 'Backend Connecting...'}
              </span>
            </>
          )}
        </div>
      </div>
    </header>
  );
};
