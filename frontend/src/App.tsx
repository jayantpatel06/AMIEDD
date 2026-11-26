import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewPage } from './components/OverviewPage';
import { StudioPage } from './components/StudioPage';
import { ArchitecturePage } from './components/ArchitecturePage';
import type { HealthResponse } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'studio' | 'architecture'>('overview');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // Poll backend health once on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data: HealthResponse = await res.json();
          setHealth(data);
        }
      } catch {
        // Backend not yet reachable; badge indicates connecting
      }
    };
    checkHealth();
    const timer = setInterval(checkHealth, 8000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#2D3330] flex flex-col font-sans selection:bg-[#DDD4C0]">
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-8 py-6">
        {activeTab === 'overview' && (
          <OverviewPage onGoToStudio={() => setActiveTab('studio')} />
        )}
        {activeTab === 'studio' && <StudioPage />}
        {activeTab === 'architecture' && <ArchitecturePage />}
      </main>

      {/* Simple Footer */}
      <footer className="border-t border-[#EBE5D8] py-6 px-4 text-center text-xs text-[#5F6862] bg-[#F5F2EB]/50">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>
            AMIEDD &bull; Adaptive Degradation-Aware Chest X-Ray Enhancement & Diagnostic Panel
          </p>
          <p className="text-[11px] text-[#A89775]">
            DenseNet-121 + EfficientNet-B0 + ResNet-50 Consensus Panel
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
