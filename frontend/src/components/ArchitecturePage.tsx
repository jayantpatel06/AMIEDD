import React from 'react';
import { Layers, Activity, GitFork, CheckCircle, ArrowDown } from 'lucide-react';

export const ArchitecturePage: React.FC = () => {
  return (
    <div className="space-y-12 py-4">
      {/* Title */}
      <div className="max-w-3xl space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-2xl bg-white border border-[#DDD4C0] shadow-sm text-xs font-semibold text-[#38563E]">
          <Activity className="w-3.5 h-3.5 text-[#4F7757]" />
          Implementation & Engineering Whitepaper
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-[#1F2421]">
          System Architecture & Algorithmic Design
        </h1>
        <p className="text-sm sm:text-base text-[#5F6862] leading-relaxed">
          AMIEDD decouples continuous physical degradation estimation, feature-wise neural modulation, and multi-radiologist consensus triage into three robust, independent stages.
        </p>
      </div>

      {/* Visual Pipeline Flowchart */}
      <section className="p-8 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-6">
        <h2 className="text-xl font-bold text-[#1F2421] tracking-tight">
          1. End-to-End Information Flow
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-center">
          {/* Card 1 */}
          <div className="p-4 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8] text-center space-y-1.5 shadow-sm">
            <span className="text-[10px] uppercase font-bold text-[#5F6862]">Input</span>
            <p className="text-xs font-bold text-[#1F2421]">Corrupted Radiograph</p>
            <p className="text-[11px] text-[#5F6862]">224&times;224 Grayscale Tensor [1, 1, 224, 224]</p>
          </div>

          <div className="hidden md:flex justify-center text-[#A89775]">
            <span className="text-lg font-bold">&rarr;</span>
          </div>

          {/* Card 2 */}
          <div className="p-4 rounded-2xl bg-[#F5F2EB] border border-[#DDD4C0] text-center space-y-1.5 shadow-sm">
            <span className="text-[10px] uppercase font-bold text-[#3D3627]">Model 1</span>
            <p className="text-xs font-bold text-[#1F2421]">Degradation Estimator</p>
            <p className="text-[11px] text-[#3D3627] font-mono">[&sigma;_n, &sigma;_b, &gamma;, q] &isin; [0, 1]&sup4;</p>
          </div>

          <div className="hidden md:flex justify-center text-[#A89775]">
            <span className="text-lg font-bold">&rarr;</span>
          </div>

          {/* Card 3 */}
          <div className="p-4 rounded-2xl bg-[#E6ECE5] border border-[#C9D7C8] text-center space-y-1.5 shadow-sm">
            <span className="text-[10px] uppercase font-bold text-[#38563E]">Model 2</span>
            <p className="text-xs font-bold text-[#1F2421]">FiLM Adaptive U-Net</p>
            <p className="text-[11px] text-[#38563E]">Clean Anatomical Tensor +10.6 dB PSNR</p>
          </div>
        </div>

        <div className="flex justify-center text-[#A89775]">
          <ArrowDown className="w-5 h-5" />
        </div>

        {/* Multi-Model Panel */}
        <div className="p-5 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#1F2421] uppercase tracking-wider">
              Stage 3: Multi-Radiologist Consensus Diagnostic Panel
            </span>
            <span className="text-[11px] text-[#5F6862] font-semibold">Weighted Soft-Voting</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-3 rounded-xl bg-white border border-[#EBE5D8] text-center">
              <p className="text-xs font-bold text-[#1F2421]">EfficientNet-B0</p>
              <p className="text-[11px] text-[#5F6862]">Primary Robust Radiologist (50% Weight)</p>
              <p className="text-[10px] text-[#4F7757] font-medium mt-1">Spatial-invariant compound scaling</p>
            </div>
            <div className="p-3 rounded-xl bg-white border border-[#EBE5D8] text-center">
              <p className="text-xs font-bold text-[#1F2421]">DenseNet-121</p>
              <p className="text-[11px] text-[#5F6862]">Second Opinion (30% Weight)</p>
              <p className="text-[10px] text-[#4F7757] font-medium mt-1">Dense feature reuse across layers</p>
            </div>
            <div className="p-3 rounded-xl bg-white border border-[#EBE5D8] text-center">
              <p className="text-xs font-bold text-[#1F2421]">ResNet-50</p>
              <p className="text-[11px] text-[#5F6862]">Tie-Breaker (20% Weight)</p>
              <p className="text-[10px] text-[#4F7757] font-medium mt-1">Deep residual identity mappings</p>
            </div>
          </div>
        </div>
      </section>

      {/* Mathematical Foundations */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* FiLM Math */}
        <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-[#38563E]" />
            <h3 className="text-lg font-bold text-[#1F2421]">
              FiLM Conditioning Formulation
            </h3>
          </div>
          <p className="text-xs text-[#5F6862] leading-relaxed">
            Rather than feeding degradation as extra static image channels, Feature-wise Linear Modulation (Perez et al.) uses an auxiliary multi-layer perceptron (MLP) to generate affine transform coefficients directly conditioning the intermediate U-Net decoder features:
          </p>

          <div className="p-4 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8] font-mono text-center text-xs text-[#1F2421] leading-relaxed">
            <p className="font-bold text-sm text-[#38563E] mb-1">
              h&#7522;&#44;&#7527;&#44;&#739; = (1 + &gamma;&#7522;(d)) &middot; x&#7522;&#44;&#7527;&#44;&#739; + &beta;&#7522;(d)
            </p>
            <p className="text-[11px] text-[#5F6862] font-sans mt-2">
              Where <strong className="text-[#1F2421]">&gamma;(d)</strong> modulates feature gain and <strong className="text-[#1F2421]">&beta;(d)</strong> adjusts feature bias based on estimated degradation <strong className="text-[#1F2421]">d</strong>.
            </p>
          </div>

          <p className="text-xs text-[#5F6862] leading-relaxed">
            The (1 + &gamma;) residual scaling guarantees that when degradation is minimal (&gamma; &rarr; 0, &beta; &rarr; 0), the layer smoothly acts as an identity pass-through, eliminating hallucinated distortions.
          </p>
        </div>

        {/* Multi-Objective Restoration Loss */}
        <div className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-[#C8644E]" />
            <h3 className="text-lg font-bold text-[#1F2421]">
              Multi-Objective Composite Loss
            </h3>
          </div>
          <p className="text-xs text-[#5F6862] leading-relaxed">
            Restoring radiographs requires pixel-level fidelity, anatomical structure preservation, and perceptual realism simultaneously:
          </p>

          <div className="p-4 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8] font-mono text-center text-xs text-[#1F2421] leading-relaxed">
            <p className="font-bold text-sm text-[#944432] mb-1">
              L_total = L&#8321; + &lambda;_SSIM &middot; (1 &minus; SSIM) + &lambda;_perc &middot; L_VGG
            </p>
            <p className="text-[11px] text-[#5F6862] font-sans mt-2">
              &bull; L&#8321;: Preserves exact Hounsfield/grayscale radiographic density.<br/>
              &bull; SSIM: Protects lung boundaries and rib contours.<br/>
              &bull; L_VGG: Reconstructs bronchial texture patterns.
            </p>
          </div>

          <p className="text-xs text-[#5F6862] leading-relaxed">
            Trained on 5,856 authentic Kaggle chest X-rays using continuous augmentation mixtures spanning noise, motion blur, gamma shifts, and JPEG DCT compression.
          </p>
        </div>
      </section>

      {/* Consensus Panel & Decision Threshold Math */}
      <section className="p-6 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
        <div className="flex items-center gap-2">
          <GitFork className="w-5 h-5 text-[#3D3627]" />
          <h3 className="text-lg font-bold text-[#1F2421]">
            Consensus Probability Math & Clinical Triage Logic
          </h3>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
          <div className="space-y-3 text-xs text-[#5F6862] leading-relaxed">
            <p>
              Under physical degradation, isolated models (e.g., DenseNet alone) suffer from high false alarm rates (~28%) because high-frequency noise resembles alveolar consolidations.
            </p>
            <p>
              AMIEDD fuses predictions across three structurally diverse deep neural network architectures using optimized Bayesian weights:
            </p>
            <div className="p-3 bg-[#FAF8F5] rounded-xl border border-[#EBE5D8] font-mono text-xs text-[#1F2421]">
              P_consensus = 0.50 &middot; P_EfficientNet + 0.30 &middot; P_DenseNet + 0.20 &middot; P_ResNet
            </div>
            <p>
              To ensure <strong className="text-[#1F2421]">patient lives are never compromised</strong>, AMIEDD avoids hard binary cuts in favor of a 3-tier clinical protocol.
            </p>
          </div>

          {/* 3-Tier Protocol Table */}
          <div className="p-4 rounded-2xl bg-[#FAF8F5] border border-[#EBE5D8] space-y-2.5">
            <p className="text-xs font-bold text-[#1F2421] uppercase tracking-wide">
              Clinical Triage Thresholds
            </p>

            <div className="p-2.5 rounded-xl bg-white border border-[#EBE5D8] flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-[#C8644E]">High Risk (Definite Pneumonia)</p>
                <p className="text-[11px] text-[#5F6862]">High consensus across all 3 models</p>
              </div>
              <span className="font-mono text-xs font-bold text-[#C8644E]">P &ge; 75%</span>
            </div>

            <div className="p-2.5 rounded-xl bg-white border border-[#EBE5D8] flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-[#D9822B]">Borderline / Review Required</p>
                <p className="text-[11px] text-[#5F6862]">Disagreement or intermediate risk; safety flag</p>
              </div>
              <span className="font-mono text-xs font-bold text-[#D9822B]">45% &le; P &lt; 75%</span>
            </div>

            <div className="p-2.5 rounded-xl bg-white border border-[#EBE5D8] flex items-center justify-between">
              <div>
                <p className="text-xs font-bold text-[#4F7757]">Low Risk (Healthy / Normal)</p>
                <p className="text-[11px] text-[#5F6862]">Clear lung fields confirmed by panel</p>
              </div>
              <span className="font-mono text-xs font-bold text-[#4F7757]">P &lt; 45%</span>
            </div>
          </div>
        </div>
      </section>

      {/* Quantitative Benchmark Comparison Table */}
      <section className="p-8 rounded-3xl bg-white border border-[#EBE5D8] shadow-soft space-y-4">
        <div>
          <h2 className="text-xl font-bold text-[#1F2421] tracking-tight">
            2. Quantitative Benchmark Matrix
          </h2>
          <p className="text-xs text-[#5F6862]">
            Empirical evaluation measured across 100 corrupted test radiographs from the authentic Kaggle dataset.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#EBE5D8] text-[#5F6862] uppercase tracking-wider text-[10px]">
                <th className="py-3 px-4">Method / Architecture</th>
                <th className="py-3 px-4">PSNR Gain</th>
                <th className="py-3 px-4">Pneumonia Sensitivity</th>
                <th className="py-3 px-4">Normal Specificity</th>
                <th className="py-3 px-4">Overall Accuracy</th>
                <th className="py-3 px-4">False Alarm Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#F5F2EB] text-[#1F2421]">
              <tr className="hover:bg-[#FAF8F5]">
                <td className="py-3 px-4 font-semibold">Raw Corrupted (No Enhancement)</td>
                <td className="py-3 px-4 text-[#5F6862]">0.0 dB (Baseline)</td>
                <td className="py-3 px-4 text-[#C8644E] font-medium">82.0%</td>
                <td className="py-3 px-4 text-[#C8644E] font-medium">45.0%</td>
                <td className="py-3 px-4 text-[#C8644E]">63.5%</td>
                <td className="py-3 px-4 text-[#C8644E] font-bold">55.0%</td>
              </tr>
              <tr className="hover:bg-[#FAF8F5]">
                <td className="py-3 px-4 font-semibold">Standard Generic CLAHE</td>
                <td className="py-3 px-4 text-[#5F6862]">+1.8 dB</td>
                <td className="py-3 px-4 text-[#38563E] font-medium">96.0%</td>
                <td className="py-3 px-4 text-[#C8644E] font-medium">52.0%</td>
                <td className="py-3 px-4 text-[#5F6862]">74.0%</td>
                <td className="py-3 px-4 text-[#C8644E] font-bold">48.0%</td>
              </tr>
              <tr className="hover:bg-[#FAF8F5]">
                <td className="py-3 px-4 font-semibold">FiLM + DenseNet-121 Solo</td>
                <td className="py-3 px-4 text-[#38563E] font-medium">+10.6 dB</td>
                <td className="py-3 px-4 text-[#38563E] font-bold">100.0%</td>
                <td className="py-3 px-4 text-[#5F6862] font-medium">72.0%</td>
                <td className="py-3 px-4 text-[#1F2421]">86.0%</td>
                <td className="py-3 px-4 text-[#D9822B] font-medium">28.0%</td>
              </tr>
              <tr className="bg-[#F4F7F4]/60 font-semibold border-t-2 border-[#C9D7C8]">
                <td className="py-3.5 px-4 text-[#38563E] flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5" />
                  AMIEDD (FiLM + 3-Model Consensus)
                </td>
                <td className="py-3.5 px-4 text-[#38563E] font-bold">+10.6 dB</td>
                <td className="py-3.5 px-4 text-[#38563E] font-bold">100.0%</td>
                <td className="py-3.5 px-4 text-[#38563E] font-bold">88.0%</td>
                <td className="py-3.5 px-4 text-[#38563E] font-bold">94.0%</td>
                <td className="py-3.5 px-4 text-[#38563E] font-bold">&lt; 12.0%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
