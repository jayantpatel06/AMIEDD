import React, { useState, useRef, useCallback } from 'react';

interface ImageComparisonSliderProps {
  originalImage: string;
  cleanedImage: string;
  originalLabel?: string;
  cleanedLabel?: string;
}

export const ImageComparisonSlider: React.FC<ImageComparisonSliderProps> = ({
  originalImage,
  cleanedImage,
  originalLabel = 'Degraded Input',
  cleanedLabel = 'FiLM-Enhanced Restored',
}) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const percentage = (x / rect.width) * 100;
      setSliderPosition(percentage);
    },
    []
  );

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  };

  return (
    <div className="flex flex-col items-center w-full select-none">
      <div
        ref={containerRef}
        onMouseDown={() => setIsDragging(true)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onMouseMove={handleMouseMove}
        onTouchStart={() => setIsDragging(true)}
        onTouchEnd={() => setIsDragging(false)}
        onTouchMove={handleTouchMove}
        className="relative w-full max-w-[480px] aspect-square rounded-3xl overflow-hidden shadow-md border-2 border-[#EBE5D8] cursor-ew-resize bg-[#2D3330]"
      >
        {/* Cleaned Restored Image (Underneath) */}
        <img
          src={cleanedImage}
          alt={cleanedLabel}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Degraded Image (Clipped on top) */}
        <div
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{ width: `${sliderPosition}%` }}
        >
          <img
            src={originalImage}
            alt={originalLabel}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ width: containerRef.current?.offsetWidth || '100%', maxWidth: 'none' }}
          />
        </div>

        {/* Split line divider */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white/90 shadow-[0_0_10px_rgba(0,0,0,0.5)] pointer-events-none"
          style={{ left: `${sliderPosition}%` }}
        >
          {/* Circular drag handle */}
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white text-[#2D3330] shadow-lg flex items-center justify-center border border-[#DDD4C0]">
            <span className="text-[11px] font-bold">⇄</span>
          </div>
        </div>

        {/* Corner Badges */}
        <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-xl bg-black/60 backdrop-blur-sm text-white text-[11px] font-medium pointer-events-none">
          {originalLabel}
        </div>
        <div className="absolute bottom-3 right-3 px-2.5 py-1 rounded-xl bg-[#4F7757]/90 backdrop-blur-sm text-white text-[11px] font-medium pointer-events-none">
          {cleanedLabel}
        </div>
      </div>

      {/* Helper caption */}
      <p className="mt-2 text-xs text-[#5F6862] font-medium">
        Drag slider left and right to compare fine alveolar markings & rib edge recovery
      </p>
    </div>
  );
};
