/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        rad: {
          darkest: '#05080f',
          dark: '#090d16',
          panel: '#0f172a',
          card: '#131d31',
          border: '#1e293b',
          borderLight: '#334155',
          muted: '#64748b',
          text: '#94a3b8',
          bright: '#f1f5f9',
          cyan: '#06b6d4',
          cyanLight: '#22d3ee',
          emerald: '#10b981',
          emeraldLight: '#34d399',
          amber: '#f59e0b',
          amberLight: '#fbbf24',
          crimson: '#ef4444',
          blue: '#3b82f6',
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Geist Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(to right, rgba(30, 41, 59, 0.3) 1px, transparent 1px), linear-gradient(to bottom, rgba(30, 41, 59, 0.3) 1px, transparent 1px)",
      },
      backgroundSize: {
        'grid-sm': '16px 16px',
        'grid-md': '24px 24px',
      }
    },
  },
  plugins: [],
}
