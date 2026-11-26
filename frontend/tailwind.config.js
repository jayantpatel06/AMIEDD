/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cream: {
          50: '#FAF8F5',
          100: '#F5F2EB',
          200: '#EBE5D8',
          300: '#DDD4C0',
          400: '#C7B99D',
          500: '#A89775',
          900: '#3D3627',
        },
        sage: {
          50: '#F4F7F4',
          100: '#E6ECE5',
          200: '#C9D7C8',
          500: '#4F7757',
          700: '#38563E',
        },
        clay: {
          50: '#FAF4F2',
          100: '#F4E7E3',
          200: '#E8CFC7',
          500: '#C8644E',
          700: '#944432',
        },
        charcoal: {
          500: '#5F6862',
          700: '#373D39',
          900: '#1F2421',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
      },
      boxShadow: {
        'soft': '0 4px 20px -2px rgba(60, 50, 40, 0.05), 0 2px 6px -1px rgba(60, 50, 40, 0.03)',
        'soft-hover': '0 10px 25px -4px rgba(60, 50, 40, 0.08), 0 4px 10px -2px rgba(60, 50, 40, 0.04)',
      }
    },
  },
  plugins: [],
}
