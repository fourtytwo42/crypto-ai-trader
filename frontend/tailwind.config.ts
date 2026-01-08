import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "var(--color-ink)",
        slate: "var(--color-slate)",
        mint: "var(--color-mint)",
        sun: "var(--color-sun)",
        sand: "var(--color-sand)",
        "surface-1": "var(--surface-1)",
        "surface-2": "var(--surface-2)",
        "surface-3": "var(--surface-3)",
        "glass-1": "var(--glass-1)",
        "glass-2": "var(--glass-2)"
      },
      boxShadow: {
        glow: "0 16px 40px -20px rgba(11, 20, 26, 0.4)",
        halo: "0 0 0 1px rgba(12, 17, 20, 0.08), 0 18px 50px -28px rgba(12, 17, 20, 0.45)",
        card: "0 30px 80px -60px rgba(12, 17, 20, 0.4)"
      },
      borderRadius: {
        xl: "1.25rem",
        "2xl": "1.75rem",
        "3xl": "2.25rem"
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" }
        },
        fade: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "0% 50%" },
          "100%": { backgroundPosition: "200% 50%" }
        }
      },
      animation: {
        float: "float 6s ease-in-out infinite",
        fade: "fade 0.6s ease-out both",
        shimmer: "shimmer 3.5s ease-in-out infinite"
      }
    }
  },
  plugins: []
};

export default config;
