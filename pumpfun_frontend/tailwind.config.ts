import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0c0f14",
        fog: "#f4f3ee",
        neon: "#54f2a5",
        lava: "#ff7b5c",
        aurora: "#3ad5ff",
        slate: "#5b6675",
      },
      boxShadow: {
        glow: "0 0 40px rgba(84, 242, 165, 0.25)",
        card: "0 24px 60px -40px rgba(12, 15, 20, 0.55)",
      },
      borderRadius: {
        xl: "20px",
        '2xl': "28px",
        '3xl': "36px",
      },
    },
  },
  plugins: [],
} satisfies Config;
