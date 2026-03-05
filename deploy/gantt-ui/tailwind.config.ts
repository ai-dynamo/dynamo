import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0d1117",
        surface: "#161b22",
        "surface-alt": "#1c2128",
        border: "#30363d",
        muted: "#7d8590",
        "text-primary": "#e6edf3",
        "text-secondary": "#7d8590",
        accent: "#58a6ff",
        "accent-muted": "#388bfd26",
      },
    },
  },
  plugins: [],
};

export default config;
