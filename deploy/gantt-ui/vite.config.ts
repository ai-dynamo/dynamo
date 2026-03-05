import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "serve-data",
      configureServer(server) {
        server.middlewares.use("/data/responses.json", (_req, res) => {
          const filePath = path.resolve(__dirname, "../../responses.json");
          if (fs.existsSync(filePath)) {
            res.setHeader("Content-Type", "application/json");
            fs.createReadStream(filePath).pipe(res);
          } else {
            res.statusCode = 404;
            res.end("Not found");
          }
        });
      },
    },
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3001,
    proxy: {
      "/api": {
        target: "http://localhost:3200",
        changeOrigin: true,
      },
    },
  },
});
