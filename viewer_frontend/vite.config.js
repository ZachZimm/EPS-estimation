import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4715,
    host: "0.0.0.0",
    allowedHosts: ["lab"],
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.VITE_BACKEND_PORT || "8100"}`,
        changeOrigin: true,
      },
    },
  },
});
