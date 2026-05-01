import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
	plugins: [react()],
	server: {
		port: 3000,
		proxy: {
			"/api": {
				target: "http://localhost:8000",
				changeOrigin: true,
			},
			// WebSocket connects directly to ws://localhost:8000 in dev mode,
			// bypassing this proxy entirely. See Dashboard.tsx.
		},
	},
});
