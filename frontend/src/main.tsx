import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";

// Recharts + StrictMode emits a benign width(-1)/height(-1) warning on the
// synchronous first render before its ResizeObserver fires. Charts render
// correctly once layout settles; suppress the noise.
if (import.meta.env.DEV) {
	const originalWarn = console.warn;
	console.warn = (...args: unknown[]) => {
		const first = args[0];
		if (
			typeof first === "string" &&
			first.includes("width(-1) and height(-1)")
		) {
			return;
		}
		originalWarn(...args);
	};
}

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<App />
	</StrictMode>,
);
