# Frontend — Pipe Clogging Detection Platform

The React + TypeScript + Vite dashboard for the platform. It connects to the
FastAPI backend over a WebSocket (live mode) and REST (batch mode) and renders
the detection charts, traffic light, forecast, and model-management UI.

See the [root README](../README.md) for the full project overview, the detection
methods, and setup instructions.

## Develop

```bash
npm install
npm run dev      # http://localhost:3000 (proxies API + WebSocket to backend on :8000)
```

The backend must be running separately (`python -m app.main` from the project root).

## Build

```bash
npm run build    # type-check (tsc -b) + production bundle into dist/
npm run preview  # serve the production build locally
```
