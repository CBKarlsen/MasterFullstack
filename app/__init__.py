"""
Pipe Clogging Detection Platform — backend package.

This package contains the FastAPI server and all signal-processing/ML logic
for detecting partial clogging in pipelines from streamed sensor data.

Module overview:
    main.py        FastAPI app: WebSocket streaming + REST API for models/data.
    engine.py      Real-time simulation loop that replays a file over a WebSocket.
    backend.py     CloggingDetector — calibration + the four detection methods.
    batch.py       Whole-file (offline) analysis used for tuning and evaluation.
    forecast.py    Growth-model fitting that projects time-to-critical (ETA).
    dataloader.py  CSV/Excel parsing with automatic column and sample-rate detection.
    models/        Pluggable ML model registry, wrappers, and built-in models.
"""
