# app/main.py
"""
FastAPI server for the Pipe Clogging Detection Platform.

Provides:
- WebSocket endpoint for real-time simulation streaming
- REST API for model management (list, upload, enable/disable)
- Health check endpoints
"""

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rf_train")

from .engine import SimulationEngine
from .models import get_registry
from .models.builtin import FFTPhysicsModel, RandomForestModel, IsolationForestModel
from .dataloader import DataStreamer
from .batch import run_batch_analysis, recompute_thresholds

# Initialize FastAPI app
app = FastAPI(
    title="Pipe Clogging Detection Platform",
    description="Modular platform for pipe clogging prediction using FFT and ML models",
    version="2.0.0",
)

# CORS configuration - allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


@app.on_event("startup")
async def startup_event():
    """Initialize model registry on server start."""
    registry = get_registry(str(MODELS_DIR))

    # Register built-in models
    registry.register(FFTPhysicsModel())
    registry.register(RandomForestModel())
    registry.register(IsolationForestModel())

    # Discover user models from models/ directory
    loaded = registry.discover_models()
    print(f"[Startup] Loaded {len(loaded)} user models: {loaded}")

    # Start watching for new model files
    registry.start_watching(poll_interval=5.0)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    registry = get_registry()
    registry.stop_watching()


# =============================================================================
# Health Check Endpoints
# =============================================================================


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Pipe Clogging Detection Platform API",
        "version": "2.0.0",
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    registry = get_registry()
    return {
        "status": "healthy",
        "models_loaded": len(registry.get_all()),
        "models_active": len(registry.get_active()),
    }


# =============================================================================
# Model Management REST API
# =============================================================================


@app.get("/api/models")
def list_models():
    """
    List all registered models with their metadata and statistics.

    Returns:
        List of model info dictionaries.
    """
    registry = get_registry()
    return registry.list_models()


@app.get("/api/models/{model_name}")
def get_model(model_name: str):
    """
    Get detailed information about a specific model.

    Args:
        model_name: Name of the model to retrieve.

    Returns:
        Model metadata and statistics.
    """
    registry = get_registry()
    model = registry.get(model_name)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Find the registered model entry for stats
    models_list = registry.list_models()
    for m in models_list:
        if m["name"] == model_name:
            return m

    return model.metadata.to_dict()


@app.put("/api/models/{model_name}/enable")
def enable_model(model_name: str, enabled: bool = True):
    """
    Enable or disable a model.

    Args:
        model_name: Name of the model.
        enabled: Whether to enable (true) or disable (false).

    Returns:
        Updated model status.
    """
    registry = get_registry()

    if not registry.get(model_name):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    success = registry.set_enabled(model_name, enabled)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update model status")

    return {
        "model": model_name,
        "enabled": enabled,
        "message": f"Model {'enabled' if enabled else 'disabled'}",
    }


@app.put("/api/models/{model_name}/weight")
def set_model_weight(model_name: str, weight: float = 1.0):
    """
    Set the ensemble trust weight for a model.

    A weight of 1.0 is normal. 0.0 effectively mutes the model from the
    ensemble without disabling its predictions. 2.0 doubles its influence.

    Args:
        model_name: Name of the model.
        weight: Trust weight between 0.0 and 5.0.

    Returns:
        Updated weight.
    """
    registry = get_registry()

    if not registry.get(model_name):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    clamped = max(0.0, min(weight, 5.0))
    registry.set_weight(model_name, clamped)

    return {"model": model_name, "weight": clamped}


@app.delete("/api/models/{model_name}")
def delete_model(model_name: str):
    """
    Remove a model from the registry.

    Note: Does not delete the file from disk for safety.

    Args:
        model_name: Name of the model to remove.

    Returns:
        Confirmation message.
    """
    registry = get_registry()

    model = registry.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Don't allow deleting built-in models
    if model.metadata.model_type.value == "builtin":
        raise HTTPException(status_code=400, detail="Cannot delete built-in models")

    success = registry.unregister(model_name)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to remove model")

    return {"model": model_name, "message": "Model unregistered (file not deleted)"}


@app.post("/api/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload a new model file.

    Accepts .pkl (sklearn), .pt/.pth (PyTorch), or .h5 (TensorFlow) files.
    Optionally accepts metadata fields.

    Args:
        file: The model file to upload.
        name: Optional model name (defaults to filename).
        author: Optional author name.
        description: Optional description.

    Returns:
        Upload confirmation with model name.
    """
    # Validate file extension
    allowed_extensions = {".pkl", ".pt", ".pth", ".h5"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = MODELS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Create metadata JSON if custom metadata provided
    if name or author or description:
        import json

        metadata = {
            "name": name or Path(file.filename).stem,
            "author": author or "Unknown",
            "description": description or "",
            "model_type": {
                ".pkl": "sklearn",
                ".pt": "pytorch",
                ".pth": "pytorch",
                ".h5": "tensorflow",
            }.get(suffix, "sklearn"),
            "input_type": "single",
            "input_features": [
                "static_score",
                "composite_score",
                "turbulence_score",
                "spectral_slope",
            ],
        }

        json_path = file_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Reload models to pick up the new one
    registry = get_registry()
    loaded = registry.discover_models()

    return {
        "message": "Model uploaded successfully",
        "filename": file.filename,
        "model_name": name or Path(file.filename).stem,
        "models_loaded": loaded,
    }


# =============================================================================
# Random Forest Training Endpoints
# =============================================================================


@app.get("/api/models/random_forest/training-info")
def get_rf_training_info():
    """Return the current training status of the built-in Random Forest."""
    from .models.builtin.random_forest import RandomForestModel

    registry = get_registry()
    model = registry.get("Random Forest")
    if not model or not isinstance(model, RandomForestModel):
        raise HTTPException(status_code=404, detail="Random Forest model not found")
    return model.get_training_info()


def _do_rf_training(
    model: "RandomForestModel",
    data_path: Path,
    filename: str,
    sigma: float,
    calibration_seconds: float,
    include_warnings: bool,
    n_estimators: int,
    max_depth: Optional[int],
) -> None:
    """
    Blocking training job executed in a background thread.
    Writes progress into model._training_state throughout.
    """
    import numpy as np

    def _set(phase: str, percent: int, message: str) -> None:
        model._training_state = {"phase": phase, "percent": percent, "message": message}

    try:
        _set("analyzing", 5, "Analyzing data…")
        results = run_batch_analysis(
            filepath=str(data_path),
            sigma=sigma,
            calibration_seconds=calibration_seconds,
        )

        if "error" in results:
            _set("error", 0, results["error"])
            model._training_error = results["error"]
            return

        _set("extracting", 68, "Extracting features…")
        features_list: list = []
        labels: list = []

        for pt in results["timeseries"]:
            if pt.get("phase") != "analysis":
                continue
            light = pt.get("traffic_light", "gray")
            if light == "green":
                label = 0
            elif light == "red":
                label = 1
            elif light == "yellow" and include_warnings:
                label = 1
            else:
                continue
            features_list.append(
                [
                    pt.get("static_score", 0.0),
                    pt.get("composite_score", 0.0),
                    pt.get("turbulence_score", 0.0),
                    pt.get("spectral_slope", 0.0),
                ]
            )
            labels.append(label)

        if len(features_list) < 20:
            msg = (
                f"Not enough labeled samples ({len(features_list)}). "
                "Try a longer file, lower sigma, or enable 'Include warnings'."
            )
            _set("error", 0, msg)
            model._training_error = msg
            return

        X = np.array(features_list)
        y = np.array(labels)
        n_healthy = int(sum(y == 0))
        n_clogged = int(sum(y == 1))

        if n_healthy < 5:
            msg = f"Too few healthy samples ({n_healthy}). Need at least 5."
            _set("error", 0, msg)
            model._training_error = msg
            return

        if n_clogged < 5:
            msg = f"Too few clogged samples ({n_clogged}). Need at least 5."
            _set("error", 0, msg)
            model._training_error = msg
            return

        _set("training", 75, f"Training Random Forest ({n_estimators} trees)…")
        model.set_training_source(f"user data: {filename}")
        stats = model.train(X, y, n_estimators=n_estimators, max_depth=max_depth)

        _set("complete", 100, "Training complete")
        model._training_result = {
            "message": "Model trained successfully",
            "file_used": filename,
            "sigma": sigma,
            **stats,
        }
        model._training_error = None

    except Exception as exc:
        msg = str(exc)
        _set("error", 0, msg)
        model._training_error = msg
        model._training_result = None


@app.post("/api/models/random_forest/train")
def train_random_forest(
    file: str,
    sigma: float = 3.0,
    calibration_seconds: float = 120.0,
    include_warnings: bool = False,
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
):
    """
    Start training the built-in Random Forest in a background thread.

    Returns immediately with ``{"status": "started"}``.
    Poll ``GET /api/models/random_forest/training-progress`` for updates.
    """
    registry = get_registry()
    model = registry.get("Random Forest")
    if not model or not isinstance(model, RandomForestModel):
        raise HTTPException(
            status_code=404, detail="Random Forest model not found in registry"
        )

    # Guard against concurrent training
    current_phase = model._training_state.get("phase", "idle")
    if current_phase not in ("idle", "complete", "error"):
        raise HTTPException(status_code=409, detail="Training already in progress")

    data_path = DATA_DIR / file
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file}' not found")

    # Reset state before handing off to the thread
    model._training_state = {"phase": "starting", "percent": 2, "message": "Starting…"}
    model._training_result = None
    model._training_error = None

    _training_executor.submit(
        _do_rf_training,
        model,
        data_path,
        file,
        sigma,
        calibration_seconds,
        include_warnings,
        n_estimators,
        max_depth,
    )

    return {"status": "started"}


@app.get("/api/models/random_forest/training-progress")
def get_rf_training_progress():
    """Poll this endpoint while training to get live progress."""
    registry = get_registry()
    model = registry.get("Random Forest")
    if not model or not isinstance(model, RandomForestModel):
        raise HTTPException(status_code=404, detail="Random Forest model not found")
    return model.get_training_state()


@app.post("/api/models/random_forest/reset")
def reset_random_forest():
    """Discard the user-trained Random Forest and revert to synthetic training."""
    from .models.builtin.random_forest import RandomForestModel

    registry = get_registry()
    model = registry.get("Random Forest")
    if not model or not isinstance(model, RandomForestModel):
        raise HTTPException(status_code=404, detail="Random Forest model not found")
    model.reset_to_synthetic()
    return {"message": "Model reset to synthetic training data"}


# =============================================================================
# Isolation Forest Training Endpoints
# =============================================================================


@app.get("/api/models/isolation_forest/training-info")
def get_if_training_info():
    """Return the current training status of the built-in Isolation Forest."""
    registry = get_registry()
    model = registry.get("Isolation Forest")
    if not model or not isinstance(model, IsolationForestModel):
        raise HTTPException(status_code=404, detail="Isolation Forest model not found")
    return model.get_training_info()


def _do_if_training(
    model: "IsolationForestModel",
    data_path: Path,
    filename: str,
    sigma: float,
    calibration_seconds: float,
    n_estimators: int,
    contamination: float,
) -> None:
    """
    Blocking Isolation Forest training job — runs in a background thread.
    Extracts only GREEN (healthy) points from batch analysis, so the model
    learns the normal operating distribution without any labelled clogged data.
    """
    import numpy as np

    def _set(phase: str, percent: int, message: str) -> None:
        model._training_state = {"phase": phase, "percent": percent, "message": message}

    try:
        _set("analyzing", 5, "Analyzing data…")
        results = run_batch_analysis(
            filepath=str(data_path),
            sigma=sigma,
            calibration_seconds=calibration_seconds,
        )

        if "error" in results:
            _set("error", 0, results["error"])
            model._training_error = results["error"]
            return

        _set("extracting", 68, "Extracting healthy samples…")
        features_list: list = []

        for pt in results["timeseries"]:
            if pt.get("phase") != "analysis":
                continue
            # Unsupervised: only use GREEN (physics-confirmed healthy) points
            if pt.get("traffic_light") != "green":
                continue
            features_list.append(
                [
                    pt.get("static_score", 0.0),
                    pt.get("composite_score", 0.0),
                    pt.get("turbulence_score", 0.0),
                    pt.get("spectral_slope", 0.0),
                ]
            )

        if len(features_list) < 20:
            msg = (
                f"Not enough healthy samples ({len(features_list)}). "
                "Try a longer file, lower sigma, or a file with more stable baseline data."
            )
            _set("error", 0, msg)
            model._training_error = msg
            return

        X_healthy = np.array(features_list)

        _set("training", 80, f"Training Isolation Forest ({n_estimators} trees)…")
        model.set_training_source(f"user data: {filename}")
        stats = model.train(
            X_healthy, n_estimators=n_estimators, contamination=contamination
        )

        _set("complete", 100, "Training complete")
        model._training_result = {
            "message": "Model trained successfully",
            "file_used": filename,
            "sigma": sigma,
            **stats,
        }
        model._training_error = None

    except Exception as exc:
        msg = str(exc)
        _set("error", 0, msg)
        model._training_error = msg
        model._training_result = None


@app.post("/api/models/isolation_forest/train")
def train_isolation_forest(
    file: str,
    sigma: float = 3.0,
    calibration_seconds: float = 120.0,
    n_estimators: int = 100,
    contamination: float = 0.05,
):
    """
    Start training the Isolation Forest in a background thread.

    Only uses GREEN-labelled (physics-confirmed healthy) samples.
    Returns immediately — poll GET /api/models/isolation_forest/training-progress.
    """
    registry = get_registry()
    model = registry.get("Isolation Forest")
    if not model or not isinstance(model, IsolationForestModel):
        raise HTTPException(status_code=404, detail="Isolation Forest model not found")

    current_phase = model._training_state.get("phase", "idle")
    if current_phase not in ("idle", "complete", "error"):
        raise HTTPException(status_code=409, detail="Training already in progress")

    data_path = DATA_DIR / file
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file}' not found")

    model._training_state = {"phase": "starting", "percent": 2, "message": "Starting…"}
    model._training_result = None
    model._training_error = None

    _training_executor.submit(
        _do_if_training,
        model,
        data_path,
        file,
        sigma,
        calibration_seconds,
        n_estimators,
        contamination,
    )

    return {"status": "started"}


@app.get("/api/models/isolation_forest/training-progress")
def get_if_training_progress():
    """Poll this endpoint while training to get live progress."""
    registry = get_registry()
    model = registry.get("Isolation Forest")
    if not model or not isinstance(model, IsolationForestModel):
        raise HTTPException(status_code=404, detail="Isolation Forest model not found")
    return model.get_training_state()


@app.post("/api/models/isolation_forest/reset")
def reset_isolation_forest():
    """Discard user-trained Isolation Forest and revert to synthetic training."""
    registry = get_registry()
    model = registry.get("Isolation Forest")
    if not model or not isinstance(model, IsolationForestModel):
        raise HTTPException(status_code=404, detail="Isolation Forest model not found")
    model.reset_to_synthetic()
    return {"message": "Model reset to synthetic training data"}


@app.get("/api/models/{model_name}/stats")
def get_model_stats(model_name: str):
    """
    Get performance statistics for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Prediction count, average inference time, last error.
    """
    registry = get_registry()

    models_list = registry.list_models()
    for m in models_list:
        if m["name"] == model_name:
            return {
                "name": model_name,
                "prediction_count": m.get("prediction_count", 0),
                "avg_inference_ms": m.get("avg_inference_ms", 0),
                "last_error": m.get("last_error"),
            }

    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


# =============================================================================
# Data File Management REST API
# =============================================================================


@app.get("/api/data")
def list_data_files():
    """
    List all available data files and folders in the data/ directory.

    Returns:
        List of data file/folder info (name, size, modified date, is_folder, file_count).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    for item_path in DATA_DIR.iterdir():
        # Handle individual files
        if item_path.is_file() and item_path.suffix.lower() in (".csv", ".xlsx"):
            stat = item_path.stat()
            items.append(
                {
                    "name": item_path.name,
                    "size_bytes": stat.st_size,
                    "size_human": _format_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "is_folder": False,
                    "file_count": 1,
                }
            )
        # Handle folders containing data files
        elif item_path.is_dir():
            # Count data files in the folder
            data_files = [
                f
                for f in item_path.iterdir()
                if f.is_file() and f.suffix.lower() in (".csv", ".xlsx")
            ]
            if data_files:  # Only include folders that have data files
                # Calculate total size and get latest modified time
                total_size = sum(f.stat().st_size for f in data_files)
                latest_modified = max(f.stat().st_mtime for f in data_files)
                items.append(
                    {
                        "name": item_path.name,
                        "size_bytes": total_size,
                        "size_human": _format_size(total_size),
                        "modified": latest_modified,
                        "is_folder": True,
                        "file_count": len(data_files),
                    }
                )

    # Sort by modified date (newest first)
    items.sort(key=lambda x: x["modified"], reverse=True)
    return items


@app.post("/api/analyze")
async def analyze_file(
    file: str,
    sigma: float = 3.0,
    calibration_seconds: float = 120.0,
):
    data_path = DATA_DIR / file
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file}' not found")
    try:
        results = run_batch_analysis(
            filepath=str(data_path),
            sigma=sigma,
            calibration_seconds=calibration_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    return results


@app.post("/api/analyze/thresholds")
async def recalculate_thresholds(
    composite_mean: float,
    composite_std: float,
    baseline_std: float,
    sigma: float,
):
    return recompute_thresholds(
        composite_mean=composite_mean,
        composite_std=composite_std,
        baseline_std=baseline_std,
        sigma=sigma,
    )


@app.post("/api/data/upload")
async def upload_data_file(file: UploadFile = File(...)):
    """
    Upload a new data file (CSV or Excel).

    Args:
        file: The data file to upload.

    Returns:
        Upload confirmation.
    """
    # Validate file extension
    allowed_extensions = {".csv", ".xlsx"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = DATA_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return {
        "message": "Data file uploaded successfully",
        "filename": file.filename,
        "size_bytes": file_path.stat().st_size,
    }


@app.delete("/api/data/{filename}")
def delete_data_file(filename: str):
    """
    Delete a data file.

    Args:
        filename: Name of the file to delete.

    Returns:
        Confirmation message.
    """
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Safety check: ensure file is in data directory
    if not file_path.resolve().parent == DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

    return {"message": f"File '{filename}' deleted"}


@app.get("/api/data/{filename}/columns")
def get_file_columns(filename: str):
    """
    Get metadata about all columns in a data file.

    Args:
        filename: Name of the data file.

    Returns:
        List of column metadata (name, type, unit).
    """
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Safety check: ensure file is in data directory
    if not file_path.resolve().parent == DATA_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        columns = DataStreamer.get_file_columns(str(file_path))
        return {"filename": filename, "columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read columns: {e}")


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# WebSocket Simulation Endpoint
# =============================================================================


@app.websocket("/ws/simulate")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time simulation streaming.

    Protocol:
    1. Client connects
    2. Client sends config message: {"action": "start", "file": "test.csv", "speed": 1.0}
    3. Server streams data until complete or client sends {"action": "stop"}

    Config options:
    - file: Name of data file in data/ directory (required)
    - speed: Simulation speed multiplier (default 1.0, higher = faster)
    """
    await websocket.accept()
    engine = None

    try:
        # Wait for configuration message from client
        config = await websocket.receive_json()

        if config.get("action") != "start":
            await websocket.send_json({"error": "Expected 'start' action"})
            return

        # Get file path
        filename = config.get("file")
        if not filename:
            await websocket.send_json({"error": "No file specified"})
            return

        data_file = DATA_DIR / filename

        if not data_file.exists():
            await websocket.send_json({"error": f"File '{filename}' not found"})
            return

        # Get speed multiplier (1.0 = real-time at the file's native sampling rate)
        speed = config.get("speed", 1.0)

        print(f"Client connected. File: {filename}, Speed: {speed}x")

        # Create simulation engine. Inter-sample delay is set against the
        # detected sampling rate during calibration, not hardcoded here.
        engine = SimulationEngine(str(data_file), speed_multiplier=speed)

        # Run simulation loop
        await engine.run_simulation(websocket)

    except WebSocketDisconnect:
        print("Client disconnected")
        if engine:
            engine.stop()
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# =============================================================================
# Entry Point (for direct running)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=None,  # Disable server-side pings (Vite proxy doesn't relay them)
        ws_ping_timeout=None,
    )
