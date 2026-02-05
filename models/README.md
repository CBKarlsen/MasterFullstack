# Models Directory

This directory is for user-provided machine learning models. The platform will automatically discover and load any compatible model files placed here.

## Supported File Formats

| Extension | Framework | How to Save |
|-----------|-----------|-------------|
| `.pkl` | scikit-learn | `joblib.dump(model, "model.pkl")` |
| `.pt` / `.pth` | PyTorch | `torch.save(model, "model.pt")` |
| `.h5` | TensorFlow/Keras | `model.save("model.h5")` |

## Quick Start

### 1. Save Your Trained Model

**For scikit-learn:**
```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save it
joblib.dump(model, "models/my_random_forest.pkl")
```

**For PyTorch:**
```python
import torch

# Train your model
model = MyLSTM()
# ... training code ...

# Save the full model (not just state_dict)
torch.save(model, "models/my_lstm.pt")
```

**For TensorFlow/Keras:**
```python
# Train your model
model = keras.Sequential([...])
model.fit(X_train, y_train)

# Save it
model.save("models/my_keras_model.h5")
```

### 2. (Optional) Add Metadata

Create a JSON file with the same name as your model to provide metadata:

**Example: `my_random_forest.json`**
```json
{
  "name": "my_random_forest",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Random Forest trained on 2024 pump data",
  "model_type": "sklearn",
  "input_type": "single",
  "input_features": ["static_score", "composite_score", "turbulence_score", "spectral_slope"],
  "output_type": "probability"
}
```

**For sequence models (LSTM, etc.):**
```json
{
  "name": "my_lstm",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "LSTM trained on 60-second windows",
  "model_type": "pytorch",
  "input_type": "sequence",
  "sequence_length": 1200,
  "input_features": ["static_score", "composite_score", "turbulence_score", "spectral_slope"],
  "output_type": "probability"
}
```

### 3. That's It!

The platform will automatically detect and load your model within 5 seconds. You can also upload models through the web dashboard.

## Input Features

Your model will receive these features (in this order for array-based inputs):

| Feature | Description | Range |
|---------|-------------|-------|
| `static_score` | Deviation from baseline pressure | 0.0 - ~0.5 |
| `composite_score` | High/low frequency energy ratio | 0.0 - ~0.1 |
| `turbulence_score` | Detrended FFT ratio | 0.0 - ~0.1 |
| `spectral_slope` | Power spectrum slope (log-log) | -3.5 to -1.0 |

**For single-input models:** Receives a 1D array `[static, composite, turbulence, slope]`

**For sequence models:** Receives a 2D array of shape `(sequence_length, 4)`

## Expected Output

Your model should output a **clogging probability** between 0.0 and 1.0.

- For classifiers: Use `predict_proba()` to get probability of class 1 (clogged)
- For regressors: Output should be in range [0, 1]

## Model Requirements

### scikit-learn Models
- Must have `predict_proba()` (classifiers) or `predict()` (regressors)
- Binary classification: class 0 = healthy, class 1 = clogged

### PyTorch Models
- Must be a full model save (not just state_dict)
- Must work in eval mode with `model.eval()`
- Single output neuron (sigmoid applied) OR two outputs (softmax applied)

### TensorFlow/Keras Models
- Standard Keras `.h5` format
- Input shape: `(batch, features)` or `(batch, sequence, features)`
- Output: probability or softmax

## Troubleshooting

**Model not appearing?**
- Check the file extension is correct
- Look at server logs for loading errors
- Ensure the model can be loaded with joblib/torch/keras

**Wrong predictions?**
- Verify your model expects features in the correct order
- Check if your model needs sequence input vs single vector
- Add a metadata JSON to specify input requirements

## API Endpoints

- `GET /api/models` - List all loaded models
- `PUT /api/models/{name}/enable?enabled=true` - Enable/disable a model
- `POST /api/models/upload` - Upload a model file
- `DELETE /api/models/{name}` - Unregister a model

## Example: Training a Simple Classifier

```python
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Simulated training data
# In practice, you'd use real labeled data from the platform
n_samples = 1000
X = np.random.rand(n_samples, 4)  # [static, composite, turbulence, slope]
y = (X[:, 0] > 0.3) | (X[:, 3] > -1.5)  # Simple clogging rule

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y.astype(int))

# Save
joblib.dump(model, "models/simple_rf.pkl")

# Create metadata
import json
with open("models/simple_rf.json", "w") as f:
    json.dump({
        "name": "simple_rf",
        "author": "Example",
        "description": "Simple Random Forest for demonstration",
        "model_type": "sklearn",
        "input_type": "single"
    }, f, indent=2)
```
