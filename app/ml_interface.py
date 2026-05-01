# ml_interface.py
import numpy as np
import os

# Try to import typical ML libs (catch errors if they aren't installed).
# `joblib` is a scikit-learn dependency, so a successful joblib import
# implies sklearn is available.
try:
    import joblib  # Common for Scikit-Learn models
except ImportError:
    joblib = None


class MLPredictor:
    def __init__(self, model_path="model/clogging_model.pkl"):
        self.model = None
        self.ready = False

        # 1. Try to load a pre-trained model if it exists
        if joblib and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.ready = True
                print(f"[ML] Model loaded from {model_path}")
            except Exception as e:
                print(f"[ML] Failed to load model: {e}")
        else:
            print("[ML] No pre-trained model found. Running in Dummy Mode.")

    def predict(self, feature_vector):
        """
        Input: feature_vector = [static_score, composite_score, turbulence_score, spectral_slope]
        Output: Probability of Clogging (0.0 to 1.0)
        """
        if self.ready and self.model:
            # REAL MODE: Use the loaded brain
            # Reshape for sklearn (1 sample, N features)
            features = np.array(feature_vector).reshape(1, -1)
            try:
                # Assuming model returns class 1 probability
                probability = self.model.predict_proba(features)[0][1]
                return probability
            except Exception:
                return 0.0
        else:
            # DUMMY MODE (Placeholder for your thesis)
            # This logic mimics a model so you can test the GUI connection

            # Unpack features
            # features = [static, composite, turbulence, slope]
            static = feature_vector[0]
            composite = feature_vector[1]
            slope = feature_vector[3]  # Using the slope

            # Simple heuristic mimicking "AI"
            if static > 0.5 and composite > 0.05:
                return 0.95  # High probability
            elif slope > -1.5 and slope != 0:
                return 0.80  # Physics says it's clogged
            elif static > 0.2:
                return 0.4
            return 0.01
