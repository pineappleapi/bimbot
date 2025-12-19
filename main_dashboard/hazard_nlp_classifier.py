#!/usr/bin/env python3
"""
NLP-Based Hazard Classifier Module
Converts sensor readings to text and classifies using trained NLP model.
"""

import numpy as np
import pickle
from pathlib import Path
from collections import deque

# Scikit-learn for model loading
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Try to load deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


def calculate_heat_index(temp_c, rh):
    """Calculate heat index in Celsius"""
    if temp_c is None or rh is None:
        return 0.0
    T = temp_c * 9 / 5 + 32.0
    R = rh
    HI_f = (-42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
            - 0.00683783 * T ** 2 - 0.05481717 * R ** 2 + 0.00122874 * T ** 2 * R
            + 0.00085282 * T * R ** 2 - 0.00000199 * T ** 2 * R ** 2)
    if R < 13 and 80 <= T <= 112:
        adj = ((13 - R) / 4) * ((17 - abs(T - 95.0)) / 17)**0.5
        HI_f -= adj
    elif R > 85 and 80 <= T <= 87:
        adj = ((R - 85) / 10) * ((87 - T) / 5)
        HI_f += adj
    return (HI_f - 32.0) * 5 / 9


class NLPHazardClassifier:
    """
    NLP-based hazard classifier that converts sensor readings to text.
    Supports multiple model types: Logistic Regression, Naive Bayes, LSTM, CNN, Transformer, etc.
    """
    
    def __init__(self, model_path='models/nlp/best_model.pkl', model_type='logistic'):
        """
        Args:
            model_path: Path to saved model
            model_type: Type of model ('logistic', 'naive_bayes', 'lstm', 'cnn', 'transformer')
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        
        # Temporal buffers for trend descriptions
        self.temp_history = deque(maxlen=2)
        self.humidity_history = deque(maxlen=2)
        self.risk_history = deque(maxlen=3)
        
        # Load model if exists
        if self.model_path.exists():
            self.load_model()
        else:
            print(f"⚠️  No trained NLP model found at {self.model_path}")
            print(f"   Using fallback threshold-based classification")
    
    def load_model(self):
        """Load pre-trained NLP model and its vectorizer/tokenizer"""
        try:
            print(f"🔍 Attempting to load: {self.model_path}")
            print(f"📁 File exists: {self.model_path.exists()}")
            print(f"🤖 Model type: {self.model_type}")
            print(f"🧠 Deep learning available: {DEEP_LEARNING_AVAILABLE}")

            if self.model_type in ['logistic', 'naive_bayes', 'decision_tree', 'random_forest']:
                # Load skleaprn model
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.vectorizer = model_data['vectorizer']
                
                print(f"✅ NLP model loaded: {self.model_type.upper()}")
            
            
            elif self.model_type in ['lstm', 'cnn', 'transformer'] and DEEP_LEARNING_AVAILABLE:
                # Load Keras model
                print(f"📥 Loading Keras model from {self.model_path}")
                self.model = load_model(self.model_path)

                parent_dir = self.model_path.parent

                # Prefer model-type–prefixed filenames, fallback to generic names
                json_candidates = [
                    parent_dir / f"{self.model_type}_tokenizer.json",  # e.g., cnn_tokenizer.json
                    parent_dir / "tokenizer.json",                     # legacy generic
                ]
                cfg_candidates = [
                    parent_dir / f"{self.model_type}_sequence_config.json",  # e.g., cnn_sequence_config.json
                    parent_dir / "sequence_config.json",                     # legacy generic
                ]

                # Defaults
                self.tokenizer = None
                self.max_len = 50  # default if no sequence_config found

                # --- Load tokenizer JSON ---
                json_path = next((p for p in json_candidates if p.exists()), None)
                if json_path is not None:
                    print(f"📥 Loading tokenizer from {json_path}")
                    with open(json_path, 'r') as f:
                        tok_json = f.read()
                    self.tokenizer = tokenizer_from_json(tok_json)
                else:
                    print(f"⚠️ No tokenizer JSON found. Checked: {', '.join(str(p) for p in json_candidates)}")

                # --- Load sequence config (optional but recommended) ---
                seq_cfg_path = next((p for p in cfg_candidates if p.exists()), None)
                if seq_cfg_path is not None:
                    print(f"📥 Loading sequence config from {seq_cfg_path}")
                    import json as _json
                    with open(seq_cfg_path, 'r') as f:
                        cfg = _json.load(f)
                    if 'max_len' in cfg:
                        try:
                            self.max_len = int(cfg['max_len'])
                        except Exception:
                            print(f"⚠️ Invalid max_len in {seq_cfg_path}; using default {self.max_len}")
                else:
                    print(f"ℹ️ No sequence config found; using default max_len={self.max_len}. "
                        f"Checked: {', '.join(str(p) for p in cfg_candidates)}")

                print(f"✅ Deep learning model loaded: {self.model_type.upper()}")

            
            else:
                print(f"⚠️  Unsupported model type or deep learning not available")
                print(f"   Model type: {self.model_type}")
                print(f"   Deep learning available: {DEEP_LEARNING_AVAILABLE}")
                self.model = None
        
        except Exception as e:
            print(f"❌ Failed to load NLP model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def textualize_reading(self, temperature, humidity):
        """
        Convert sensor reading to natural language text.
        Uses multiple templates for robustness.
        """
        heat_index = calculate_heat_index(temperature, humidity)
        
        # Describe heat index
        if heat_index < 27:
            hi_desc = "heat index is normal"
        elif heat_index < 32:
            hi_desc = "heat index is elevated"
        elif heat_index < 40:
            hi_desc = "heat index is high"
        else:
            hi_desc = "heat index exceeds threshold"
        
        # Create text description (multiple templates for variety)
        templates = [
            f"Temperature is {temperature:.1f}°C, humidity is {humidity:.1f}%, {hi_desc}.",
            f"Current conditions: {temperature:.1f}°C temperature, {humidity:.1f}% humidity, {hi_desc}.",
            f"Sensor readings show {temperature:.1f} degrees Celsius, {humidity:.1f} percent humidity, {hi_desc}.",
        ]
        
        # Add temporal context if available
        if len(self.temp_history) > 0:
            temp_change = temperature - self.temp_history[-1]
            hum_change = humidity - self.humidity_history[-1]
            
            if abs(temp_change) > 1:
                trend = "rising" if temp_change > 0 else "dropping"
                templates.append(
                    f"Temperature {trend} to {temperature:.1f}°C. Humidity at {humidity:.1f}%. {hi_desc.capitalize()}."
                )
        
        # Update history
        self.temp_history.append(temperature)
        self.humidity_history.append(humidity)
        
        # Return first template (most consistent for production)
        return templates[0]
    
    
    def predict(self, temperature, humidity):
        """
        Predict hazard level from sensor readings.

        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in %

        Returns:
            tuple: (risk_label, risk_score, confidence, probabilities_dict, method, interpretation)
        """
        # Convert sensor readings to natural-language text
        text = self.textualize_reading(temperature, humidity)

        print(f"🔍 Model status: {self.model is not None}, Type: {self.model_type}")

        # --- NLP / ML Prediction ---
        if self.model is not None:
            try:
                # --- SKLEARN MODELS ---
                if self.model_type in ['logistic', 'naive_bayes', 'decision_tree', 'random_forest']:
                    X = self.vectorizer.transform([text])
                    hazard_class = self.model.predict(X)[0]

                    # Probabilities
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba(X)[0]
                    else:
                        probabilities = np.zeros(3)
                        probabilities[hazard_class] = 1.0

                    confidence = probabilities[hazard_class]

                # --- DEEP LEARNING MODELS ---
                
                elif self.model_type in ['lstm', 'cnn', 'transformer'] and self.tokenizer:
                    seq = self.tokenizer.texts_to_sequences([text])
                    X = pad_sequences(seq, maxlen=self.max_len, padding='post')
                    probabilities = self.model.predict(X, verbose=0)[0]
                    hazard_class = np.argmax(probabilities)
                    confidence = probabilities[hazard_class]  # <-- ADD THIS


                else:
                    raise Exception("Model or tokenizer not properly loaded")

                # --- Convert to human-readable labels & scores ---
                labels = ['Low Risk', 'Moderate Risk', 'Severe Risk']
                scores = [25, 60, 90]
                risk_label = labels[hazard_class]
                risk_score = scores[hazard_class]

                # --- Build probability dictionary ---
                prob_dict = {
                    "low": float(probabilities[0]),
                    "moderate": float(probabilities[1]),
                    "severe": float(probabilities[2])
                }

                # --- Interpretation templates ---
                base_interpretations = {
                    'Low Risk': (
                        "Environmental conditions are within safe limits. "
                        "No immediate action is required."
                    ),
                    'Moderate Risk': (
                        "Moderate environmental stress detected. "
                        "Prolonged exposure may lead to discomfort."
                    ),
                    'Severe Risk': (
                        "Severe heat stress detected. "
                        "Immediate precautions are strongly recommended."
                    )
                }

                interpretation = base_interpretations[risk_label]

                # --- Trend-aware language ---
                trend = self._risk_trend(hazard_class)
                if trend == "decreasing":
                    interpretation += " Environmental changes indicate that risk levels are decreasing."
                elif trend == "increasing":
                    interpretation += " Environmental changes indicate that risk levels are increasing."
                elif trend == "stable":
                    # Context-aware stable messages
                    if hazard_class == 0:  # Low Risk
                        interpretation += " Environmental conditions remain relatively stable."
                    elif hazard_class == 1:  # Moderate Risk
                        interpretation += " Current moderate conditions are persisting."
                    else:  # Severe Risk
                        interpretation += " Severe conditions remain unchanged - monitor accordingly."

                # Store current risk for trend analysis
                self.risk_history.append(hazard_class)

                return (
                    risk_label,
                    risk_score,
                    float(confidence),
                    prob_dict,    
                    f'nlp_{self.model_type}',
                    interpretation
                )

            except Exception as e:
                print(f"NLP prediction failed: {e}, falling back to threshold")

        # --- FALLBACK: Threshold-based classification ---
        risk_label, risk_score, conf, probs_list, method = self._threshold_fallback(
            temperature, humidity, calculate_heat_index(temperature, humidity)
        )

        # Convert list to dictionary for backward compatibility
        prob_dict = {
            "low": float(probs_list[0]),
            "moderate": float(probs_list[1]),
            "severe": float(probs_list[2])
        }

        # FIX: Generate proper interpretation for threshold fallback
        base_interpretations = {
            'Low Risk': (
                "Environmental conditions are within safe limits. "
                "No immediate action is required."
            ),
            'Moderate Risk': (
                "Moderate environmental stress detected. "
                "Prolonged exposure may lead to discomfort."
            ),
            'Severe Risk': (
                "Severe heat stress detected. "
                "Immediate precautions are strongly recommended."
            )
        }

        interpretation = base_interpretations[risk_label]

        # Add trend analysis for threshold fallback too
        hazard_class = ['Low Risk', 'Moderate Risk', 'Severe Risk'].index(risk_label)
        trend = self._risk_trend(hazard_class)

        if trend == "decreasing":
            interpretation += " Environmental changes indicate that risk levels are decreasing."
        elif trend == "increasing":
            interpretation += " Environmental changes indicate that risk levels are increasing."
        elif trend == "stable":
            interpretation += " Environmental conditions remain relatively stable."

        self.risk_history.append(hazard_class)

        return (risk_label, risk_score, conf, prob_dict, method, interpretation)  # <-- Now returns interpretation

    
    def _risk_trend(self, current_class):
        """Determine risk trend based on history"""
        if len(self.risk_history) < 2:
            return None

        previous_class = self.risk_history[-2]

        if current_class > previous_class:
            return "increasing"
        elif current_class < previous_class:
            return "decreasing"
        else:
            return "stable"
    
    def _threshold_fallback(self, temperature, humidity, heat_index):
        """Fallback threshold-based classification"""
        THRESHOLDS = {
            "temperature_c": {"low_max": 27.0, "moderate_max": 32.0},
            "humidity_pct": {"low_max": 60.0, "moderate_max": 80.0},
            "heat_index_c": {"low_max": 32.0, "moderate_max": 40.0}
        }
        
        def classify_metric(value, bounds):
            if value <= bounds["low_max"]:
                return 0
            elif value <= bounds["moderate_max"]:
                return 1
            return 2
        
        t_risk = classify_metric(temperature, THRESHOLDS["temperature_c"])
        h_risk = classify_metric(humidity, THRESHOLDS["humidity_pct"])
        hi_risk = classify_metric(heat_index, THRESHOLDS["heat_index_c"])
        
        overall_risk = max(t_risk, h_risk, hi_risk)
        
        labels = ['Low Risk', 'Moderate Risk', 'Severe Risk']
        scores = [25, 60, 90]
        
        probs = [0.0, 0.0, 0.0]
        probs[overall_risk] = 1.0
        
        return (
            labels[overall_risk],
            scores[overall_risk],
            1.0,
            probs,
            'threshold'
        )
    
    def reset(self):
        """Clear temporal history"""
        self.temp_history.clear()
        self.humidity_history.clear()


def get_classifier(model_path='models/nlp/logistic_regression.pkl', model_type='logistic'):
    """Always create a fresh classifier with the provided configuration."""
    return NLPHazardClassifier(model_path, model_type)