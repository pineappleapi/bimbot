#!/usr/bin/env python3
"""
XGBoost Hazard Classifier Module
Provides ML-based hazard classification for temperature/humidity data.
"""

import numpy as np
import xgboost as xgb
from pathlib import Path
from collections import deque

# Same heat index calculation from your app.py
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


class XGBoostHazardClassifier:
    """
    XGBoost-based hazard classifier with temporal feature tracking.
    """
    
    def __init__(self, model_path='models/hazard_xgboost_model.json', window_size=5):
        self.model_path = Path(model_path)
        self.model = None
        self.window_size = window_size
        
        # Temporal buffers for computing change rates and rolling stats
        self.temp_history = deque(maxlen=window_size)
        self.humidity_history = deque(maxlen=window_size)
        
        # Feature names (must match training order)
        self.feature_names = [
            'temperature_c',
            'humidity_pct',
            'heat_index_c',
            'temp_change_rate',
            'humidity_change_rate',
            'temp_rolling_std',
            'humidity_rolling_std'
        ]
        
        # Load model if exists
        if self.model_path.exists():
            self.load_model()
        else:
            print(f"⚠️  No trained model found at {self.model_path}")
            print(f"   Using fallback threshold-based classification")
    
    def load_model(self):
        """Load pre-trained XGBoost model (handles both sklearn and native formats)"""
        try:
            # Try loading as sklearn-style model first
            self.model = xgb.XGBClassifier()
            try:
                self.model.load_model(str(self.model_path))
                print(f"✅ XGBoost model loaded from {self.model_path} (sklearn format)")
            except (TypeError, AttributeError):
                # Fallback: Load as native XGBoost booster
                print(f"⚠️  Loading as native XGBoost format...")
                booster = xgb.Booster()
                booster.load_model(str(self.model_path))
                self.model._Booster = booster
                
                # Set sklearn compatibility attributes
                self.model.n_classes_ = 3
                self.model.classes_ = np.array([0, 1, 2])
                self.model._estimator_type = "classifier"
                
                print(f"✅ XGBoost model loaded from {self.model_path} (native format)")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def update_history(self, temperature, humidity):
        """Update temporal buffers"""
        self.temp_history.append(temperature)
        self.humidity_history.append(humidity)
    
    def compute_temporal_features(self):
        """Compute change rates and rolling statistics"""
        if len(self.temp_history) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        # Change rates (difference between last two readings)
        temp_change = self.temp_history[-1] - self.temp_history[-2]
        humidity_change = self.humidity_history[-1] - self.humidity_history[-2]
        
        # Rolling standard deviation (measure of instability)
        temp_std = np.std(self.temp_history) if len(self.temp_history) > 1 else 0.0
        humidity_std = np.std(self.humidity_history) if len(self.humidity_history) > 1 else 0.0
        
        return temp_change, humidity_change, temp_std, humidity_std
    
    def predict(self, temperature, humidity):
        """
        Predict hazard level using XGBoost model.
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity in %
        
        Returns:
            tuple: (risk_label, risk_score, confidence, probabilities, method)
                - risk_label: 'Low Risk', 'Moderate Risk', or 'Severe Risk'
                - risk_score: Numeric score (25, 60, or 90)
                - confidence: Probability of predicted class (0-1)
                - probabilities: List of [p_low, p_moderate, p_severe]
                - method: 'xgboost' or 'threshold' (fallback)
        """
        # Update history
        self.update_history(temperature, humidity)
        
        # Compute features
        heat_index = calculate_heat_index(temperature, humidity)
        temp_change, hum_change, temp_std, hum_std = self.compute_temporal_features()
        
        # Build feature vector
        features = np.array([[
            temperature,
            humidity,
            heat_index,
            temp_change,
            hum_change,
            temp_std,
            hum_std
        ]])
        
        # Predict using XGBoost if available
        if self.model is not None:
            try:
                # Get prediction
                hazard_class = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]
                confidence = probabilities[hazard_class]
                
                # Convert to labels
                labels = ['Low Risk', 'Moderate Risk', 'Severe Risk']
                scores = [25, 60, 90]
                
                return (
                    labels[hazard_class],
                    scores[hazard_class],
                    float(confidence),
                    probabilities.tolist(),
                    'xgboost'
                )
            except Exception as e:
                print(f"XGBoost prediction failed: {e}, falling back to threshold")
        
        # Fallback: Use threshold-based classification
        return self._threshold_fallback(temperature, humidity, heat_index)
    
    def _threshold_fallback(self, temperature, humidity, heat_index):
        """Fallback threshold-based classification (your original logic)"""
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
        
        # Fake confidence/probabilities for consistency
        probs = [0.0, 0.0, 0.0]
        probs[overall_risk] = 1.0
        
        return (
            labels[overall_risk],
            scores[overall_risk],
            1.0,  # Full confidence in threshold
            probs,
            'threshold'
        )
    
    def reset(self):
        """Clear temporal history"""
        self.temp_history.clear()
        self.humidity_history.clear()
        


# Singleton instance for app.py to use
_classifier_instance = None

def get_classifier():
    """Get or create singleton classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = XGBoostHazardClassifier()
    return _classifier_instance

