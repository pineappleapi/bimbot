#!/usr/bin/env python3
"""
Train XGBoost model for hazard classification using historical sensor data.

Usage:
    python train_hazard_model.py --input data/training/sensor_data.csv
    python train_hazard_model.py --input data/training/sensor_data.csv --test-size 0.3
    python train_hazard_model.py --input data/training/sensor_data.csv --output models/custom_model.json

Run this script once to generate models/hazard_xgboost_model.json
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from collections import deque
import sys

# Import heat index calculation
def calculate_heat_index(temp_c, rh):
    """Calculate heat index in Celsius"""
    if temp_c is None or rh is None or pd.isna(temp_c) or pd.isna(rh):
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


def load_and_validate_csv(csv_path):
    """
    Load CSV and validate required columns.
    
    Expected columns (case-insensitive):
        - temperature (or temp, temperature_c)
        - humidity (or humidity_pct, rh)
        - risk_level (or hazard_level, hazard_overall)
    
    Optional columns:
        - heat_index (will be computed if missing)
        - timestamp (for temporal ordering)
    """
    print(f"📂 Loading CSV from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column name variations
    column_mapping = {}
    
    # Temperature mapping
    temp_candidates = ['temperature', 'temp', 'temperature_c', 'temp_c']
    for col in df.columns:
        if col in temp_candidates:
            column_mapping[col] = 'temperature_c'
            break
    
    # Humidity mapping
    humidity_candidates = ['humidity', 'humidity_pct', 'rh', 'relative_humidity']
    for col in df.columns:
        if col in humidity_candidates:
            column_mapping[col] = 'humidity_pct'
            break
    
    # Risk level mapping
    risk_candidates = ['risk_level', 'hazard_level', 'hazard_overall']
    for col in df.columns:
        if col in risk_candidates:
            column_mapping[col] = 'risk_level'
            break
    
    # Apply mapping
    df.rename(columns=column_mapping, inplace=True)
    
    # Validate required columns exist
    required = ['temperature_c', 'humidity_pct', 'risk_level']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Expected: temperature, humidity, risk_level")
        sys.exit(1)
    
    # Convert to numeric (handle any string values)
    df['temperature_c'] = pd.to_numeric(df['temperature_c'], errors='coerce')
    df['humidity_pct'] = pd.to_numeric(df['humidity_pct'], errors='coerce')
    
    # Remove rows with invalid numeric values
    initial_len = len(df)
    df.dropna(subset=['temperature_c', 'humidity_pct'], inplace=True)
    if len(df) < initial_len:
        print(f"⚠️  Dropped {initial_len - len(df)} rows with invalid numeric values")
    
    # Compute heat index if not present
    if 'heat_index' not in df.columns:
        print("🔥 Computing heat index (not found in CSV)...")
        df['heat_index'] = df.apply(
            lambda row: calculate_heat_index(row['temperature_c'], row['humidity_pct']),
            axis=1
        )
    
    # Normalize risk level labels
    risk_mapping = {
        'low risk': 'Low Risk',
        'lowrisk': 'Low Risk',
        'low': 'Low Risk',
        'moderate risk': 'Moderate Risk',
        'moderaterisk': 'Moderate Risk',
        'moderate': 'Moderate Risk',
        'mod': 'Moderate Risk',
        'severe risk': 'Severe Risk',
        'severerisk': 'Severe Risk',
        'severe': 'Severe Risk',
        'high': 'Severe Risk',
        'high risk': 'Severe Risk'
    }
    
    df['risk_level'] = df['risk_level'].str.strip().str.lower().map(risk_mapping)
    
    # Remove any rows with unmapped risk levels
    invalid_risks = df['risk_level'].isna().sum()
    if invalid_risks > 0:
        print(f"⚠️  Dropped {invalid_risks} rows with invalid risk_level values")
        df.dropna(subset=['risk_level'], inplace=True)
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.sort_values('timestamp', inplace=True)
            print(f"✅ Sorted by timestamp")
        except:
            print(f"⚠️  Could not parse timestamp column")
    
    print(f"✅ Final dataset: {len(df)} valid rows")
    return df


def engineer_features(df, window_size=5):
    """
    Engineer temporal features from sensor data.
    
    Features created:
        - temperature_c (original)
        - humidity_pct (original)
        - heat_index_c (original or computed)
        - temp_change_rate (difference from previous reading)
        - humidity_change_rate (difference from previous reading)
        - temp_rolling_std (rolling standard deviation)
        - humidity_rolling_std (rolling standard deviation)
    """
    print(f"🔧 Engineering temporal features (window={window_size})...")
    
    # Create copies for feature computation
    df = df.copy()
    
    # 1. Change rates (gradient)
    df['temp_change_rate'] = df['temperature_c'].diff().fillna(0)
    df['humidity_change_rate'] = df['humidity_pct'].diff().fillna(0)
    
    # 2. Rolling statistics (measure of instability)
    df['temp_rolling_std'] = df['temperature_c'].rolling(
        window=window_size, min_periods=1, center=False
    ).std().fillna(0)
    
    df['humidity_rolling_std'] = df['humidity_pct'].rolling(
        window=window_size, min_periods=1, center=False
    ).std().fillna(0)
    
    # Feature names in order (must match model)
    feature_columns = [
        'temperature_c',
        'humidity_pct',
        'heat_index',
        'temp_change_rate',
        'humidity_change_rate',
        'temp_rolling_std',
        'humidity_rolling_std'
    ]
    
    # Extract feature matrix
    X = df[feature_columns].values
    
    # Convert labels to numeric
    label_mapping = {
        'Low Risk': 0,
        'Moderate Risk': 1,
        'Severe Risk': 2
    }
    y = df['risk_level'].map(label_mapping).values
    
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"   Features: {feature_columns}")
    
    return X, y, feature_columns


def train_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train XGBoost classifier.
    """
    print(f"\n Training XGBoost Classifier...")
    print(f"   Parameters: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print(f"✅ Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    print(f"\n📊 Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.2%}\n")
    
    # Classification report
    target_names = ['Low Risk', 'Moderate Risk', 'Severe Risk']
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  Low   Mod   Sev")
    for i, label in enumerate(target_names):
        print(f"Actual {label:12s}  {cm[i][0]:3d}   {cm[i][1]:3d}   {cm[i][2]:3d}")
    
    # Feature importance
    print("\n🔍 Feature Importance:")
    feature_names = [
        'temperature_c', 'humidity_pct', 'heat_index',
        'temp_change_rate', 'humidity_change_rate',
        'temp_rolling_std', 'humidity_rolling_std'
    ]
    
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"   {name:25s}: {importance:.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Train XGBoost hazard classifier from sensor data CSV'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (e.g., data/training/sensor_data.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/hazard_xgboost_model.json',
        help='Path to save trained model (default: models/hazard_xgboost_model.json)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Window size for rolling statistics (default: 5)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of boosting rounds (default: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 60)
    print("  XGBoost Hazard Classifier Training")
    print("=" * 60 + "\n")
    
    # 1. Load and validate data
    df = load_and_validate_csv(args.input)
    
    if len(df) < 100:
        print(f"⚠️  WARNING: Only {len(df)} samples available.")
        print(f"   Consider collecting more data for better model performance.")
        print(f"   Recommended: At least 500-1000 samples")
    
    # 2. Engineer features
    X, y, feature_names = engineer_features(df, window_size=args.window_size)
    
    # 3. Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = ['Low Risk', 'Moderate Risk', 'Severe Risk']
    print(f"\n📋 Class Distribution:")
    for label_id, count in zip(unique, counts):
        print(f"   {label_names[label_id]:15s}: {count:4d} ({count/len(y)*100:5.1f}%)")
    
    # Warn if highly imbalanced
    min_count = min(counts)
    if min_count < 10:
        print(f"⚠️  WARNING: Class imbalance detected!")
        print(f"   Smallest class has only {min_count} samples.")
        print(f"   Model may not perform well on rare classes.")
    
    # 4. Train/test split
    print(f"\n🔀 Splitting data: {int((1-args.test_size)*100)}% train, {int(args.test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y  # Maintain class balance in splits
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    
    # 5. Train model
    model = train_model(
        X_train, y_train, X_test, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    
    # 6. Evaluate
    evaluate_model(model, X_test, y_test)
    
    # 7. Save model - UPDATED SECTION
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try sklearn-style save first (for compatibility)
        model.save_model(str(output_path))
        print(f"\n💾 Model saved to: {output_path}")
    except (TypeError, AttributeError) as e:
        # Fallback to native XGBoost save (more robust)
        print(f"⚠️  sklearn-style save failed: {e}")
        print(f"   Using native XGBoost format instead...")
        model.get_booster().save_model(str(output_path))
        print(f"💾 Model saved to: {output_path}")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model file: {output_path}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.1%}")
    print("\nNext steps:")
    print(f"1. Model is ready at: {output_path}")
    print(f"2. Restart app.py to use new model")
    print(f"3. Monitor confidence scores in dashboard")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()