#!/usr/bin/env python3
"""
Convert sensor CSV data into textualized format for NLP training.
This creates the bridge between your numeric sensor data and text-based classification.

Usage:
    python generate_text_dataset.py --input data/training/sensor_data.csv --output data/training/text_hazard_data.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import random

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


def describe_heat_index(hi):
    """Generate descriptive text for heat index"""
    if hi < 27:
        return "heat index is normal"
    elif hi < 32:
        return "heat index is elevated"
    elif hi < 40:
        return "heat index is high"
    else:
        return "heat index exceeds threshold"


def describe_temperature(temp):
    """Generate descriptive text for temperature"""
    if temp < 24:
        return "temperature is cool"
    elif temp < 27:
        return "temperature is comfortable"
    elif temp < 32:
        return "temperature is warm"
    else:
        return "temperature is hot"


def describe_humidity(hum):
    """Generate descriptive text for humidity"""
    if hum < 50:
        return "humidity is low"
    elif hum < 60:
        return "humidity is moderate"
    elif hum < 70:
        return "humidity is high"
    elif hum < 80:
        return "humidity is very high"
    else:
        return "humidity is extreme"


def describe_trend(current, previous, metric_name):
    """Describe temporal trends"""
    if previous is None or pd.isna(previous):
        return ""
    
    change = current - previous
    
    if abs(change) < 0.5:  # Stable
        return f"{metric_name} stable"
    elif change > 0:
        if change > 2:
            return f"{metric_name} rising rapidly"
        else:
            return f"{metric_name} rising"
    else:
        if change < -2:
            return f"{metric_name} dropping rapidly"
        else:
            return f"{metric_name} dropping"


def textualize_reading(row, prev_row=None, style='detailed'):
    """
    Convert a sensor reading into natural language text.
    
    Multiple template styles for data augmentation:
    - 'detailed': Full description with all metrics
    - 'concise': Brief statement
    - 'trend': Focus on changes
    - 'alarm': Warning-style language
    """
    temp = row['temperature_c']
    hum = row['humidity_pct']
    hi = row.get('heat_index', calculate_heat_index(temp, hum))
    
    # Get previous values for trends
    prev_temp = prev_row['temperature_c'] if prev_row is not None else None
    prev_hum = prev_row['humidity_pct'] if prev_row is not None else None
    
    templates = {
        'detailed': [
            f"Temperature is {temp:.1f}°C, humidity is {hum:.1f}%, {describe_heat_index(hi)}.",
            f"Current conditions: {temp:.1f}°C temperature, {hum:.1f}% humidity, {describe_heat_index(hi)}.",
            f"Sensor readings show {temp:.1f} degrees Celsius, {hum:.1f} percent humidity, {describe_heat_index(hi)}.",
        ],
        'concise': [
            f"{describe_temperature(temp).capitalize()} at {temp:.1f}°C, {describe_humidity(hum)}.",
            f"Environment: {temp:.1f}°C, {hum:.1f}%, {describe_heat_index(hi)}.",
            f"{temp:.1f}°C and {hum:.1f}% humidity detected.",
        ],
        'trend': [
            f"{describe_trend(temp, prev_temp, 'Temperature')} to {temp:.1f}°C. {describe_trend(hum, prev_hum, 'Humidity')} to {hum:.1f}%.",
            f"Temperature at {temp:.1f}°C ({describe_trend(temp, prev_temp, 'trend')}), humidity at {hum:.1f}%.",
        ],
        'alarm': [
            f"Alert: Temperature {temp:.1f}°C, humidity {hum:.1f}%. {describe_heat_index(hi).capitalize()}.",
            f"Warning: {describe_temperature(temp)}, {describe_humidity(hum)}. Heat index concern.",
        ]
    }
    
    # Select random template from chosen style
    if style in templates:
        return random.choice(templates[style])
    else:
        # Fallback to detailed
        return random.choice(templates['detailed'])


def generate_text_dataset(csv_path, output_path, augment=True):
    """
    Convert sensor CSV to text-based dataset.
    
    Args:
        csv_path: Path to input CSV with sensor data
        output_path: Path to save text dataset
        augment: If True, create multiple text variations per reading
    """
    print(f"📂 Loading sensor data from: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    
    # Normalize column names
    column_mapping = {}
    temp_candidates = ['temperature', 'temp', 'temperature_c', 'temp_c']
    for col in df.columns:
        if col in temp_candidates:
            column_mapping[col] = 'temperature_c'
            break
    
    humidity_candidates = ['humidity', 'humidity_pct', 'rh', 'relative_humidity']
    for col in df.columns:
        if col in humidity_candidates:
            column_mapping[col] = 'humidity_pct'
            break
    
    risk_candidates = ['risk_level', 'hazard_level', 'hazard_overall']
    for col in df.columns:
        if col in risk_candidates:
            column_mapping[col] = 'risk_level'
            break
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Convert to numeric
    df['temperature_c'] = pd.to_numeric(df['temperature_c'], errors='coerce')
    df['humidity_pct'] = pd.to_numeric(df['humidity_pct'], errors='coerce')
    df.dropna(subset=['temperature_c', 'humidity_pct'], inplace=True)
    
    # Compute heat index if missing
    if 'heat_index' not in df.columns:
        df['heat_index'] = df.apply(
            lambda row: calculate_heat_index(row['temperature_c'], row['humidity_pct']),
            axis=1
        )
    
    # Normalize risk levels
    risk_mapping = {
        'low risk': 'Low',
        'lowrisk': 'Low',
        'low': 'Low',
        'moderate risk': 'Moderate',
        'moderaterisk': 'Moderate',
        'moderate': 'Moderate',
        'mod': 'Moderate',
        'severe risk': 'Severe',
        'severerisk': 'Severe',
        'severe': 'Severe',
        'high': 'Severe',
        'high risk': 'Severe'
    }
    df['risk_level'] = df['risk_level'].str.strip().str.lower().map(risk_mapping)
    df.dropna(subset=['risk_level'], inplace=True)
    
    print(f"✅ Loaded {len(df)} valid sensor readings")
    
    # Generate text descriptions
    text_data = []
    styles = ['detailed', 'concise', 'trend', 'alarm'] if augment else ['detailed']
    
    print(f"🔤 Generating text descriptions (augment={augment})...")
    
    for idx, row in df.iterrows():
        prev_row = df.iloc[idx - 1] if idx > 0 else None
        
        if augment:
            # Create multiple variations per reading
            for style in styles:
                text = textualize_reading(row, prev_row, style)
                text_data.append({
                    'text': text,
                    'label': row['risk_level'],
                    'temperature': row['temperature_c'],
                    'humidity': row['humidity_pct'],
                    'heat_index': row['heat_index']
                })
        else:
            # Single variation
            text = textualize_reading(row, prev_row, 'detailed')
            text_data.append({
                'text': text,
                'label': row['risk_level'],
                'temperature': row['temperature_c'],
                'humidity': row['humidity_pct'],
                'heat_index': row['heat_index']
            })
    
    # Create DataFrame
    text_df = pd.DataFrame(text_data)
    
    # Shuffle
    text_df = text_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Text dataset created!")
    print(f"   Output: {output_path}")
    print(f"   Total samples: {len(text_df)}")
    print(f"\n📊 Class Distribution:")
    print(text_df['label'].value_counts())
    print(f"\n📝 Sample texts:")
    for i, row in text_df.head(5).iterrows():
        print(f"   [{row['label']}] {row['text']}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert sensor CSV to text-based NLP dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with sensor data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/training/text_hazard_data.csv',
        help='Path to save text dataset (default: data/training/text_hazard_data.csv)'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation (creates only one text per reading)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  Text Dataset Generator for NLP Hazard Classification")
    print("=" * 60 + "\n")
    
    generate_text_dataset(
        csv_path=args.input,
        output_path=args.output,
        augment=not args.no_augment
    )
    
    print("\n" + "=" * 60)
    print("✅ GENERATION COMPLETE!")
    print("=" * 60)
    print("Next step: Train NLP models using train_nlp_models.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()