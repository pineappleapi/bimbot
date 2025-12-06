
#!/usr/bin/env python3
"""
Underground Mine Environmental Hazard Classifier
------------------------------------------------
Reads temperature & humidity CSV data collected by a robot, preprocesses,
compares against safety thresholds, and classifies each record into:
  - Low Hazard
  - Moderate Hazard
  - Severe Risk

Usage (examples):
    python hazard_classification.py --input robot_env.csv --output robot_env_classified.csv
    python hazard_classification.py --input robot_env.csv --resample 30S --smoothing-window 5
    python hazard_classification.py --input robot_env.csv --thresholds thresholds.json
    python hazard_classification.py --input robot_env.csv --no-heat-index

CSV schema (flexible):
    timestamp, temperature_c, humidity_pct
    2025-11-29T12:00:05Z, 28.4, 72.1
"""

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Default thresholds (PLACEHOLDERS — replace with site-approved limits)
# ---------------------------
DEFAULT_THRESHOLDS = {
    "temperature_c": {
        "low_max": 27.0,      # ≤ 27°C → Low Hazard
        "moderate_max": 32.0  # 27–32°C → Moderate Hazard; > 32°C → Severe Risk
    },
    "humidity_pct": {
        "low_max": 60.0,      # ≤ 60% → Low Hazard
        "moderate_max": 80.0  # 60–80% → Moderate Hazard; > 80% → Severe Risk
    },
    "heat_index_c": {
        "low_max": 32.0,      # ≤ 32°C → Low Hazard
        "moderate_max": 40.0  # 32–40°C → Moderate Hazard; > 40°C → Severe Risk
    }
}

# Physically plausible clamps (example bounds)
DEFAULT_CLAMPS = {
    "temperature_c": (-10.0, 60.0),
    "humidity_pct": (0.0, 100.0)
}

# Sensor calibration offsets
DEFAULT_CALIBRATION = {
    "temperature_c": 0.0,
    "humidity_pct": 0.0
}

# Column candidates for flexible schema resolution
COLUMN_MAP = {
    "timestamp": ["timestamp", "time", "ts", "datetime"],
    "temperature_c": ["temperature_c", "temp_c", "temperature", "temp"],
    "humidity_pct": ["humidity_pct", "rh", "humidity", "relative_humidity"]
}

ORDER = {"Low Hazard": 0, "Moderate Hazard": 1, "Severe Risk": 2}
RISK_SCORE = {"Low Hazard": 25, "Moderate Hazard": 60, "Severe Risk": 90}


# ---------------------------
# Logging setup
# ---------------------------
def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------
# Utility: resolve columns
# ---------------------------
def resolve_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the expected columns found: {candidates}. Got columns: {list(df.columns)}")


# ---------------------------
# Heat Index (Rothfusz regression)
# ---------------------------
def heat_index_c(temp_c: float, rh: float) -> float:
    """
    Compute Heat Index in °C from temperature (°C) and relative humidity (%).
    Uses Rothfusz regression (Fahrenheit base) with conservative adjustments.
    Returns NaN if inputs are NaN.
    """
    if pd.isna(temp_c) or pd.isna(rh):
        return np.nan
    T = temp_c * 9 / 5 + 32.0
    R = rh
    HI_f = (-42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
            - 0.00683783 * T ** 2 - 0.05481717 * R ** 2 + 0.00122874 * T ** 2 * R
            + 0.00085282 * T * R ** 2 - 0.00000199 * T ** 2 * R ** 2)
    # Adjustments
    if R < 13 and 80 <= T <= 112:
        adj = ((13 - R) / 4) * np.sqrt((17 - abs(T - 95.0)) / 17)
        HI_f -= adj
    elif R > 85 and 80 <= T <= 87:
        adj = ((R - 85) / 10) * ((87 - T) / 5)
        HI_f += adj
    return (HI_f - 32.0) * 5 / 9


# ---------------------------
# Hazard classification helpers
# ---------------------------
def classify_metric(value: float, bounds: Dict[str, float]) -> str:
    """
    Returns: 'Low Hazard' | 'Moderate Hazard' | 'Severe Risk'
    """
    if pd.isna(value):
        return np.nan
    if value <= bounds["low_max"]:
        return "Low Hazard"
    elif value <= bounds["moderate_max"]:
        return "Moderate Hazard"
    else:
        return "Severe Risk"


def aggregate_classification(labels: List[Optional[str]]) -> Optional[str]:
    labels = [lbl for lbl in labels if isinstance(lbl, str)]
    if not labels:
        return np.nan
    return max(labels, key=lambda x: ORDER.get(x, -1))


# ---------------------------
# Preprocess pipeline
# ---------------------------
def preprocess_environment(
    df: pd.DataFrame,
    calibration: Dict[str, float],
    clamps: Dict[str, Tuple[float, float]],
    smoothing_window: int = 5,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """
    - Parse timestamp (UTC), sort, deduplicate
    - Coerce numerics
    - Apply calibration & clamps
    - Interpolate gaps
    - Optional resample by time interval (e.g., '30S', '1Min')
    - Rolling median smoothing
    Returns DataFrame with columns: ['temperature_c', 'humidity_pct'] indexed by timestamp.
    """
    ts_col = resolve_column(df, COLUMN_MAP["timestamp"])
    t_col = resolve_column(df, COLUMN_MAP["temperature_c"])
    h_col = resolve_column(df, COLUMN_MAP["humidity_pct"])

    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    before = len(df)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(subset=[ts_col])
    logging.info(f"Time parsed. Dropped {before - len(df)} rows due to invalid timestamps/duplicates.")

    df = df.set_index(ts_col)

    # Numerics
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[h_col] = pd.to_numeric(df[h_col], errors="coerce")

    # Calibration
    df[t_col] = df[t_col] + calibration.get("temperature_c", 0.0)
    df[h_col] = df[h_col] + calibration.get("humidity_pct", 0.0)

    # Clamps
    t_min, t_max = clamps.get("temperature_c", (-np.inf, np.inf))
    h_min, h_max = clamps.get("humidity_pct", (-np.inf, np.inf))
    df[t_col] = df[t_col].clip(t_min, t_max)
    df[h_col] = df[h_col].clip(h_min, h_max)

    # Interpolate
    df[t_col] = df[t_col].interpolate(limit_direction="both")
    df[h_col] = df[h_col].interpolate(limit_direction="both")

    # Optional resample
    if resample:
        df = df.resample(resample).mean().interpolate(limit_direction="both")
        logging.info(f"Resampled to '{resample}'.")

    # Rolling median smoothing
    df[t_col] = df[t_col].rolling(window=smoothing_window, min_periods=1, center=True).median()
    df[h_col] = df[h_col].rolling(window=smoothing_window, min_periods=1, center=True).median()
    logging.info(f"Smoothing applied (window={smoothing_window}).")

    # Return with canonical names
    return df.rename(columns={t_col: "temperature_c", h_col: "humidity_pct"})


# ---------------------------
# Classification pipeline
# ---------------------------
def classify_hazards(
    df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    compute_heat_index: bool = True,
) -> pd.DataFrame:
    """
    Adds columns:
      - heat_index_c (optional)
      - hazard_temperature
      - hazard_humidity
      - hazard_heat_index (optional)
      - hazard_overall
      - risk_score (0–100 style bucket)
    """
    if compute_heat_index:
        df["heat_index_c"] = [heat_index_c(t, rh) for t, rh in zip(df["temperature_c"], df["humidity_pct"])]

    df["hazard_temperature"] = df["temperature_c"].apply(lambda v: classify_metric(v, thresholds["temperature_c"]))
    df["hazard_humidity"] = df["humidity_pct"].apply(lambda v: classify_metric(v, thresholds["humidity_pct"]))
    metrics = ["hazard_temperature", "hazard_humidity"]

    if compute_heat_index:
        df["hazard_heat_index"] = df["heat_index_c"].apply(lambda v: classify_metric(v, thresholds["heat_index_c"]))
        metrics.append("hazard_heat_index")

    df["hazard_overall"] = df[metrics].apply(lambda row: aggregate_classification(list(row.values)), axis=1)
    df["risk_score"] = df["hazard_overall"].map(RISK_SCORE)

    return df


# ---------------------------
# Thresholds loader (JSON optional)
# ---------------------------
def load_thresholds(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return DEFAULT_THRESHOLDS
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Basic validation
        for key in ["temperature_c", "humidity_pct"]:
            if key not in data or not all(k in data[key] for k in ["low_max", "moderate_max"]):
                raise ValueError(f"Thresholds JSON missing required keys for '{key}'.")
        # heat_index optional but recommended
        if "heat_index_c" not in data:
            logging.warning("No 'heat_index_c' thresholds provided; using defaults for heat index.")
            data["heat_index_c"] = DEFAULT_THRESHOLDS["heat_index_c"]
        return data
    except Exception as e:
        logging.error(f"Failed to load thresholds from '{path}': {e}. Using defaults.")
        return DEFAULT_THRESHOLDS


# ---------------------------
# CLI
# ---------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify underground mine environmental hazards from robot-collected temperature & humidity data."
    )
    parser.add_argument("--input", required=True, help="Input CSV file path (e.g., robot_env.csv)")
    parser.add_argument("--output", default=None, help="Output CSV file path (e.g., robot_env_classified.csv)")
    parser.add_argument("--thresholds", default=None, help="Path to thresholds JSON (optional)")
    parser.add_argument("--resample", default=None, help="Time resample interval (e.g., '30S', '1Min') (optional)")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Rolling median window size (default: 5)")
    parser.add_argument("--no-heat-index", action="store_true", help="Disable heat index computation")
    parser.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    return parser.parse_args(argv)


# ---------------------------
# Main
# ---------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=not args.quiet)

    logging.info("Starting hazard classification pipeline.")
    logging.info("Flow: Read -> Preprocess -> Threshold Comparison -> Hazard Classification")

    # Load CSV
    try:
        raw = pd.read_csv(args.input)
        logging.info(f"Loaded input CSV: {args.input} (rows={len(raw)})")
    except Exception as e:
        logging.error(f"Failed to read input CSV '{args.input}': {e}")
        return 1

    # Load thresholds
    thresholds = load_thresholds(args.thresholds)
    if args.thresholds:
        logging.info(f"Using thresholds from: {args.thresholds}")
    else:
        logging.info("Using default thresholds (PLACEHOLDER — confirm with safety officer).")

    # Preprocess
    pre = preprocess_environment(
        raw,
        calibration=DEFAULT_CALIBRATION,
        clamps=DEFAULT_CLAMPS,
        smoothing_window=args.smoothing_window,
        resample=args.resample,
    )

    # Classify
    out = classify_hazards(
        pre,
        thresholds=thresholds,
        compute_heat_index=(not args.no_heat_index),
    )

    # Output
    if args.output:
        try:
            out.reset_index().to_csv(args.output, index=False)
            logging.info(f"Saved classified output: {args.output}")
        except Exception as e:
            logging.error(f"Failed to write output CSV '{args.output}': {e}")
            return 1
    else:
        # Show a preview if no output path specified
        logging.info("No output path provided. Showing preview:")
        print(out.head(10))

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
