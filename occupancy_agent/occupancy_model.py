"""
occupancy_model.py
===================
Trains and saves two separate scikit-learn models:

1. SENSOR ESTIMATOR — predicts how many people are in a room RIGHT NOW
   from live ambient sensor readings (temperature, light, sound, CO2,
   motion). Trained on your real Occupancy_Estimation.csv (UCI Room
   Occupancy Estimation dataset — one instrumented room, 10,129
   readings). This is genuine sensor-based occupancy detection.

2. FORECASTER — predicts a zone's expected occupancy % for a future
   time slot, given zone, day-of-week, and hour. Trained on the
   synthetic multi-zone history from generate_occupancy_data.py. This
   is what answers "what will tomorrow's occupancy look like" per your
   Milestone 3 forecasting example.

Run `python generate_occupancy_data.py` first if dataset/occupancy_zones.csv
doesn't exist yet, then run this file.

Usage:
    python occupancy_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
MODELS_DIR = BASE_DIR / "models"

SENSOR_CSV_PATH = DATASET_DIR / "Occupancy_Estimation.csv"
ZONES_CSV_PATH = DATASET_DIR / "occupancy_zones.csv"

SENSOR_MODEL_PATH = MODELS_DIR / "sensor_estimator.joblib"
FORECAST_MODEL_PATH = MODELS_DIR / "forecast_model.joblib"

SENSOR_FEATURE_COLUMNS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope",
    "S6_PIR", "S7_PIR",
]


# ==========================================================
# MODEL 1: SENSOR-BASED LIVE OCCUPANCY ESTIMATOR
# ==========================================================

def train_sensor_estimator():
    df = pd.read_csv(SENSOR_CSV_PATH)

    X = df[SENSOR_FEATURE_COLUMNS]
    y = df["Room_Occupancy_Count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": SENSOR_FEATURE_COLUMNS},
        SENSOR_MODEL_PATH,
    )

    print(f"[Sensor Estimator] test accuracy: {accuracy:.3f}  "
          f"(trained on {len(df):,} readings)")
    print(f"[Sensor Estimator] saved to {SENSOR_MODEL_PATH}")

    return model, accuracy


# ==========================================================
# MODEL 2: PER-ZONE OCCUPANCY FORECASTER
# ==========================================================

def build_forecast_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["occupancy_pct"] = (df["occupancy_count"] / df["capacity"] * 100).clip(0, 100)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def train_forecast_model():
    df = pd.read_csv(ZONES_CSV_PATH)
    df = build_forecast_features(df)

    feature_columns = ["zone", "floor", "hour", "day_of_week", "is_weekend"]
    X = df[feature_columns]
    y = df["occupancy_pct"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("zone_onehot", OneHotEncoder(handle_unknown="ignore"), ["zone"]),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=14,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": pipeline, "feature_columns": feature_columns},
        FORECAST_MODEL_PATH,
    )

    print(f"[Forecaster] test MAE: {mae:.2f} percentage points  "
          f"(trained on {len(df):,} zone-hour records)")
    print(f"[Forecaster] saved to {FORECAST_MODEL_PATH}")

    return pipeline, mae


if __name__ == "__main__":
    train_sensor_estimator()
    print()
    train_forecast_model()