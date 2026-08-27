import os
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(
    title="Agentic FacilityOps AI Platform"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

ENERGY_CSV_PATH = (
    BASE_DIR.parent
    / "dataset"
    / "cleaned_energy_consumption.csv"
)

MAINTENANCE_CSV_PATH = (
    BASE_DIR.parent
    / "dataset"
    / "predictive_maintenance_dataset.csv"
)

OCCUPANCY_CSV_PATH = (
    BASE_DIR.parent
    / "dataset"
    / "occupancy_zones.csv"
)

SENSOR_MODEL_PATH = (
    BASE_DIR.parent
    / "occupancy_agent"
    / "models"
    / "sensor_estimator.joblib"
)

FORECAST_MODEL_PATH = (
    BASE_DIR.parent
    / "occupancy_agent"
    / "models"
    / "forecast_model.joblib"
)


# ==========================================================
# IN-MEMORY STATE
# ==========================================================

energy_df = pd.DataFrame()
maintenance_df = pd.DataFrame()
occupancy_df = pd.DataFrame()

average_energy = 0.0

_energy_mtime = None
_maintenance_mtime = None
_occupancy_mtime = None

sensor_estimator_bundle = None
forecast_model_bundle = None


# ==========================================================
# DATA HELPERS
# ==========================================================

def safe_records(df):
    """
    Convert DataFrame rows to JSON-safe records.
    NaN values are converted to None. Datetime columns are converted
    to ISO strings explicitly first, since pandas' NaT.isoformat()
    quirkily returns the literal string "NaT" instead of raising or
    being treated as null by FastAPI's default encoder.
    """
    df = df.copy()

    for column in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[column] = df[column].apply(
            lambda value: value.isoformat() if pd.notnull(value) else None
        )

    return (
        df.where(pd.notnull(df), None)
        .to_dict(orient="records")
    )


def get_hour_column(df):
    """
    Return the hour for each energy record.
    Uses Hour when available, otherwise extracts it from Timestamp.
    """
    if "Hour" in df.columns:
        return pd.to_numeric(
            df["Hour"],
            errors="coerce"
        )

    if "Timestamp" in df.columns:
        timestamp = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )
        return timestamp.dt.hour

    return pd.Series(
        np.nan,
        index=df.index
    )


def get_weekend_column(df):
    """
    Return True or False for each energy record.
    Supports Weekend, DayOfWeek, or Timestamp columns.
    """
    if "Weekend" in df.columns:
        values = (
            df["Weekend"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return values.isin([
            "yes",
            "true",
            "1",
            "y"
        ])

    if "DayOfWeek" in df.columns:
        day_numbers = pd.to_numeric(
            df["DayOfWeek"],
            errors="coerce"
        )

        return day_numbers >= 5

    if "Timestamp" in df.columns:
        timestamps = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        return timestamps.dt.dayofweek >= 5

    return pd.Series(
        False,
        index=df.index
    )


# ==========================================================
# ENERGY PREDICTION
# ==========================================================

def predict_energy_consumption(
    df,
    hour,
    outdoor_temperature,
    occupancy,
    weekend
):
    work = df.copy()

    work["_hour"] = get_hour_column(work)

    if "Temperature" in work.columns:
        work["_temperature"] = pd.to_numeric(
            work["Temperature"],
            errors="coerce"
        )
    else:
        work["_temperature"] = np.nan

    if "Occupancy" in work.columns:
        work["_occupancy"] = pd.to_numeric(
            work["Occupancy"],
            errors="coerce"
        )
    else:
        work["_occupancy"] = np.nan

    work["_weekend"] = get_weekend_column(work)

    work["EnergyConsumption"] = pd.to_numeric(
        work["EnergyConsumption"],
        errors="coerce"
    )

    work = work.dropna(
        subset=[
            "_hour",
            "_temperature",
            "_occupancy",
            "EnergyConsumption"
        ]
    )

    if work.empty:
        return 0.0

    temperature_std = max(
        float(work["_temperature"].std()),
        1.0
    )

    occupancy_std = max(
        float(work["_occupancy"].std()),
        1.0
    )

    hour_difference = np.minimum(
        abs(work["_hour"] - hour),
        24 - abs(work["_hour"] - hour)
    )

    distance = (
        (hour_difference / 6.0) ** 2
        + (
            (
                work["_temperature"]
                - outdoor_temperature
            ) / temperature_std
        ) ** 2
        + (
            (
                work["_occupancy"]
                - occupancy
            ) / occupancy_std
        ) ** 2
        + np.where(
            work["_weekend"] == weekend,
            0,
            2
        )
    )

    nearest = (
        work.assign(_distance=distance)
        .nsmallest(
            min(50, len(work)),
            "_distance"
        )
    )

    weights = 1 / (
        nearest["_distance"] + 0.25
    )

    prediction = np.average(
        nearest["EnergyConsumption"],
        weights=weights
    )

    return round(float(prediction), 2)


# ==========================================================
# HEALTH SCORE
# ==========================================================

def add_health_score(df):
    score = pd.Series(
        100,
        index=df.index
    )

    score = score - np.where(df["failure"] == 1, 50, 0)
    score = score - np.where(df["metric5"] > 15, 10, 0)
    score = score - np.where(df["metric6"] > 300000, 15, 0)
    score = score - np.where(df["metric9"] > 500, 10, 0)
    score = score - np.where(df["metric3"] > 100, 10, 0)
    score = score - np.where(df["metric4"] > 20, 5, 0)

    df["Health Score"] = score.clip(lower=0)

    return df


def status_series(score_series):
    return np.select(
        [score_series >= 80, score_series >= 50],
        ["Healthy", "Warning"],
        default="Critical"
    )


# ==========================================================
# MAINTENANCE RECORD HELPERS (id + numeric coercion)
# ==========================================================

NON_METRIC_COLUMNS = ["id", "device", "failure", "Health Score", "Status"]


def ensure_id_column(df):
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df["id"] = df.index + 1

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    return df


def metric_columns(df):
    return [column for column in df.columns if column not in NON_METRIC_COLUMNS]


def coerce_maintenance_types(df):
    for column in metric_columns(df):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if "failure" in df.columns:
        df["failure"] = pd.to_numeric(df["failure"], errors="coerce").fillna(0).astype(int)

    if "device" in df.columns:
        df["device"] = df["device"].astype(str)

    return df


def persist_maintenance_df(df):
    global maintenance_df
    global _maintenance_mtime

    df = coerce_maintenance_types(df)
    df = df.drop(columns=["Health Score"], errors="ignore")
    df = add_health_score(df)

    df.to_csv(MAINTENANCE_CSV_PATH, index=False)

    maintenance_df = df
    _maintenance_mtime = os.path.getmtime(MAINTENANCE_CSV_PATH)

    return df


# ==========================================================
# DATA LOADING
# ==========================================================

def load_energy_data(force=False):
    global energy_df
    global average_energy
    global _energy_mtime

    if not ENERGY_CSV_PATH.exists():
        raise FileNotFoundError(f"Energy dataset not found at {ENERGY_CSV_PATH}")

    current_mtime = os.path.getmtime(ENERGY_CSV_PATH)

    if force or _energy_mtime != current_mtime:
        df = pd.read_csv(ENERGY_CSV_PATH)
        df = ensure_id_column(df)
        df["EnergyConsumption"] = pd.to_numeric(df["EnergyConsumption"], errors="coerce")

        energy_df = df
        average_energy = round(float(df["EnergyConsumption"].mean()), 4)
        _energy_mtime = current_mtime

    return energy_df


def column_default_value(df, column):
    series = df[column].dropna()

    if series.empty:
        return None

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())

    mode = series.mode()

    return mode.iloc[0] if not mode.empty else None


def persist_energy_df(df):
    df.to_csv(ENERGY_CSV_PATH, index=False)
    return load_energy_data(force=True)


def load_maintenance_data(force=False):
    global maintenance_df
    global _maintenance_mtime

    if not MAINTENANCE_CSV_PATH.exists():
        raise FileNotFoundError(f"Maintenance dataset not found at {MAINTENANCE_CSV_PATH}")

    current_mtime = os.path.getmtime(MAINTENANCE_CSV_PATH)

    if force or _maintenance_mtime != current_mtime:
        df = pd.read_csv(MAINTENANCE_CSV_PATH)
        df = ensure_id_column(df)
        df = add_health_score(df)

        maintenance_df = df
        _maintenance_mtime = current_mtime

    return maintenance_df


def ensure_occupancy_id_column(df):
    if "occupancy_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["occupancy_id"] = df.index + 1

    df["occupancy_id"] = pd.to_numeric(df["occupancy_id"], errors="coerce")
    df = df.dropna(subset=["occupancy_id"])
    df["occupancy_id"] = df["occupancy_id"].astype(int)

    return df


def load_occupancy_data(force=False):
    global occupancy_df
    global _occupancy_mtime

    if not OCCUPANCY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Occupancy dataset not found at {OCCUPANCY_CSV_PATH}. "
            f"Run occupancy_agent/generate_occupancy_data.py first."
        )

    current_mtime = os.path.getmtime(OCCUPANCY_CSV_PATH)

    if force or _occupancy_mtime != current_mtime:
        df = pd.read_csv(OCCUPANCY_CSV_PATH)
        df = ensure_occupancy_id_column(df)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["occupancy_count"] = pd.to_numeric(df["occupancy_count"], errors="coerce")
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

        df["occupancy_pct"] = ((df["occupancy_count"] / df["capacity"]) * 100).round(2)

        occupancy_df = df
        _occupancy_mtime = current_mtime

    return occupancy_df


def persist_occupancy_df(df):
    df = df.drop(columns=["occupancy_pct"], errors="ignore")
    df.to_csv(OCCUPANCY_CSV_PATH, index=False)
    return load_occupancy_data(force=True)


def load_occupancy_models():
    global sensor_estimator_bundle
    global forecast_model_bundle

    if SENSOR_MODEL_PATH.exists():
        sensor_estimator_bundle = joblib.load(SENSOR_MODEL_PATH)

    if FORECAST_MODEL_PATH.exists():
        forecast_model_bundle = joblib.load(FORECAST_MODEL_PATH)


@app.on_event("startup")
def startup_load():
    load_energy_data(force=True)
    load_maintenance_data(force=True)
    load_occupancy_data(force=True)
    load_occupancy_models()
    load_security_data(force=True)


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():
    return {"message": "Welcome to Agentic FacilityOps AI Platform"}


@app.get("/refresh")
def refresh():
    load_energy_data(force=True)
    load_maintenance_data(force=True)

    return {
        "message": "Datasets reloaded",
        "energy_records": len(energy_df),
        "maintenance_records": len(maintenance_df)
    }


# ==========================================================
# ENERGY ENDPOINTS
# ==========================================================

@app.get("/energy")
def get_energy():
    df = load_energy_data()
    return safe_records(df)


@app.get("/statistics")
def statistics():
    df = load_energy_data()
    return {
        "Average Energy": round(df["EnergyConsumption"].mean(), 2),
        "Maximum Energy": round(df["EnergyConsumption"].max(), 2),
        "Minimum Energy": round(df["EnergyConsumption"].min(), 2)
    }


@app.get("/energy/simulate")
def simulate_energy(
    hour: int = Query(..., ge=0, le=23),
    outdoor_temperature: float = Query(...),
    occupancy: int = Query(..., ge=0),
    weekend: str = Query("No")
):
    df = load_energy_data()

    weekend_value = weekend.strip().lower() in ["yes", "true", "1", "y"]

    predicted_consumption = predict_energy_consumption(
        df=df,
        hour=hour,
        outdoor_temperature=outdoor_temperature,
        occupancy=occupancy,
        weekend=weekend_value
    )

    return {
        "Timestamp": datetime.now().isoformat(),
        "Hour": hour,
        "Temperature": outdoor_temperature,
        "Occupancy": occupancy,
        "Weekend": "Yes" if weekend_value else "No",
        "EnergyConsumption": predicted_consumption,
        "HVACUsage": "On" if (outdoor_temperature < 20 or outdoor_temperature > 26) else "Off",
        "LightingUsage": "On" if occupancy > 0 else "Off",
        "Simulated": True
    }


@app.get("/alerts")
def alerts():
    df = load_energy_data()
    threshold = average_energy * 1.2

    flagged = df[df["EnergyConsumption"] > threshold].copy()
    flagged["Alert"] = "High Energy Consumption"

    if "Timestamp" in flagged.columns:
        flagged["Timestamp"] = flagged["Timestamp"].astype(str)

    return flagged[["Timestamp", "Alert"]].to_dict(orient="records")


@app.get("/recommendations")
def recommendations():
    df = load_energy_data()
    threshold = average_energy * 1.2

    hvac_mask = (df["HVACUsage"] == "On") & (df["Temperature"] < 22)
    light_mask = (df["LightingUsage"] == "On") & (df["Occupancy"] < 10)
    renewable_mask = df["RenewableEnergy"] < 10
    high_mask = df["EnergyConsumption"] > threshold

    any_mask = hvac_mask | light_mask | renewable_mask | high_mask

    timestamps = df.loc[any_mask, "Timestamp"].astype(str).to_numpy()
    hvac_flags = hvac_mask[any_mask].to_numpy()
    light_flags = light_mask[any_mask].to_numpy()
    renewable_flags = renewable_mask[any_mask].to_numpy()
    high_flags = high_mask[any_mask].to_numpy()

    recommendation_list = []

    for timestamp, hvac, lights, renewable, high in zip(
        timestamps, hvac_flags, light_flags, renewable_flags, high_flags
    ):
        current_recommendations = []

        if hvac:
            current_recommendations.append("Turn OFF HVAC")
        if lights:
            current_recommendations.append("Switch OFF unnecessary lights")
        if renewable:
            current_recommendations.append("Increase renewable energy usage")
        if high:
            current_recommendations.append("Inspect equipment with high energy consumption")

        recommendation_list.append({"Timestamp": timestamp, "Recommendations": current_recommendations})

    return recommendation_list


@app.get("/dashboard")
def dashboard():
    df = load_energy_data()
    threshold = average_energy * 1.2

    high_alerts = int(len(df[df["EnergyConsumption"] > threshold]))

    return {
        "Total Records": len(df),
        "Average Energy": round(df["EnergyConsumption"].mean(), 2),
        "Maximum Energy": round(df["EnergyConsumption"].max(), 2),
        "Minimum Energy": round(df["EnergyConsumption"].min(), 2),
        "High Energy Alerts": high_alerts
    }


# ==========================================================
# ENERGY RECORDS — ADD / REMOVE LIVE DATA
# ==========================================================

@app.post("/energy-records")
def add_energy_record(payload: dict = Body(...)):
    df = load_energy_data().copy()

    try:
        hour = int(payload.get("hour"))
        outdoor_temperature = float(payload.get("outdoor_temperature"))
        occupancy = int(payload.get("occupancy"))
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"error": "hour, outdoor_temperature and occupancy are required numeric fields"}
        )

    if not (0 <= hour <= 23):
        return JSONResponse(status_code=400, content={"error": "hour must be between 0 and 23"})

    weekend_raw = str(payload.get("weekend", "No")).strip().lower()
    weekend_value = weekend_raw in ["yes", "true", "1", "y"]

    predicted_consumption = predict_energy_consumption(
        df=df, hour=hour, outdoor_temperature=outdoor_temperature,
        occupancy=occupancy, weekend=weekend_value
    )

    new_row = {column: column_default_value(df, column) for column in df.columns if column != "id"}

    record_timestamp = datetime.now().replace(hour=hour, minute=datetime.now().minute, second=0, microsecond=0)

    if "Timestamp" in new_row:
        new_row["Timestamp"] = record_timestamp.isoformat()
    if "Hour" in new_row:
        new_row["Hour"] = hour
    if "Temperature" in new_row:
        new_row["Temperature"] = outdoor_temperature
    if "Occupancy" in new_row:
        new_row["Occupancy"] = occupancy

    if "Humidity" in new_row:
        humidity_raw = payload.get("humidity")
        try:
            new_row["Humidity"] = float(humidity_raw)
        except (TypeError, ValueError):
            pass

    if "Weekend" in new_row:
        new_row["Weekend"] = "Yes" if weekend_value else "No"
    elif "DayOfWeek" in df.columns:
        existing_days = df["DayOfWeek"].dropna()
        if not existing_days.empty and pd.api.types.is_numeric_dtype(existing_days):
            new_row["DayOfWeek"] = 5 if weekend_value else 1
        else:
            new_row["DayOfWeek"] = "Saturday" if weekend_value else "Monday"

    if "HVACUsage" in new_row:
        new_row["HVACUsage"] = "On" if (outdoor_temperature < 20 or outdoor_temperature > 26) else "Off"

    if "LightingUsage" in new_row:
        new_row["LightingUsage"] = "On" if occupancy > 0 else "Off"

    new_row["EnergyConsumption"] = predicted_consumption

    new_id = int(df["id"].max()) + 1 if not df.empty else 1
    new_row["id"] = new_id

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = persist_energy_df(df)

    created = df[df["id"] == new_id]

    return safe_records(created)[0]


@app.delete("/energy-records/{record_id}")
def delete_energy_record(record_id: int):
    df = load_energy_data().copy()

    if record_id not in df["id"].values:
        return JSONResponse(status_code=404, content={"error": f"Record {record_id} not found"})

    df = df[df["id"] != record_id].reset_index(drop=True)
    df = persist_energy_df(df)

    return {"message": "Record deleted", "id": record_id, "remaining_records": len(df)}


# ==========================================================
# PREDICTIVE MAINTENANCE
# ==========================================================

@app.get("/health-score")
def health_score():
    df = load_maintenance_data().copy()
    df["Status"] = status_series(df["Health Score"])

    output = df[["device", "Health Score", "Status"]].rename(columns={"device": "Device"})

    return output.to_dict(orient="records")


@app.get("/maintenance-schedule")
def maintenance_schedule():
    df = load_maintenance_data().copy()
    today = datetime.today()

    days_offset = np.select(
        [df["Health Score"] < 50, df["Health Score"] < 80], [3, 7], default=30
    )
    priority = np.select(
        [df["Health Score"] < 50, df["Health Score"] < 80], ["High", "Medium"], default="Low"
    )

    df["Maintenance Date"] = (today + pd.to_timedelta(days_offset, unit="D")).strftime("%Y-%m-%d")
    df["Priority"] = priority

    output = df[["device", "Health Score", "Maintenance Date", "Priority"]].rename(columns={"device": "Device"})

    return output.to_dict(orient="records")


@app.get("/maintenance-alerts")
def maintenance_alerts():
    df = load_maintenance_data().copy()

    df["Alert"] = np.select(
        [df["Health Score"] < 50, df["Health Score"] < 80],
        ["Immediate maintenance required", "Inspection recommended"],
        default=None
    )

    flagged = df[df["Alert"].notna()]
    output = flagged[["device", "Alert"]].rename(columns={"device": "Device"})

    return output.to_dict(orient="records")


@app.get("/maintenance-report")
def maintenance_report():
    df = load_maintenance_data().copy()
    df["Status"] = status_series(df["Health Score"])
    df["Failure"] = df["failure"].astype(int)

    output = df[["device", "Health Score", "Status", "Failure"]].rename(columns={"device": "Device"})

    return output.to_dict(orient="records")


@app.get("/maintenance-dashboard")
def maintenance_dashboard():
    df = load_maintenance_data()

    healthy = int(len(df[df["Health Score"] >= 80]))
    warning = int(len(df[(df["Health Score"] >= 50) & (df["Health Score"] < 80)]))
    critical = int(len(df[df["Health Score"] < 50]))

    avg_health_score = float(df["Health Score"].mean())
    lowest_health_score = float(df["Health Score"].min())

    risky_mask = df["Health Score"] < 50

    if "failure" in df.columns:
        risky_mask = risky_mask | (pd.to_numeric(df["failure"], errors="coerce").fillna(0) == 1)

    high_risk_readings = int(risky_mask.sum())

    return {
        "Total Devices": int(df["device"].nunique()),
        "Total Records": int(len(df)),
        "Healthy": healthy,
        "Warning": warning,
        "Critical": critical,
        "Failures": int(df["failure"].sum()),
        "Avg Health Score": round(avg_health_score, 2),
        "Lowest Health Score": round(lowest_health_score, 2),
        "High-Risk Readings": high_risk_readings
    }


@app.get("/device-health-summary")
def device_health_summary():
    df = load_maintenance_data()

    grouped = (
        df.groupby("device")
        .agg(**{
            "Health Score": ("Health Score", "mean"),
            "Failures": ("failure", "sum"),
            "Readings": ("Health Score", "count")
        })
        .reset_index()
        .rename(columns={"device": "Device"})
    )

    grouped["Health Score"] = grouped["Health Score"].round(2)
    grouped["Status"] = status_series(grouped["Health Score"])
    grouped["Failures"] = grouped["Failures"].astype(int)
    grouped["Readings"] = grouped["Readings"].astype(int)

    return grouped.to_dict(orient="records")


@app.get("/maintenance-records")
def get_maintenance_records(limit: int = Query(None, ge=1)):
    df = load_maintenance_data()

    if limit is not None:
        df = df.sort_values("id").tail(limit)

    return safe_records(df)


@app.get("/maintenance-fields")
def get_maintenance_fields():
    df = load_maintenance_data()
    return {"metric_columns": metric_columns(df)}


@app.post("/maintenance-records")
def add_maintenance_record(payload: dict = Body(...)):
    df = load_maintenance_data().copy()

    device = str(payload.get("device", "")).strip()

    if not device:
        return JSONResponse(status_code=400, content={"error": "device is required"})

    new_row = {column: 0.0 for column in metric_columns(df)}

    for key, value in payload.items():
        if key in new_row and value not in (None, ""):
            new_row[key] = value

    new_row["device"] = device

    failure_value = payload.get("failure", 0)

    if isinstance(failure_value, str):
        failure_value = 1 if failure_value.strip().lower() in ["yes", "true", "1", "y"] else 0

    new_row["failure"] = int(failure_value or 0)

    new_id = int(df["id"].max()) + 1 if not df.empty else 1
    new_row["id"] = new_id

    df = pd.concat(
        [df.drop(columns=["Health Score"], errors="ignore"), pd.DataFrame([new_row])],
        ignore_index=True
    )

    df = persist_maintenance_df(df)
    created = df[df["id"] == new_id]

    return safe_records(created)[0]


@app.delete("/maintenance-records/{record_id}")
def delete_maintenance_record(record_id: int):
    df = load_maintenance_data().copy()

    if record_id not in df["id"].values:
        return JSONResponse(status_code=404, content={"error": f"Record {record_id} not found"})

    df = df[df["id"] != record_id].reset_index(drop=True)
    df = persist_maintenance_df(df)

    return {"message": "Record deleted", "id": record_id, "remaining_records": len(df)}


# ==========================================================
# OCCUPANCY AGENT — MILESTONE 3
# ==========================================================

OCCUPANCY_ALERT_HIGH_THRESHOLD = 90
OCCUPANCY_ALERT_MEDIUM_THRESHOLD = 75

OCCUPANCY_UTIL_HIGH_THRESHOLD = 60
OCCUPANCY_UTIL_MODERATE_THRESHOLD = 35

SENSOR_FEATURE_COLUMNS = [
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2", "S5_CO2_Slope",
    "S6_PIR", "S7_PIR",
]


def latest_reading_per_zone(df):
    return (
        df.sort_values("timestamp")
        .groupby(["floor", "zone"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def classify_alert(occupancy_pct):
    if occupancy_pct >= OCCUPANCY_ALERT_HIGH_THRESHOLD:
        return "HIGH", "exceeds configured capacity"
    if occupancy_pct >= OCCUPANCY_ALERT_MEDIUM_THRESHOLD:
        return "MEDIUM", "approaching configured capacity"
    return None, None


def classify_utilization(avg_pct):
    if avg_pct >= OCCUPANCY_UTIL_HIGH_THRESHOLD:
        return "Highly Utilized"
    if avg_pct >= OCCUPANCY_UTIL_MODERATE_THRESHOLD:
        return "Moderately Utilized"
    return "Underutilized"


@app.get("/occupancy/dashboard")
def occupancy_dashboard():
    df = load_occupancy_data()
    latest = latest_reading_per_zone(df)

    total_people = int(latest["occupancy_count"].sum())
    total_capacity = int(latest["capacity"].sum())
    avg_occupancy_pct = float(latest["occupancy_pct"].mean())

    high_alerts = int((latest["occupancy_pct"] >= OCCUPANCY_ALERT_HIGH_THRESHOLD).sum())
    medium_alerts = int(
        (
            (latest["occupancy_pct"] >= OCCUPANCY_ALERT_MEDIUM_THRESHOLD)
            & (latest["occupancy_pct"] < OCCUPANCY_ALERT_HIGH_THRESHOLD)
        ).sum()
    )

    return {
        "Total Zones": int(latest.shape[0]),
        "Total Floors": int(latest["floor"].nunique()),
        "Current Occupants": total_people,
        "Total Capacity": total_capacity,
        "Avg Occupancy %": round(avg_occupancy_pct, 2),
        "High Alerts": high_alerts,
        "Medium Alerts": medium_alerts,
        "Last Updated": latest["timestamp"].max().isoformat() if not latest.empty else None
    }


@app.get("/occupancy/heatmap")
def occupancy_heatmap():
    df = load_occupancy_data()
    latest = latest_reading_per_zone(df)

    output = latest[["floor", "zone", "occupancy_count", "capacity", "occupancy_pct"]].sort_values(["floor", "zone"])

    return output.to_dict(orient="records")


@app.get("/occupancy/alerts")
def occupancy_alerts():
    df = load_occupancy_data()
    latest = latest_reading_per_zone(df)

    alerts = []

    for _, row in latest.iterrows():
        severity, reason = classify_alert(row["occupancy_pct"])

        if severity is None:
            continue

        alerts.append({
            "Alert Type": "OCCUPANCY",
            "Severity": severity,
            "Floor": int(row["floor"]),
            "Zone": row["zone"],
            "Occupancy %": row["occupancy_pct"],
            "Message": f'{row["zone"]} {reason}.'
        })

    severity_order = {"HIGH": 0, "MEDIUM": 1}
    alerts.sort(key=lambda a: severity_order.get(a["Severity"], 2))

    return alerts


@app.get("/occupancy/utilization")
def occupancy_utilization():
    df = load_occupancy_data()

    business_hours = df[
        (df["timestamp"].dt.hour >= 9)
        & (df["timestamp"].dt.hour < 18)
        & (df["timestamp"].dt.dayofweek < 5)
    ]

    if business_hours.empty:
        return []

    utilization = (
        business_hours.groupby(["floor", "zone"], as_index=False)["occupancy_pct"]
        .mean()
        .round(2)
    )

    utilization["Classification"] = utilization["occupancy_pct"].apply(classify_utilization)

    utilization = utilization.rename(columns={
        "floor": "Floor",
        "zone": "Zone",
        "occupancy_pct": "Avg Occupancy % (Business Hours)"
    })

    return utilization.sort_values("Avg Occupancy % (Business Hours)", ascending=False).to_dict(orient="records")


@app.get("/occupancy/zones")
def occupancy_zone_list():
    df = load_occupancy_data()

    zones = df[["floor", "zone", "capacity"]].drop_duplicates().sort_values(["floor", "zone"])

    return zones.to_dict(orient="records")


@app.get("/occupancy/records")
def get_occupancy_records(limit: int = Query(None, ge=1), zone: str = Query(None)):
    df = load_occupancy_data()

    if zone:
        df = df[df["zone"] == zone]

    if limit is not None:
        df = df.sort_values("occupancy_id").tail(limit)

    return safe_records(df)


@app.post("/occupancy/records")
def add_occupancy_record(payload: dict = Body(...)):
    df = load_occupancy_data().copy()

    zone = str(payload.get("zone", "")).strip()

    if not zone:
        return JSONResponse(status_code=400, content={"error": "zone is required"})

    existing_zone_rows = df[df["zone"] == zone]

    if existing_zone_rows.empty:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown zone '{zone}'. Use /occupancy/zones to see valid zones."}
        )

    try:
        occupancy_count = int(payload.get("occupancy_count"))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "occupancy_count is a required integer field"})

    reading_timestamp = datetime.now()

    reference_row = existing_zone_rows.iloc[-1]
    capacity = int(reference_row["capacity"])

    new_id = int(df["occupancy_id"].max()) + 1 if not df.empty else 1

    max_allowed = int(capacity * 1.5)

    new_row = {
        "occupancy_id": new_id,
        "facility_id": int(reference_row.get("facility_id", 1)),
        "floor": int(reference_row["floor"]),
        "zone": zone,
        "timestamp": reading_timestamp.isoformat(),
        "occupancy_count": max(0, min(occupancy_count, max_allowed)),
        "capacity": capacity
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = persist_occupancy_df(df)

    created = df[df["occupancy_id"] == new_id]

    return safe_records(created)[0]


@app.delete("/occupancy/records/{record_id}")
def delete_occupancy_record(record_id: int):
    df = load_occupancy_data().copy()

    if record_id not in df["occupancy_id"].values:
        return JSONResponse(status_code=404, content={"error": f"Record {record_id} not found"})

    df = df[df["occupancy_id"] != record_id].reset_index(drop=True)
    df = persist_occupancy_df(df)

    return {"message": "Record deleted", "occupancy_id": record_id, "remaining_records": len(df)}


@app.get("/occupancy/forecast")
def occupancy_forecast(zone: str = Query(...), day_of_week: int = Query(None, ge=0, le=6)):
    if forecast_model_bundle is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Forecast model not trained yet. Run occupancy_agent/occupancy_model.py first."}
        )

    df = load_occupancy_data()
    zone_rows = df[df["zone"] == zone]

    if zone_rows.empty:
        return JSONResponse(status_code=404, content={"error": f"Unknown zone '{zone}'"})

    floor = int(zone_rows.iloc[-1]["floor"])

    target_day = day_of_week if day_of_week is not None else (datetime.now().weekday() + 1) % 7

    model = forecast_model_bundle["model"]
    feature_columns = forecast_model_bundle["feature_columns"]

    hours = list(range(24))

    request_df = pd.DataFrame([
        {
            "zone": zone,
            "floor": floor,
            "hour": hour,
            "day_of_week": target_day,
            "is_weekend": 1 if target_day >= 5 else 0
        }
        for hour in hours
    ])[feature_columns]

    predictions = model.predict(request_df)
    predictions = np.clip(predictions, 0, 100).round(1)

    hourly_forecast = [
        {"hour": hour, "predicted_occupancy_pct": float(pct)}
        for hour, pct in zip(hours, predictions)
    ]

    business_hours_avg = float(
        np.mean([p["predicted_occupancy_pct"] for p in hourly_forecast if 9 <= p["hour"] < 18])
    )

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return {
        "zone": zone,
        "floor": floor,
        "day_of_week": target_day,
        "day_name": day_names[target_day],
        "predicted_avg_occupancy_pct_business_hours": round(business_hours_avg, 1),
        "hourly_forecast": hourly_forecast
    }


@app.post("/occupancy/estimate-live")
def occupancy_estimate_live(payload: dict = Body(...)):
    if sensor_estimator_bundle is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Sensor estimator not trained yet. Run occupancy_agent/occupancy_model.py first."}
        )

    missing = [col for col in SENSOR_FEATURE_COLUMNS if col not in payload]

    if missing:
        return JSONResponse(status_code=400, content={"error": f"Missing required sensor fields: {missing}"})

    try:
        input_row = {col: float(payload[col]) for col in SENSOR_FEATURE_COLUMNS}
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "All sensor fields must be numeric"})

    model = sensor_estimator_bundle["model"]
    feature_columns = sensor_estimator_bundle["feature_columns"]

    request_df = pd.DataFrame([input_row])[feature_columns]

    predicted_count = int(model.predict(request_df)[0])

    return {"predicted_occupancy_count": predicted_count, "sensor_readings": input_row}


# ==========================================================
# SECURITY AGENT — MILESTONE 3, PART B
# ==========================================================

SECURITY_EMPLOYEES_CSV_PATH = BASE_DIR.parent / "dataset" / "security_employees.csv"
SECURITY_ACCESS_LOGS_CSV_PATH = BASE_DIR.parent / "dataset" / "security_access_logs.csv"
SECURITY_VISITORS_CSV_PATH = BASE_DIR.parent / "dataset" / "security_visitors.csv"
SECURITY_VISITOR_MOVEMENTS_CSV_PATH = BASE_DIR.parent / "dataset" / "security_visitor_movements.csv"
SECURITY_CCTV_EVENTS_CSV_PATH = BASE_DIR.parent / "dataset" / "security_cctv_events.csv"
SECURITY_EVENTS_CSV_PATH = BASE_DIR.parent / "dataset" / "security_events.csv"

security_employees_df = pd.DataFrame()
security_access_logs_df = pd.DataFrame()
security_visitors_df = pd.DataFrame()
security_visitor_movements_df = pd.DataFrame()
security_cctv_events_df = pd.DataFrame()
security_events_df = pd.DataFrame()

_security_mtimes = {
    "employees": None,
    "access_logs": None,
    "visitors": None,
    "visitor_movements": None,
    "cctv_events": None,
    "security_events": None,
}

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _load_csv_if_changed(path, key, datetime_columns=None):
    if not path.exists():
        raise FileNotFoundError(
            f"Security dataset not found at {path}. Run security_agent/generate_security_data.py first."
        )

    current_mtime = os.path.getmtime(path)

    if _security_mtimes[key] == current_mtime:
        return None

    df = pd.read_csv(path)

    for column in (datetime_columns or []):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    _security_mtimes[key] = current_mtime

    return df


def load_security_data(force=False):
    global security_employees_df, security_access_logs_df, security_visitors_df
    global security_visitor_movements_df, security_cctv_events_df, security_events_df

    if force:
        for key in _security_mtimes:
            _security_mtimes[key] = None

    employees = _load_csv_if_changed(SECURITY_EMPLOYEES_CSV_PATH, "employees")
    if employees is not None:
        security_employees_df = employees

    access_logs = _load_csv_if_changed(
        SECURITY_ACCESS_LOGS_CSV_PATH, "access_logs", datetime_columns=["timestamp"]
    )
    if access_logs is not None:
        security_access_logs_df = access_logs

    visitors = _load_csv_if_changed(
        SECURITY_VISITORS_CSV_PATH, "visitors", datetime_columns=["entry_time", "exit_time"]
    )
    if visitors is not None:
        security_visitors_df = visitors

    movements = _load_csv_if_changed(
        SECURITY_VISITOR_MOVEMENTS_CSV_PATH, "visitor_movements", datetime_columns=["timestamp"]
    )
    if movements is not None:
        security_visitor_movements_df = movements

    cctv_events = _load_csv_if_changed(
        SECURITY_CCTV_EVENTS_CSV_PATH, "cctv_events", datetime_columns=["timestamp"]
    )
    if cctv_events is not None:
        security_cctv_events_df = cctv_events

    security_events = _load_csv_if_changed(
        SECURITY_EVENTS_CSV_PATH, "security_events", datetime_columns=["timestamp"]
    )
    if security_events is not None:
        security_events_df = security_events


@app.get("/security/access-logs")
def get_access_logs(zone: str = Query(None), granted: bool = Query(None), limit: int = Query(200, ge=1, le=2000)):
    load_security_data()
    df = security_access_logs_df

    if zone:
        df = df[df["zone"] == zone]

    if granted is not None:
        df = df[df["access_granted"] == granted]

    df = df.sort_values("timestamp", ascending=False).head(limit)

    return safe_records(df)


@app.get("/security/access-logs/summary")
def access_logs_summary():
    load_security_data()
    df = security_access_logs_df

    grouped = (
        df.groupby(["floor", "zone"], as_index=False)
        .agg(
            total_attempts=("log_id", "count"),
            denied=("access_granted", lambda s: int((~s).sum()))
        )
        .sort_values(["floor", "zone"])
    )

    grouped["denial_rate_pct"] = ((grouped["denied"] / grouped["total_attempts"]) * 100).round(1)

    return grouped.to_dict(orient="records")


@app.get("/security/events")
def get_security_events(
    severity: str = Query(None),
    source: str = Query(None),
    zone: str = Query(None),
    limit: int = Query(200, ge=1, le=2000)
):
    load_security_data()
    df = security_events_df

    if severity:
        df = df[df["severity"] == severity.upper()]

    if source:
        df = df[df["source"] == source.upper()]

    if zone:
        df = df[df["zone"] == zone]

    df = df.sort_values("timestamp", ascending=False).head(limit)

    return safe_records(df)


@app.get("/security/alerts")
def security_alerts():
    load_security_data()
    df = security_events_df

    flagged = df[df["severity"].isin(["HIGH", "CRITICAL"])].copy()
    flagged["_order"] = flagged["severity"].map(SEVERITY_ORDER)
    flagged = flagged.sort_values(["_order", "timestamp"], ascending=[True, False])
    flagged = flagged.drop(columns=["_order"])

    return safe_records(flagged)


@app.get("/security/events/summary")
def security_events_summary():
    load_security_data()
    df = security_events_df

    counts = df["severity"].value_counts().to_dict()

    return {
        "CRITICAL": int(counts.get("CRITICAL", 0)),
        "HIGH": int(counts.get("HIGH", 0)),
        "MEDIUM": int(counts.get("MEDIUM", 0)),
        "LOW": int(counts.get("LOW", 0)),
        "Total": int(len(df))
    }


@app.get("/security/cctv-events")
def get_cctv_events(zone: str = Query(None), event_type: str = Query(None), limit: int = Query(200, ge=1, le=2000)):
    load_security_data()
    df = security_cctv_events_df

    if zone:
        df = df[df["zone"] == zone]

    if event_type:
        df = df[df["event_type"] == event_type]

    df = df.sort_values("timestamp", ascending=False).head(limit)

    return safe_records(df)


@app.get("/security/visitors")
def get_visitors(flagged_only: bool = Query(False), limit: int = Query(200, ge=1, le=2000)):
    load_security_data()
    df = security_visitors_df

    if flagged_only:
        df = df[df["flagged_unescorted_restricted_access"] == True]  # noqa: E712

    df = df.sort_values("entry_time", ascending=False).head(limit)

    return safe_records(df)


@app.get("/security/visitors/{visitor_id}/path")
def get_visitor_path(visitor_id: str):
    load_security_data()

    visitor_rows = security_visitors_df[security_visitors_df["visitor_id"] == visitor_id]

    if visitor_rows.empty:
        return JSONResponse(status_code=404, content={"error": f"Visitor {visitor_id} not found"})

    path = (
        security_visitor_movements_df[security_visitor_movements_df["visitor_id"] == visitor_id]
        .sort_values("sequence")
    )

    return {"visitor": safe_records(visitor_rows)[0], "path": safe_records(path)}


@app.get("/security/incidents/{event_id}")
def investigate_incident(event_id: int, window_minutes: int = Query(30, ge=1, le=1440)):
    load_security_data()

    event_rows = security_events_df[security_events_df["event_id"] == event_id]

    if event_rows.empty:
        return JSONResponse(status_code=404, content={"error": f"Security event {event_id} not found"})

    event = event_rows.iloc[0]
    event_zone = event["zone"]
    event_time = event["timestamp"]

    window_start = event_time - timedelta(minutes=window_minutes)
    window_end = event_time + timedelta(minutes=window_minutes)

    related_access = security_access_logs_df[
        (security_access_logs_df["zone"] == event_zone)
        & (security_access_logs_df["timestamp"] >= window_start)
        & (security_access_logs_df["timestamp"] <= window_end)
    ].sort_values("timestamp")

    related_cctv = security_cctv_events_df[
        (security_cctv_events_df["zone"] == event_zone)
        & (security_cctv_events_df["timestamp"] >= window_start)
        & (security_cctv_events_df["timestamp"] <= window_end)
    ].sort_values("timestamp")

    related_movements = security_visitor_movements_df[
        (security_visitor_movements_df["zone"] == event_zone)
        & (security_visitor_movements_df["timestamp"] >= window_start)
        & (security_visitor_movements_df["timestamp"] <= window_end)
    ].sort_values("timestamp")

    return {
        "event": safe_records(event_rows)[0],
        "window_minutes": window_minutes,
        "related_access_logs": safe_records(related_access),
        "related_cctv_events": safe_records(related_cctv),
        "related_visitor_movements": safe_records(related_movements)
    }


@app.get("/security/dashboard")
def security_dashboard():
    load_security_data()

    events = security_events_df
    logs = security_access_logs_df
    visitors = security_visitors_df

    severity_counts = events["severity"].value_counts().to_dict()

    return {
        "Total Access Attempts": int(len(logs)),
        "Denied Attempts": int((~logs["access_granted"]).sum()),
        "Total Visitors": int(len(visitors)),
        "Flagged Visitor Visits": int(visitors["flagged_unescorted_restricted_access"].sum()),
        "Total Security Events": int(len(events)),
        "Critical Alerts": int(severity_counts.get("CRITICAL", 0)),
        "High Alerts": int(severity_counts.get("HIGH", 0)),
        "Last Event": events["timestamp"].max().isoformat() if not events.empty else None
    }