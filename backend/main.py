import os
from pathlib import Path
from datetime import datetime

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


# ==========================================================
# IN-MEMORY STATE
# ==========================================================

energy_df = pd.DataFrame()
maintenance_df = pd.DataFrame()

average_energy = 0.0

_energy_mtime = None
_maintenance_mtime = None


# ==========================================================
# DATA HELPERS
# ==========================================================

def safe_records(df):
    """
    Convert DataFrame rows to JSON-safe records.
    NaN values are converted to None.
    """
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
    """
    Predict consumption by finding historical records that are
    most similar to the entered hour, temperature, occupancy,
    and weekend value.

    This is a similarity-based prediction and does not require
    a separate machine-learning model.
    """

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

    # Circular distance handles the fact that hour 23 and hour 0
    # are close to each other.
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

    score = score - np.where(
        df["failure"] == 1,
        50,
        0
    )

    score = score - np.where(
        df["metric5"] > 15,
        10,
        0
    )

    score = score - np.where(
        df["metric6"] > 300000,
        15,
        0
    )

    score = score - np.where(
        df["metric9"] > 500,
        10,
        0
    )

    score = score - np.where(
        df["metric3"] > 100,
        10,
        0
    )

    score = score - np.where(
        df["metric4"] > 20,
        5,
        0
    )

    df["Health Score"] = score.clip(
        lower=0
    )

    return df


def status_series(score_series):
    return np.select(
        [
            score_series >= 80,
            score_series >= 50
        ],
        [
            "Healthy",
            "Warning"
        ],
        default="Critical"
    )


# ==========================================================
# MAINTENANCE RECORD HELPERS (id + numeric coercion)
# ==========================================================

# Columns that must never be treated as a numeric "metric" input,
# because they are either identifiers or computed fields.
NON_METRIC_COLUMNS = [
    "id",
    "device",
    "failure",
    "Health Score",
    "Status"
]


def ensure_id_column(df):
    """
    Guarantee every maintenance row has a stable integer id.
    If the CSV does not already have one, create it once from
    the row order so ids stay stable across reloads.
    """
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df["id"] = df.index + 1

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce"
    )

    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    return df


def metric_columns(df):
    """
    Every numeric reading column that isn't an id, label,
    or computed field. Used to build the Add Record form and
    to coerce new rows to the right dtype.
    """
    return [
        column
        for column in df.columns
        if column not in NON_METRIC_COLUMNS
    ]


def coerce_maintenance_types(df):
    for column in metric_columns(df):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0)

    if "failure" in df.columns:
        df["failure"] = pd.to_numeric(
            df["failure"],
            errors="coerce"
        ).fillna(0).astype(int)

    if "device" in df.columns:
        df["device"] = df["device"].astype(str)

    return df


def persist_maintenance_df(df):
    """
    Recompute health score, save to disk, and update the
    in-memory cache so every endpoint reflects the change
    immediately (no manual /refresh needed).
    """
    global maintenance_df
    global _maintenance_mtime

    df = coerce_maintenance_types(df)

    df = df.drop(
        columns=["Health Score"],
        errors="ignore"
    )

    df = add_health_score(df)

    df.to_csv(
        MAINTENANCE_CSV_PATH,
        index=False
    )

    maintenance_df = df
    _maintenance_mtime = os.path.getmtime(
        MAINTENANCE_CSV_PATH
    )

    return df


# ==========================================================
# DATA LOADING
# ==========================================================

def load_energy_data(force=False):
    global energy_df
    global average_energy
    global _energy_mtime

    if not ENERGY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Energy dataset not found at "
            f"{ENERGY_CSV_PATH}"
        )

    current_mtime = os.path.getmtime(
        ENERGY_CSV_PATH
    )

    if force or _energy_mtime != current_mtime:
        df = pd.read_csv(
            ENERGY_CSV_PATH
        )

        df = ensure_id_column(
            df
        )

        df["EnergyConsumption"] = pd.to_numeric(
            df["EnergyConsumption"],
            errors="coerce"
        )

        energy_df = df

        average_energy = round(
            float(
                df["EnergyConsumption"].mean()
            ),
            4
        )

        _energy_mtime = current_mtime

    return energy_df


def column_default_value(df, column):
    """
    A sensible fallback for a column we don't explicitly control
    when adding a new energy record: the median for numeric
    columns, the most common value for everything else.
    """
    series = df[column].dropna()

    if series.empty:
        return None

    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())

    mode = series.mode()

    return mode.iloc[0] if not mode.empty else None


def persist_energy_df(df):
    """
    Save to disk and reload through load_energy_data so the id
    column, dtypes, and average_energy all stay in sync with
    what every other energy endpoint reads.
    """
    df.to_csv(
        ENERGY_CSV_PATH,
        index=False
    )

    return load_energy_data(
        force=True
    )


def load_maintenance_data(force=False):
    global maintenance_df
    global _maintenance_mtime

    if not MAINTENANCE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Maintenance dataset not found at "
            f"{MAINTENANCE_CSV_PATH}"
        )

    current_mtime = os.path.getmtime(
        MAINTENANCE_CSV_PATH
    )

    if force or _maintenance_mtime != current_mtime:
        df = pd.read_csv(
            MAINTENANCE_CSV_PATH
        )

        df = ensure_id_column(
            df
        )

        df = add_health_score(
            df
        )

        maintenance_df = df
        _maintenance_mtime = current_mtime

    return maintenance_df


@app.on_event("startup")
def startup_load():
    load_energy_data(
        force=True
    )

    load_maintenance_data(
        force=True
    )


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():
    return {
        "message": (
            "Welcome to Agentic FacilityOps "
            "AI Platform"
        )
    }


@app.get("/refresh")
def refresh():
    load_energy_data(
        force=True
    )

    load_maintenance_data(
        force=True
    )

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

    return safe_records(
        df
    )


@app.get("/statistics")
def statistics():
    df = load_energy_data()

    return {
        "Average Energy": round(
            df["EnergyConsumption"].mean(),
            2
        ),
        "Maximum Energy": round(
            df["EnergyConsumption"].max(),
            2
        ),
        "Minimum Energy": round(
            df["EnergyConsumption"].min(),
            2
        )
    }


@app.get("/energy/simulate")
def simulate_energy(
    hour: int = Query(
        ...,
        ge=0,
        le=23
    ),
    outdoor_temperature: float = Query(
        ...
    ),
    occupancy: int = Query(
        ...,
        ge=0
    ),
    weekend: str = Query(
        "No"
    )
):
    df = load_energy_data()

    weekend_value = (
        weekend.strip().lower()
        in [
            "yes",
            "true",
            "1",
            "y"
        ]
    )

    predicted_consumption = (
        predict_energy_consumption(
            df=df,
            hour=hour,
            outdoor_temperature=outdoor_temperature,
            occupancy=occupancy,
            weekend=weekend_value
        )
    )

    return {
        "Timestamp": datetime.now().isoformat(),
        "Hour": hour,
        "Temperature": outdoor_temperature,
        "Occupancy": occupancy,
        "Weekend": (
            "Yes"
            if weekend_value
            else "No"
        ),
        "EnergyConsumption": predicted_consumption,
        "HVACUsage": (
            "On"
            if (
                outdoor_temperature < 20
                or outdoor_temperature > 26
            )
            else "Off"
        ),
        "LightingUsage": (
            "On"
            if occupancy > 0
            else "Off"
        ),
        "Simulated": True
    }


@app.get("/alerts")
def alerts():
    df = load_energy_data()

    threshold = average_energy * 1.2

    flagged = df[
        df["EnergyConsumption"] > threshold
    ].copy()

    flagged["Alert"] = (
        "High Energy Consumption"
    )

    if "Timestamp" in flagged.columns:
        flagged["Timestamp"] = (
            flagged["Timestamp"].astype(str)
        )

    return flagged[
        [
            "Timestamp",
            "Alert"
        ]
    ].to_dict(
        orient="records"
    )


@app.get("/recommendations")
def recommendations():
    df = load_energy_data()

    threshold = average_energy * 1.2

    hvac_mask = (
        (df["HVACUsage"] == "On")
        & (df["Temperature"] < 22)
    )

    light_mask = (
        (df["LightingUsage"] == "On")
        & (df["Occupancy"] < 10)
    )

    renewable_mask = (
        df["RenewableEnergy"] < 10
    )

    high_mask = (
        df["EnergyConsumption"] > threshold
    )

    any_mask = (
        hvac_mask
        | light_mask
        | renewable_mask
        | high_mask
    )

    timestamps = (
        df.loc[any_mask, "Timestamp"]
        .astype(str)
        .to_numpy()
    )

    hvac_flags = (
        hvac_mask[any_mask]
        .to_numpy()
    )

    light_flags = (
        light_mask[any_mask]
        .to_numpy()
    )

    renewable_flags = (
        renewable_mask[any_mask]
        .to_numpy()
    )

    high_flags = (
        high_mask[any_mask]
        .to_numpy()
    )

    recommendation_list = []

    for timestamp, hvac, lights, renewable, high in zip(
        timestamps,
        hvac_flags,
        light_flags,
        renewable_flags,
        high_flags
    ):
        current_recommendations = []

        if hvac:
            current_recommendations.append(
                "Turn OFF HVAC"
            )

        if lights:
            current_recommendations.append(
                "Switch OFF unnecessary lights"
            )

        if renewable:
            current_recommendations.append(
                "Increase renewable energy usage"
            )

        if high:
            current_recommendations.append(
                "Inspect equipment with high energy consumption"
            )

        recommendation_list.append(
            {
                "Timestamp": timestamp,
                "Recommendations": (
                    current_recommendations
                )
            }
        )

    return recommendation_list


@app.get("/dashboard")
def dashboard():
    df = load_energy_data()

    threshold = average_energy * 1.2

    high_alerts = int(
        len(
            df[
                df["EnergyConsumption"]
                > threshold
            ]
        )
    )

    return {
        "Total Records": len(df),
        "Average Energy": round(
            df["EnergyConsumption"].mean(),
            2
        ),
        "Maximum Energy": round(
            df["EnergyConsumption"].max(),
            2
        ),
        "Minimum Energy": round(
            df["EnergyConsumption"].min(),
            2
        ),
        "High Energy Alerts": high_alerts
    }


# ==========================================================
# ENERGY RECORDS — ADD / REMOVE LIVE DATA
# ==========================================================

@app.post("/energy-records")
def add_energy_record(payload: dict = Body(...)):
    """
    Adds a real record to the energy dataset using the same
    Hour / Outdoor Temperature / Occupancy / Weekend inputs as
    the sidebar simulator. EnergyConsumption is estimated with
    the same similarity model used by /energy/simulate, then the
    row is persisted to the CSV so every chart, alert, and
    recommendation on the dashboard reflects it going forward.
    """
    df = load_energy_data().copy()

    try:
        hour = int(payload.get("hour"))
        outdoor_temperature = float(
            payload.get("outdoor_temperature")
        )
        occupancy = int(payload.get("occupancy"))
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "hour, outdoor_temperature and occupancy "
                    "are required numeric fields"
                )
            }
        )

    if not (0 <= hour <= 23):
        return JSONResponse(
            status_code=400,
            content={"error": "hour must be between 0 and 23"}
        )

    weekend_raw = str(
        payload.get("weekend", "No")
    ).strip().lower()

    weekend_value = weekend_raw in ["yes", "true", "1", "y"]

    predicted_consumption = predict_energy_consumption(
        df=df,
        hour=hour,
        outdoor_temperature=outdoor_temperature,
        occupancy=occupancy,
        weekend=weekend_value
    )

    # Start every column from a sensible existing-data default so
    # nothing downstream sees an unexpected NaN, then override the
    # columns we actually control.
    new_row = {
        column: column_default_value(df, column)
        for column in df.columns
        if column != "id"
    }

    record_timestamp = datetime.now().replace(
        hour=hour,
        minute=datetime.now().minute,
        second=0,
        microsecond=0
    )

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
            pass  # keep the column_default_value fallback

    if "Weekend" in new_row:
        new_row["Weekend"] = "Yes" if weekend_value else "No"
    elif "DayOfWeek" in df.columns:
        existing_days = df["DayOfWeek"].dropna()

        if (
            not existing_days.empty
            and pd.api.types.is_numeric_dtype(existing_days)
        ):
            new_row["DayOfWeek"] = 5 if weekend_value else 1
        else:
            new_row["DayOfWeek"] = (
                "Saturday" if weekend_value else "Monday"
            )

    if "HVACUsage" in new_row:
        new_row["HVACUsage"] = (
            "On"
            if (
                outdoor_temperature < 20
                or outdoor_temperature > 26
            )
            else "Off"
        )

    if "LightingUsage" in new_row:
        new_row["LightingUsage"] = (
            "On" if occupancy > 0 else "Off"
        )

    new_row["EnergyConsumption"] = predicted_consumption

    new_id = (
        int(df["id"].max()) + 1
        if not df.empty
        else 1
    )

    new_row["id"] = new_id

    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True
    )

    df = persist_energy_df(df)

    created = df[df["id"] == new_id]

    return safe_records(created)[0]


@app.delete("/energy-records/{record_id}")
def delete_energy_record(record_id: int):
    df = load_energy_data().copy()

    if record_id not in df["id"].values:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Record {record_id} not found"
            }
        )

    df = df[
        df["id"] != record_id
    ].reset_index(drop=True)

    df = persist_energy_df(df)

    return {
        "message": "Record deleted",
        "id": record_id,
        "remaining_records": len(df)
    }


# ==========================================================
# PREDICTIVE MAINTENANCE
# ==========================================================

@app.get("/health-score")
def health_score():
    df = load_maintenance_data().copy()

    df["Status"] = status_series(
        df["Health Score"]
    )

    output = (
        df[
            [
                "device",
                "Health Score",
                "Status"
            ]
        ]
        .rename(
            columns={
                "device": "Device"
            }
        )
    )

    return output.to_dict(
        orient="records"
    )


@app.get("/maintenance-schedule")
def maintenance_schedule():
    df = load_maintenance_data().copy()

    today = datetime.today()

    days_offset = np.select(
        [
            df["Health Score"] < 50,
            df["Health Score"] < 80
        ],
        [
            3,
            7
        ],
        default=30
    )

    priority = np.select(
        [
            df["Health Score"] < 50,
            df["Health Score"] < 80
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    df["Maintenance Date"] = (
        today
        + pd.to_timedelta(
            days_offset,
            unit="D"
        )
    ).strftime(
        "%Y-%m-%d"
    )

    df["Priority"] = priority

    output = (
        df[
            [
                "device",
                "Health Score",
                "Maintenance Date",
                "Priority"
            ]
        ]
        .rename(
            columns={
                "device": "Device"
            }
        )
    )

    return output.to_dict(
        orient="records"
    )


@app.get("/maintenance-alerts")
def maintenance_alerts():
    df = load_maintenance_data().copy()

    df["Alert"] = np.select(
        [
            df["Health Score"] < 50,
            df["Health Score"] < 80
        ],
        [
            "Immediate maintenance required",
            "Inspection recommended"
        ],
        default=None
    )

    flagged = df[
        df["Alert"].notna()
    ]

    output = (
        flagged[
            [
                "device",
                "Alert"
            ]
        ]
        .rename(
            columns={
                "device": "Device"
            }
        )
    )

    return output.to_dict(
        orient="records"
    )


@app.get("/maintenance-report")
def maintenance_report():
    df = load_maintenance_data().copy()

    df["Status"] = status_series(
        df["Health Score"]
    )

    df["Failure"] = (
        df["failure"].astype(int)
    )

    output = (
        df[
            [
                "device",
                "Health Score",
                "Status",
                "Failure"
            ]
        ]
        .rename(
            columns={
                "device": "Device"
            }
        )
    )

    return output.to_dict(
        orient="records"
    )


@app.get("/maintenance-dashboard")
def maintenance_dashboard():
    df = load_maintenance_data()

    healthy = int(
        len(
            df[
                df["Health Score"] >= 80
            ]
        )
    )

    warning = int(
        len(
            df[
                (
                    df["Health Score"] >= 50
                )
                & (
                    df["Health Score"] < 80
                )
            ]
        )
    )

    critical = int(
        len(
            df[
                df["Health Score"] < 50
            ]
        )
    )

    return {
        "Total Devices": int(
            df["device"].nunique()
        ),
        "Total Records": int(
            len(df)
        ),
        "Healthy": healthy,
        "Warning": warning,
        "Critical": critical,
        "Failures": int(
            df["failure"].sum()
        )
    }


# ==========================================================
# MAINTENANCE RECORDS — ADD / REMOVE LIVE DATA
# ==========================================================

@app.get("/maintenance-records")
def get_maintenance_records():
    """
    Full raw maintenance dataset (one row per reading), including
    id, device, failure, every metric column, and the computed
    Health Score. This is what the "Add / Remove Live Data" panel
    in the dashboard reads and writes.
    """
    df = load_maintenance_data()

    return safe_records(
        df
    )


@app.get("/maintenance-fields")
def get_maintenance_fields():
    """
    Tells the frontend which metric columns exist in the current
    dataset, so the Add Record form can be built dynamically
    instead of hardcoding column names.
    """
    df = load_maintenance_data()

    return {
        "metric_columns": metric_columns(df)
    }


@app.post("/maintenance-records")
def add_maintenance_record(payload: dict = Body(...)):
    df = load_maintenance_data().copy()

    device = str(
        payload.get("device", "")
    ).strip()

    if not device:
        return JSONResponse(
            status_code=400,
            content={
                "error": "device is required"
            }
        )

    new_row = {
        column: 0.0
        for column in metric_columns(df)
    }

    for key, value in payload.items():
        if key in new_row and value not in (None, ""):
            new_row[key] = value

    new_row["device"] = device

    failure_value = payload.get("failure", 0)

    if isinstance(failure_value, str):
        failure_value = (
            1
            if failure_value.strip().lower()
            in ["yes", "true", "1", "y"]
            else 0
        )

    new_row["failure"] = int(failure_value or 0)

    new_id = (
        int(df["id"].max()) + 1
        if not df.empty
        else 1
    )

    new_row["id"] = new_id

    df = pd.concat(
        [
            df.drop(columns=["Health Score"], errors="ignore"),
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

    df = persist_maintenance_df(
        df
    )

    created = df[df["id"] == new_id]

    return safe_records(created)[0]


@app.delete("/maintenance-records/{record_id}")
def delete_maintenance_record(record_id: int):
    df = load_maintenance_data().copy()

    if record_id not in df["id"].values:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Record {record_id} not found"
            }
        )

    df = df[
        df["id"] != record_id
    ].reset_index(drop=True)

    df = persist_maintenance_df(
        df
    )

    return {
        "message": "Record deleted",
        "id": record_id,
        "remaining_records": len(df)
    }