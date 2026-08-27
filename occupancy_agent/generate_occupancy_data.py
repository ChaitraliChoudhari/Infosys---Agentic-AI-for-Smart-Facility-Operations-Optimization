"""
generate_occupancy_data.py
===========================
Builds a realistic HISTORICAL occupancy dataset across every zone in
zones_config.ZONES, matching the OCCUPANCY_RECORDS schema you sketched:
    occupancy_id, facility_id, zone, floor, timestamp, occupancy_count, capacity

Why this exists: your real sensor CSV (Occupancy_Estimation.csv) covers
one room's live sensor readings, not a multi-zone building's history.
This script simulates the building-wide history needed for zone
analytics, heatmaps, and forecasting, using believable patterns:
  - Business hours (9am-6pm, Mon-Fri) are busiest
  - Nights and weekends are near-empty
  - Each zone type (Office/Meeting Room/Storage/etc.) has its own
    typical "how full does it get" behavior, from ZONE_TYPE_BUSY_FRACTION
  - Random noise so it isn't a perfectly repeating pattern

Run this once to produce dataset/occupancy_zones.csv. Re-run any time
you change zones_config.py or want a fresh history.

Usage:
    python generate_occupancy_data.py
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from zones_config import ZONES, ZONE_TYPE_BUSY_FRACTION, zone_type

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "dataset" / "occupancy_zones.csv"

DAYS_OF_HISTORY = 60          # how many days back to simulate
READING_INTERVAL_MINUTES = 30  # one reading every 30 minutes
RANDOM_SEED = 42

# ==========================================================
# SURGE EVENTS — recurring overcrowding scenarios
# ==========================================================
# Real buildings have predictable crunch points: everyone hits the
# cafeteria at lunch, meeting rooms overbook around common meeting
# times. These rules make that happen in the generated history, so
# HIGH/MEDIUM alerts occur naturally and repeatedly — not just as a
# one-off forced value. `applies_to` matches a zone TYPE (see
# zone_type() in zones_config.py), so it affects every zone of that
# type (e.g. every "Meeting Room").
SURGE_RULES = [
    # Cafeteria packed at lunch — regularly tips over capacity (HIGH).
    # target_fraction_range overrides the base office-hours curve entirely
    # (cafeterias peak exactly when offices dip for lunch, so reusing the
    # office curve and just multiplying it isn't enough to cross capacity).
    {"applies_to": "Cafeteria", "hours": [12, 13], "weekday_only": True,
     "probability": 0.75, "target_fraction_range": (0.85, 1.30)},
    # Meeting rooms overbooked around common meeting slots (HIGH/MEDIUM)
    {"applies_to": "Meeting Room", "hours": [10, 11, 14, 15], "weekday_only": True,
     "probability": 0.40, "target_fraction_range": (0.70, 1.25)},
    {"applies_to": "Conference Room", "hours": [9, 10, 14], "weekday_only": True,
     "probability": 0.35, "target_fraction_range": (0.70, 1.20)},
    # Lobby gets busy right at the start of the day (MEDIUM)
    {"applies_to": "Lobby", "hours": [8, 9], "weekday_only": True,
     "probability": 0.30, "target_fraction_range": (0.70, 1.05)},
]

# A capacity of 100 people doesn't mean physically impossible above
# that — real rooms get overcrowded past their configured limit. This
# caps the SIMULATED value at 150% of capacity so overcrowding shows
# up as a genuine >100% occupancy_pct (matches "exceeds configured
# capacity" in your alert wording) rather than being clipped away.
MAX_OCCUPANCY_MULTIPLE_OF_CAPACITY = 1.5

# Forces the LAST (most recent) reading for a few zones to specific
# levels, so the alerts panel always shows a HIGH, a MEDIUM, and some
# normal zones the moment you generate data — regardless of what real
# hour you happen to run this script at. Everything else in the
# dataset is left as naturally simulated. Remove/edit freely.
DEMO_SEED_OVERRIDES = [
    {"zone": "Cafeteria",        "target_pct": 108},  # HIGH — over capacity
    {"zone": "Conference Room",  "target_pct": 94},   # HIGH
    {"zone": "Meeting Room 1",   "target_pct": 82},   # MEDIUM
    {"zone": "Office A",         "target_pct": 55},   # normal
    {"zone": "Storage",          "target_pct": 8},    # normal / low
]


def hourly_business_factor(hour, is_weekend):
    """
    Returns a 0-1 multiplier for how busy a typical business zone is at
    this hour. Peaks mid-morning and mid-afternoon, dips at lunch,
    near-zero outside 7am-8pm, and heavily reduced on weekends.
    """
    if is_weekend:
        # Weekends: mostly empty, a small chance of a few people around midday
        if 10 <= hour <= 15:
            return 0.08
        return 0.01

    if hour < 7 or hour > 20:
        return 0.02

    # Business-hours curve: ramps up, dips at lunch (13:00), ramps down
    morning_peak = np.exp(-((hour - 10) ** 2) / 8)
    afternoon_peak = np.exp(-((hour - 15) ** 2) / 10)
    lunch_dip = 1 - 0.35 * np.exp(-((hour - 13) ** 2) / 2)

    factor = max(morning_peak, afternoon_peak) * lunch_dip
    return float(np.clip(factor, 0.03, 1.0))


def apply_surge(expected_occupancy, capacity, zone_name, ts, rng):
    """
    If a surge rule matches this zone/hour, OVERRIDE expected occupancy
    with a fraction of capacity (can push it above capacity — that's
    the point). This deliberately ignores the base business-hours curve,
    since e.g. cafeterias peak exactly when offices dip for lunch.
    """
    z_type = zone_type(zone_name)
    is_weekend = ts.weekday() >= 5

    for rule in SURGE_RULES:
        if rule["applies_to"] != z_type:
            continue
        if rule["weekday_only"] and is_weekend:
            continue
        if ts.hour not in rule["hours"]:
            continue
        if rng.random() >= rule["probability"]:
            continue

        target_fraction = rng.uniform(*rule["target_fraction_range"])
        return capacity * target_fraction

    return expected_occupancy


def simulate_zone_history(zone_row, timestamps, rng):
    z_type = zone_type(zone_row["zone"])
    base_busy_fraction = ZONE_TYPE_BUSY_FRACTION.get(z_type, 0.4)
    capacity = zone_row["capacity"]
    max_occupancy = capacity * MAX_OCCUPANCY_MULTIPLE_OF_CAPACITY

    records = []

    for ts in timestamps:
        is_weekend = ts.weekday() >= 5
        business_factor = hourly_business_factor(ts.hour, is_weekend)

        expected_occupancy = capacity * base_busy_fraction * business_factor
        expected_occupancy = apply_surge(expected_occupancy, capacity, zone_row["zone"], ts, rng)

        # Add realistic noise. Floor at 0, ceiling at 150% of capacity so
        # surge events can genuinely exceed capacity instead of being
        # silently clipped back down to 100%.
        noise = rng.normal(loc=0, scale=max(1.0, expected_occupancy * 0.2))
        occupancy_count = int(np.clip(round(expected_occupancy + noise), 0, max_occupancy))

        records.append(
            {
                "facility_id": zone_row["facility_id"],
                "floor": zone_row["floor"],
                "zone": zone_row["zone"],
                "timestamp": ts.isoformat(),
                "occupancy_count": occupancy_count,
                "capacity": capacity,
            }
        )

    return records


def apply_demo_seed_overrides(df):
    """
    Overwrites the most recent reading for a few zones so the alerts
    panel is guaranteed to show a HIGH, a MEDIUM, and some normal zones
    right after generation — see DEMO_SEED_OVERRIDES above.
    """
    latest_timestamp = df["timestamp"].max()

    for override in DEMO_SEED_OVERRIDES:
        mask = (df["zone"] == override["zone"]) & (df["timestamp"] == latest_timestamp)

        if not mask.any():
            continue

        capacity = df.loc[mask, "capacity"].iloc[0]
        target_count = round(capacity * override["target_pct"] / 100)
        df.loc[mask, "occupancy_count"] = target_count

    return df


def generate():
    rng = np.random.default_rng(RANDOM_SEED)

    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=DAYS_OF_HISTORY)

    timestamps = pd.date_range(
        start=start, end=end, freq=f"{READING_INTERVAL_MINUTES}min"
    ).to_pydatetime().tolist()

    all_records = []

    for zone_row in ZONES:
        all_records.extend(simulate_zone_history(zone_row, timestamps, rng))

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = apply_demo_seed_overrides(df)
    df["timestamp"] = df["timestamp"].apply(lambda t: t.isoformat())

    df.insert(0, "occupancy_id", range(1, len(df) + 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df):,} records across {len(ZONES)} zones")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    generate()