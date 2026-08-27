"""
generate_security_data.py
==========================
Generates a synthetic-but-realistic Security Agent dataset, aligned to
zones_config.py so it joins with the Occupancy Agent data on
(facility_id, floor, zone).

Outputs (CSV, written to ./data/):
    EMPLOYEES.csv          badge_id, name, department, access_level
    ACCESS_LOGS.csv        badge-in/out attempts at each zone's door
    VISITORS.csv            one row per visitor visit (check-in/out)
    VISITOR_MOVEMENTS.csv   zone-by-zone path each visitor took
    CCTV_EVENTS.csv         camera-detected events per zone
    SECURITY_EVENTS.csv     rule-derived alerts (the SECURITY_EVENTS table)

Run:  python generate_security_data.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "occupancy_agent"))

from zones_config import ZONES
from security_config import (
    ACCESS_LEVELS,
    CCTV_EVENT_TYPES,
    camera_id_for_zone,
    door_id_for_zone,
    is_restricted_zone,
    is_within_working_hours,
    required_level_for_zone,
)

random.seed(42)
np.random.seed(42)

FACILITY_ID = 1
DAYS_OF_HISTORY = 30
START_DATE = datetime(2025, 12, 1)  # 30 days ending ~Dec 30 2025

# Matches how OCCUPANCY_CSV_PATH etc. are defined in backend/main.py:
# BASE_DIR.parent / "dataset" / "<name>.csv". Adjust if this script
# doesn't live at security_agent/ (i.e. one level under project root).
OUT_DIR = str(Path(__file__).resolve().parent.parent / "dataset")

CSV_NAMES = {
    "employees": "security_employees.csv",
    "access_logs": "security_access_logs.csv",
    "visitors": "security_visitors.csv",
    "visitor_movements": "security_visitor_movements.csv",
    "cctv_events": "security_cctv_events.csv",
    "security_events": "security_events.csv",
}

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Facilities", "IT", "Management"]
FIRST_NAMES = ["Aakash", "Ryan", "Satha", "Harsh", "Cathy", "Priya", "Rahul", "Neha", "Kiran",
               "Divya", "Arjun", "Meera", "Vikram", "Sneha", "Rohan", "Anita", "Suresh", "Pooja"]
LAST_NAMES = ["Sharma", "Gosavi", "Sivam", "Khurana", "Verma", "Iyer", "Nair", "Rao",
              "Reddy", "Patel", "Singh", "Gupta", "Menon", "Das", "Chatterjee"]

VISITOR_FIRST = ["Alex", "Jordan", "Sam", "Taylor", "Chris", "Morgan", "Jamie", "Riley"]
VISITOR_LAST = ["Fernandes", "D'Souza", "Pinto", "Rao", "Mehta", "Shah", "Costa", "Almeida"]


# ---------------------------------------------------------------------------
# EMPLOYEES
# ---------------------------------------------------------------------------
def generate_employees(n=90):
    rows = []
    for i in range(1, n + 1):
        dept = random.choice(DEPARTMENTS)
        if dept == "IT":
            level = 4
        elif dept == "Management":
            level = 3
        elif dept == "Facilities":
            level = random.choice([2, 4])
        else:
            level = random.choice([1, 1, 1, 2])  # mostly level 1
        rows.append({
            "badge_id": f"BADGE-{i:04d}",
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "department": dept,
            "access_level": level,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ACCESS LOGS  (badge swipes at each zone's door)
# ---------------------------------------------------------------------------
def generate_access_logs(employees_df):
    rows = []
    log_id = 1
    open_zones = [z for z in ZONES if not is_restricted_zone(z["zone"])]
    restricted_zones = [z for z in ZONES if is_restricted_zone(z["zone"])]

    for day in range(DAYS_OF_HISTORY):
        date = START_DATE + timedelta(days=day)
        is_workday = date.weekday() < 5

        for _, emp in employees_df.iterrows():
            # most employees skip weekends; a few (Facilities/IT) still show up
            if not is_workday and emp["access_level"] < 4 and random.random() > 0.05:
                continue

            # each employee visits a handful of ordinary (non-restricted) zones
            n_visits = random.randint(2, 4)
            visited_zones = random.sample(open_zones, k=min(n_visits, len(open_zones)))

            # restricted-zone visits are rare and mostly belong to staff who
            # actually have business there (IT/Facilities). A small fraction
            # of visits from unauthorized staff simulate real policy violations
            # worth flagging, rather than uniform random access to every door.
            if emp["access_level"] >= 4:
                if random.random() < 0.35:
                    visited_zones.append(random.choice(restricted_zones))
            else:
                if random.random() < 0.02:
                    visited_zones.append(random.choice(restricted_zones))

            # base arrival time, normal ~9am with some spread; occasional after-hours
            base_hour = np.clip(np.random.normal(9, 1.3), 6, 22)
            minutes_offset = 0

            for z in visited_zones:
                ts = date.replace(hour=0, minute=0, second=0) + timedelta(
                    minutes=int(base_hour * 60) + minutes_offset + random.randint(0, 45)
                )
                minutes_offset += random.randint(20, 180)

                required_level = required_level_for_zone(z["zone"])
                has_clearance = emp["access_level"] >= required_level

                # failed attempts happen occasionally even with clearance
                # (mistyped PIN, forgotten badge, etc.) but far more often
                # when the employee lacks the required clearance
                if has_clearance:
                    granted = random.random() > 0.03
                else:
                    granted = random.random() > 0.85  # rare accidental grant (misconfig)

                rows.append({
                    "log_id": log_id,
                    "facility_id": FACILITY_ID,
                    "floor": z["floor"],
                    "zone": z["zone"],
                    "door_id": door_id_for_zone(FACILITY_ID, z["floor"], z["zone"]),
                    "badge_id": emp["badge_id"],
                    "access_granted": bool(granted),
                    "required_level": required_level,
                    "badge_level": emp["access_level"],
                    "timestamp": ts.isoformat(sep=" "),
                })
                log_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VISITORS + VISITOR MOVEMENTS
# ---------------------------------------------------------------------------
def generate_visitors_and_movements(employees_df, n_visitors=60):
    visitor_rows = []
    movement_rows = []
    move_id = 1

    normal_paths = [
        ["Lobby", "Meeting Room 1"],
        ["Lobby", "Conference Room"],
        ["Lobby", "Office A", "Meeting Room 1"],
        ["Lobby", "Common Area", "Meeting Room 2"],
        ["Lobby", "Meeting Room 3"],
    ]
    # a small fraction of visits deviate into a restricted zone unescorted
    flagged_paths = [
        ["Lobby", "Common Area", "Storage"],
        ["Lobby", "Office E", "Storage"],
        ["Lobby", "Meeting Room 2", "Storage"],
    ]

    for i in range(1, n_visitors + 1):
        day = random.randint(0, DAYS_OF_HISTORY - 1)
        date = START_DATE + timedelta(days=day)
        if date.weekday() >= 5:
            date -= timedelta(days=date.weekday() - 4)  # push weekend visits to Friday

        host = employees_df.sample(1).iloc[0]
        entry_hour = np.clip(np.random.normal(11, 2.5), 7, 18)
        entry_time = date.replace(hour=0, minute=0, second=0) + timedelta(minutes=int(entry_hour * 60))

        flagged = random.random() < 0.08  # ~8% of visits deviate into a restricted zone
        path = random.choice(flagged_paths) if flagged else random.choice(normal_paths)

        visit_minutes = random.randint(30, 150)
        exit_time = entry_time + timedelta(minutes=visit_minutes)

        floor_by_zone = {z["zone"]: z["floor"] for z in ZONES}

        visitor_rows.append({
            "visitor_id": f"VIS-{i:04d}",
            "name": f"{random.choice(VISITOR_FIRST)} {random.choice(VISITOR_LAST)}",
            "host_badge_id": host["badge_id"],
            "host_name": host["name"],
            "facility_id": FACILITY_ID,
            "entry_time": entry_time.isoformat(sep=" "),
            "exit_time": exit_time.isoformat(sep=" "),
            "flagged_unescorted_restricted_access": flagged,
        })

        step_minutes = visit_minutes / max(len(path), 1)
        for seq, zone in enumerate(path, start=1):
            ts = entry_time + timedelta(minutes=step_minutes * (seq - 1))
            movement_rows.append({
                "movement_id": move_id,
                "visitor_id": f"VIS-{i:04d}",
                "facility_id": FACILITY_ID,
                "sequence": seq,
                "floor": floor_by_zone.get(zone, 1),
                "zone": zone,
                "timestamp": ts.isoformat(sep=" "),
            })
            move_id += 1

    return pd.DataFrame(visitor_rows), pd.DataFrame(movement_rows)


# ---------------------------------------------------------------------------
# CCTV EVENTS
# ---------------------------------------------------------------------------
def generate_cctv_events(n_events=500):
    rows = []
    event_types = list(CCTV_EVENT_TYPES.keys())
    # weight toward benign motion detection, rarer for higher-severity types
    weights = [0.45, 0.18, 0.10, 0.12, 0.10, 0.05]

    for i in range(1, n_events + 1):
        z = random.choice(ZONES)
        day = random.randint(0, DAYS_OF_HISTORY - 1)
        date = START_DATE + timedelta(days=day)

        # tailgating/loitering skew toward restricted zones + after-hours
        etype = random.choices(event_types, weights=weights, k=1)[0]
        if is_restricted_zone(z["zone"]) and random.random() < 0.4:
            etype = random.choice(["tailgating", "loitering", "motion_detected"])

        if etype in ("tailgating", "loitering") and random.random() < 0.5:
            hour = random.choice([random.randint(20, 23), random.randint(0, 6)])
        else:
            hour = random.randint(7, 20)

        ts = date.replace(hour=0, minute=0, second=0) + timedelta(
            hours=hour, minutes=random.randint(0, 59)
        )

        rows.append({
            "cctv_event_id": i,
            "facility_id": FACILITY_ID,
            "floor": z["floor"],
            "zone": z["zone"],
            "camera_id": camera_id_for_zone(FACILITY_ID, z["floor"], z["zone"]),
            "event_type": etype,
            "confidence": round(random.uniform(0.55, 0.99), 2),
            "timestamp": ts.isoformat(sep=" "),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SECURITY EVENTS  (derived from the three sources above via rules)
# matches the spec's table: event_id (PK), facility_id (FK), event_type,
# severity, timestamp  — plus a few extra descriptive columns.
# ---------------------------------------------------------------------------
def derive_security_events(access_logs, visitors, movements, cctv_events):
    events = []
    eid = 1

    # 1) Access control: denied swipes
    denied = access_logs[~access_logs["access_granted"]].copy()
    denied["ts"] = pd.to_datetime(denied["timestamp"])
    for badge_id, grp in denied.groupby("badge_id"):
        grp = grp.sort_values("ts")
        # flag repeated failures by the same badge within a rolling 2-hour window
        window_start = None
        window_count = 0
        for _, row in grp.iterrows():
            if window_start is None or (row["ts"] - window_start) > timedelta(hours=2):
                window_start = row["ts"]
                window_count = 1
            else:
                window_count += 1

            restricted = is_restricted_zone(row["zone"])
            after_hours = not is_within_working_hours(row["ts"])

            if restricted and window_count >= 2 and after_hours:
                severity, etype = "CRITICAL", "REPEATED_UNAUTHORIZED_ACCESS"
            elif restricted:
                severity, etype = "HIGH", "UNAUTHORIZED_ACCESS_RESTRICTED_AREA"
            elif window_count >= 2:
                severity, etype = "MEDIUM", "MULTIPLE_FAILED_ACCESS_ATTEMPTS"
            else:
                severity, etype = "LOW", "DOOR_ACCESS_DENIED"

            events.append({
                "event_id": eid, "facility_id": FACILITY_ID, "floor": row["floor"],
                "zone": row["zone"], "event_type": etype, "severity": severity,
                "source": "ACCESS_CONTROL",
                "description": f"Badge {badge_id} denied entry at {row['zone']} "
                                f"(door {row['door_id']})",
                "timestamp": row["timestamp"],
            })
            eid += 1

    # 2) Access control: after-hours entry to a restricted zone even if granted
    granted = access_logs[access_logs["access_granted"]].copy()
    granted["ts"] = pd.to_datetime(granted["timestamp"])
    for _, row in granted.iterrows():
        if is_restricted_zone(row["zone"]) and not is_within_working_hours(row["ts"]):
            events.append({
                "event_id": eid, "facility_id": FACILITY_ID, "floor": row["floor"],
                "zone": row["zone"], "event_type": "AFTER_HOURS_RESTRICTED_ACCESS",
                "severity": "MEDIUM", "source": "ACCESS_CONTROL",
                "description": f"Badge {row['badge_id']} accessed {row['zone']} outside working hours",
                "timestamp": row["timestamp"],
            })
            eid += 1

    # 3) CCTV: escalate tailgating / after-hours loitering in restricted zones
    for _, row in cctv_events.iterrows():
        base_severity = CCTV_EVENT_TYPES.get(row["event_type"], "LOW")
        restricted = is_restricted_zone(row["zone"])
        ts = pd.to_datetime(row["timestamp"])
        after_hours = not is_within_working_hours(ts)

        severity = base_severity
        if row["event_type"] == "tailgating" and restricted:
            severity = "CRITICAL" if after_hours else "HIGH"
        elif row["event_type"] == "loitering" and restricted and after_hours:
            severity = "HIGH"
        elif restricted and base_severity == "LOW":
            severity = "MEDIUM"

        # only surface CCTV events as security alerts when they're at least
        # MEDIUM, or restricted-zone related — keep pure benign motion out of the log
        if severity in ("MEDIUM", "HIGH", "CRITICAL") or (restricted and row["event_type"] != "motion_detected"):
            events.append({
                "event_id": eid, "facility_id": FACILITY_ID, "floor": row["floor"],
                "zone": row["zone"], "event_type": f"CCTV_{row['event_type'].upper()}",
                "severity": severity, "source": "CCTV",
                "description": f"Camera {row['camera_id']} detected {row['event_type'].replace('_',' ')} "
                                f"(confidence {row['confidence']})",
                "timestamp": row["timestamp"],
            })
            eid += 1

    # 4) Visitor movement: unescorted entry into a restricted zone
    flagged_visitors = visitors[visitors["flagged_unescorted_restricted_access"]]
    for _, v in flagged_visitors.iterrows():
        v_moves = movements[movements["visitor_id"] == v["visitor_id"]].sort_values("sequence")
        restricted_hits = v_moves[v_moves["zone"].apply(is_restricted_zone)]
        for _, m in restricted_hits.iterrows():
            events.append({
                "event_id": eid, "facility_id": FACILITY_ID, "floor": m["floor"],
                "zone": m["zone"], "event_type": "VISITOR_UNESCORTED_RESTRICTED_ZONE",
                "severity": "HIGH", "source": "VISITOR_TRACKING",
                "description": f"Visitor {v['name']} ({v['visitor_id']}, hosted by {v['host_name']}) "
                                f"entered restricted zone {m['zone']} unescorted",
                "timestamp": m["timestamp"],
            })
            eid += 1

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["event_id"] = range(1, len(df) + 1)
    return df


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating employees...")
    employees = generate_employees()
    employees.to_csv(f"{OUT_DIR}/{CSV_NAMES['employees']}", index=False)

    print("Generating access logs...")
    access_logs = generate_access_logs(employees)
    access_logs.to_csv(f"{OUT_DIR}/{CSV_NAMES['access_logs']}", index=False)

    print("Generating visitors + movements...")
    visitors, movements = generate_visitors_and_movements(employees)
    visitors.to_csv(f"{OUT_DIR}/{CSV_NAMES['visitors']}", index=False)
    movements.to_csv(f"{OUT_DIR}/{CSV_NAMES['visitor_movements']}", index=False)

    print("Generating CCTV events...")
    cctv_events = generate_cctv_events()
    cctv_events.to_csv(f"{OUT_DIR}/{CSV_NAMES['cctv_events']}", index=False)

    print("Deriving security events...")
    security_events = derive_security_events(access_logs, visitors, movements, cctv_events)
    security_events.to_csv(f"{OUT_DIR}/{CSV_NAMES['security_events']}", index=False)

    print(f"\nDone. Files written to {OUT_DIR}/")
    print("Row counts:")
    print(f"  {CSV_NAMES['employees']:<32}{len(employees)}")
    print(f"  {CSV_NAMES['access_logs']:<32}{len(access_logs)}")
    print(f"  {CSV_NAMES['visitors']:<32}{len(visitors)}")
    print(f"  {CSV_NAMES['visitor_movements']:<32}{len(movements)}")
    print(f"  {CSV_NAMES['cctv_events']:<32}{len(cctv_events)}")
    print(f"  {CSV_NAMES['security_events']:<32}{len(security_events)}")
    print("\nSecurity events by severity:")
    print(security_events["severity"].value_counts())


if __name__ == "__main__":
    main()