"""
security_config.py
===================
Single source of truth for the Security Agent's access-control rules.
Builds directly on zones_config.py so Security Agent data joins cleanly
with Occupancy Agent data on (facility_id, floor, zone).

Edit this file to match real building security policy — the data
generator and the rule-based alerting logic both read from here.
"""

from zones_config import ZONES, zone_type

# ---------------------------------------------------------------------------
# Access levels (higher = more privileged)
#   1 = General staff        — public/common areas, own-department offices
#   2 = Department lead      — project rooms, cross-department offices
#   3 = Management           — conference rooms, admin areas
#   4 = IT / Facilities Ops  — restricted/server areas (Storage zones)
# ---------------------------------------------------------------------------
ACCESS_LEVELS = {1: "General Staff", 2: "Department Lead", 3: "Management", 4: "IT / Facilities"}

# Required access level to enter each zone TYPE. Zone types not listed
# default to level 1 (open to all badged staff).
ZONE_REQUIRED_LEVEL = {
    "Storage": 4,          # treated as restricted/server-equivalent room per floor
    "Conference Room": 3,
    "Project Room": 2,
}

# Zones treated as "restricted areas" for unauthorized-access-detection logic.
# (Currently == the zone types requiring level >= 3, kept separate in case
# the policy diverges later, e.g. a restricted zone open to level 2+.)
RESTRICTED_ZONE_TYPES = {"Storage", "Conference Room"}

# Business hours (24h). Access/CCTV activity outside this window on a
# restricted zone is treated as higher-severity.
WORKING_HOURS = {"start_hour": 8, "end_hour": 19}  # 08:00–19:00
WORKING_DAYS = {0, 1, 2, 3, 4}  # Mon-Fri (Python weekday(): Mon=0)

# CCTV event types the synthetic generator can emit, with a base severity
# hint used when a CCTV event is escalated into a SECURITY_EVENTS row.
CCTV_EVENT_TYPES = {
    "motion_detected": "LOW",
    "loitering": "MEDIUM",
    "tailgating": "HIGH",
    "object_left_unattended": "MEDIUM",
    "crowd_detected": "LOW",
    "camera_offline": "MEDIUM",
}

# Severity tiers exactly as defined by the project spec.
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def required_level_for_zone(zone_name: str) -> int:
    ztype = zone_type(zone_name)
    return ZONE_REQUIRED_LEVEL.get(ztype, 1)


def is_restricted_zone(zone_name: str) -> bool:
    return zone_type(zone_name) in RESTRICTED_ZONE_TYPES


def is_within_working_hours(dt) -> bool:
    return (
        dt.weekday() in WORKING_DAYS
        and WORKING_HOURS["start_hour"] <= dt.hour < WORKING_HOURS["end_hour"]
    )


# Doors: one door per zone, ids derived from zone (deterministic, so they
# join cleanly with ZONES). Cameras: one camera per zone, same idea.
def door_id_for_zone(facility_id: int, floor: int, zone: str) -> str:
    slug = zone.replace(" ", "")
    return f"DOOR-F{floor}-{slug}"


def camera_id_for_zone(facility_id: int, floor: int, zone: str) -> str:
    slug = zone.replace(" ", "")
    return f"CAM-F{floor}-{slug}"