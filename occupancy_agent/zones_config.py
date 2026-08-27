"""
zones_config.py
================
Single source of truth for the building's floor/zone layout and each
zone's configured capacity. Edit this file to match your real building
— everything else (data generation, occupancy %, overcrowding alerts,
heatmap, forecasting) reads from here.

facility_id is fixed at 1 for now (single facility). Add more entries
with a different facility_id if you ever have multiple buildings.
"""

ZONES = [
    # Floor 1
    {"facility_id": 1, "floor": 1, "zone": "Office A",        "capacity": 25},
    {"facility_id": 1, "floor": 1, "zone": "Office B",        "capacity": 20},
    {"facility_id": 1, "floor": 1, "zone": "Office C",        "capacity": 18},
    {"facility_id": 1, "floor": 1, "zone": "Office D",        "capacity": 15},
    {"facility_id": 1, "floor": 1, "zone": "Meeting Room 1",  "capacity": 12},
    {"facility_id": 1, "floor": 1, "zone": "Conference Room", "capacity": 20},
    {"facility_id": 1, "floor": 1, "zone": "Cafeteria",       "capacity": 40},
    {"facility_id": 1, "floor": 1, "zone": "Lobby",           "capacity": 30},
    {"facility_id": 1, "floor": 1, "zone": "Storage",         "capacity": 10},

    # Floor 2
    {"facility_id": 1, "floor": 2, "zone": "Office E",        "capacity": 20},
    {"facility_id": 1, "floor": 2, "zone": "Office F",        "capacity": 20},
    {"facility_id": 1, "floor": 2, "zone": "Meeting Room 2",  "capacity": 10},
    {"facility_id": 1, "floor": 2, "zone": "Mobile Zone",     "capacity": 15},
    {"facility_id": 1, "floor": 2, "zone": "Common Area",     "capacity": 25},
    {"facility_id": 1, "floor": 2, "zone": "Storage",         "capacity": 10},

    # Floor 3
    {"facility_id": 1, "floor": 3, "zone": "Office G",        "capacity": 18},
    {"facility_id": 1, "floor": 3, "zone": "Office H",        "capacity": 18},
    {"facility_id": 1, "floor": 3, "zone": "Office I",        "capacity": 15},
    {"facility_id": 1, "floor": 3, "zone": "Meeting Room 3",  "capacity": 10},
    {"facility_id": 1, "floor": 3, "zone": "Project Room",    "capacity": 12},
    {"facility_id": 1, "floor": 3, "zone": "Common Area",     "capacity": 20},
    {"facility_id": 1, "floor": 3, "zone": "Break Area",      "capacity": 15},
    {"facility_id": 1, "floor": 3, "zone": "Storage",         "capacity": 10},
]

# Roughly how "busy" each zone type tends to be, as a fraction of capacity,
# during business hours. Used only by the synthetic data generator to make
# the historical dataset look realistic (offices busiest, storage nearly
# empty, meeting rooms spiky). Purely a simulation knob — has no effect
# once you're feeding in real occupancy readings.
ZONE_TYPE_BUSY_FRACTION = {
    "Office": 0.75,
    "Meeting Room": 0.45,
    "Conference Room": 0.55,
    "Cafeteria": 0.50,
    "Lobby": 0.60,
    "Common Area": 0.45,
    "Mobile Zone": 0.35,
    "Project Room": 0.40,
    "Break Area": 0.30,
    "Storage": 0.15,
}


def zone_type(zone_name):
    """Map a specific zone name ('Office A') to its general type ('Office')."""
    for type_name in ZONE_TYPE_BUSY_FRACTION:
        if zone_name.startswith(type_name):
            return type_name
    return "Common Area"