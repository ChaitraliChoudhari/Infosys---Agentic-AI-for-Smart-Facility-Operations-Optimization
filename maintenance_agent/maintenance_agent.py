import pandas as pd
from datetime import datetime, timedelta

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv("../dataset/predictive_maintenance_dataset.csv")

# -------------------------------------
# Health Score Calculation
# -------------------------------------
def calculate_health_score(row):

    score = 100

    if row["failure"] == 1:
        score -= 50

    if row["metric5"] > 15:
        score -= 10

    if row["metric6"] > 300000:
        score -= 15

    if row["metric9"] > 500:
        score -= 10

    if row["metric3"] > 100:
        score -= 10

    if row["metric4"] > 20:
        score -= 5

    return max(score, 0)

df["Health Score"] = df.apply(calculate_health_score, axis=1)

# -------------------------------------
# Equipment Status
# -------------------------------------
def equipment_status(score):
    if score >= 80:
        return "Healthy"
    elif score >= 50:
        return "Warning"
    else:
        return "Critical"

df["Status"] = df["Health Score"].apply(equipment_status)

# -------------------------------------
# Maintenance Schedule
# -------------------------------------
today = datetime.today()

def schedule(score):
    if score < 50:
        return (today + timedelta(days=3)).strftime("%Y-%m-%d"), "High"
    elif score < 80:
        return (today + timedelta(days=7)).strftime("%Y-%m-%d"), "Medium"
    else:
        return (today + timedelta(days=30)).strftime("%Y-%m-%d"), "Low"

maintenance = df["Health Score"].apply(schedule)

df["Maintenance Date"] = maintenance.apply(lambda x: x[0])
df["Priority"] = maintenance.apply(lambda x: x[1])

# -------------------------------------
# AI Recommendation
# -------------------------------------
def recommendation(row):

    if row["Priority"] == "High":
        return "Immediate maintenance required."

    elif row["Priority"] == "Medium":
        return "Schedule maintenance within one week."

    else:
        return "Equipment operating normally."

df["Recommendation"] = df.apply(recommendation, axis=1)

# -------------------------------------
# Display Report
# -------------------------------------
report = df[
    [
        "device",
        "Health Score",
        "Status",
        "Maintenance Date",
        "Priority",
        "Recommendation"
    ]
]

# Show highest-risk equipment first
report = report.sort_values(by="Health Score")

print("=" * 80)
print("PREDICTIVE MAINTENANCE REPORT")
print("=" * 80)

print(report.head(20))

print("\nStatus Summary")
print(df["Status"].value_counts())

print("\nPriority Summary")
print(df["Priority"].value_counts())