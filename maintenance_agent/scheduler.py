import pandas as pd
from datetime import datetime, timedelta

# Load dataset
df = pd.read_csv("../dataset/predictive_maintenance_dataset.csv")

# -----------------------------
# Health Score Function
# -----------------------------
def calculate_health_score(row):

    score = 100

    if row["failure"] == 1:
        score -= 50

    if row["metric5"] > 20:
        score -= 15

    if row["metric6"] > 350000:
        score -= 10

    if row["metric9"] > 1000:
        score -= 15

    return max(score, 0)

df["Health Score"] = df.apply(calculate_health_score, axis=1)

# -----------------------------
# Maintenance Scheduler
# -----------------------------
today = datetime.today()

def maintenance_schedule(score):

    if score < 50:
        return today + timedelta(days=3), "High"

    elif score < 80:
        return today + timedelta(days=7), "Medium"

    else:
        return today + timedelta(days=30), "Low"

schedule = df["Health Score"].apply(maintenance_schedule)

df["Maintenance Date"] = schedule.apply(lambda x: x[0].strftime("%Y-%m-%d"))
df["Priority"] = schedule.apply(lambda x: x[1])

print("=" * 60)
print("PREDICTIVE MAINTENANCE SCHEDULE")
print("=" * 60)

print(df[
    [
        "device",
        "Health Score",
        "Maintenance Date",
        "Priority"
    ]
].head(20))

print("\nPriority Count")
print(df["Priority"].value_counts())