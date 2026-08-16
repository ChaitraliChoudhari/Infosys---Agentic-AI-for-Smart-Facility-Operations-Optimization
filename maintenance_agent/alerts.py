import pandas as pd

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
# Generate Alerts
# -----------------------------
print("=" * 60)
print("MAINTENANCE ALERTS")
print("=" * 60)

alerts_found = False

for _, row in df.iterrows():

    if row["Health Score"] < 50:
        print(f"🔴 CRITICAL: {row['device']} requires immediate maintenance.")
        alerts_found = True

    elif row["Health Score"] < 80:
        print(f"🟡 WARNING: {row['device']} should be inspected soon.")
        alerts_found = True

if not alerts_found:
    print("🟢 No maintenance alerts. All equipment is operating normally.")