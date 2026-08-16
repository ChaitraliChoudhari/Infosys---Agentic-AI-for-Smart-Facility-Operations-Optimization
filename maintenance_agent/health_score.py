import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/predictive_maintenance_dataset.csv")

# Calculate Health Score
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


# Apply health score
df["Health Score"] = df.apply(calculate_health_score, axis=1)

# Equipment Status
def equipment_status(score):

    if score >= 80:
        return "Healthy"

    elif score >= 50:
        return "Warning"

    else:
        return "Critical"


df["Status"] = df["Health Score"].apply(equipment_status)

print("=" * 50)
print("Equipment Health Scores")
print("=" * 50)

print(df[["device", "Health Score", "Status"]].head(20))

print("\nStatus Count")
print(df["Status"].value_counts())