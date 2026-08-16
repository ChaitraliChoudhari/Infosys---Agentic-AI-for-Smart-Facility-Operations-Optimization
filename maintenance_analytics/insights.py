import pandas as pd

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv("../dataset/predictive_maintenance_dataset.csv")

print("=" * 60)
print("PREDICTIVE MAINTENANCE INSIGHTS")
print("=" * 60)

# Total Records
print(f"\nTotal Records: {len(df)}")

# Total Devices
print(f"Total Devices: {df['device'].nunique()}")

# Total Failures
total_failures = df["failure"].sum()
print(f"Total Failures: {total_failures}")

# Failure Percentage
failure_percentage = (total_failures / len(df)) * 100
print(f"Failure Percentage: {failure_percentage:.2f}%")

# Top 5 Devices with Most Failures
print("\nTop 5 Devices with Highest Failures")

top_devices = (
    df.groupby("device")["failure"]
      .sum()
      .sort_values(ascending=False)
      .head(5)
)

print(top_devices)

# Average Sensor Values
print("\nAverage Sensor Values")

sensor_columns = [
    "metric1",
    "metric2",
    "metric3",
    "metric4",
    "metric5",
    "metric6",
    "metric7",
    "metric8",
    "metric9"
]

print(df[sensor_columns].mean())

# Devices That Experienced Failure
failed_devices = df[df["failure"] == 1]["device"].nunique()

print(f"\nDevices That Experienced Failure: {failed_devices}")

print("\nMaintenance Insights")

if failure_percentage < 1:
    print("✔ Failure rate is very low.")
else:
    print("⚠ High failure rate detected.")

print("✔ Monitor devices with repeated failures.")
print("✔ Schedule preventive maintenance for high-risk devices.")
print("✔ Continuously monitor sensor metrics.")
print("✔ Use health scores to prioritize maintenance.")

print("\nInsights Generated Successfully.")