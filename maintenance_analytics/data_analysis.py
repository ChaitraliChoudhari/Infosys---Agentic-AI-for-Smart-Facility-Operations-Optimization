import pandas as pd

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv("../dataset/predictive_maintenance_dataset.csv")

# -------------------------------------
# Dataset Information
# -------------------------------------
print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print(df.info())

# -------------------------------------
# Missing Values
# -------------------------------------
print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------------
# Summary Statistics
# -------------------------------------
print("\nSummary Statistics")
print(df.describe())

# -------------------------------------
# Column Names
# -------------------------------------
print("\nDataset Columns")
print(df.columns.tolist())

# -------------------------------------
# Failure Count
# -------------------------------------
print("\nFailure Count")
print(df["failure"].value_counts())

# -------------------------------------
# Total Devices
# -------------------------------------
print("\nTotal Devices")
print(df["device"].nunique())

# -------------------------------------
# Total Records
# -------------------------------------
print("\nTotal Records")
print(len(df))

# -------------------------------------
# Failures Per Device
# -------------------------------------
print("\nFailures Per Device")
print(
    df.groupby("device")["failure"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

# -------------------------------------
# Average Sensor Values
# -------------------------------------
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

# -------------------------------------
# Correlation Matrix
# -------------------------------------
print("\nCorrelation Matrix")

print(df[sensor_columns + ["failure"]].corr())

print("\nData Analysis Completed Successfully.")