import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv("../dataset/predictive_maintenance_dataset.csv")

# -------------------------------------
# Failure Distribution
# -------------------------------------
plt.figure(figsize=(6,4))
df["failure"].value_counts().plot(kind="bar")
plt.title("Failure Distribution")
plt.xlabel("Failure")
plt.ylabel("Count")
plt.grid(axis="y")
plt.show()

# -------------------------------------
# Top 10 Devices with Failures
# -------------------------------------
device_failure = (
    df.groupby("device")["failure"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,5))
device_failure.plot(kind="bar")
plt.title("Top 10 Devices with Highest Failures")
plt.xlabel("Device")
plt.ylabel("Number of Failures")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

# -------------------------------------
# Metric 1 Distribution
# -------------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["metric1"], bins=30)
plt.title("Metric1 Distribution")
plt.xlabel("Metric1")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# -------------------------------------
# Metric 5 Distribution
# -------------------------------------
plt.figure(figsize=(8,5))
plt.hist(df["metric5"], bins=30)
plt.title("Metric5 Distribution")
plt.xlabel("Metric5")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# -------------------------------------
# Correlation Heatmap
# -------------------------------------
plt.figure(figsize=(10,8))

corr = df[
    [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9",
        "failure"
    ]
].corr()

plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("Visualizations Generated Successfully.")