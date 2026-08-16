import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("../dataset/cleaned_energy_consumption.csv")

# Convert Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Create Hour column if not present
if "Hour" not in df.columns:
    df["Hour"] = df["Timestamp"].dt.hour

# -------- Graph 1 --------
# Average Energy Consumption by Hour

hourly = df.groupby("Hour")["EnergyConsumption"].mean()

plt.figure(figsize=(10,5))
plt.plot(hourly.index, hourly.values, marker="o")
plt.title("Average Energy Consumption by Hour")
plt.xlabel("Hour")
plt.ylabel("Energy Consumption")
plt.grid(True)
plt.show()

# -------- Graph 2 --------
# Average Energy Consumption by Day
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

df["Day"] = pd.Categorical(
    df["Timestamp"].dt.day_name(),
    categories=day_order,
    ordered=True
)

daily = df.groupby("Day", observed=True)["EnergyConsumption"].mean()

plt.figure(figsize=(8,5))
plt.bar(daily.index, daily.values)
plt.title("Average Energy Consumption by Day")
plt.xlabel("Day")
plt.ylabel("Energy Consumption")
plt.xticks(rotation=45)
plt.show()

# -------- Graph 3 --------
# Temperature vs Energy Consumption
plt.figure(figsize=(8,5))
plt.scatter(df["Temperature"], df["EnergyConsumption"])
plt.title("Temperature vs Energy Consumption")
plt.xlabel("Temperature")
plt.ylabel("Energy Consumption")
plt.show()

# -------- Graph 4 --------
# Occupancy vs Energy Consumption
plt.figure(figsize=(8,5))
plt.scatter(df["Occupancy"], df["EnergyConsumption"])
plt.title("Occupancy vs Energy Consumption")
plt.xlabel("Occupancy")
plt.ylabel("Energy Consumption")
plt.show()

# -------- Graph 5 --------
# HVAC Usage vs Energy Consumption
plt.figure(figsize=(8,5))
plt.scatter(df["HVACUsage"], df["EnergyConsumption"])
plt.title("HVAC Usage vs Energy Consumption")
plt.xlabel("HVAC Usage")
plt.ylabel("Energy Consumption")
plt.show()
