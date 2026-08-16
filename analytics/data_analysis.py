import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/Energy_consumption.csv")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Create new columns
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day_name()
df["Month"] = df["Timestamp"].dt.month

# Save cleaned dataset
df.to_csv("../dataset/cleaned_energy_consumption.csv", index=False)

print("\nDataset cleaned successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nAverage Energy Consumption:")
print(df["EnergyConsumption"].mean())

print("\nMaximum Energy Consumption:")
print(df["EnergyConsumption"].max())

print("\nMinimum Energy Consumption:")
print(df["EnergyConsumption"].min())

hourly = df.groupby("Hour")["EnergyConsumption"].mean()

print("\nAverage Energy Consumption by Hour")
print(hourly)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

df["Day"] = pd.Categorical(df["Day"], categories=day_order, ordered=True)

daily = df.groupby("Day", observed=True)["EnergyConsumption"].mean().sort_index()

print("\nAverage Energy Consumption by Day")
print(daily)