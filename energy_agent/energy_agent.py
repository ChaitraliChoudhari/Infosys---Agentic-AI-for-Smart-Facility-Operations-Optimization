import pandas as pd

# Load cleaned dataset
df = pd.read_csv("../dataset/cleaned_energy_consumption.csv")

# Calculate average energy consumption
average_energy = df["EnergyConsumption"].mean()

print(f"Average Energy Consumption: {average_energy:.2f}\n")

# Analyse each record
for index, row in df.iterrows():

    energy = row["EnergyConsumption"]

    if energy > average_energy * 1.20:
        status = "HIGH"

    elif energy < average_energy * 0.80:
        status = "LOW"

    else:
        status = "NORMAL"

    print(f"Row {index+1}: {status}")