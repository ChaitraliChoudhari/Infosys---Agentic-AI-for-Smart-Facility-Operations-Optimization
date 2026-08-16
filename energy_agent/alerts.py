import pandas as pd

df = pd.read_csv("../dataset/cleaned_energy_consumption.csv")

average = df["EnergyConsumption"].mean()

alerts = []

for index, row in df.iterrows():

    if row["EnergyConsumption"] > average * 1.2:

        alerts.append({
            "Timestamp": row["Timestamp"],
            "Alert": "High Energy Consumption"
        })

alerts_df = pd.DataFrame(alerts)

print(alerts_df)