import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/cleaned_energy_consumption.csv")

average_energy = df["EnergyConsumption"].mean()

for index, row in df.iterrows():

    recommendations = []

    # HVAC Recommendation
    if row["HVACUsage"] == "On" and row["Temperature"] < 22:
        recommendations.append(
            "HVAC is ON even though temperature is low. Consider turning it OFF."
        )

    # Lighting Recommendation
    if row["LightingUsage"] == "On" and row["Occupancy"] < 10:
        recommendations.append(
            "Lighting is ON but occupancy is low. Switch off unnecessary lights."
        )

    # Renewable Energy Recommendation
    if row["RenewableEnergy"] < 10:
        recommendations.append(
            "Renewable energy generation is low."
        )

    # High Energy Consumption Recommendation
    if row["EnergyConsumption"] > average_energy * 1.2:
        recommendations.append(
            "Energy consumption is unusually high. Inspect HVAC and lighting."
        )

    if recommendations:
        print(f"\nTimestamp: {row['Timestamp']}")
        for rec in recommendations:
            print("•", rec)