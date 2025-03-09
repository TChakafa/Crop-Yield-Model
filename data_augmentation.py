import pandas as pd

# Load the dataset
df = pd.read_csv("weather_data.csv")

# Define min and max values for each variable
temp_min, temp_max = -19.96, 39.999
humid_min, humid_max = 30, 89
precip_min, precip_max = 8.90415, 14.971
wind_min, wind_max = 0.000050648, 29.999

# Normalize the variables
df["Normalized_Temp"] = (df["Temperature_C"] - temp_min) / (temp_max - temp_min)
df["Normalized_Humid"] = (df["Humidity_pct"] - humid_min) / (humid_max - humid_min)
df["Normalized_Precip"] = (df["Precipitation_mm"] - precip_min) / (precip_max - precip_min)
df["Normalized_Wind_S"] = (df["Wind_Speed_kmh"] - wind_min) / (wind_max - wind_min)

# Calculate crop yield using weighted contributions
df["Crop_Yield"] = (
    0.4 * df["Normalized_Temp"] +
    0.3 * df["Normalized_Humid"] +
    0.2 * df["Normalized_Precip"] +
    0.1 * df["Normalized_Wind_S"]
)

# Scale crop yield to the desired range (0.0 to 5.0)
df["Crop_Yield"] = df["Crop_Yield"] * 5.0

# Drop normalized columns (optional)
df.drop(columns=["Normalized_Temp", "Normalized_Humid", "Normalized_Precip", "Normalized_Wind_S"], inplace=True)

# Add a new column "Season(Jan-May)" with all values set to 1
df["Season(Jan-May)"] = 1
# Save the updated dataset
df.to_csv("weather_data_augmented.csv", index=False)

print("Crop yield column added and dataset saved!")