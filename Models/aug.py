import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define the number of rows in the dataset
num_rows = 1000000  

# Define min and max values for each variable
temp_min, temp_max = -19.96, 39.999
humid_min, humid_max = 30, 89
precip_min, precip_max = 8.90415, 14.971
wind_min, wind_max = 0.000050648, 29.999
yield_min, yield_max = 0.0, 5  # Crop yield in tonnes

# Function to generate crop yield based on weather variables
def generate_crop_yield(temp, humid, precip, wind):
    # Normalize the weather variables
    normalized_temp = (temp - temp_min) / (temp_max - temp_min)
    normalized_humid = (humid - humid_min) / (humid_max - humid_min)
    normalized_precip = (precip - precip_min) / (precip_max - precip_min)
    normalized_wind = (wind - wind_min) / (wind_max - wind_min)

    # Simulate crop yield based on normalized variables
    # Higher temperature and moderate humidity increase yield
    # High precipitation and wind speed decrease yield
    crop_yield = (
        0.4 * normalized_temp +  # Temperature contribution
        0.3 * (1 - abs(normalized_humid - 0.5)) +  # Ideal humidity is around 50%
        0.2 * (1 - normalized_precip) +  # Less precipitation is better
        0.1 * (1 - normalized_wind)  # Less wind is better
    )

    # Scale crop yield to the desired range (0.1 to 5 tonnes)
    crop_yield = crop_yield * (yield_max - yield_min) + yield_min

    # Add some random noise to make the data more realistic
    crop_yield += random.uniform(-0.1, 0.1)

    # Ensure crop yield stays within bounds
    return max(yield_min, min(yield_max, crop_yield))

# Generate synthetic data
data = []
for _ in range(num_rows):
    temp = random.uniform(temp_min, temp_max)
    humid = random.uniform(humid_min, humid_max)
    precip = random.uniform(precip_min, precip_max)
    wind = random.uniform(wind_min, wind_max)
    crop_yield = generate_crop_yield(temp, humid, precip, wind)

    data.append([temp, humid, precip, wind, crop_yield])

# Create a DataFrame
df = pd.DataFrame(data, columns=["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh", "Crop_Yield_tonnes"])

# Save the dataset to a CSV file
df.to_csv("synthetic_weather_data.csv", index=False)

print("Synthetic dataset generated and saved!")