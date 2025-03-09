import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()  # Authenticate using the kaggle.json file

# Define the dataset you want to download
dataset = "prasad22/weather-data"  # Dataset name from the Kaggle URL

# Create a directory to store the downloaded data
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Download the dataset
api.dataset_download_files(dataset, path=output_dir, unzip=True)

print(f"Dataset downloaded and extracted to {output_dir}")