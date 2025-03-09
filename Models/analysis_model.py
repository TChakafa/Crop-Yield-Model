#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import folium
#import geopandas as gpd
import plotly.express as px
#import geodatasets
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display


# # Importing the Dataset  for analysis

# In[171]:


# Load the dataset
df = pd.read_csv("weather_data_augmented.csv")
#View the first 10 rows
df.head(10)


# # Renaming colum names

# In[174]:


# Rename columns
df.rename(columns={
    "Date_Time": "Date",
    "Temperature_C": "Temperature(C)",
    "Humidity_pct": "Humidity(pct)",
    "Precipitation_mm": "Precipitation(mm)",
    "Wind_Speed_kmh": "Wind_Speed(km/h)"
}, inplace=True)


# # Removing the time component in the date column

# In[177]:


# Remove time from the Date column
df["Date"] = pd.to_datetime(df["Date"]).dt.date


# In[179]:


#Check if changes have been applied
df.head(5)


# In[181]:


# Dictionary mapping locations to their coordinates
coordinates = {
    'Dallas': (32.7767, -96.7970),
    'Phoenix': (33.4484, -112.0740),
    'San Diego': (32.7157, -117.1611),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'San Antonio': (29.4241, -98.4936),
    'New York': (40.7128, -74.0060),
    'Philadelphia': (39.9526, -75.1652),
    'San Jose': (37.3382, -121.8863),
    'Los Angeles': (34.0522, -118.2437)
}


# In[183]:


aggregated_data = df.groupby('Location').agg({
    'Temperature(C)': 'mean', 
    'Humidity(pct)': 'mean', 
    'Precipitation(mm)': 'sum', 
    'Wind_Speed(km/h)': 'mean'
}).reset_index()


# In[185]:


#Add latitude and longitude based on the dictionary
aggregated_data['Latitude'] = aggregated_data['Location'].map(lambda x: coordinates[x][0] if x in coordinates else None)
aggregated_data['Longitude'] = aggregated_data['Location'].map(lambda x: coordinates[x][1] if x in coordinates else None)


# In[187]:


#Filter the DataFrame to include only specified locations
filtered_data = aggregated_data[aggregated_data['Location'].isin(coordinates.keys())]

# Create a Folium map centered around North America
m = folium.Map(location=[37.0, -95.0], zoom_start=4)  # Centered on the continental US

# Define colors for each location
color_map = {
    'Dallas': 'blue',
    'Phoenix': 'orange',
    'San Diego': 'green',
    'Chicago': 'red',
    'Houston': 'purple',
    'San Antonio': 'darkblue',
    'New York': 'darkgreen',
    'Philadelphia': 'darkred',
    'San Jose': 'black',
    'Los Angeles': 'yellow'
}

# Add markers for each location with specific colors
for _, row in filtered_data.iterrows():
    if row['Location'] in color_map:
        folium.Marker(
            location=[coordinates[row['Location']][0], coordinates[row['Location']][1]],
            popup=f"Location: {row['Location']}<br>"
                  f"Temp: {row['Temperature(C)']:.1f} °C<br>"
                  f"Humidity: {row['Humidity(pct)']:.1f} %<br>"
                  f"Precipitation: {row['Precipitation(mm)']:.1f} mm<br>"
                  f"Wind Speed: {row['Wind_Speed(km/h)']:.1f} km/h",
            icon=folium.Icon(color=color_map[row['Location']])
        ).add_to(m)

# Add a legend to the map with clickable links
legend_html = '''
<div style="position: fixed; 
            top: 10px; left: 10px; width: 150px; height: auto; 
            border:2px solid grey; z-index:1000; font-size:14px;
            background-color: white;">
    <p style="text-align: center; font-weight: bold;">Locations Legend</p>
'''

# Add clickable locations to the legend
for location, color in color_map.items():
    lat, lon = coordinates[location]
    legend_html += f'<p><a href="#" onclick="map.setView([{lat}, {lon}], 8);" style="color: {color};">{location}</a></p>'

legend_html += '</div>'

# Add the legend to the map
m.get_root().html.add_child(folium.Element(legend_html))

# Display the map in a Jupyter Notebook
display(m)

# Save the map to an HTML file (optional)
m.save('weather_analysis_map.html')


# In[189]:


# Create a standalone Matplotlib map
# Generate a bar chart for the aggregated data
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Temperature
axs[0, 0].bar(filtered_data['Location'], filtered_data['Temperature(C)'], color='blue')
axs[0, 0].set_title('Average Temperature by Location')
axs[0, 0].set_ylabel('Temperature (°C)')
axs[0, 0].tick_params(axis='x', rotation=45)

# Humidity
axs[0, 1].bar(filtered_data['Location'], filtered_data['Humidity(pct)'], color='orange')
axs[0, 1].set_title('Average Humidity by Location')
axs[0, 1].set_ylabel('Humidity (%)')
axs[0, 1].tick_params(axis='x', rotation=45)

# Precipitation
axs[1, 0].bar(filtered_data['Location'], filtered_data['Precipitation(mm)'], color='green')
axs[1, 0].set_title('Total Precipitation by Location')
axs[1, 0].set_ylabel('Precipitation (mm)')
axs[1, 0].tick_params(axis='x', rotation=45)

# Wind Speed
axs[1, 1].bar(filtered_data['Location'], filtered_data['Wind_Speed(km/h)'], color='red')
axs[1, 1].set_title('Average Wind Speed by Location')
axs[1, 1].set_ylabel('Wind Speed (km/h)')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Save the Matplotlib figure to an image file
matplotlib_map_file = 'weather_analysis_map.png'
plt.savefig(matplotlib_map_file)
plt.show()


# In[190]:


total_yield = df.groupby('Location')['Crop_Yield'].sum().reset_index()

#Create a bar graph for total crop yield by location
fig_total_yield = px.bar(total_yield, 
                          x='Location', 
                          y='Crop_Yield', 
                          title='Total Crop Yield by Location',
                          labels={'Crop_Yield': 'Total Crop Yield'},
                          color='Crop_Yield',
                          color_continuous_scale=px.colors.sequential.Plasma)

#Calculate average crop yield by location for the line graph
average_yield = df.groupby('Location')['Crop_Yield'].mean().reset_index()

# Create a line graph for average crop yield by location
fig_yield_trend = px.line(average_yield, 
                           x='Location', 
                           y='Crop_Yield', 
                           title='Average Crop Yield by Location',
                           labels={'Crop_Yield': 'Average Crop Yield'},
                           markers=True)

# Show the figures
fig_total_yield.show()
fig_yield_trend.show()


# In[193]:


# Ensure the 'Date' column is in datetime format (specifically for mm/dd/yyyy format)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

# Check for any rows that could not be converted
if df['Date'].isnull().any():
    print("Warning: Some dates could not be parsed. Check the data for inconsistencies.")

# Select relevant columns for correlation analysis
correlation_data = df[['Date', 'Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']].copy()

# Convert 'Date' to ordinal for correlation calculation
correlation_data['Date_Ordinal'] = correlation_data['Date'].map(pd.Timestamp.toordinal)

# Calculate the correlation matrix
correlation_matrix = correlation_data[['Date_Ordinal', 'Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']].corr()

# Display the correlation matrix
print(correlation_matrix)

# Create a heatmap for better visualization
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Weather Variables')
plt.show()


# In[194]:


# Group by location and calculate average climate conditions and crop yield
average_conditions = df.groupby('Location').agg({
    'Temperature(C)': 'mean',
    'Humidity(pct)': 'mean',
    'Precipitation(mm)': 'mean',
    'Wind_Speed(km/h)': 'mean',
    'Crop_Yield': 'mean'  # Assuming Crop_Yield is also available per location
}).reset_index()

# Calculate the correlation matrix
correlation_matrix = average_conditions[['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)', 'Crop_Yield']].corr()

# Display the correlation matrix
print(correlation_matrix)

# Create a heatmap for better visualization
plt.figure(figsize=(8, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Climate Conditions and Crop Yield')
plt.show()


# In[195]:


# Prepare input features (X) and target variable (y)
X = average_conditions[['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']]
y = average_conditions['Crop_Yield']

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients and feature names
coefficients = model.coef_
features = X.columns

# Create a DataFrame for variable importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': coefficients})

# Use absolute values for importance
importance_df['Importance'] = importance_df['Importance'].abs()
importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()  # Normalize importance

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(importance_df['Importance'], labels=importance_df['Feature'], autopct='%1.1f%%', startangle=140)
plt.title('Variable Importance to Total Crop Yield')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.show()


# # Simulating large dataset challenges: Trend Analysis of weather conditions by date

# In[ ]:


#df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

# Set the date as the index
df.set_index('Date', inplace=True)

# Plotting
plt.figure(figsize=(14, 7))

# Plot each variable using the renamed columns
plt.plot(df.index, df['Temperature(C)'], label='Temperature (°C)', color='red', linewidth=2)
plt.plot(df.index, df['Humidity(pct)'], label='Humidity (%)', color='blue', linewidth=2)
plt.plot(df.index, df['Precipitation(mm)'], label='Precipitation (mm)', color='green', linewidth=2)
plt.plot(df.index, df['Wind_Speed(km/h)'], label='Wind Speed (km/h)', color='orange', linewidth=2)

# Adding titles and labels
plt.title('Trend Analysis of Weather Variables Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend()
plt.grid()

# Show plot
plt.tight_layout()
plt.show()


# In[ ]:


# Select the first 20 samples
df_subset = df.head(20)

# Plotting
plt.figure(figsize=(14, 7))

# Plot each variable 
plt.plot(df_subset.index, df_subset['Temperature(C)'], label='Temperature (°C)', color='red', linewidth=2)
plt.plot(df_subset.index, df_subset['Humidity(pct)'], label='Humidity (%)', color='blue', linewidth=2)
plt.plot(df_subset.index, df_subset['Precipitation(mm)'], label='Precipitation (mm)', color='green', linewidth=2)
plt.plot(df_subset.index, df_subset['Wind_Speed(km/h)'], label='Wind Speed (km/h)', color='orange', linewidth=2)

# Adding titles and labels
plt.title('Trend Analysis of Weather Variables (First 20 Samples)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(loc='upper right')
plt.grid()

# Show plot
plt.tight_layout()
plt.show()


# In[ ]:




