import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px



# Set page title and layout
st.set_page_config(
    page_title="Weather Analysis Model",
    layout="wide",
    page_icon="📊",  # Default icon
)
# Load the dataset
df = pd.read_csv("weather_data_augmented.csv")

# Rename columns
df.rename(columns={
    "Date_Time": "Date",
    "Temperature_C": "Temperature(C)",
    "Humidity_pct": "Humidity(pct)",
    "Precipitation_mm": "Precipitation(mm)",
    "Wind_Speed_kmh": "Wind_Speed(km/h)"
}, inplace=True)

# Remove time from the Date column
df["Date"] = pd.to_datetime(df["Date"]).dt.date

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


# Custom CSS to change the sidebar color
st.markdown(
    """
    <style>
    .st-emotion-cache-6qob1r {
        background-color: #ADD8E6; /* pale blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Dashboard",  # Title of the menu
        options=["Home", "Processed Data View", "Weather Distribution Map", 
                 "Correlation Analysis", "Analysis", "Complexity of Analysis"],  # Menu options
        icons=["house", "table", "map", "bar-chart", "graph-up", "gear"],  # Icons for each option
        menu_icon="cast",  # Menu icon
        default_index=0,  # Default selected option
        styles={
        "nav-link-selected": {"background-color": "black"},
        "container": {"background-color": "#ADD8E6"},
    }
)
    

if selected == "Home":
    # Include Bootstrap 5 CSS
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

    # Centered heading using Bootstrap
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6">Objectives and Tools</h1>
    </div>
    """, unsafe_allow_html=True)

    # Custom HTML with Bootstrap 5 classes and faint purple background
    st.markdown("""
    <div style="background-color: #f3e5f5; border: 2px solid #ce93d8; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
        <p class="fs-5 text-secondary">
            This application simulates a data pipeline analysis of weather data to understand climate conditions and their impact on agriculture crop yield. 
            Tools such as <span class="fw-bold">Kaggle API</span> for data scrapping, <span class="fw-bold">Pandas</span> for data manipulation, <span class="fw-bold">Folium</span> for mapping, 
            <span class="fw-bold">Matplotlib</span> and <span class="fw-bold">Seaborn</span> for visualization, 
             <span class="fw-bold">Scikit-learn</span> for feature importance coefficient analysis as well as <span class="fw-bold">Streamlit</span>  for dashboard were used.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Members Section
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6">Group Members</h1>
    </div>
    """, unsafe_allow_html=True)


    # Create a DataFrame for members
    members_data = {
        "Full Name": ["Sandra Mabhanda", "Nqobani Dube", "Trish Vutabwarova", "Tynos Chakafa"],
        "Reg Number": ["R226376Q", "R227709F", "R225049A", "R228140B"]
    }
    members_df = pd.DataFrame(members_data)

    # Display the table
    st.table(members_df)

    # Close the bordered container
    st.markdown("</div>", unsafe_allow_html=True)
    
# Processed Data View
elif selected == "Processed Data View":
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6">Data Preprocessing & Augmentation</h1>
    </div>
    """, unsafe_allow_html=True)

    # Custom HTML with Bootstrap 5 classes and faint purple background
    st.markdown("""
    <div style="background-color: #f3e5f5; border: 2px solid #ce93d8; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
        <p class="fs-5 text-secondary">
            This included renaming column names, removing time component from <span class="fw-bold">Date Column</span> as well as adding crop yield and season through standardization and normalization.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6"></h1>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head())

# Weather Distribution
elif selected == "Weather Distribution Map":
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6">Weather Distribution by Location</h1>
    </div>
    """, unsafe_allow_html=True)

    locations = st.multiselect("SELECT LOCATIONS", options=list(coordinates.keys()), default=list(coordinates.keys()))
    filtered_data = df[df['Location'].isin(locations)]
    
    # Aggregated data
    aggregated_data = filtered_data.groupby('Location').agg({
        'Temperature(C)': 'mean', 
        'Humidity(pct)': 'mean', 
        'Precipitation(mm)': 'sum', 
        'Wind_Speed(km/h)': 'mean'
    }).reset_index()

    # Create a Folium map
    m = folium.Map(location=[37.0, -95.0], zoom_start=4)
    for _, row in aggregated_data.iterrows():
        folium.Marker(
            location=coordinates[row['Location']],
            popup=f"{row['Location']}<br>Temp: {row['Temperature(C)']:.1f} °C<br>Humidity: {row['Humidity(pct)']:.1f} %<br>Precipitation: {row['Precipitation(mm)']:.1f} mm<br>Wind Speed: {row['Wind_Speed(km/h)']:.1f} km/h"
        ).add_to(m)

    # Save the map to an HTML file
    map_file = "weather_map.html"
    m.save(map_file)
    st.info("**_Click at any selected location to view weather conditions_**")

    st.components.v1.html(open(map_file, 'r').read(), height=500)

elif selected == "Correlation Analysis":
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6">Correlation Analysis</h1>
    </div>
    """, unsafe_allow_html=True)


    # Allow users to select between tables or graphs
    analysis_option = st.selectbox(
        "**Select Analysis Type**",
        options=["Correlation Tables", "Correlation Graphs"],
        index=0
    )

    # Calculate correlation matrices
    correlation_data = df[['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']]
    correlation_matrix_weather = correlation_data.corr()

    correlation_yield_data = df[['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)', 'Crop_Yield']]
    correlation_matrix_yield = correlation_yield_data.corr()

    # Display Correlation Tables
    if analysis_option == "Correlation Tables":
        st.info("**Correlation Matrix of Weather Variables**")
        st.table(correlation_matrix_weather)

        st.info("**Correlation Matrix of Weather Conditions and Crop Yield**")
        st.table(correlation_matrix_yield)

    # Display Correlation Graphs (Heatmaps using Seaborn)
    elif analysis_option == "Correlation Graphs":
        st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)
    
        st.markdown("""
         <div class="text-center">
         <h1 class="display-6">Correlation Heatmaps</h1>
         </div>
    """, unsafe_allow_html=True)
#        st.subheader("Correlation Heatmaps")

        # Create columns to display heatmaps side by side
        col1, col2 = st.columns(2)

        with col1:
            st.info("Correlation Heatmap of Weather Variables")
            plt.figure(figsize=(6, 4))
            sns.heatmap(correlation_matrix_weather, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap of Weather Variables', fontsize=12)
            st.pyplot(plt)

        with col2:
            st.info("Correlation Heatmap of Weather Conditions and Crop Yield")
            plt.figure(figsize=(6, 4))
            sns.heatmap(correlation_matrix_yield, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap of Weather Conditions and Crop Yield', fontsize=12)
            st.pyplot(plt)

# Analysis Section
elif selected == "Analysis":
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    st.markdown("""
         <div class="text-center">
         <h1 class="display-6">Analysis</h1>
         </div>
    """, unsafe_allow_html=True)


    # Allow users to select a graph to view
    graph_option = st.selectbox(
        "**SElECT ANALYSIS TO VIEW**",
        options=["Total Yield by Location Bargraph", "Average Yield by Location Trend", "Variable Importance Analysis"],
        index=0
    )

    # Total Yield by Location (Bar Chart)
    if graph_option == "Total Yield by Location Bargraph":
        total_yield = df.groupby('Location')['Crop_Yield'].sum().reset_index()
        labels = total_yield['Location'].tolist()
        data = total_yield['Crop_Yield'].tolist()

        # Chart.js Bar Chart with improved axis visibility
        bar_chart_js = f"""
        <canvas id="barChart" width="400" height="200"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            var ctx = document.getElementById('barChart').getContext('2d');
            var barChart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        label: 'Total Yield',
                        data: {data},
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Total Yield',
                                font: {{
                                    size: 14
                                }}
                            }},
                            ticks: {{
                                font: {{
                                    size: 12
                                }}
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Location',
                                font: {{
                                    size: 14
                                }}
                            }},
                            ticks: {{
                                font: {{
                                    size: 12
                                }},
                                autoSkip: false,
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Total Crop Yield by Location',
                            font: {{
                                size: 16
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """
        st.components.v1.html(bar_chart_js, height=400)

    # Average Yield by Location (Line Chart)
    elif graph_option == "Average Yield by Location Trend":
        average_yield = df.groupby('Location')['Crop_Yield'].mean().reset_index()
        labels = average_yield['Location'].tolist()
        data = average_yield['Crop_Yield'].tolist()

        # Chart.js Line Chart with improved axis visibility
        line_chart_js = f"""
        <canvas id="lineChart" width="400" height="200"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            var ctx = document.getElementById('lineChart').getContext('2d');
            var lineChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        label: 'Average Yield',
                        data: {data},
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 2,
                        fill: false
                    }}]
                }},
                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Average Yield',
                                font: {{
                                    size: 14
                                }}
                            }},
                            ticks: {{
                                font: {{
                                    size: 12
                                }}
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Location',
                                font: {{
                                    size: 14
                                }}
                            }},
                            ticks: {{
                                font: {{
                                    size: 12
                                }},
                                autoSkip: false,
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Average Crop Yield by Location',
                            font: {{
                                size: 16
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """
        st.components.v1.html(line_chart_js, height=400)

    # Variable Importance (Pie Chart)
    elif graph_option == "Variable Importance Analysis":
        X = df[['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']]
        y = df['Crop_Yield']
        model = LinearRegression()
        model.fit(X, y)
        importance = model.coef_

        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        importance_df['Importance'] = importance_df['Importance'].abs()
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()

        labels = importance_df['Feature'].tolist()
        data = importance_df['Importance'].tolist()

        # Chart.js Pie Chart with percentages
        pie_chart_js = f"""
        <canvas id="pieChart" width="450" height="450"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.0.0"></script>
        <script>
            var ctx = document.getElementById('pieChart').getContext('2d');
            var pieChart = new Chart(ctx, {{
                type: 'pie',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        data: {data},
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(255, 206, 86, 0.6)',
                            'rgba(75, 192, 192, 0.6)'
                        ],
                        borderColor: [
                            'rgba(255, 99, 132, 1)',
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(75, 192, 192, 1)'
                        ],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Variable Importance to Total Crop Yield',
                            font: {{
                                size: 16
                            }}
                        }},
                        legend: {{
                            labels: {{
                                font: {{
                                    size: 14
                                }}
                            }}
                        }},
                        datalabels: {{
                            color: '#fff',
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }},
                            formatter: function(value, context) {{
                                return (value * 100).toFixed(1) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: [ChartDataLabels]
            }});
        </script>
        """
        st.components.v1.html(pie_chart_js, height=580)

# Complex Analysis
elif selected == "Complexity of Analysis":
    # Centered heading using Bootstrap
    st.markdown("""
    <div class="text-center">
        <h1 class="display-6">Large Dataset Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)


    # Custom HTML with Bootstrap 5 classes and faint purple background
    st.markdown("""
    <div style="background-color: #f3e5f5; border: 2px solid #ce93d8; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
        <p class="fs-5 text-secondary">
            This section provide two challenges encountered when working with large datasets. 
            The challenges encountered include <span class="fw-bold">computational intensity </span> in terms of storage and execution time and <span class="fw-bold">interpretibility</span> challenge of some graphs. 
        </p>
    </div>
    """, unsafe_allow_html=True)
    

    # Trend analysis for large datasets
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df.set_index('Date', inplace=True)

    # User selection for trend analysis
    analysis_choice = st.selectbox("**SELECT ANALYSIS**", 
                                ("Trend Analysis-Large Dataset", "Trend Analysis-First 20 Rows"))

    if analysis_choice == "Trend Analysis-Large Dataset":
      try:
        fig, ax = plt.subplots(figsize=(14, 7))
        for col in ['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']:
            ax.plot(df.index, df[col], label=col)
        ax.set_title('Trend Analysis of Weather Variables Over Time', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Values', fontsize=14)
        ax.legend()
        ax.grid()
        st.pyplot(fig)
      except Exception as e:
        st.error(f"An error occurred while plotting the trend analysis: {e}")

    elif analysis_choice == "Trend Analysis-First 20 Rows":
        # Select the first 20 samples
        df_subset = df.head(20)

        fig_subset, ax_subset = plt.subplots(figsize=(14, 7))
        for col in ['Temperature(C)', 'Humidity(pct)', 'Precipitation(mm)', 'Wind_Speed(km/h)']:
            ax_subset.plot(df_subset.index, df_subset[col], label=col)
        ax_subset.set_title('Trend Analysis of Weather Variables (First 20 Samples)', fontsize=16)
        ax_subset.set_xlabel('Date', fontsize=14)
        ax_subset.set_ylabel('Values', fontsize=14)
        ax_subset.legend(loc='upper right')
        ax_subset.grid()
        st.pyplot(fig_subset)

# Run the app
if __name__ == "__main__":
    st.write("------------------------------ ")