import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Load the trained model
model = joblib.load('greenhouse_gas_model.joblib')

# OpenWeatherMap API key
API_KEY = "b7e6baf62f44d6f3052581fd70c52e2b"  # Replace with your actual API key

def get_air_quality(lat, lon):
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['list'][0]['components']
    else:
        st.error("Failed to fetch air quality data")
        return None

def get_air_quality_forecast(lat, lon):
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['list']
    else:
        st.error("Failed to fetch air quality forecast data")
        return None

st.title('Predicting future emissions with respect to present emissions ')

st.write("""
This app predicts CO2 , CH4, and CO concentrations based on location and date, fetches real-time air quality data of CO2, CH4, CO, NO,O3, SO2, NH3 and N2O,
and provides various visualizations to help understand the data.
""")

# Create input fields
latitude = st.slider('Latitude', min_value=-90.0, max_value=90.0, value=40.7128, step=0.1)
longitude = st.slider('Longitude', min_value=-180.0, max_value=180.0, value=-74.0060, step=0.1)
date = datetime.now()

# Calculate day of year
day_of_year = date.timetuple().tm_yday

# Make prediction
input_data = np.array([[latitude, longitude, day_of_year]])
prediction = model.predict(input_data)

st.subheader('Model Prediction')
st.success(f'The predicted CO2 concentration is: {prediction[0]:.2f} ppm')

# Fetch real-time data
st.subheader('Real-time Air Quality Data')
air_quality = get_air_quality(latitude, longitude)

if air_quality:
    st.write("Current air quality components:")
    for component, value in air_quality.items():
        st.write(f"{component}: {value}")
    
    # Bar chart of air quality components
    fig_bar = px.bar(x=list(air_quality.keys()), y=list(air_quality.values()),
                     labels={'x': 'Component', 'y': 'Concentration (μg/m³)'},
                     title='Air Quality Components')
    st.plotly_chart(fig_bar)

# Fetch and visualize air quality forecast
forecast_data = get_air_quality_forecast(latitude, longitude)

if forecast_data:
    # Prepare data for time series chart
    times = [datetime.fromtimestamp(item['dt']) for item in forecast_data]
    co_values = [item['components']['co'] for item in forecast_data]
    
    # Time series chart
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=times, y=co_values, mode='lines+markers', name='CO'))
    fig_time.update_layout(title='CO Concentration Forecast',
                           xaxis_title='Time',
                           yaxis_title='Concentration (μg/m³)')
    st.plotly_chart(fig_time)

    # Generate synthetic data for comparison (replace this with your actual historical data if available)
    synthetic_data = [model.predict([[latitude, longitude, (date + timedelta(hours=i)).timetuple().tm_yday]])[0] 
                      for i in range(len(forecast_data))]
    
    # Scatter plot of predicted vs "actual" (forecast) values
    fig_scatter = px.scatter(x=synthetic_data, y=co_values, 
                             labels={'x': 'Predicted CO2 (ppm)', 'y': 'Forecast CO (μg/m³)'},
                             title='Predicted CO2 vs Forecast CO')
    fig_scatter.add_trace(go.Scatter(x=[min(synthetic_data), max(synthetic_data)],
                                     y=[min(synthetic_data), max(synthetic_data)],
                                     mode='lines', name='y=x'))
    st.plotly_chart(fig_scatter)

# Map to visualize the location
st.subheader('Location')
df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
st.map(df)

# Information about the project
st.subheader('About this project')
st.write("""
This project was developed for the 2024 NASA Space Apps Challenge. It uses a machine learning model
trained on synthetic data to predict CO2 concentrations at different locations and times of the year.
The predictions are compared with real-time air quality data and forecasts from OpenWeatherMap.
""")

# Model performance metrics
st.subheader('Model Performance')
st.write("""
The model's performance metrics on the test set:
- Mean Squared Error: X
- R-squared Score: Y

Replace X and Y with your actual model performance metrics from the training notebook.
""")

# Disclaimer
st.sidebar.success("""
This is a demonstration app using a model trained on real-world events data from Satellite and NASA organizations.

Disclaimer:Real-world CO2 concentrations may vary significantly from these predictions.

""")
