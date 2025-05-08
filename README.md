# 🌵 Drought Monitor

An interactive visualization tool for drought prediction and analysis across US counties.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drought-monitor.streamlit.app/)

## Features

- Geographic visualization of drought levels using county-level FIPS codes
- Comparison between predicted and actual drought conditions
- Interactive filters for specific drought severity levels
- County search functionality via FIPS codes
- Multi-week prediction view
- Statistical analysis with accuracy metrics and confusion matrices

## Screenshots

![image](https://github.com/user-attachments/assets/5d52ccbc-722d-4268-9942-cde0224fc009)

## Data

The application uses US county-level drought data with the following components:
- Predicted drought levels (machine learning model outputs)
- Actual drought conditions (ground truth)
- Geographic county boundaries

## How to run it on your own machine

1. Clone this repository
   ```
   git clone https://github.com/l1aF-2027/drought-monitor.git
   cd drought-monitor
   ```

2. Install the requirements
   ```
   pip install -r requirements.txt
   ```

3. Run the app
   ```
   streamlit run streamlit_app.py
   ```

## Requirements

The application requires the following libraries:
- streamlit
- pandas
- geopandas
- pydeck
- matplotlib
- seaborn
- scikit-learn
- numpy

## Data Structure

Make sure you have the following data files in the `/data` directory:
- `dict_map.csv`: Contains drought predictions and actual values
- `counties.geojson`: Geographic boundary data for US counties
