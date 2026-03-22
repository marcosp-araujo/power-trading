import streamlit as st
from src import plot
from src.plot import plot_obj
from src.app.load import load_app_data

config, data, model = load_app_data()

st.header("Power Trading Forecasting Dashboard")

st.write(
"This dashboard showcases a Short-Term Wind Power Forecasting model designed for the Netherlands' Intraday Energy Market. "
f"Utilizing a Neural Network architecture, the system predicts generation {data.config.horizon_string} ahead by analyzing a 12-hour rolling window of historical data."
)

st.write(
"The dataset used in this project was obtained from the "
"[Open Power System Data](https://data.open-power-system-data.org/time_series/) platform, "
"which provides free and open datasets for power system analysis. "
"The selected data consists of wind power generation in the Netherlands from 1 January to 31 March, "
"recorded at a 15-minute frequency."
)

st.write("Use the sidebar to audit the model’s convergence, visualize real-time forecasts, and evaluate performance against industry-standard benchmarks.")

# Dataset
plot.series(
    plot_obj(
            data.df_clean[config.time_column],
            data.df_clean[config.series_column],
            config.series_column
            ),
    title = "Wind power generation in the Netherlands.",
    streamlit=True,
    ylabel=config.series_column
)