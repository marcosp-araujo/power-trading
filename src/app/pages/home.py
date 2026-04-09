import streamlit as st
from src.libs import plot
from src.libs.plot import plot_obj
from src.app.load import load_app_data
from src.app.app_config import load_config

config = load_config()

st.header("Power Trading Forecasting Dashboard")

st.write(
"This dashboard template showcases a Short-Term Wind Power Forecasting model to support Intraday Energy Market. "
f"Utilizing a Neural Network architecture, the system predicts generation {config.horizon_string} ahead by analyzing a 12-hour rolling window of historical data."
)

st.write(
"The example dataset used in this project was obtained from the "
"[Open Power System Data](https://data.open-power-system-data.org/time_series/) platform, "
"which provides free and open datasets for power system analysis. "
"The selected data consists of wind power generation in the Netherlands from 1 January to 31 March, recorded at a 15-minute frequency."
)

st.write("Use the sidebar to audit the model’s convergence, forecast, and evaluate its performance.")

model = load_app_data()

# Dataset
plot.series(
    plot_obj(
            model.data.df_clean[config.time_column],
            model.data.df_clean[config.series_column],
            config.series_column
            ),
    title = "Wind power generation in the Netherlands.",
    streamlit=True,
    ylabel=config.series_column
)