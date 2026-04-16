import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal
from src.libs.model_tools import Model_Output
from src.libs.data_tools import Data_Manager
from src.libs.config_tools import Config_Manager, variables_dictionary

class plot_obj:
    time: pd.Series
    serie: pd.Series
    label: str
    mode: Literal['lines','markers']
    color: str
    def __init__(self, time, serie, label, mode = 'lines', color = None):
        self.time = time
        self.serie = serie
        self.label = label
        self.mode = mode
        self.color = color

def series(
    *plot_args: plot_obj,
    ylabel: str = "Value",
    xlabel: str = "Time",
    xlim: tuple = None,
    ylim: tuple = None,
    title: str = None,
    streamlit: bool = False
) -> None:
    """Plots time series data using Plotly.
       Accepts multiple (time, series, label) sets as arguments.
    """

    fig = go.Figure()
    kwargs = {} # for keyword arguments
    for p in plot_args:
        if p.color:
            kwargs["line"] = dict(color=p.color)
        fig.add_trace(
            go.Scatter(
                x=p.time,
                y=p.serie,
                mode=p.mode,
                name=p.label,
                line=dict(color=p.color)
            )
        )
    # Updating label based on dictionary
    xlabel = variables_dictionary.get(xlabel, xlabel)
    ylabel = variables_dictionary.get(ylabel, ylabel)

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=400,
        width=600,
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=1.15,
            x=0.5,
            xanchor="center",
            yanchor="bottom"
        ),
        margin=dict(t=80, b=100)  # space for legend and title
    )

    if title:
        fig.add_annotation(
            text=title,
            x=0.5,
            y=-0.35,              # below plot
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=16)
        )
    if xlim:
        fig.update_xaxes(range=xlim)

    if ylim:
        fig.update_yaxes(range=ylim)
    # Diplay the figure using Streamlit or directly
    if streamlit:
        st.plotly_chart(fig, width='stretch')
    else:
        fig.show()

def forecast(model_output: Model_Output, 
             streamlit:bool=False
             ) -> None:
    '''Plots the actual vs forecasted values using Plotly.'''
    ajust = model_output.data.config.horizon
    time_valid = model_output.data.time_valid[ajust:]
    x_valid = model_output.data.x_valid[ajust:]
    time_forecast = model_output.time_forecast
    forecast = model_output.forecast

    # Correcting ylabel
    ylabel = model_output.data.config.series_column
    ylabel = variables_dictionary.get(ylabel, ylabel)
    ylabel = ylabel.replace("Actual","")
    series(
        plot_obj(time_valid, x_valid, "Validation"),
        plot_obj(time_forecast, forecast, "Forecast"),
        xlabel="Time",
        ylabel=ylabel,
        title="Comparison between forecast and validation time series.",
        streamlit=streamlit
    )

def train_test(data:Data_Manager, 
               config:Config_Manager,
               streamlit:bool=False
               ) -> None:
    '''Plots the training and validation sets using Plotly.'''

    time_train = data.time_train
    x_train = data.x_train
    time_valid = data.time_valid
    x_valid = data.x_valid

    series(
        plot_obj(time_train, x_train, "Train"),
        plot_obj(time_valid, x_valid, "Validation"),
        xlabel="Time",
        ylabel=config.series_column,
        title="Comparison between training and validation sets.",
        streamlit=streamlit
    )

def metrics_history(history, streamlit:bool=False) -> None:
    """Plots key performance indicators for both training and validation sets."""
    
    # 1. Extract Metrics
    # We use .get() to avoid errors if validation data wasn't provided
    mae = history.get("mae", [])
    val_mae = history.get("val_mae", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    
    epochs = np.arange(len(loss)) + 1 # For 1-based epoch numbering in the plot
    title_base = "Model training and validation history "

    # 2. Full History Plot
    # Grouping related metrics helps visualize the gap between Train and Val
    series(
        plot_obj(epochs, mae, "Train MAE", mode='markers+lines'),
        plot_obj(epochs, loss, "Train Loss", mode='markers+lines'),
        plot_obj(epochs, val_mae, "Val MAE", mode='markers+lines'),
        plot_obj(epochs, val_loss, "Val Loss", mode='markers+lines'),
        xlabel="Epochs",
        ylabel="Metric Value",
        title=title_base + "(full).",
        streamlit=streamlit
    )

    # 3. Last 50% Zoom
    zoom_split = int(len(epochs) * 0.5)
    epochs_zoom = epochs[zoom_split:]
    
    series(
        plot_obj(epochs_zoom, mae[zoom_split:], "Train MAE", mode='markers+lines'),
        plot_obj(epochs_zoom, val_mae[zoom_split:], "Val MAE", mode='markers+lines'),
        plot_obj(epochs_zoom, loss[zoom_split:], "Train Loss", mode='markers+lines'),
        plot_obj(epochs_zoom, val_loss[zoom_split:], "Val Loss", mode='markers+lines'),
        xlabel="Epochs",
        ylabel="Metric Value",
        title=title_base + "(zoomed - last 50% epochs).",
        streamlit=streamlit
    )
def plot_sliding_window(data:Data_Manager, 
                        config:Config_Manager
                        ) -> None:

    window_size = config.window_size

    # Full training series
    time = data.time_train.values
    series_values = data.x_train.values

    # Choose region to display
    start = window_size * 2
    end = start + window_size * 3

    # Window location
    w_start = start + window_size
    w_end = w_start + window_size

    # Data slices
    time_series = time[start:end]
    series_segment = series_values[start:end]

    time_window = time[w_start:w_end]
    window_values = series_values[w_start:w_end]

    time_target = time[w_end:w_end + config.horizon]
    target_value = series_values[w_end:w_end + config.horizon]

    series(
        plot_obj(
            time_series,
            series_segment,
            "Training series"
        ),
        plot_obj(
            time_window,
            window_values,
            "Input window"
        ),
        plot_obj(
            time_target,
            target_value,
            "Prediction target",
            mode='markers+lines',
            color = "black"
        ),
        ylabel=config.series_column,
        xlabel="Time",
        title="Sliding window representation for time series forecasting.",
        streamlit=True
    )

def scatter(model_output: Model_Output, 
            x_label="Validation", 
            y_label="Forecast",
            xlim=[0, 3000], 
            ylim=[0, 3000],
            streamlit=False):
    """
    Creates and displays an interactive scatter plot using Plotly with 
    linear regression and forced squared aspect ratio.
    """
    x = model_output.x_valid_adjusted
    y = model_output.forecast

    # 1. Calculate Linear Regression
    x_arr = np.array(x)
    y_arr = np.array(y)

    m, b = np.polyfit(x_arr, y_arr, 1)
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(x_arr, y_arr)
    r_squared = correlation_matrix[0, 1]**2
    
    # Create trendline data using xlim to ensure it spans the full plot
    x_fit = np.array(xlim)
    y_fit = m * x_fit + b
    
    # 2. Build the Plot
    fig = go.Figure()

    # Add the raw data
    fig.add_trace(go.Scatter(
        x=x, y=y, 
        mode='markers', 
        name='Data',
        marker=dict(opacity=0.7)
    ))

    # Add the regression line with stats in the legend
    stats_label = f'Slope: {m:.2f}<br>Intercept: {b:.2f}<br>R²: {r_squared:.3f}'
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit, 
        mode='lines', 
        name=stats_label,
        line=dict(color='red', dash='dash')
    ))

    # 3. Force Squared Aspect Ratio and Set Limits
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            range=xlim,
            constrain='domain',
            scaleanchor="y", 
            scaleratio=1,
            automargin=True
        ),
        yaxis=dict(
            range=ylim,
            constrain='domain',
            scaleanchor="x", # Explicitly link both ways
            scaleratio=1,
            automargin=True
        ),
        width=650, 
        height=500,
            
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02, # Moves legend just to the right of the plot boundary
                bgcolor="rgba(255, 255, 255, 0.5)"
            ),
        margin=dict(l=50, r=150, t=50, b=50) # Extra 'r' (right) margin for the legend
        )

    if streamlit:
        st.plotly_chart(fig, width='stretch') # False to respect the 600x600 square
    else:
        fig.show()

    return r_squared