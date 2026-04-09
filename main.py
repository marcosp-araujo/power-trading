#%% Imports ------------------------------------------------
import os
os.environ["KERAS_BACKEND"] = "jax"

from src.libs.config_tools import Config_Manager
from src.libs import plot
from src.libs.processing_chain import processing_chain

if __name__ == "__main__":

    config = Config_Manager(
                    mode="load", # "train" or "load"
                    model_name="model_v1_7_w12h_1h_ahead",
                    data_path="data/time_series_15min.parquet",
                    time_resolution=15, # Data time resolution in minutes
                    series_column="NL_wind_generation_actual",
                    time_column="cet_cest_timestamp",
                    start_time="2016-01-01",
                    end_time="2016-04-30",
                    horizon=4, # 1h ahead forecasting
                    window_size=4*12, # 12 hours of data with 15min frequency
                    batch_size=32,
                    shuffle_buffer=1000,
                    epochs=10,
                    train_size=0.7, # Beginning of the validation set
                    learning_rate=1e-6,
                )
    config.run()

    #%% Preprocessing the data --------------

    model = processing_chain(config)

    #%% Plotting the training and validation datasets

    plot.train_test(model.data, config)

    #%% Plotting the training history

    plot.metrics_history(model.history)
    
    #%% Plotting the forecast

    plot.forecast(model)     

    #%% Showing the evaluation metrics

    results = model.metrics()
    for key, item in results.items():
        print(f"{key}: {item:.2f}")

# %%
