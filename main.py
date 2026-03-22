#%% Imports ------------------------------------------------

from src.config_tools import Config_Manager
from src import data_tools, model_tools, plot

if __name__ == "__main__":

    config = Config_Manager(
                    mode="train", # "train" or "load"
                    model_name="model_v1_6_w12h_1h_ahead",
                    data_path="data/time_series_15min.parquet",
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

    data = data_tools.Data_Manager(config)
    plot.train_test(data, config)
    
    #%% Creating a model manager object

    model = model_tools.Model_Manager(data, config)

    #%%

    plot.metrics_history(model.history)
    
    #%% Forecasting ----------------------------

    model.compute_forecast(delay=1)
    plot.forecast(model)     
    
    #%% PERFORMANCE

    results = model.metrics()
    for key, item in results.items():
        print(f"{key}: {item:.2f}")

# %%
