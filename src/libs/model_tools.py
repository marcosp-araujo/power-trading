from xml.parsers.expat import model

import pandas as pd
import numpy as np
from src.libs.config_tools import Config_Manager
from src.libs.data_tools import Data_Manager

class Model_Manager:
  config: Config_Manager
  data: Data_Manager
  
  def __init__(self, 
               data: Data_Manager, 
               config: Config_Manager
               ) -> None:
    self.config = config
    self.data = data
    if config.mode == "train":
      from src.libs.model_train import model_train
      self = model_train(self)

  def compute_forecast_numpy(self, delay=1):
    """Uses an input model to generate predictions on data windows without TF data pipelines"""

    # Ensure it's a numpy array. RNNs expect (samples, timesteps, features)
    series_values = self.data.x_valid.values.reshape(-1, 1)
    
    # Create a sliding window view of the data
    window_size = self.config.window_size
    num_windows = len(series_values) - window_size + 1
    
    # Each window is series_values[i : i + window_size]
    windows = [series_values[i : i + window_size] for i in range(num_windows)]
    
    # Resulting shape: (num_windows, window_size, 1)
    x_predict = np.array(windows)

    # Keras models accept numpy arrays directly in .predict()
    forecast = self.model.predict(x_predict, batch_size=self.config.batch_size, verbose=0)
    forecast = forecast.squeeze()

    # Computing forecasting delay index
    forecast_delay = self.config.window_size - delay

    # Validation data and forecast time adjusted to forecast delay
    # Pandas slicing works the same as before
    x_valid_adjusted = self.data.x_valid.iloc[forecast_delay:]
    time_forecast = self.data.time_valid.iloc[forecast_delay:]

    # Computing mean between overlapped forecast values (Multi-horizon handling)
    if forecast.ndim > 1 and forecast.shape[1] > 1:
        forecast = forecast.mean(axis=1)

    # Storing results in the object
    self.forecast = forecast  
    self.time_forecast = time_forecast
    self.forecast_delay = forecast_delay
    self.x_valid_adjusted = x_valid_adjusted

  def metrics(self):
    '''Calculates metrics comparing the forecasted values 
      with the actual values.'''

    # Mean absolute error
    mae = np.mean(np.abs(self.x_valid_adjusted - self.forecast))
   
    # Normalized mean absolute error
    capacity = np.max(self.data.df_clean[self.config.series_column])  # max power
    nmae = mae / capacity * 100

    # Root mean squared error
    rmse = np.sqrt(np.mean(
                  (self.x_valid_adjusted - self.forecast)**2)
                  ).item()

    results = {"mae": mae, 
               "nmae": nmae,
               "rmse": rmse,
               "capacity": capacity}

    return results
  
class Model_Output:
  def __init__(self, data, model:Model_Manager):
    # Build dataframe
    self.data = data
    self.time_forecast = model.time_forecast
    self.forecast = model.forecast  
    self.x_valid_adjusted = model.x_valid_adjusted
    self.history = model.history.history
    self.results = model.metrics()