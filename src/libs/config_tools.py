import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(BASE_DIR)

import json
import humanize
import datetime as dt
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Literal

@dataclass(slots=True)
class Config_Manager:
  # --- Model Action ---
  model_name: str         # name of the model to be used or created
  mode: Literal['train','load'] = "train"  # 'train' or 'load' to train a new model or load an existing one from disk

  # --- Data ---
  data_path: str = ""         # path to the raw data file
  df: pd.DataFrame | None = None       # loaded from data_path
  df_clean: pd.DataFrame | None = None # to be filled after preprocessing
  time_resolution: int = 15    # Time resolution in minutes
  time_column: str = ""       # name of the time column
  series_column: str = ""     # name of the target series column 
  start_time: str = ""      # First time for slicing the raw data
  end_time: str = ""        # Last time for slicing the raw data
  train_size: float = 0.7 # train size between 0 and 1
  horizon_string: str = field(init=False) # To save horizon in human readable format
  
  # --- Windowing / Dataset ---
  window_size: int = 48       # size of the input window for the model
  batch_size: int = 32        # batch size for training 
  shuffle_buffer: int = 1000  # buffer size for shuffling the dataset
  horizon:int = 4             # Time steps ahead to be forecasted
  
  # --- Training ---
  output_size: float = 1      # number of output features (1 for univariate forecasting)
  epochs: int = 100           # number of training epochs
  learning_rate: float = 1e-6 # optimizer learning rate
  momentum: float = 0.8       # optimizer momentum
  
  # --- Paths for saving model ---
  model_folder: str = field(init=False)
  tf_model_path: str = field(init=False)
  model_path: str = field(init=False)
  history_path: str = field(init=False)
  config_path: str = field(init=False)
  model_out_path: str = field(init=False)

  def run(self):
    '''Main method to run the whole process.'''
    self.set_paths()
    self.horizon_to_string()
    self.save_or_load_config()

  def save_or_load_config(self):
    if self.mode == 'train':
      with open(self.config_path, "w") as f:
          self.mode = 'load' # Avoid to recompute the model when loaded further
          json.dump(asdict(self), f, indent=4)
          self.mode = 'train'
    elif self.mode == 'load':
      with open(self.config_path, "r") as f:
          data = json.load(f)

      for key, value in data.items():
          setattr(self, key, value)
  
  def set_paths(self):
    self.model_folder = f"models/{self.model_name}"
    os.makedirs(f"{self.model_folder}", exist_ok=True)
    self.tf_model_path = f"{self.model_folder}/{self.model_name}.keras"
    self.model_path = f"{self.model_folder}/{self.model_name}.pkl"
    self.history_path = f"{self.model_folder}/{self.model_name}_history.pkl"
    self.config_path = f"{self.model_folder}/{self.model_name}_config.json"
    self.model_out_path = f"{self.model_folder}/{self.model_name}_forecast.pkl"
    print(f"Model folder set to: {self.model_folder}")
    print(f"TensorFlow model path set to: {self.tf_model_path}")
    print(f"History path set to: {self.history_path}")
    print(f"Config path set to: {self.config_path}")
    print(f"Forecast path set to: {self.model_out_path}")

  def horizon_to_string(self):
    "Converts horizon time step into human readable string"

    # Total duration
    total_min = self.horizon * self.time_resolution
    delta = dt.timedelta(minutes=total_min)

    # 'precisedelta' turns 60 min" into "1 hour"
    self.horizon_string = humanize.precisedelta(delta)

# Dictionary to tranlate raw columns names in the database
# into labels for plots
variables_dictionary = {'NL_wind_generation_actual':
                       'Actual NL Wind Generation [MW]'}