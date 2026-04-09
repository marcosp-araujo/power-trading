import os
import keras
import pickle
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
      self.model_train()

  def train_or_load_model(self):
    '''Trains a new model or loads an existing one from disk.'''

    # Loading an existing model
    if self.config.mode == 'load':
      if not os.path.exists(self.config.tf_model_path):
        raise FileNotFoundError(f"Model object file not found at: \n{self.config.tf_model_path}")
      print("Loading a model from:")
      print(self.config.model_folder)
      pickle.load(open(f"{self.config.model_folder}/model.pkl", "rb"))

    # Start the training a new model
    else: 
      print(f"Training a new model:")
      print(self.config.model_name)
      self.model_train()

      # Save the model object to disk
      self.model.save(self.config.tf_model_path)
      pickle.dump(self, open(f"{self.config.model_folder}/model.pkl", "wb"))

      print(f"Model dependencies were saved to:")
      print(self.config.model_folder) 

  def model_train(self):
    '''Build and train a model'''
    # Set the learning rate
    output_size = self.config.horizon
    learning_rate = self.config.learning_rate
    momentum = self.config.momentum
    window_size = self.config.window_size
  
    model = keras.Sequential([ 
      keras.Input(shape=(window_size, 1)), 
      keras.layers.Conv1D(32, 5, activation='relu'), 
      keras.layers.Conv1D(16, 5, activation='relu'), 
      keras.layers.Flatten(), 
      keras.layers.Dense(output_size),
      keras.layers.Reshape((output_size, 1))]
      )
    
    # Set the optimizer 
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, 
                                        momentum=momentum)

    # Set the training parameters
    model.compile(loss=keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"]
    )

    self.model = model # Storing model temporarily for the callbacks 

    forecast_callback = ForecastHistoryCallback(self)

    # Model fitting
    history = model.fit(
              self.data.x_train_window,
              self.data.y_train_window,
              epochs=self.config.epochs,
              batch_size=self.config.batch_size,
              validation_data=(
                  self.data.x_valid_window,
                  self.data.y_valid_window
              ),
              callbacks=[forecast_callback]
    )
    # Storing the model and history in the object
    self.model = model
    self.history = history
    self.forecast_per_epoch = forecast_callback.epoch_forecasts

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
    mae_metric = keras.metrics.MeanAbsoluteError()
    mae = mae_metric(self.x_valid_adjusted, 
                     self.forecast
                     )
   
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
  
class ForecastHistoryCallback(keras.callbacks.Callback):
    def __init__(self, model_manager:Model_Manager):
        super().__init__()
        self.model_manager = model_manager
        self.epoch_forecasts = []

    def on_epoch_end(self, epoch, logs=None):
        # We temporarily point the manager's model to the current state
        # then run the existing forecast logic
        self.model_manager.compute_forecast_numpy()
        # Store a copy of the forecast result for this epoch
        self.epoch_forecasts.append(self.model_manager.forecast.copy())
