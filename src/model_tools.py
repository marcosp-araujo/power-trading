import os
import pickle
import tensorflow as tf
import numpy as np
from src.config_tools import Config_Manager
from src.data_tools import Data_Manager

class Model_Manager:
  config: Config_Manager
  data: Data_Manager
  
  def __init__(self, 
               data: Data_Manager, 
               config: Config_Manager
               ) -> None:
    self.config = config
    self.data = data
    self.train_or_load_model()

  def train_or_load_model(self):
    '''Trains a new model or loads an existing one from disk.'''

    # Loading an existing model
    if self.config.mode == 'load':
      if not os.path.exists(self.config.tf_model_path):
        raise FileNotFoundError(f"Model object file not found at: \n{self.config.tf_model_path}")
      print("Loading a model from:")
      print(self.config.model_folder)
      self.model = tf.keras.models.load_model(f'{self.config.tf_model_path}')
      self.history = pickle.load(open(self.config.history_path, "rb"))

    # Start the training a new model
    else: 
      print(f"Training a new model:")
      print(self.config.model_name)
      self.model_train()

      # Save the model object to disk
      self.model.save(self.config.tf_model_path)
      pickle.dump(self.history, open(self.config.history_path, "wb"))
      print(f"Model dependencies were saved to:")
      print(self.config.model_folder) 

  def model_train(self):
    '''Build and train a model'''
    # Set the learning rate
    output_size = self.config.horizon
    learning_rate = self.config.learning_rate
    momentum = self.config.momentum
    epochs = self.config.epochs
    window_size = self.config.window_size
  
    model = tf.keras.Sequential([ 
      tf.keras.Input(shape=(window_size, 1)), 
      tf.keras.layers.Conv1D(32, 5, activation='relu'), 
      tf.keras.layers.Conv1D(16, 5, activation='relu'), 
      tf.keras.layers.Flatten(), 
      tf.keras.layers.Dense(output_size),
      tf.keras.layers.Reshape((output_size, 1))]
      )
    
    # Set the optimizer 
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, 
                                        momentum=momentum)

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"]
    )

    self.model = model # Storing model temporarily for the callbacks 

    forecast_callback = ForecastHistoryCallback(self)
    # Train the model
    history = model.fit(
            self.data.x_train_window, 
            epochs=epochs,
            validation_data=self.data.x_valid_window, # <--- ADD THIS LINE
            callbacks=[forecast_callback]
    )
    # Storing the model and history in the object
    self.model = model
    self.history = history
    self.forecast_per_epoch = forecast_callback.epoch_forecasts

  def compute_forecast(self, delay=1):
    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Add an axis for the feature dimension of RNN layers
    series_valid = tf.expand_dims(self.data.x_valid, axis=-1)
    
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series_valid)

    # Window the data but only take those with the specified size
    dataset = dataset.window(self.config.window_size, 
                              shift=1, 
                              drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    @tf.autograph.experimental.do_not_convert
    def _batch_window(w):
        return w.batch(self.config.window_size)

    dataset = dataset.flat_map(_batch_window)
    
    # Create batches of windows
    dataset = dataset.batch(self.config.batch_size).prefetch(1)
    
    # Get predictions on the entire dataset
    forecast = self.model.predict(dataset, verbose=0)
    forecast = forecast.squeeze()

    # Computing forecasting delay index
    forecast_delay = self.config.window_size-delay

    # Validation data and forecast time ajusted to forecast delay
    x_valid_adjusted = self.data.x_valid[forecast_delay:]
    time_forecast = self.data.time_valid[forecast_delay:]

    # Computing mean between overlaped forecast values
    if forecast.ndim > 1 and forecast.shape[1] > 1:
      forecast = forecast.mean(axis=1)

    # Storing the forecast, time_forecast and forecast_delay in the object
    self.forecast = forecast  
    self.time_forecast = time_forecast
    self.forecast_delay = forecast_delay
    self.x_valid_adjusted = x_valid_adjusted

  def metrics(self):
    '''Calculates metrics comparing the forecasted values 
      with the actual values.'''

    # Mean absolute error
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    mae = mae_metric(self.x_valid_adjusted, 
                     self.forecast
                     ).numpy()
   
    # Normalized mean absolute error
    capacity = np.max(self.data.df_clean[self.config.series_column])  # max power
    nmae = mae / capacity * 100

    # Root mean squared error
    rmse = np.sqrt(
                   np.mean(
                      (self.x_valid_adjusted - self.forecast)**2)
                      ).item()

    results = {"mae":mae, 
               "nmae": nmae,
               "rmse":rmse,
               "capacity":capacity}

    return results
  
class ForecastHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.epoch_forecasts = []

    def on_epoch_end(self, epoch, logs=None):
        # We temporarily point the manager's model to the current state
        # then run the existing forecast logic
        self.model_manager.compute_forecast()
        # Store a copy of the forecast result for this epoch
        self.epoch_forecasts.append(self.model_manager.forecast.copy())
