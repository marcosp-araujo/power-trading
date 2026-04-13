import keras
from src.libs.model_tools import Model_Manager

def model_train(self:Model_Manager):
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
    return self

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
