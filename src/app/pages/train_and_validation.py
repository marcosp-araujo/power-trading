import streamlit as st
from src import plot
from src.app.load import load_app_data
from src.app.app_config import load_config


data, model = load_app_data()

st.header("Train and Test Datasets")

st.write("The whole dataset was split into 70% for training and 30% for validation as show in the figure below.")

config = load_config()

plot.train_test(data, config, streamlit=True)

st.subheader("Windowed Dataset Construction")

st.write(
    "The training data is transformed into a windowed time-series dataset to allow the neural network to learn temporal patterns. "
    "A sliding window of size `window_size + 1` is applied across the series with a shift of one time step. "
    "Each window is then split into two parts: the first `window_size` values are used as the input sequence, "
    "while the final value becomes the prediction target. The resulting `(input_window, target)` pairs are cached, "
    "shuffled, batched according to `batch_size`, and prefetched to optimize the training "
    "pipeline."
)

plot.plot_sliding_window(data, config)
