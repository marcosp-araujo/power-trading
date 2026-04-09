import streamlit as st
from src.app.app_config import load_config

st.markdown("""
### Model Description

This model is a **1D CNN for time-series forecasting** that learns temporal patterns from sliding windows of historical data. It consists of **two 1D convolutional layers** (32 filters and 16 filters, kernel size 5) with ReLU activation to extract features from the input sequence. The convolutional outputs are **flattened** and passed through a **dense layer** to produce the forecast, which is then **reshaped** to match the output sequence length. The model is designed to predict multiple future steps simultaneously and is trained using Huber loss function, and mean absolute error (MAE) metric for evaluation.
""")

config = load_config()
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Model Configuration")
    st.write(f"Window Size: {config.window_size}")
    st.write(f"Batch Size: {config.batch_size}")
    st.write(f"Shuffle Buffer: {config.shuffle_buffer}")

with col2:
    st.markdown("#### Training Parameters")
    st.write(f"Output Size: {config.output_size}")
    st.write(f"Epochs: {config.epochs}")
    st.write(f"Learning Rate: {config.learning_rate}")
    st.write(f"Momentum: {config.momentum}")