import streamlit as st
from src import plot
from src.app.load import load_app_data

st.subheader("Training Evaluation")

st.write("This section presents the Loss and Mean Absolute Error (MAE) for the training and validation datasets across all training epochs. " 
    "The upper plot provides the full training history, while the lower plot zooms in on the final 50% of epochs to highlight fine-grained trends.")

config, data, model = load_app_data()

plot.metrics_history(model.history, streamlit=True)

st.write("The model demonstrates convergence and stability, as both the training and validation metrics decrease consistently without significant divergence or volatility. " 
"This suggests the model is effectively learning the underlying patterns without overfitting to the training data.")