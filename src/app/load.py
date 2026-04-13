import streamlit as st
import pickle
from src.app.app_config import load_config

@st.cache_resource
def load_model(config):
    return pickle.load(open(f"{config.model_out_path}", "rb"))

def load_app_data():

    config = load_config()

    if "model" not in st.session_state:
        st.session_state.model_output = load_model(config)

    model_output = st.session_state.model_output

    return model_output