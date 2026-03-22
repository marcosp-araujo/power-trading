import streamlit as st
from src.config_tools import Config_Manager
from src import data_tools, model_tools

def load_data(config):
    return data_tools.Data_Manager(config)

def load_model(_data, config):
    return model_tools.Model_Manager(_data, config)

def load_app_data():

    config = Config_Manager(
        mode="load",
        model_name="model_v1_6_w12h_1h_ahead",
    )

    config.run()

    if "data" not in st.session_state:
        st.session_state.data = load_data(config)

    data = st.session_state.data

    if "model" not in st.session_state:
        st.session_state.model = load_model(data, config)

    model = st.session_state.model
    model.compute_forecast()

    return config, data, model