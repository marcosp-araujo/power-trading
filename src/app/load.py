import streamlit as st
from src import data_tools, model_tools
from src.app.app_config import config

@st.cache_resource
def load_data(config):
    return data_tools.Data_Manager(config)

@st.cache_resource
def load_model(_data, config):
    return model_tools.Model_Manager(_data, config)

def load_app_data():

    if "data" not in st.session_state:
        st.session_state.data = load_data(config)

    data = st.session_state.data

    if "model" not in st.session_state:
        st.session_state.model = load_model(data, config)

    model = st.session_state.model
    model.compute_forecast()

    return data, model