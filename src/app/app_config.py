import streamlit as st
from src.config_tools import Config_Manager

config = Config_Manager(
    mode="load",
    model_name="model_v1_6_w12h_1h_ahead",
)

@st.cache_resource
def run_config():
    return  config.run()

run_config()