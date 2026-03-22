import streamlit as st
from src.config_tools import Config_Manager

@st.cache_resource
def load_config():
    config = Config_Manager(
        mode="load",
        model_name="model_v1_6_w12h_1h_ahead"
    )
    config.run()
    return config
