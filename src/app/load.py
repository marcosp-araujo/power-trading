import streamlit as st
from src.app.app_config import load_config
from src.libs.processing_chain import processing_chain

@st.cache_resource
def load_model(config):
    return processing_chain(config)

def load_app_data():

    config = load_config()

    if "model" not in st.session_state:
        st.session_state.model = load_model(config)

    model = st.session_state.model

    return model