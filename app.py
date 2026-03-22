import streamlit as st
from pathlib import Path

if __name__ == "__main__":

    st.markdown("""<style>.block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)

    st.set_page_config(page_title="Time Series Forecasting", 
                       page_icon="📈")

    pages_dir = Path("src/app/pages")

    pg = st.navigation([
        st.Page(pages_dir / "home.py", 
                title="Home"),
        st.Page(pages_dir / "model_description.py", 
                title = "Model Architecture"),
        st.Page(pages_dir / "train_and_validation.py", 
                title = "Training Process"),
        st.Page(pages_dir / "training_history.py", 
                title = "Training Evaluation"),
        st.Page(pages_dir / "results.py", 
                title = "Forecasting Result and Performance"),
    ])

    pg.run()


    

