import streamlit as st
from src import plot
from src.app.load import load_app_data

data, model = load_app_data()
results = model.metrics()

st.subheader("Forecasting Result")

st.write("The forecasted time series accurately captures the trends observed in the validation data.")

plot.forecast(model, streamlit=True)

st.subheader("Performance")
st.write(f"A coefficient of determination close to 1 was obtained when comparing the forecasted and reference data, indicating strong model performance.")

plot.scatter(model=model, streamlit=True)

st.markdown(f"""
#### Model KPIs

The key performance indicators below highlight the goodness of fit of the model regardin mean absolute error (MAE), The Normalized Mean Absolute Error (NMAE), and Root Mean Squared Error (RMSE).

The NMAE is calculated as:

        NMAE = MAE / Capacity,

where the capacity = {results["capacity"]:.1f} MW, defined as the maximum wind power observed in the dataset.
""")

# Metrics display
col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{results['mae']:.2f} MW")
col2.metric("NMAE", f"{results['nmae']:.2f} %")
col3.metric("RMSE", f"{results['rmse']:.2f} MW")

