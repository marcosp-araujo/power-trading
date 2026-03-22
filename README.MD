# 🌬️ Wind Power Forecasting

This repository provides an Artificial Intelligence (AI) solution for wind power forecasting in the **Netherlands**. By analyzing 12-hour rolling windows of historical data, the model predicts generation for the next **15-minute**. 

This tool can be used to assist energy traders in optimizing intraday positions and minimizing imbalance settlement costs on the Dutch grid.

---

## 🚀 Features
* **Time Horizon:** 15-minute ahead forecasting.
* **Architecture:** Deep Neural Network optimized with SGD & Momentum.
* **Robust Metrics:** Performance tracking **Huber Loss** to handle high-capacity volatility (0-3000 MW).
* **Interactive Dashboard:** A Streamlit interface to audit model stability and visualize real-time predictions.

---

## 🛠️ Installation & Setup

Creating a virtual environment is highly recommended to ensure dependency stability.

Install `uv` (Fast Package Manager)

```bash
pip install uv
```
Create a virtual environment
```
uv venv --python 3.12.10
```
Activate the virtual environment

Windows (PowerShell):
```
.\.venv\Scripts\activate.ps1
```
Linux/macOS:
```
source .venv/bin/activate
```

# 💻 Running the Software
## Model Training & Backend
To train a new model or load existing weights for evaluation, execute the main script:

```
uv run .\main.py
```

## Interactive Dashboard

To launch the Streamlit-based visualization and performance dashboard:
```
streamlit run .\app.py
```