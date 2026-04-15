import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


@st.cache_resource
def load_models():
    lstm_model = load_model("model/tuned_lstm_energy_model.keras")
    arima_model = joblib.load("arima_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return lstm_model, arima_model, scaler

lstm_model, arima_model, scaler = load_models()


st.title("Energy Demand Forecasting")
st.write("Compare ARIMA and LSTM predictions")


model_choice = st.selectbox("Choose Model", ["ARIMA", "LSTM"])

steps = st.slider("Forecast Steps (hours)", 1, 48, 24)


# ARIMA PREDICTION
if model_choice == "ARIMA":
    st.subheader("ARIMA Forecast")

    forecast = arima_model.forecast(steps=steps)

    fig, ax = plt.subplots()
    ax.plot(forecast, label="Forecast")
    ax.set_title("ARIMA Forecast")
    ax.legend()

    st.pyplot(fig)


# LSTM PREDICTION
if model_choice == "LSTM":
    st.subheader("LSTM Forecast")

    # Dummy input (replace later with real last sequence)
    input_seq = np.random.rand(1, 24, 1)

    preds = []

    for _ in range(steps):
        pred = lstm_model.predict(input_seq, verbose=0)
        preds.append(pred[0][0])

        # slide window
        input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.plot(preds, label="Forecast")
    ax.set_title("LSTM Forecast")
    ax.legend()

    st.pyplot(fig)