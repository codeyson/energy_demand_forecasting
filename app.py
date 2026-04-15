import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# LOAD MODELS
@st.cache_resource
def load_models():
    lstm_model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(30, 1)),
        Dropout(0.0),
        LSTM(16, return_sequences=False),
        Dropout(0.0),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])

    lstm_model(np.zeros((1, 30, 1), dtype=np.float32))
    lstm_model.load_weights("model/lstm_model.h5")

    arima_model = joblib.load("model/arima_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    return lstm_model, arima_model, scaler


lstm_model, arima_model, scaler = load_models()


# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/energy_dataset.csv")
    return df

df = load_data()


def get_target_column(df):
    possible_cols = [
        "energy",
        "demand",
        "energy_demand",
        "load",
        "generation"
    ]
    for col in possible_cols:
        if col in df.columns:
            return col
    return df.columns[-1]  


target_col = get_target_column(df)

st.title("Energy Demand Forecasting")
st.write("Compare ARIMA and LSTM predictions")

model_choice = st.selectbox("Choose Model", ["ARIMA", "LSTM"])
steps = st.slider("Forecast Steps (hours)", 1, 48, 24)


# ARIMA PREDICTION
if model_choice == "ARIMA":
    st.subheader("ARIMA Forecast")

    forecast = arima_model.forecast(steps=steps)

    fig, ax = plt.subplots()
    ax.plot(range(1, steps + 1), forecast, marker="o", label="ARIMA Forecast")
    ax.set_xlabel("Forecast Step")
    ax.set_ylabel("Energy Demand")
    ax.set_title("ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)

    st.write("Forecast values:")
    st.dataframe(pd.DataFrame({"Step": range(1, steps + 1), "Forecast": forecast}))


# LSTM PREDICTION
if model_choice == "LSTM":
    st.subheader("LSTM Forecast")

    series = df[target_col].dropna().values.reshape(-1, 1)

    if len(series) < 30:
        st.error("Dataset must contain at least 30 rows for LSTM prediction.")
    else:
        # Scale using saved scaler
        scaled_series = scaler.transform(series)

        # Last 30 timesteps
        input_seq = scaled_series[-30:].reshape(1, 30, 1)

        preds_scaled = []

        for _ in range(steps):
            pred_scaled = lstm_model.predict(input_seq, verbose=0)
            preds_scaled.append(pred_scaled[0, 0])

            # slide window
            input_seq = np.concatenate(
                [input_seq[:, 1:, :], pred_scaled.reshape(1, 1, 1)],
                axis=1
            )

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).flatten()

        fig, ax = plt.subplots()
        ax.plot(range(1, steps + 1), preds, marker="o", label="LSTM Forecast")
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel("Energy Demand")
        ax.set_title("LSTM Forecast")
        ax.legend()
        st.pyplot(fig)

        st.write("Forecast values:")
        st.dataframe(pd.DataFrame({"Step": range(1, steps + 1), "Forecast": preds}))