import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        # Rebuild LSTM architecture
        lstm_model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(30, 1)),
            Dropout(0.0),
            LSTM(16, return_sequences=False),
            Dropout(0.0),
            Dense(16, activation="relu"),
            Dense(1, activation="linear")
        ])

        # Initialize model
        lstm_model(np.zeros((1, 30, 1), dtype=np.float32))

        # Load weights (IMPORTANT: must be weights file)
        lstm_model.load_weights("model/lstm_model.h5")

        arima_model = joblib.load("model/arima_model.pkl")
        scaler = joblib.load("model/scaler.pkl")

        return lstm_model, arima_model, scaler

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None


lstm_model, arima_model, scaler = load_models()

# Stop app if models failed
if lstm_model is None:
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset/energy_dataset.csv")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None


df = load_data()

if df is None:
    st.stop()

# -----------------------------
# TARGET COLUMN
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.title("⚡ Energy Demand Forecasting")
st.write("Compare ARIMA and LSTM predictions")

model_choice = st.selectbox("Choose Model", ["ARIMA", "LSTM"])
steps = st.slider("Forecast Steps (hours)", 1, 48, 24)

# -----------------------------
# ARIMA
# -----------------------------
if model_choice == "ARIMA":
    st.subheader("ARIMA Forecast")

    try:
        forecast = arima_model.forecast(steps=steps)

        fig, ax = plt.subplots()
        ax.plot(range(1, steps + 1), forecast, marker="o")
        ax.set_title("ARIMA Forecast")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy Demand")

        st.pyplot(fig)

        st.dataframe(pd.DataFrame({
            "Step": range(1, steps + 1),
            "Forecast": forecast
        }))

    except Exception as e:
        st.error(f"❌ ARIMA error: {e}")

# -----------------------------
# LSTM
# -----------------------------
if model_choice == "LSTM":
    st.subheader("LSTM Forecast")

    try:
        series = df[target_col].dropna().values.reshape(-1, 1)

        if len(series) < 30:
            st.error("Dataset must have at least 30 rows")
        else:
            # Scale input
            scaled_series = scaler.transform(series)

            # Get last 30 values
            input_seq = scaled_series[-30:].reshape(1, 30, 1)

            preds_scaled = []

            for _ in range(steps):
                pred = lstm_model.predict(input_seq, verbose=0)
                preds_scaled.append(pred[0, 0])

                # Slide window
                input_seq = np.concatenate(
                    [input_seq[:, 1:, :], pred.reshape(1, 1, 1)],
                    axis=1
                )

            preds_scaled = np.array(preds_scaled).reshape(-1, 1)
            preds = scaler.inverse_transform(preds_scaled).flatten()

            fig, ax = plt.subplots()
            ax.plot(range(1, steps + 1), preds, marker="o")
            ax.set_title("LSTM Forecast")
            ax.set_xlabel("Step")
            ax.set_ylabel("Energy Demand")

            st.pyplot(fig)

            st.dataframe(pd.DataFrame({
                "Step": range(1, steps + 1),
                "Forecast": preds
            }))

    except Exception as e:
        st.error(f"❌ LSTM error: {e}")