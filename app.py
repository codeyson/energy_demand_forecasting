import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import load_model

# BASE PATH (important for deployment)
BASE_DIR = os.path.dirname(__file__)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    try:
        lstm_model = load_model(os.path.join(BASE_DIR, "model/lstm_model.h5"))
        arima_model = joblib.load(os.path.join(BASE_DIR, "model/arima_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "model/scaler.pkl"))

        return lstm_model, arima_model, scaler

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


lstm_model, arima_model, scaler = load_models()


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "dataset/energy_dataset.csv"))
        return df
    except:
        st.error("Dataset not found. Check your dataset folder.")
        st.stop()


df = load_data()


# =========================
# TARGET COLUMN DETECTION
# =========================
def get_target_column(df):
    possible_cols = ["energy", "demand", "energy_demand", "load", "generation"]
    for col in possible_cols:
        if col in df.columns:
            return col
    return df.columns[-1]


target_col = get_target_column(df)


# =========================
# UI
# =========================
st.title("Energy Demand Forecasting")
st.write("Compare ARIMA and LSTM predictions")

model_choice = st.selectbox("Choose Model", ["ARIMA", "LSTM"])
steps = st.slider("Forecast Steps (hours)", 1, 48, 24)


# =========================
# ARIMA
# =========================
if model_choice == "ARIMA":
    st.subheader("ARIMA Forecast")

    try:
        forecast = arima_model.forecast(steps=steps)

        fig, ax = plt.subplots()
        ax.plot(range(1, steps + 1), forecast, marker="o", label="ARIMA Forecast")
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel("Energy Demand")
        ax.legend()

        st.pyplot(fig)

        st.dataframe(pd.DataFrame({
            "Step": range(1, steps + 1),
            "Forecast": forecast
        }))

    except Exception as e:
        st.error(f"ARIMA error: {e}")


# =========================
# LSTM
# =========================
if model_choice == "LSTM":
    st.subheader("LSTM Forecast")

    try:
        series = df[target_col].dropna().values.reshape(-1, 1)

        if len(series) < 30:
            st.error("Dataset must contain at least 30 rows.")
        else:
            scaled_series = scaler.transform(series)

            input_seq = scaled_series[-30:].reshape(1, 30, 1)

            preds_scaled = []

            for _ in range(steps):
                pred_scaled = lstm_model.predict(input_seq, verbose=0)
                preds_scaled.append(pred_scaled[0, 0])

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
            ax.legend()

            st.pyplot(fig)

            st.dataframe(pd.DataFrame({
                "Step": range(1, steps + 1),
                "Forecast": preds
            }))

    except Exception as e:
        st.error(f"LSTM error: {e}")