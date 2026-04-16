import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="LSTM Energy Demand Forecast", layout="wide")

# =============================
# MODEL DEFINITION
# Must match the training architecture exactly
# =============================
def build_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(30, 1)),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.LSTM(16, return_sequences=False),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    return model

# =============================
# LOAD RESOURCES
# =============================
@st.cache_resource
def load_resources():
    try:
        model = build_lstm_model()

        # Build model first before loading weights
        dummy_input = np.zeros((1, 30, 1), dtype=np.float32)
        model(dummy_input)

        model.load_weights("model/lstm_model.h5")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/energy_dataset.csv")
    return df

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

def forecast_lstm(model, scaler, series, steps=24, window_size=30):
    scaled_series = scaler.transform(series.reshape(-1, 1))
    input_seq = scaled_series[-window_size:].reshape(1, window_size, 1)

    preds_scaled = []
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)
        preds_scaled.append(pred[0, 0])

        input_seq = np.concatenate(
            [input_seq[:, 1:, :], pred.reshape(1, 1, 1)],
            axis=1
        )

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

# =============================
# APP
# =============================
st.title("⚡ LSTM Energy Demand Forecasting")
st.write("Forecast future energy demand using the saved LSTM model.")

model, scaler, load_error = load_resources()

if load_error:
    st.error(f"Model/scaler loading failed: {load_error}")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

target_col = get_target_column(df)

if target_col not in df.columns:
    st.error("Target column not found in dataset.")
    st.stop()

series = df[target_col].dropna().values

if len(series) < 30:
    st.error("Dataset must contain at least 30 rows for forecasting.")
    st.stop()

st.subheader("Forecast Settings")
steps = st.slider("Forecast horizon (hours)", min_value=1, max_value=48, value=24)

if st.button("Run Forecast"):
    try:
        preds = forecast_lstm(model, scaler, series, steps=steps, window_size=30)

        result_df = pd.DataFrame({
            "Hour": np.arange(1, steps + 1),
            "Predicted Energy Demand": preds
        })

        st.subheader("Forecast Results")
        st.dataframe(result_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result_df["Hour"], result_df["Predicted Energy Demand"], marker="o")
        ax.set_title("LSTM Forecast")
        ax.set_xlabel("Forecast Hour")
        ax.set_ylabel("Predicted Energy Demand")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Latest Input Window")
        st.line_chart(pd.DataFrame({"Recent Actual Values": series[-30:]}))

    except Exception as e:
        st.error(f"Forecast failed: {e}")