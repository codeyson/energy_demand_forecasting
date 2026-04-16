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

    # Keep preprocessing closer to notebook
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)

    # Removed df.asfreq('D') because it can break alignment with training data
    df["price actual"] = df["price actual"].interpolate(method="linear")
    df["price actual"] = df["price actual"].ffill().bfill()

    return df

# =============================
# FORECAST FUNCTION
# Notebook-style prediction:
# uses actual previous values for each test step
# =============================
def forecast_lstm(model, scaler, train_series, test_series, window_size=30):
    full_series = np.concatenate([train_series, test_series])
    scaled_full = scaler.transform(full_series.reshape(-1, 1)).flatten()

    X_test = []
    for i in range(len(train_series), len(full_series)):
        X_test.append(scaled_full[i - window_size:i])

    X_test = np.array(X_test).reshape(-1, window_size, 1)

    preds_scaled = model.predict(X_test, verbose=0)
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

df = load_data()

# =============================
# SLIDER — same as ARIMA app
# =============================
forecast_pct = st.slider(
    "Forecast portion of data (%)",
    min_value=5,
    max_value=40,
    value=20,
    step=5,
    help="Controls how much of the data is used for forecasting"
)

split_idx = int(len(df) * (1 - forecast_pct / 100))
train_display = df.iloc[:split_idx]
actual_pct = df.iloc[split_idx:]
steps = len(actual_pct)

st.info(f"Historical: {split_idx} days ({100 - forecast_pct}%) → Forecast: {steps} days ({forecast_pct}%)")

# =============================
# FORECAST
# =============================
if st.button("Generate Forecast"):
    with st.spinner("Forecasting..."):

        # Use notebook-style prediction instead of recursive forecasting
        train_series = train_display["price actual"].values
        test_series = actual_pct["price actual"].values
        preds = forecast_lstm(model, scaler, train_series, test_series, window_size=30)

        forecast_df = pd.DataFrame({
            'Forecast': preds
        }, index=actual_pct["time"])

    st.subheader("Forecast Results")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_display["time"], train_display["price actual"], label=f'Historical ({100 - forecast_pct}%)', color='steelblue')
    ax.plot(actual_pct["time"], actual_pct["price actual"], label=f'Actual ({forecast_pct}%)', color='gray', linestyle=':')
    ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='tomato')
    ax.axvline(x=train_display["time"].iloc[-1], color='black', linestyle='--', alpha=0.4, label='Split point')
    ax.legend()
    ax.set_title(f"LSTM Energy Price Forecast — {100 - forecast_pct}/{forecast_pct} Split")
    ax.set_xlabel("time")
    ax.set_ylabel("Energy Price")
    st.pyplot(fig)

    st.subheader("Forecast Values")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "lstm_forecast.csv", "text/csv")