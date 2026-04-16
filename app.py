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
    df = pd.read_csv("dataset/energy-dataset.csv", index_col='time')
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.asfreq('D')
    df['price actual'] = df['price actual'].interpolate(method='linear')
    return df

# =============================
# FORECAST FUNCTION
# =============================
def forecast_lstm(model, scaler, series, steps=30, window_size=30):
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

        # Use training portion as input window
        series = train_display['price actual'].values
        preds = forecast_lstm(model, scaler, series, steps=steps, window_size=30)

        forecast_df = pd.DataFrame({
            'Forecast': preds
        }, index=actual_pct.index)

    st.subheader("Forecast Results")
    fig, ax = plt.subplots(figsize=(12, 4))
    train_display['price actual'].plot(ax=ax, label=f'Historical ({100 - forecast_pct}%)', color='steelblue')
    actual_pct['price actual'].plot(ax=ax, label=f'Actual ({forecast_pct}%)', color='gray', linestyle=':')
    forecast_df['Forecast'].plot(ax=ax, label='Forecast', color='tomato')
    ax.axvline(x=train_display.index[-1], color='black', linestyle='--', alpha=0.4, label='Split point')
    ax.legend()
    ax.set_title(f"LSTM Energy Price Forecast — {100 - forecast_pct}/{forecast_pct} Split")
    st.pyplot(fig)

    st.subheader("Forecast Values")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "lstm_forecast.csv", "text/csv")