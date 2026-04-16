import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(page_title="Energy Demand Forecasting", layout="wide")


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

@st.cache_resource
def load_resources():
    try:
        model = build_lstm_model()

        # Build model before loading weights
        dummy_input = np.zeros((1, 30, 1), dtype=np.float32)
        model(dummy_input)

        model.load_weights("model/lstm_model.h5")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/energy_dataset.csv", index_col="time")
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.sort_index()
    df = df.asfreq("D")
    return df

def get_target_column(df):
    possible_cols = [
        "price actual",
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

def forecast_lstm_over_test(model, scaler, train_series, test_len, window_size=30):
    scaled_train = scaler.transform(train_series.reshape(-1, 1))
    input_seq = scaled_train[-window_size:].reshape(1, window_size, 1)

    preds_scaled = []

    for _ in range(test_len):
        pred = model.predict(input_seq, verbose=0)
        preds_scaled.append(pred[0, 0])

        input_seq = np.concatenate(
            [input_seq[:, 1:, :], pred.reshape(1, 1, 1)],
            axis=1
        )

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

# APP
st.title("Energy Demand Forecasting")

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

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df[target_col] = df[target_col].interpolate(method="linear")
df = df.dropna(subset=[target_col])

if len(df) < 60:
    st.error("Dataset must contain at least 60 valid rows for forecasting.")
    st.stop()

st.write("Model: LSTM")

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
test_actual = df.iloc[split_idx:]
steps = len(test_actual)

if len(train_display) < 30:
    st.error("Training portion is too small. Increase the dataset size or reduce forecast percentage.")
    st.stop()

if st.button("Generate Forecast"):
    with st.spinner("Forecasting..."):
        train_series = train_display[target_col].values
        preds = forecast_lstm_over_test(
            model=model,
            scaler=scaler,
            train_series=train_series,
            test_len=steps,
            window_size=30
        )

        forecast_df = pd.DataFrame({
            "Forecast": preds
        }, index=test_actual.index)

    st.subheader("Forecast Results")
    fig, ax = plt.subplots(figsize=(12, 4))
    train_display[target_col].plot(ax=ax, label="Historical", color="steelblue")
    forecast_df["Forecast"].plot(ax=ax, label="Forecast", color="tomato")
    ax.legend()
    ax.set_title("Energy Demand Forecast")
    st.pyplot(fig)
