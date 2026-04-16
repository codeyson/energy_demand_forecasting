import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMAResults


@st.cache_resource
def load_model():
    model = ARIMAResults.load('model/arima_model.pkl')
    params = joblib.load('model/arima_params.joblib')
    return model, params

@st.cache_data
def load_data():
    df = pd.read_csv('dataset/energy_dataset.csv', index_col='time')
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.asfreq('D')
    df['price actual'] = df['price actual'].interpolate(method='linear')
    return df


model, params = load_model()
df = load_data()

st.title("Energy Price Forecasting")
st.write(f"Model: ARIMA({params['optimal_p']}, {params['optimal_d']}, {params['optimal_q']})")

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
steps = len(df) - split_idx

if st.button("Generate Forecast"):
    with st.spinner("Forecasting..."):

        actual_pct = df.iloc[split_idx:]

        updated_model = model.apply(actual_pct['price actual'])
        pred_obj = updated_model.get_prediction(
            start=actual_pct.index[0],
            end=actual_pct.index[-1],
            dynamic=False
        )
        forecast = pred_obj.predicted_mean
        forecast.index = actual_pct.index

        forecast_df = pd.DataFrame({
            'Forecast': forecast.values
        }, index=actual_pct.index)

    st.subheader("Forecast Results")
    fig, ax = plt.subplots(figsize=(12, 4))
    train_display['price actual'].plot(ax=ax, label='Historical', color='steelblue')
    forecast_df['Forecast'].plot(ax=ax, label='Forecast', color='tomato')
    ax.legend()
    ax.set_title(f"Energy Price Forecast")
    st.pyplot(fig)