import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.arima.model import ARIMAResults


@st.cache_resource  # loads once, stays in memory
def load_model():
    model = ARIMAResults.load('arima_model.pkl')
    params = joblib.load('arima_params.joblib')
    return model, params

model, params = load_model()


st.title("Energy Price Forecasting")
st.write(f"Model: ARIMA({params['optimal_p']}, {params['optimal_d']}, {params['optimal_q']})")


uploaded_file = st.file_uploader("Upload CSV with 'time' and 'price actual' columns", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['time'], index_col='time')
    st.subheader("Uploaded Data")
    st.line_chart(df['price actual'])


    steps = st.slider("Forecast how many days ahead?", min_value=1, max_value=60, value=30)

    if st.button("Generate Forecast"):
        with st.spinner("Forecasting..."):

            updated_model = model.apply(df['price actual'])
            forecast = updated_model.forecast(steps=steps)


            forecast_index = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=steps
            )
            forecast_df = pd.DataFrame({
                'Forecast': forecast.values
            }, index=forecast_index)


        st.subheader("Forecast Results")
        fig, ax = plt.subplots(figsize=(12, 4))
        df['price actual'].iloc[-60:].plot(ax=ax, label='Historical', color='steelblue')
        forecast_df['Forecast'].plot(ax=ax, label='Forecast', color='tomato', linestyle='--')
        ax.legend()
        ax.set_title("Energy Price Forecast")
        st.pyplot(fig)


        st.subheader("Forecast Values")
        st.dataframe(forecast_df)


        csv = forecast_df.to_csv().encode('utf-8')
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")