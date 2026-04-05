import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.preprocessing import Preprocessor
from src.features import create_sequences

st.set_page_config(page_title="Traffic Prediction System", layout="wide")

st.title("🚦 Traffic Prediction Dashboard (LSTM)")

# Upload file
file = st.file_uploader("Upload Traffic Dataset (CSV)")

if file:
    pre = Preprocessor()

    # Process data
    df = pd.read_csv(file)
    df['date_time'] = df['date_time'].str.replace('.', ':', regex=False)
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    df = df.dropna(subset=['date_time'])

    df = df.sort_values('date_time')

    st.subheader("📊 Traffic Volume Over Time")
    st.line_chart(df.set_index('date_time')['traffic_volume'])

    # Scale data
    scaler = joblib.load('models/lstm_scaler.pkl')
    df['traffic_volume'] = scaler.transform(df[['traffic_volume']])

    # Load model
    model = load_model('models/lstm_model.keras')

    # Prediction settings
    steps = st.slider("Predict future hours", 1, 48, 24)

    if st.button("Predict Future Traffic"):
        data = df['traffic_volume'].values.tolist()
        seq_len = 24
        preds = []

        for _ in range(steps):
            seq = np.array(data[-seq_len:])
            seq = seq.reshape((1, seq_len, 1))

            pred = model.predict(seq)[0][0]

            data.append(pred)
            preds.append(pred)

        # Inverse scale
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        preds = preds.flatten()

        # Create future timestamps
        last_time = df['date_time'].iloc[-1]
        future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(steps)]

        pred_df = pd.DataFrame({
            "date_time": future_times,
            "traffic_volume": preds
        })

        st.subheader("🔮 Future Predictions")
        st.line_chart(pred_df.set_index('date_time'))

        st.dataframe(pred_df)