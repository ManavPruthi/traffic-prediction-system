import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        os.makedirs("models", exist_ok=True)

    def process(self, path, year=None):
        df = pd.read_csv(path)
        # Fix format: "02-10-2012 9.00" → "02-10-2012 09:00"
        df['date_time'] = df['date_time'].str.replace('.', ':', regex=False)

        # Parse explicitly (day-first format)
        df['date_time'] = pd.to_datetime(
            df['date_time'],
            format="%d-%m-%Y %H:%M",
            errors='coerce'
        )
        df = df.dropna(subset=['date_time'])
        df = df.sort_values('date_time')

        # Optional filter
        if year:
            df = df[df['date_time'].dt.year == year]

        # Keep only needed column
        df = df[['date_time', 'traffic_volume']]

        # Handle missing
        df = df.ffill()

        # Scale
        df['traffic_volume'] = self.scaler.fit_transform(
            df[['traffic_volume']]
        )

        joblib.dump(self.scaler, 'models/lstm_scaler.pkl')

        return df