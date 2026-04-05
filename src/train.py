import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def split_data(df, split_ratio=0.8):
    split = int(len(df) * split_ratio)
    return df[:split], df[split:]

def train_lstm(train, seq_len=24):

    data = train['traffic_volume'].values

    from src.features import create_sequences
    X, y = create_sequences(data, seq_len)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_len, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    model.save('models/lstm_model.keras')

    return model