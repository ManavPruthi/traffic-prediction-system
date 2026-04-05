import numpy as np
import joblib
from tensorflow.keras.models import load_model

def predict_future(df, steps=24, seq_len=24):

    model = load_model('models/lstm_model.h5')
    scaler = joblib.load('models/lstm_scaler.pkl')

    data = df['traffic_volume'].values.tolist()

    preds = []

    for _ in range(steps):
        seq = np.array(data[-seq_len:])
        seq = seq.reshape((1, seq_len, 1))

        pred = model.predict(seq)[0][0]

        data.append(pred)
        preds.append(pred)

    # Inverse scaling
    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1,1)
    )

    return preds.flatten()