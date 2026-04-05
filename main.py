import numpy as np
import joblib

from src.preprocessing import Preprocessor
from src.train import split_data, train_lstm
from src.features import create_sequences
from src.evaluate import evaluate

print("🔹 Loading data...")

pre = Preprocessor()
df = pre.process("data/traffic.csv", year=2018)

train, test = split_data(df)

# Train model
model = train_lstm(train)

# Prepare test sequences
test_data = test['traffic_volume'].values
X_test, y_test = create_sequences(test_data)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict
preds = model.predict(X_test).flatten()

# Evaluate
metrics = evaluate(y_test, preds)

print("\n📊 LSTM RESULTS:")
print(metrics)