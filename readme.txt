# 🚦 Traffic Prediction System (LSTM)

## 📌 Overview

This project is a Traffic Prediction System built using LSTM (Long Short-Term Memory) neural networks.

It predicts future traffic volume based on historical time-series data.

The system includes:
- Data preprocessing pipeline
- Deep learning model (LSTM)
- Evaluation metrics
- Interactive dashboard using Streamlit

---

## 🚀 Features

- Time-series data preprocessing
- Sequence-based LSTM modeling
- Future traffic prediction (multi-step forecasting)
- Model evaluation (MAE, RMSE)
- Interactive visualization dashboard
- Modular project structure

---

## 🏗️ Project Structure

traffic_prediction_engine/

├── data/
├── models/
│   ├── lstm_model.keras
│   └── lstm_scaler.pkl

├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py

├── main.py
├── dashboard.py
├── requirements.txt
└── README.md

---

## 📊 Dataset

This project uses the Metro Interstate Traffic Volume Dataset.

Required columns:
- date_time
- traffic_volume

---

## ⚙️ Installation

1. Clone the repository:

git clone <your-repo-link>
cd traffic_prediction_engine

2. Create virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run Training

python main.py

This will:
- preprocess data
- train the LSTM model
- save model and scaler in models/

---

## 🌐 Run Dashboard

streamlit run dashboard.py

Then open:
http://localhost:8501

---

## 🧠 Model Details

- Model: LSTM (Deep Learning)
- Input: Previous 24 hours of traffic data
- Output: Next-hour prediction
- Loss Function: Mean Squared Error
- Scaling: MinMaxScaler

---

## 📈 Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## 🔮 Future Improvements

- Add weather and holiday features
- Hyperparameter tuning
- GRU / Transformer models
- Real-time prediction API (FastAPI)
- Docker deployment

---

## ⚠️ Notes

- Dataset timestamps must be cleaned before parsing
- Model performance improves with more data
- GPU is not required (CPU training supported)

---

## 👨‍💻 Author

Traffic Prediction ML Project  
Built for learning production-level machine learning systems