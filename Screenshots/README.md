# Stock Market Prediction using LSTM and FastAPI

# Project Overview

This project predicts the next closing price of Apple (AAPL) stock using a Long Short-Term Memory (LSTM) deep learning model. The application automatically downloads the latest stock prices using the yfinance library and displays the predicted next closing price through a FastAPI web application.

# Objective
To build an end-to-end stock market prediction system using Deep Learning and deploy it as a web application.

# Features
- Predicts the next closing price of AAPL stock
- Downloads the latest stock data automatically
- Uses an LSTM Deep Learning model
- FastAPI backend
- Interactive web interface
- Clean and responsive UI
- Uses yfinance for real-time historical stock data

# Technologies Used
- Python
- FastAPI
- TensorFlow / Keras
- LSTM (Long Short-Term Memory)
- NumPy
- Pandas
- Scikit-learn
- Joblib
- yfinance
- HTML
- CSS
- JavaScript

# Project Structure
```
Stock_market_prediction/
│
├── app/
├── data/
├── models/
├── notebooks/
├── screenshots/
├── static/
├── templates/
├── requirements.txt
├── Dockerfile
└── README.md
```
# How to Run

# Clone the repository

```bash
git clone <your-github-repository-link>
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Start the application

```bash
uvicorn app.main:app --reload
```

Open your browser and visit:

```
http://127.0.0.1:8000
```

# Screenshots

# Home Page

Add:

```
screenshots/home.png
```

# Prediction Result

Add:

```
screenshots/prediction.png
```

# Future Enhancements

- Support multiple stock symbols
- Improve prediction accuracy
- Display historical price charts
- Deploy on a cloud platform
- Add confidence metrics for predictions

# Developed By

Usha Reddy N