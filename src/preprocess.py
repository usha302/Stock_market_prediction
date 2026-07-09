import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

print("="*60)
print("Downloading Stock Market Data")
print("="*60)

# Download Apple stock data
df = yf.download("AAPL", start="2015-01-01", end="2025-01-01")

# Save original data
df.to_csv("data/stock_data.csv")

print("\nDataset Shape:", df.shape)

# Use only the Close price
data = df[['Close']]

print("\nFirst 5 Rows:")
print(data.head())

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print("\nScaled Data Shape:", scaled_data.shape)

# Save scaler
import joblib
joblib.dump(scaler, "models/scaler.pkl")

# Create sequences
sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print("\nInput Shape:", X.shape)
print("Output Shape:", y.shape)

# Train/Test Split
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Save processed arrays
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("\nPreprocessing Completed Successfully!")