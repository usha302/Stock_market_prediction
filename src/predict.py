import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

print("=" * 60)
print("Loading Model")
print("=" * 60)

# Load trained model
model = load_model("models/lstm_model.keras")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Load test data
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print("Predicting...")

# Predict
predictions = model.predict(X_test)

# Convert back to original stock prices
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# Plot graph
plt.figure(figsize=(12,6))

plt.plot(actual, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.savefig("models/prediction_graph.png")

print("\nPrediction graph saved successfully!")

# Print first 10 predictions
print("\nFirst 10 Predictions:\n")

for i in range(10):
    print(f"Actual: {actual[i][0]:.2f}  |  Predicted: {predictions[i][0]:.2f}")