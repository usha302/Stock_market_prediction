import mlflow
import mlflow.tensorflow
import numpy as np

from tensorflow.keras.models import load_model

# Set experiment
mlflow.set_experiment("Stock Market Prediction")

# Load model
model = load_model("models/lstm_model.keras")

# Load data
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Start MLflow run
with mlflow.start_run():

    loss = model.evaluate(X_test, y_test, verbose=0)

    mlflow.log_param("Model", "LSTM")
    mlflow.log_param("Epochs", 20)
    mlflow.log_param("Batch Size", 32)

    mlflow.log_metric("Test Loss", loss)

    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="lstm_model"
    )

print("MLflow logging completed successfully!")