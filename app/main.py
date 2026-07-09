from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model

app = FastAPI(title="Stock Market Prediction")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and scaler
model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler.pkl")


class StockInput(BaseModel):
    symbol: str = "AAPL"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )


@app.post("/predict")
def predict(data: StockInput):

    symbol = data.symbol.upper()

    try:
        # Download the latest data
        df = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if len(df) < 60:
            return {
                "error": "Not enough historical data."
            }

        # Last 60 closing prices
        prices = df["Close"].values[-60:]

        prices = prices.reshape(-1, 1)

        prices_scaled = scaler.transform(prices)

        X = np.array([prices_scaled])

        prediction = model.predict(X, verbose=0)

        prediction = scaler.inverse_transform(prediction)

        latest_price = float(prices[-1][0])

        return {
            "Stock": symbol,
            "Latest Close": round(latest_price, 2),
            "Predicted Next Close": round(float(prediction[0][0]), 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }