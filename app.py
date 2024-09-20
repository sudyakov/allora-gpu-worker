import torch
import pandas as pd
from flask import Flask, Response, json
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from config import TARGET_SYMBOL, PREDICTION_MINUTES, MODEL_FILENAME, MODEL_PARAMS
from download_data import DownloadData
from BiLSTMModel import SimpleBiLSTMModel, get_device, predict_future_price

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Initialize the DownloadData class
download_data = DownloadData()

# Initialize the model with the same architecture as during training
device = get_device()
model = SimpleBiLSTMModel(**MODEL_PARAMS).to(device)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
model.eval()

@app.route("/inference")
def get_inference():
    symbol = TARGET_SYMBOL
    interval = PREDICTION_MINUTES

    # Get current price data using the DownloadData class
    df = download_data.get_current_price(symbol, interval)
    if not df.empty:
        current_price = df.iloc[-1]["close"]
        current_time = df.iloc[-1]["timestamp"]
        readable_time = datetime.fromtimestamp(current_time / 1000).isoformat()
        
        logger.info(f"Current Price: {current_price} at {readable_time}")

        # Prepare data for the BiLSTM model
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

        seq = torch.FloatTensor(scaled_data).view(1, -1, 1).to(device)

        # Make prediction
        with torch.no_grad():
            y_pred = model(seq)

        # Inverse transform the predictions to get the actual prices
        predicted_prices = scaler.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
        predicted_price = round(float(predicted_prices[-1][0]), 2)

        # Log the prediction
        logger.info(f"Prediction: {predicted_price}")

        # Return the current price and predicted price in JSON response
        return Response(json.dumps({"current_price": current_price, "timestamp": readable_time, "predicted_price": predicted_price}), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data"}), status=500, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
