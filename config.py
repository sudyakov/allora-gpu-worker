from datetime import datetime, timezone
from typing import Dict, List, Union
import requests
import time
from typing import Optional

# Base URL for Binance API
API_BASE_URL = "https://api.binance.com/api/v3"
# Parameters for request retries
MAX_RETRIES = 3
RETRY_DELAY = 5
BINANCE_LIMIT_STRING = 1000
BINANCE_INTERVAL_REQUEST = 1 # 1 minute
REQUEST_DELAY = 1 # second

# Target symbol and list of symbols for analysis
TARGET_SYMBOL = 'ETHUSDT'
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ARBUSDT"]
SEQ_LENGTH = 10

# Parameters for prediction
PREDICTION_MINUTES = 5
# Интервал для текущих данных
CURRENT_MINUTES = 1  

# Intervals and periods for data collection
INTERVALS_PERIODS = {
    "1m": {"interval": "1m", "days": 1, "minutes": 1, "milliseconds": 1 * 60 * 1000},
    "5m": {"interval": "5m", "days": 5, "minutes": 5, "milliseconds": 5 * 60 * 1000},
    "15m": {"interval": "15m", "days": 15, "minutes": 15, "milliseconds": 15 * 60 * 1000},
}

# Feature names for the model
FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
]
# Binance API columns
BINANCE_API_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]

# Feature names for the model
FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
]

# Data types for columns
DATA_TYPES = {
    'timestamp': int, 'open': float, 'high': float, 'low': float, 'close': float,
    'volume': float, 'quote_asset_volume': float,
    'number_of_trades': int, 'taker_buy_base_asset_volume': float,
    'taker_buy_quote_asset_volume': float, 'symbol': str, 'interval': str
}

# Model version and its parameters
MODEL_VERSION = "2.0"
MODEL_PARAMS = {
    'input_size': len(FEATURE_NAMES),
    'hidden_layer_size': 512,
    'num_layers': 6,
    'dropout': 0.3
}

# Parameters for model training
TRAINING_PARAMS = {
    'batch_size': 4096, 'initial_epochs': 10, 'initial_lr': 0.001,
    'max_epochs': 50, 'min_lr': 0.0001, 'use_mixed_precision': True,
    'num_workers': 16
}

# Paths to files and directories
PATHS = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
}

# Date and time format
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
# Columns for the dataset
DATASET_COLUMNS = ['timestamp', 'symbol', 'interval'] + FEATURE_NAMES

# Function to get Binance server time offset
def get_binance_time_offset() -> Optional[int]:
    try:
        response = requests.get(f"{API_BASE_URL}/time")
        server_time = response.json()['serverTime']
        local_time = int(time.time() * 1000)
        return server_time - local_time
    except:
        return None

TIME_OFFSET = get_binance_time_offset()
