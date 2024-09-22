from datetime import datetime, timezone
from typing import Dict, List, Union
import requests
import time
from typing import Optional
import numpy as np

API_BASE_URL = "https://api.binance.com/api/v3"
MAX_RETRIES = 3
RETRY_DELAY = 5
BINANCE_LIMIT_STRING = 1000
BINANCE_INTERVAL_REQUEST = 1
REQUEST_DELAY = 1

TARGET_SYMBOL = 'ETHUSDT'
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ARBUSDT"]
SEQ_LENGTH = 300

PREDICTION_MINUTES = 5
CURRENT_MINUTES = 1

INTERVALS_PERIODS = {
    "1m": {"interval": "1m", "days": 7, "minutes": 1, "milliseconds": 1 * 60 * 1000},
    "5m": {"interval": "5m", "days": 14, "minutes": 5, "milliseconds": 5 * 60 * 1000},
    "15m": {"interval": "15m", "days": 28, "minutes": 15, "milliseconds": 15 * 60 * 1000},
    # "30m": {"interval": "30m", "days": 72, "minutes": 30, "milliseconds": 30 * 60 * 1000}
}

FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
]

BINANCE_API_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]

DATA_TYPES = {
    'timestamp': int, 'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64,
    'volume': np.float64, 'quote_asset_volume': np.float64,
    'number_of_trades': int, 'taker_buy_base_asset_volume': np.float64,
    'taker_buy_quote_asset_volume': np.float64, 'symbol': str, 'interval': str
}

MODEL_VERSION = "2.0"
MODEL_PARAMS = {
    'input_size': len(FEATURE_NAMES),
    'hidden_layer_size': 256,
    'num_layers': 4,
    'dropout': 0.2
}

TRAINING_PARAMS = {
    'batch_size': 512,
    'initial_epochs': 5,
    'initial_lr': 0.0005,
    'max_epochs': 100,
    'min_lr': 0.00001,
    'use_mixed_precision': True,
    'num_workers': 8
}

PATHS = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
}

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATASET_COLUMNS = ['timestamp', 'symbol', 'interval'] + FEATURE_NAMES

def get_binance_time_offset() -> Optional[int]:
    try:
        response = requests.get(f"{API_BASE_URL}/time")
        server_time = response.json()['serverTime']
        local_time = int(time.time() * 1000)
        return server_time - local_time
    except:
        return None

TIME_OFFSET = get_binance_time_offset()
