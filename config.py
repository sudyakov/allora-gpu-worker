import logging
from typing import Optional, Dict, Literal, TypedDict, Union, Tuple
import os
from collections import OrderedDict
import requests
import time
import numpy as np
from datetime import datetime, timezone

API_BASE_URL: str = "https://api.binance.com/api/v3"
MAX_RETRIES: int = 3
RETRY_DELAY: int = 5
BINANCE_LIMIT_STRING: int = 1000
SEQ_LENGTH: int = 50

class IntervalConfig(TypedDict):
    days: int
    minutes: int
    milliseconds: int

IntervalKey = int

SYMBOL_MAPPING: OrderedDict[str, int] = OrderedDict([
    ("ETHUSDT", 0),
    ("BTCUSDT", 1)
])

TARGET_SYMBOL: str = "ETHUSDT"
PREDICTION_MINUTES: int = 1

INTERVAL_MAPPING: OrderedDict[IntervalKey, IntervalConfig] = OrderedDict([
    (1, {"days": 15, "minutes": 1, "milliseconds": 60000}),
    (5, {"days": 90, "minutes": 5, "milliseconds": 300000}),
    (15, {"days": 180, "minutes": 15, "milliseconds": 900000})
])

RAW_FEATURES = OrderedDict([
    ('symbol', str),
    ('interval', np.int64)
])

TIME_FEATURES = OrderedDict([
    ('hour', np.float32),
    ('dayofweek', np.float32),
    ('timestamp', np.int64)
])

SCALABLE_FEATURES = OrderedDict([
    ('open', np.float32),
    ('high', np.float32),
    ('low', np.float32),
    ('close', np.float32),
    ('volume', np.float32),
    ('quote_asset_volume', np.float32),
    ('number_of_trades', np.float32),
    ('taker_buy_base_asset_volume', np.float32),
    ('taker_buy_quote_asset_volume', np.float32)
])

ADD_FEATURES = OrderedDict([
    ('sin_hour', np.float32),
    ('cos_hour', np.float32),
    ('sin_day', np.float32),
    ('cos_day', np.float32)
])

MODEL_FEATURES = OrderedDict()
MODEL_FEATURES.update(RAW_FEATURES)
MODEL_FEATURES.update(TIME_FEATURES)
MODEL_FEATURES.update(SCALABLE_FEATURES)
MODEL_FEATURES.update(ADD_FEATURES)

class ModelParams(TypedDict):
    input_size: int
    hidden_layer_size: int
    num_layers: int
    dropout: float
    embedding_dim: int
    num_symbols: int
    num_intervals: int
    timestamp_embedding_dim: int

MODEL_PARAMS: ModelParams = {
    "input_size": len(MODEL_FEATURES),
    "hidden_layer_size": 128,
    "num_layers": 16,
    "dropout": 0.2,
    "embedding_dim": 128,
    "num_symbols": len(SYMBOL_MAPPING.keys()),
    "num_intervals": len(INTERVAL_MAPPING.keys()),
    "timestamp_embedding_dim": 128
}

class TrainingParams(TypedDict):
    batch_size: int
    initial_epochs: int
    fine_tune_epochs: int
    initial_lr: float
    max_epochs: int
    min_lr: float
    use_mixed_precision: bool
    num_workers: int

TRAINING_PARAMS: TrainingParams = {
    "batch_size": 1024,
    "initial_epochs": 3,
    "fine_tune_epochs": 3,
    "initial_lr": 0.0005,
    "max_epochs": 100,
    "min_lr": 0.00001,
    "use_mixed_precision": True,
    "num_workers": 8
}

PATHS: Dict[str, str] = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data'
}

MODEL_VERSION = "2.0"

MODEL_FILENAME = os.path.join(PATHS["models_dir"], f"enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth")
DATA_PROCESSOR_FILENAME = os.path.join(PATHS["models_dir"], f"data_processor_{TARGET_SYMBOL}_v{MODEL_VERSION}.pkl")

DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

def get_binance_time_offset() -> Optional[int]:
    try:
        response = requests.get(f"{API_BASE_URL}/time")
        server_time: int = response.json()['serverTime']
        local_time: int = int(time.time() * 1000)
        return server_time - local_time
    except requests.RequestException:
        return None

TIME_OFFSET: Optional[int] = get_binance_time_offset()

def get_interval(minutes: int) -> Optional[IntervalKey]:
    for key, config in INTERVAL_MAPPING.items():
        if config["minutes"] == minutes:
            return key
    logging.error("Interval for %d minutes not found.", minutes)
    return None

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time
