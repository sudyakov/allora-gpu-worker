from typing import Optional, Dict, Literal, TypedDict, Union
import os

import requests
import time
import numpy as np

API_BASE_URL: str = "https://api.binance.com/api/v3"

MAX_RETRIES: int = 3
RETRY_DELAY: int = 5

BINANCE_LIMIT_STRING: int = 1000

SEQ_LENGTH: int = 100

class IntervalConfig(TypedDict):
    days: int
    minutes: int
    milliseconds: int

IntervalKey = Literal["1m", "5m", "15m"]

SYMBOL_MAPPING: Dict[str, int] = {
    "ETHUSDT": 0,
    "BTCUSDT": 1,
}

TARGET_SYMBOL: str = "ETHUSDT"
PREDICTION_MINUTES: int = 5

INTERVAL_MAPPING: Dict[IntervalKey, IntervalConfig] = {
    "1m": {"days": 30, "minutes": 1, "milliseconds": 60000},
    "5m": {"days": 60, "minutes": 5, "milliseconds": 300000},
    "15m": {"days": 90, "minutes": 15, "milliseconds": 900000},
}

RAW_FEATURES: Dict[str, type] = {
    'symbol': str,  # Categorical data; consider encoding later
    'interval_str': str,  # Categorical data; consider encoding later
    'interval': np.int64,
    'timestamp': np.int64,
}

SCALABLE_FEATURES: Dict[str, type] = {
    'open': np.float32,
    'high': np.float32,
    'low': np.float32,
    'close': np.float32,
    'volume': np.float32,
    'quote_asset_volume': np.float32,
    'number_of_trades': np.int64,
    'taker_buy_base_asset_volume': np.float32,
    'taker_buy_quote_asset_volume': np.float32
}

ADD_FEATURES: Dict[str, type] = {
    "hour": np.int64,
    "dayofweek": np.int64,
    "sin_hour": np.float32,
    "cos_hour": np.float32,
    "sin_day": np.float32,
    "cos_day": np.float32,
}

MODEL_FEATURES: Dict[str, type] = {
    **RAW_FEATURES,
    **SCALABLE_FEATURES,
    **ADD_FEATURES,
}

PATHS: Dict[str, str] = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
}

MODEL_VERSION = "2.0"

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
    "input_size": len(MODEL_FEATURES)-1,
    "hidden_layer_size": 256,
    "num_layers": 4,
    "dropout": 0.2,
    "embedding_dim": 128,
    "num_symbols": len(SYMBOL_MAPPING),
    "num_intervals": 3,
    "timestamp_embedding_dim": 64,
}

class TrainingParams(TypedDict):
    batch_size: int
    initial_epochs: int
    initial_lr: float
    max_epochs: int
    min_lr: float
    use_mixed_precision: bool
    num_workers: int

TRAINING_PARAMS: TrainingParams = {
    "batch_size": 512,
    "initial_epochs": 5,
    "initial_lr": 0.0005,
    "max_epochs": 100,
    "min_lr": 0.00001,
    "use_mixed_precision": True,
    "num_workers": 8,
}

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
