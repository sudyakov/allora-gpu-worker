from typing import Optional, Dict, Literal, TypedDict, Union
import os
from collections import OrderedDict
import requests
import time
import numpy as np

API_BASE_URL: str = "https://api.binance.com/api/v3"

MAX_RETRIES: int = 3
RETRY_DELAY: int = 5

BINANCE_LIMIT_STRING: int = 1000

SEQ_LENGTH: int = 300

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
    "1m": {"days": 9, "minutes": 1, "milliseconds": 60000},
    "5m": {"days": 18, "minutes": 5, "milliseconds": 300000},
    "15m": {"days": 36, "minutes": 15, "milliseconds": 900000},
}

RAW_FEATURES = OrderedDict([
    ('symbol', str),
    ('interval', str),
])

SCALABLE_FEATURES = OrderedDict([
    ('timestamp', np.int64),
    ('open', np.float32),
    ('high', np.float32),
    ('low', np.float32),
    ('close', np.float32),
    ('volume', np.float32),
    ('quote_asset_volume', np.float32),
    ('number_of_trades', np.int64),
    ('taker_buy_base_asset_volume', np.float32),
    ('taker_buy_quote_asset_volume', np.float32),
])

ADD_FEATURES = OrderedDict([
    ('hour', np.int64),
    ('dayofweek', np.int64),
    ('sin_hour', np.float32),
    ('cos_hour', np.float32),
    ('sin_day', np.float32),
    ('cos_day', np.float32),
])

MODEL_FEATURES = OrderedDict()
MODEL_FEATURES.update(RAW_FEATURES)
MODEL_FEATURES.update(SCALABLE_FEATURES)
MODEL_FEATURES.update(ADD_FEATURES)

PATHS: Dict[str, str] = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
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
