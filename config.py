from typing import Optional, Dict, Literal, TypedDict, Union

import requests
import time

# Базовый URL для API Binance
API_BASE_URL: str = "https://api.binance.com/api/v3"

# Параметры для повторных попыток запросов
MAX_RETRIES: int = 3
RETRY_DELAY: int = 5  # Задержка между попытками в секундах

# Ограничения Binance API
BINANCE_LIMIT_STRING: int = 1000

SEQ_LENGTH: int = 60

class IntervalConfig(TypedDict):
    days: int
    minutes: int
    milliseconds: int

IntervalKey = Literal["1m", "5m", "15m"]

SYMBOL_MAPPING: Dict[str, int] = {
    "BTCUSDT": 0,
    "ETHUSDT": 1,
}

TARGET_SYMBOL: str = "ETHUSDT"
PREDICTION_MINUTES: int = 5

INTERVAL_MAPPING: Dict[IntervalKey, IntervalConfig] = {
    "1m": {"days": 7, "minutes": 1, "milliseconds": 60000},
    "5m": {"days": 14, "minutes": 5, "milliseconds": 300000},
    "15m": {"days": 28, "minutes": 15, "milliseconds": 900000},
}

RAW_FEATURES: Dict[str, type] = {
    'symbol': str,
    'interval_str': str,
    'interval': int,
    'timestamp': int,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float,
    'quote_asset_volume': float,
    'number_of_trades': int,
    'taker_buy_base_asset_volume': float,
    'taker_buy_quote_asset_volume': float
}

MODEL_FEATURES: Dict[str, type] = {
    **RAW_FEATURES,
    "hour": int,
    "dayofweek": int,
    "sin_hour": float,
    "cos_hour": float,
    "sin_day": float,
    "cos_day": float,
}

# Пути к файлам и директориям
PATHS: Dict[str, str] = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
}

# Формат даты и времени
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Функция для получения разницы во времени между локальным и серверным временем Binance
def get_binance_time_offset() -> Optional[int]:
    try:
        response = requests.get(f"{API_BASE_URL}/time")
        server_time: int = response.json()['serverTime']
        local_time: int = int(time.time() * 1000)
        return server_time - local_time
    except requests.RequestException:
        return None

# Получение разницы во времени
TIME_OFFSET: Optional[int] = get_binance_time_offset()
