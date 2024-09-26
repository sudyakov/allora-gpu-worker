# Импорт необходимых библиотек
from datetime import datetime, timezone
from typing import Dict, Union, Optional
import requests
import time
import pandas as pd

# Базовый URL для API Binance
API_BASE_URL = "https://api.binance.com/api/v3"

# Параметры для повторных попыток запросов
MAX_RETRIES = 3
RETRY_DELAY = 5

# Ограничения Binance API
BINANCE_LIMIT_STRING = 1000
BINANCE_INTERVAL_REQUEST = 1
REQUEST_DELAY = 1
# Временной интевал, выражает текушее время в минутах
CURRENT_MINUTES = 1
# Длина последовательности для модели
SEQ_LENGTH = 30

# Целевая и дополнительные торговые пары
TARGET_SYMBOL = 'ETHUSDT'
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Временные интервалы для прогнозирования
PREDICTION_MINUTES = 5

# Периоды для различных интервалов
INTERVALS_PERIODS = {
    "1m": {"days": 7, "minutes": 1, "milliseconds": 1 * 60 * 1000},
    "5m": {"days": 14, "minutes": 5, "milliseconds": 5 * 60 * 1000},
    "15m": {"days": 28, "minutes": 15, "milliseconds": 15 * 60 * 1000},
}

# Определение типов данных для признаков
FEATURE_NAMES: Dict[str, Union[type, str]] = {
    'symbol': str,
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


# Столбцы, возвращаемые API Binance
BINANCE_API_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]

# Версия и параметры модели
MODEL_VERSION = "2.0"
MODEL_PARAMS = {
    'input_size': len(FEATURE_NAMES),
    'hidden_layer_size': 256,
    'num_layers': 4,
    'dropout': 0.2
}

# Параметры для обучения модели
TRAINING_PARAMS = {
    'batch_size': 512,
    'initial_epochs': 10,
    'initial_lr': 0.0005,
    'max_epochs': 100,
    'min_lr': 0.00001,
    'use_mixed_precision': True,
    'num_workers': 8
}

# Пути к файлам и директориям
PATHS = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
}

# Формат даты и времени
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Функция для получения разницы во времени между локальным и серверным временем Binance
def get_binance_time_offset() -> Optional[int]:
    try:
        response = requests.get(f"{API_BASE_URL}/time")
        server_time = response.json()['serverTime']
        local_time = int(time.time() * 1000)
        return server_time - local_time
    except requests.RequestException:
        return None

# Получение разницы во времени
TIME_OFFSET = get_binance_time_offset()
