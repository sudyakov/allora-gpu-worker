from typing import Optional
import requests
import time

# Базовый URL для API Binance
API_BASE_URL = "https://api.binance.com/api/v3"

# Параметры для повторных попыток запросов
MAX_RETRIES = 3
RETRY_DELAY = 5

# Ограничения Binance API
BINANCE_LIMIT_STRING = 1000

# Длина последовательности для модели
SEQ_LENGTH = 100

# Временные интервалы для прогнозирования
PREDICTION_MINUTES = 5

# Целевая и дополнительные торговые пары
TARGET_SYMBOL = 'ETHUSDT'

# Маппинги символов
SYMBOL_MAPPING = {
    "BTCUSDT": 0,
    "ETHUSDT": 1,
    # Добавьте другие символы по необходимости
}

# Периоды для различных интервалов
INTERVAL_MAPPING = {
    "1m": {"days": 7, "minutes": 1, "milliseconds": 1 * 60 * 1000},
    "5m": {"days": 14, "minutes": 5, "milliseconds": 5 * 60 * 1000},
    "15m": {"days": 28, "minutes": 15, "milliseconds": 15 * 60 * 1000},
}

# Сырые признаки, получаемые из Binance API с указанием типов данных
RAW_FEATURES = {
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
# Признаки, используемые моделью (включают дополнительные признаки) с указанием типов данных
MODEL_FEATURES = {
    **RAW_FEATURES,
    'hour': int,
    'dayofweek': int,
    'sin_hour': float,
    'cos_hour': float,
    'sin_day': float,
    'cos_day': float
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
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

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