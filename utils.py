import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Tuple
from config import *
from sklearn.preprocessing import MinMaxScaler
from rich.table import Table
from rich.console import Console

console = Console()

# Функция для предобработки данных Binance
def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['timestamp', 'close_time']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

# Функция для сортировки DataFrame
def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values('timestamp', ascending=False)

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime(DATETIME_FORMAT)

def readable_time_to_timestamp(readable_time: str) -> int:
    dt = datetime.strptime(readable_time, DATETIME_FORMAT)
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def get_current_time() -> Tuple[int, str]:
    response = requests.get("https://api.binance.com/api/v3/time")
    server_time = response.json()['serverTime']
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time

# Функция для очистки кэша путем удаления всех CSV файлов данных
def clear_cache(self):
    for file in os.listdir(self.PATHS['data_dir']):
        if file.endswith('_data.csv'):
            os.remove(os.path.join(self.PATHS['data_dir'], file))
    self.logger.info("Cache cleared")
