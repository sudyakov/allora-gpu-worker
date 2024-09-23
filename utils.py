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
import logging

console = Console()

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_NAMES:
        if col in df.columns:
            if col == 'symbol':
                df[col] = df[col].astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(FEATURE_NAMES[col])
    for col in ['timestamp', 'close_time']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.int64)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logging.info(f"Data after preprocessing in preprocess_binance_data: {df.head()}")  # Добавлено для отладки
    logging.info(f"Columns after preprocessing in preprocess_binance_data: {df.columns}")  # Добавлено для отладки
    return df


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

def clear_cache(self):
    for file in os.listdir(self.PATHS['data_dir']):
        if file.endswith('_data.csv'):
            os.remove(os.path.join(self.PATHS['data_dir'], file))
    self.logger.info("Cache cleared")
