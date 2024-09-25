import os
import pandas as pd
from pandas.api.types import is_string_dtype
import requests
from datetime import datetime, timezone
from typing import Tuple, Dict
from config import *
from sklearn.preprocessing import MinMaxScaler
from rich.table import Table
from rich.console import Console
import logging
import torch
from torch.utils.data import TensorDataset
import numpy as np

console = Console()

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[list(FEATURE_NAMES.keys())]
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)
    for col, dtype in FEATURE_NAMES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    logging.debug(f"Preprocessed DataFrame: {df}")
    return df

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values('timestamp', ascending=False)
    logging.debug(f"Sorted DataFrame: {sorted_df}")
    return sorted_df

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

def clear_cache(paths):
    for file in os.listdir(paths['data_dir']):
        if file.endswith('_data.csv'):
            os.remove(os.path.join(paths['data_dir'], file))
    logging.info("Cache cleared")

def ensure_file_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')
        if 'predictions' in filepath or 'differences' in filepath:
            df = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
            df.to_csv(filepath, index=False)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
