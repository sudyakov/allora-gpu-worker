import os
import logging
from typing import Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from config import (
    API_BASE_URL,
    ADD_FEATURES,
    MODEL_FEATURES,
    RAW_FEATURES,
    SCALABLE_FEATURES,
)
from torch.utils.data import DataLoader, TensorDataset


def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = df['timestamp'].astype(int)
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)

    for feature_dict in [RAW_FEATURES, SCALABLE_FEATURES, ADD_FEATURES]:
        for col, dtype in feature_dict.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    logging.debug(f"Preprocessed DataFrame:\n{df.head()}")
    return df


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values('timestamp', ascending=False)
    logging.debug(f"Sorted DataFrame:\n{sorted_df.head()}")
    return sorted_df


def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fill_missing_add_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df = df.ffill().bfill()
    logging.debug(f"Filled DataFrame:\n{df.head()}")
    return df


def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time


def ensure_file_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(MODEL_FEATURES.keys()))
        df.to_csv(filepath, index=False)
