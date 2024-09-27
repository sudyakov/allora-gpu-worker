import os
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Tuple, Optional
import logging
import torch
import numpy as np

from config import (
    RAW_FEATURES,
    SYMBOL_MAPPING,
    INTERVAL_MAPPING,
    PATHS,
    PREDICTION_MINUTES,
    SEQ_LENGTH,
    DATETIME_FORMAT
)

from rich.console import Console

console = Console()

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных Binance."""
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)
    for col, dtype in RAW_FEATURES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    logging.debug(f"Preprocessed DataFrame: {df.head()}")
    return df

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Сортировка DataFrame по временной метке."""
    sorted_df = df.sort_values('timestamp', ascending=False)
    logging.debug(f"Sorted DataFrame: {sorted_df.head()}")
    return sorted_df

def timestamp_to_readable_time(timestamp: int) -> str:
    """Преобразование временной метки в читаемый формат."""
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime(DATETIME_FORMAT)

def get_current_time() -> Tuple[int, str]:
    """Получение текущего времени с сервера Binance."""
    response = requests.get("https://api.binance.com/api/v3/time")
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time

def ensure_file_exists(filepath: str) -> None:
    """Убедиться, что файл существует, иначе создать его."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(RAW_FEATURES.keys()))
        df.to_csv(filepath, index=False)

def clear_cache(paths: dict) -> None:
    """Очистка кэша данных."""
    data_dir = paths.get('data_dir', '')
    for file in os.listdir(data_dir):
        if file.endswith('_data.csv'):
            os.remove(os.path.join(data_dir, file))
    logging.info("Cache cleared")

def get_latest_value(data_file: str, target_symbol: str) -> pd.DataFrame:
    """Получение последнего значения для указанного символа."""
    if target_symbol not in SYMBOL_MAPPING:
        console.print(f"Символ {target_symbol} не найден в SYMBOL_MAPPING", style="red")
        return pd.DataFrame(columns=RAW_FEATURES.keys())
    
    df = pd.read_csv(data_file)
    df = preprocess_binance_data(df)
    df = df[(df['symbol'] == target_symbol) & (df['interval'] == PREDICTION_MINUTES)]
    
    if df.empty:
        console.print(f"No data found for symbol: {target_symbol}", style="blue")
        return pd.DataFrame(columns=RAW_FEATURES.keys())
    
    latest_value_row = df.sort_values('timestamp', ascending=False).iloc[0]
    return latest_value_row.to_frame().T

def get_difference_row(current_time: int, symbol: str) -> pd.Series:
    """Получение строки разницы для указанного времени и символа."""
    if symbol not in SYMBOL_MAPPING:
        console.print(f"Символ {symbol} не найден в SYMBOL_MAPPING", style="red")
        return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())
    
    if not os.path.exists(PATHS.get('differences', '')):
        return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())
    
    differences_data = pd.read_csv(PATHS['differences'])
    differences_data = preprocess_binance_data(differences_data)
    difference_row = differences_data[
        (differences_data['symbol'] == symbol) &
        (differences_data['interval'] == PREDICTION_MINUTES) &
        (differences_data['timestamp'] == current_time)
    ]
    
    if not difference_row.empty:
        return difference_row.iloc[0]
    return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())

def get_latest_timestamp(data_file: str, target_symbol: str, prediction_minutes: int) -> Optional[int]:
    """Получение последней временной метки для указанного символа и интервала."""
    if target_symbol not in SYMBOL_MAPPING:
        console.print(f"Символ {target_symbol} не найден в SYMBOL_MAPPING", style="red")
        return None
    if not os.path.exists(data_file):
        return None
    df = pd.read_csv(data_file)
    df = preprocess_binance_data(df)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == prediction_minutes)]
    if filtered_df.empty:
        return None
    return filtered_df['timestamp'].max()

def get_sequence_for_timestamp(timestamp: int, target_symbol: str, prediction_minutes: int) -> Optional[torch.Tensor]:
    """Получение последовательности данных для указанной временной метки."""
    if target_symbol not in SYMBOL_MAPPING:
        console.print(f"Символ {target_symbol} не найден в SYMBOL_MAPPING", style="red")
        return None
    if not os.path.exists(PATHS.get('combined_dataset', '')):
        return None
    
    df = pd.read_csv(PATHS['combined_dataset'])
    df = preprocess_binance_data(df)
    filtered_df = df[
        (df['symbol'] == target_symbol) &
        (df['interval'] == prediction_minutes) &
        (df['timestamp'] <= timestamp)
    ]
    
    if len(filtered_df) >= SEQ_LENGTH:
        sequence = filtered_df.sort_values('timestamp', ascending=False).head(SEQ_LENGTH)
        sequence = sequence[list(RAW_FEATURES.keys())].values[::-1]
        return torch.tensor(sequence, dtype=torch.float32)
    return None

def get_interval(minutes: int) -> Optional[str]:
    """Получение интервала по количеству минут."""
    return next((k for k, v in INTERVAL_MAPPING.items() if v['minutes'] == minutes), None)

def get_device() -> torch.device:
    """Получение устройства для выполнения вычислений (CPU или GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fill_missing_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Заполнение недостающих признаков модели."""
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
        df['dayofweek'] = pd.to_datetime(df['timestamp'], unit='ms').dt.dayofweek
        df['sin_hour'] = np.sin(2 * np.pi * df['hour']/24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour']/24)
        df['sin_day'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['cos_day'] = np.cos(2 * np.pi * df['dayofweek']/7)
    df = df.ffill().bfill()
    return df
