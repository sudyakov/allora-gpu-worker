import os
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Tuple
import logging
import torch

from config import *
from rich.console import Console
from rich.table import Table

console = Console()

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    # Убедитесь, что 'timestamp' остается целым числом
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)
    for col, dtype in RAW_FEATURES.items():
        if col in df.columns:
            if dtype == str:
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(dtype)
    logging.debug(f"Preprocessed DataFrame: {df.head()}")
    return df

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values('timestamp', ascending=False)
    logging.debug(f"Sorted DataFrame: {sorted_df.head()}")
    return sorted_df

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime(DATETIME_FORMAT)

def get_current_time() -> Tuple[int, str]:
    response = requests.get("https://api.binance.com/api/v3/time")
    server_time = response.json()['serverTime']
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time

def ensure_file_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(RAW_FEATURES.keys()))
        df.to_csv(filepath, index=False)

def clear_cache(paths):
    for file in os.listdir(paths['data_dir']):
        if file.endswith('_data.csv'):
            os.remove(os.path.join(paths['data_dir'], file))
    logging.info("Cache cleared")

def get_latest_value(data_file, target_symbol):
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
    if symbol not in SYMBOL_MAPPING:
        console.print(f"Символ {symbol} не найден в SYMBOL_MAPPING", style="red")
        return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())
    
    if not os.path.exists(PATHS['differences']):
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
    else:
        return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())

def get_latest_timestamp(data_file, target_symbol, prediction_minutes):
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

def get_sequence_for_timestamp(timestamp, target_symbol, prediction_minutes):
    if target_symbol not in SYMBOL_MAPPING:
        console.print(f"Символ {target_symbol} не найден в SYMBOL_MAPPING", style="red")
        return None
    df = pd.read_csv(PATHS['combined_dataset'])
    df = preprocess_binance_data(df)
    filtered_df = df[
        (df['symbol'] == target_symbol) &
        (df['interval'] == PREDICTION_MINUTES) &
        (df['timestamp'] <= timestamp)
    ]
    if len(filtered_df) >= SEQ_LENGTH:
        sequence = filtered_df.sort_values('timestamp', ascending=False).head(SEQ_LENGTH)
        sequence = sequence[list(RAW_FEATURES.keys())].values[::-1]
        return sequence
    return None

def get_interval(minutes):
    return next((k for k, v in INTERVAL_MAPPING.items() if v['minutes'] == minutes), None)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
