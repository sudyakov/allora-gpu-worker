import os
import pandas as pd
import numpy as np
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

console = Console()

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_NAMES:
        if col not in df.columns:
            if col == 'symbol':
                df[col] = ''
            elif col == 'interval':
                df[col] = 0
            else:
                df[col] = np.nan
    
    df = df[list(FEATURE_NAMES.keys())]  # Оставляем только нужные столбцы
    df = df.astype(FEATURE_NAMES)  # Приводим типы к нужным
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
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

def load_and_prepare_data():
    combined_data = pd.read_csv(PATHS['combined_dataset'])
    predictions_data = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
    differences_data = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
    
    if os.path.exists(PATHS['predictions']):
        predictions_data = pd.read_csv(PATHS['predictions'])
    if os.path.exists(PATHS['differences']):
        differences_data = pd.read_csv(PATHS['differences'])
    
    for df in [combined_data, predictions_data, differences_data]:
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = preprocess_binance_data(df)
    
    logging.debug(f"Combined Data: {combined_data}")
    logging.debug(f"Predictions Data: {predictions_data}")
    logging.debug(f"Differences Data: {differences_data}")
    
    return combined_data, predictions_data, differences_data

def prepare_dataset(csv_file, seq_length=SEQ_LENGTH, target_symbol=TARGET_SYMBOL):
    df = pd.read_csv(csv_file)
    df = df[df['symbol'] == target_symbol].sort_values('timestamp')
    df = preprocess_binance_data(df)
    
    df['symbol'] = df['symbol'].astype('category')
    symbol_codes = df['symbol'].cat.codes.values
    
    numeric_columns = [col for col in FEATURE_NAMES if col not in ['symbol']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns, index=df.index)
    
    scaler.scaled_columns = numeric_columns
    
    scaled_data['symbol'] = symbol_codes
    
    sequences, labels = [], []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data.values[i:i+seq_length].astype(np.float64))
        labels.append(scaled_data.values[i+seq_length].astype(np.float64))
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    logging.debug(f"Prepared Dataset: sequences shape {sequences.shape}, labels shape {labels.shape}")
    
    return TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(labels)), scaler, df

def save_predictions_to_csv(predictions, filename, current_time):
    df = pd.DataFrame(predictions, columns=list(FEATURE_NAMES.keys()))
    prediction_milliseconds = INTERVALS_PERIODS[get_interval(PREDICTION_MINUTES)]['milliseconds']
    next_interval = (current_time // prediction_milliseconds + 1) * prediction_milliseconds
    df['timestamp'] = next_interval
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = PREDICTION_MINUTES
    df = preprocess_binance_data(df)
    
    if not pd.io.common.file_exists(filename):
        df.to_csv(filename, index=False, float_format='%.10f')
    else:
        existing_df = pd.read_csv(filename)
        existing_df = preprocess_binance_data(existing_df)
        combined_df = pd.concat([df, existing_df], ignore_index=True).sort_values('timestamp', ascending=False)
        combined_df.to_csv(filename, index=False, float_format='%.10f')
    
    logging.debug(f"Saved Predictions DataFrame: {df}")
    
    return df.iloc[0]

def get_latest_prediction(predictions_file, target_symbol):
    if not os.path.exists(predictions_file):
        return pd.Series(dtype='float64')
    df = pd.read_csv(predictions_file).sort_values('timestamp', ascending=False)
    df = preprocess_binance_data(df)
    
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == PREDICTION_MINUTES)]
    
    if filtered_df.empty:
        return pd.Series(dtype='float64')
    
    logging.debug(f"Latest Prediction: {filtered_df.iloc[0]}")
    
    return filtered_df.iloc[0]

def get_latest_value(data_file, target_symbol):
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    df = preprocess_binance_data(df)
    interval = get_interval(PREDICTION_MINUTES)
    console.print(f"Filtering data for symbol: {target_symbol} and interval: {interval}", style="blue")
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == PREDICTION_MINUTES)]
    
    if filtered_df.empty:
        console.print(f"No data found for symbol: {target_symbol} and interval: {interval}", style="blue")
        return pd.DataFrame(columns=FEATURE_NAMES.keys())
    
    latest_value_row = filtered_df.iloc[0]
    return latest_value_row.to_frame().T

def get_difference_row(current_time: int, symbol: str) -> pd.Series:
    if not os.path.exists(PATHS['differences']):
        return pd.Series([None] * len(FEATURE_NAMES), index=FEATURE_NAMES.keys())
    differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
    differences_data = preprocess_binance_data(differences_data)
    difference_row = differences_data[
        (differences_data['symbol'] == symbol) &
        (differences_data['interval'] == PREDICTION_MINUTES) &
        (differences_data['timestamp'] == current_time)
    ]
    if not difference_row.empty:
        return difference_row.iloc[0]
    else:
        return pd.Series([None] * len(FEATURE_NAMES), index=FEATURE_NAMES.keys())

def save_difference_to_csv(predictions, actuals, filename, current_time):
    difference = actuals - predictions
    df = pd.DataFrame(difference.reshape(1, -1), columns=list(FEATURE_NAMES.keys()))
    df['timestamp'] = current_time
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = PREDICTION_MINUTES
    df = preprocess_binance_data(df)
    
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, float_format='%.10f')
    else:
        existing_df = pd.read_csv(filename)
        existing_df = preprocess_binance_data(existing_df)
        combined_df = pd.concat([df, existing_df], ignore_index=True).sort_values('timestamp', ascending=False)
        combined_df.to_csv(filename, index=False, float_format='%.10f')
    
    logging.debug(f"Saved Differences DataFrame: {df}")
    
    return df.iloc[0]

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

def get_latest_timestamp(data_file, target_symbol, prediction_minutes):
    if not os.path.exists(data_file):
        return None
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    df = preprocess_binance_data(df)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == prediction_minutes)]
    
    if filtered_df.empty:
        return None
    
    return filtered_df['timestamp'].max()

def get_sequence_for_timestamp(timestamp, target_symbol, prediction_minutes):
    df = pd.read_csv(PATHS['combined_dataset'])
    df = preprocess_binance_data(df)
    
    filtered_df = df[(df['symbol'] == target_symbol) &
                    (df['interval'] == PREDICTION_MINUTES) &
                    (df['timestamp'] <= timestamp)]
    
    if len(filtered_df) >= SEQ_LENGTH:
        sequence = filtered_df.sort_values('timestamp', ascending=False).head(SEQ_LENGTH)[list(FEATURE_NAMES.keys())].values
        return sequence[::-1]
    return None

def get_interval(minutes):
    return next(k for k, v in INTERVALS_PERIODS.items() if v['minutes'] == minutes)

def print_combined_row(current_row, difference_row, predicted_next_row):
    table = Table(title="Current vs Predicted")
    table.add_column("Field", style="cyan")
    table.add_column("Current Value", style="magenta")
    table.add_column("Difference", style="yellow")
    table.add_column("Predicted next", style="green")
    
    for col in FEATURE_NAMES:
        table.add_row(
            col,
            str(current_row[col].iloc[0]) if not current_row.empty else "N/A",
            str(difference_row[col]) if difference_row[col] is not None else "N/A",
            str(predicted_next_row[col]) if predicted_next_row[col] is not None else "N/A"
        )
    
    console.print(table)

