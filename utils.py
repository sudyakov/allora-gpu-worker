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
    
    # Убедимся, что столбец 'interval' имеет тип int
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)
    
    # Приведем типы данных всех столбцов в соответствие с FEATURE_NAMES
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

def prepare_dataset(combined_dataset_path, seq_length=SEQ_LENGTH, target_symbol=TARGET_SYMBOL):
    df = pd.read_csv(combined_dataset_path)
    df = df[df['symbol'] == target_symbol].sort_values('timestamp')
    df = preprocess_binance_data(df)
    df['symbol'] = df['symbol'].astype('category')
    symbol_codes = df['symbol'].cat.codes
    df['interval'] = df['interval'].astype('category')
    numeric_columns = [col for col, dtype in FEATURE_NAMES.items() if dtype in [float, int]]
    categorical_columns = [col for col, dtype in FEATURE_NAMES.items() if dtype == str]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_numeric_data = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns, index=df.index)

    scaled_data = pd.concat([scaled_numeric_data, df[categorical_columns]], axis=1)

    scaler.scaled_columns = numeric_columns

    scaled_data['symbol'] = symbol_codes.values
    sequences = []
    labels = []
    for i in range(len(scaled_data) - seq_length):
        seq = scaled_data.iloc[i:i+seq_length].to_numpy().astype(float)
        label = scaled_data.iloc[i+seq_length].to_numpy().astype(float)
        sequences.append(seq)
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    logging.debug(f"Prepared Dataset: sequences shape {sequences_tensor.shape}, labels shape {labels_tensor.shape}")

    return TensorDataset(sequences_tensor, labels_tensor), scaler, df

def save_predictions_to_csv(predictions, filename):
    df = pd.DataFrame(predictions, columns=list(FEATURE_NAMES.keys()))
    prediction_milliseconds = INTERVALS_PERIODS[get_interval(PREDICTION_MINUTES)]['milliseconds']
    next_interval = (current_time // prediction_milliseconds + 1) * prediction_milliseconds
    df['timestamp'] = next_interval
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = PREDICTION_MINUTES

    for col, dtype in FEATURE_NAMES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
            logging.debug(f"Column {col} value: {df[col].iloc[0]}, type: {type(df[col].iloc[0])}")
            if dtype == str:
                assert is_string_dtype(df[col]), f"Column {col} has incorrect type {df[col].dtype}, expected {dtype}"
            else:
                assert df[col].dtype.type == np.dtype(dtype).type, f"Column {col} has incorrect type {df[col].dtype}, expected {dtype}"

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(filename, index=False)
    return df

def get_latest_prediction(predictions_file, target_symbol):
    if not os.path.exists(predictions_file):
        return pd.Series(dtype='float64')
    df = pd.read_csv(predictions_file).sort_values('timestamp', ascending=False)
    df = preprocess_binance_data(df)

    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == PREDICTION_MINUTES)]

    if filtered_df.empty:
        return pd.Series(dtype='float64')

    latest_prediction = filtered_df.iloc[0]

    for col in FEATURE_NAMES.keys():
        assert col in latest_prediction.index, f"Column {col} is missing in the latest prediction"

    for col, dtype in FEATURE_NAMES.items():
        if col in latest_prediction.index:
            logging.debug(f"Column {col} value: {latest_prediction[col]}, type: {type(latest_prediction[col])}")
            if dtype == str:
                assert isinstance(latest_prediction[col], str), f"Column {col} has incorrect type {type(latest_prediction[col])}, expected {dtype}"
            elif dtype == int:
                assert isinstance(latest_prediction[col], (int, np.integer)), f"Column {col} has incorrect type {type(latest_prediction[col])}, expected {dtype}"
            elif dtype == float:
                assert isinstance(latest_prediction[col], (float, np.floating)), f"Column {col} has incorrect type {type(latest_prediction[col])}, expected {dtype}"
            else:
                assert isinstance(latest_prediction[col], dtype), f"Column {col} has incorrect type {type(latest_prediction[col])}, expected {dtype}"

    logging.debug(f"Latest Prediction: {latest_prediction}")

    return latest_prediction

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

def save_difference_to_csv(predictions_file, actuals_file, differences_file):
    # Проверяем наличие файлов
    if not os.path.exists(predictions_file):
        logging.warning(f"Файл с предсказаниями {predictions_file} не найден.")
        return
    if not os.path.exists(actuals_file):
        logging.warning(f"Файл с архивными данными {actuals_file} не найден.")
        return

    # Читаем предсказания и архивные данные
    predictions = pd.read_csv(predictions_file)
    actuals = pd.read_csv(actuals_file)

    # Предобрабатываем данные
    predictions = preprocess_binance_data(predictions)
    actuals = preprocess_binance_data(actuals)

    # Проверяем, есть ли новые предсказания
    if predictions.empty:
        logging.info("Нет новых предсказаний для сравнения.")
        return

    # Объединяем предсказания и фактические данные по timestamp
    merged_df = pd.merge(predictions, actuals, on='timestamp', suffixes=('_pred', '_act'))

    if merged_df.empty:
        logging.info("Нет совпадений между предсказаниями и архивными данными по timestamp.")
        return

    # Вычисляем разницу по числовым колонкам, исключая 'symbol', 'interval' и 'timestamp'
    exclude_columns = ['symbol', 'interval', 'timestamp']
    numeric_columns = [col for col, dtype in FEATURE_NAMES.items() if dtype in [int, float] and col not in exclude_columns]

    # Создаем DataFrame для хранения разниц
    differences = pd.DataFrame()
    differences['timestamp'] = merged_df['timestamp']
    differences['symbol'] = merged_df['symbol_pred']
    differences['interval'] = merged_df['interval_pred']

    # Вычисляем разницу по каждому числовому столбцу
    for col in numeric_columns:
        differences[col] = merged_df[f"{col}_act"] - merged_df[f"{col}_pred"]

    # Добавляем нечисловые колонки, если требуется
    for col in FEATURE_NAMES.keys():
        if FEATURE_NAMES[col] == str and col not in exclude_columns:
            differences[col] = merged_df[f"{col}_act"]

    # Сохраняем результаты в файл
    if not os.path.exists(differences_file):
        differences.to_csv(differences_file, index=False, float_format='%.10f')
    else:
        existing_differences = pd.read_csv(differences_file)
        combined_differences = pd.concat([existing_differences, differences], ignore_index=True)
        combined_differences.to_csv(differences_file, index=False, float_format='%.10f')

    logging.info(f"Сохранены разницы в файл: {differences_file}")

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
        return sequence[::-1].copy()
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
        current_value = str(current_row[col].iloc[0]) if not current_row.empty else "N/A"
        difference_value = str(difference_row[col]) if difference_row[col] is not None else "N/A"
        predicted_value = str(predicted_next_row[col]) if predicted_next_row[col] is not None else "N/A"
        table.add_row(col, current_value, difference_value, predicted_value)

    console.print(table)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
