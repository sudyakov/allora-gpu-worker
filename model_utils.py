import logging
import os
from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock  # Добавляем импорт библиотеки для блокировки файлов

from config import (
    ADD_FEATURES,
    DATA_PROCESSOR_FILENAME,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    PATHS,
    PREDICTION_MINUTES,
    get_interval,
)
from data_utils import shared_data_processor

def predict_future_price(
    model: nn.Module,
    latest_df: pd.DataFrame,
    device: torch.device,
    prediction_minutes: int = PREDICTION_MINUTES,
    last_prediction_timestamp: int | None = None
) -> pd.DataFrame:
    model.eval()
    predictions_list = []
    with torch.no_grad():
        if len(latest_df) < SEQ_LENGTH:
            logging.info("Insufficient data for prediction.")
            return pd.DataFrame()
        # Получаем последнюю метку времени из данных Binance
        last_binance_timestamp = latest_df["timestamp"].iloc[-1]
        if pd.isna(last_binance_timestamp):
            logging.error("Invalid last Binance timestamp value.")
            return pd.DataFrame()
        # Если нет предыдущих предсказаний, начинаем с последней метки времени модели
        interval = get_interval(prediction_minutes)
        if interval is None:
            logging.error("Invalid prediction interval.")
            return pd.DataFrame()
        interval_ms = INTERVAL_MAPPING[interval]["milliseconds"]
        if last_prediction_timestamp is None:
            last_prediction_timestamp = last_binance_timestamp - interval_ms
            if last_prediction_timestamp is None:
                logging.error("last_prediction_timestamp is None after computation.")
                return pd.DataFrame()

        # Дополнительная проверка перед использованием int()
        if last_prediction_timestamp is None or last_binance_timestamp is None:
            logging.error("Timestamp values cannot be None.")
            return pd.DataFrame()

        timestamps_to_predict = np.arange(
            int(last_prediction_timestamp) + interval_ms,
            int(last_binance_timestamp) + interval_ms,
            interval_ms
        )
        for next_timestamp in timestamps_to_predict:
            # Подготовка данных для текущей временной метки
            # Фильтруем данные для текущей последовательности
            current_df = latest_df[latest_df["timestamp"] <= next_timestamp].tail(SEQ_LENGTH)
            if len(current_df) < SEQ_LENGTH:
                logging.info(f"Insufficient data for timestamp {next_timestamp}.")
                continue
            current_df_transformed = shared_data_processor.transform(current_df)
            inputs = torch.tensor(current_df_transformed.values, dtype=torch.float32).unsqueeze(0).to(device)
            predictions = model(inputs).cpu().numpy()
            predictions_df = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))
            predictions_df_denormalized = shared_data_processor.inverse_transform(predictions_df)
            predictions_df_denormalized["symbol"] = TARGET_SYMBOL
            predictions_df_denormalized["interval"] = prediction_minutes
            predictions_df_denormalized["timestamp"] = int(next_timestamp)
            predictions_df_denormalized = shared_data_processor.fill_missing_add_features(predictions_df_denormalized)
            final_columns = list(MODEL_FEATURES.keys())
            predictions_df_denormalized = predictions_df_denormalized[final_columns]
            predictions_list.append(predictions_df_denormalized)
    if predictions_list:
        all_predictions = pd.concat(predictions_list, ignore_index=True)
        return all_predictions
    else:
        return pd.DataFrame()


def update_differences(
    differences_path: str,
    predictions_path: str,
    combined_dataset_path: str
) -> None:
    if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
        predictions_df = pd.read_csv(predictions_path)
    else:
        logging.info("No predictions available to process.")
        return
    # Используем блокировку при доступе к combined_dataset.csv
    lock_path = f"{combined_dataset_path}.lock"
    with FileLock(lock_path):
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            combined_df = pd.read_csv(combined_dataset_path)
        else:
            logging.error("Combined dataset not found.")
            return
    if os.path.exists(differences_path) and os.path.getsize(differences_path) > 0:
        existing_differences = pd.read_csv(differences_path)
    else:
        existing_differences = pd.DataFrame(columns=predictions_df.columns)
    required_columns = predictions_df.columns.tolist()
    missing_columns_pred = set(required_columns) - set(predictions_df.columns)
    missing_columns_actual = set(required_columns) - set(combined_df.columns)
    if missing_columns_pred:
        logging.error(f"Missing columns in predictions DataFrame: {missing_columns_pred}")
        return
    if missing_columns_actual:
        logging.error(f"Missing columns in actual DataFrame: {missing_columns_actual}")
        return
    actual_df = combined_df[
        (combined_df['symbol'].isin(predictions_df['symbol'].unique())) &
        (combined_df['interval'].isin(predictions_df['interval'].unique())) &
        (combined_df['hour'].isin(predictions_df['hour'].unique())) &
        (combined_df['dayofweek'].isin(predictions_df['dayofweek'].unique())) &
        (combined_df['timestamp'].isin(predictions_df['timestamp'].unique()))
    ]
    if actual_df.empty:
        logging.info("No matching actual data found for predictions.")
        return
    merged_df = pd.merge(
        predictions_df,
        actual_df,
        on=['symbol', 'interval', 'hour', 'dayofweek', 'timestamp'],
        suffixes=('_pred', '_actual')
    )
    if merged_df.empty:
        logging.info("No matching timestamps between predictions and actual data.")
        return
    if not existing_differences.empty:
        merged_df = pd.merge(
            merged_df,
            existing_differences[['symbol', 'interval', 'hour', 'dayofweek', 'timestamp']],
            on=['symbol', 'interval', 'hour', 'dayofweek', 'timestamp'],
            how='left',
            indicator=True
        )
        merged_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        if merged_df.empty:
            logging.info("All differences have already been processed.")
            return
    key_columns = ['symbol', 'interval', 'hour', 'dayofweek', 'timestamp']
    pred_columns = [col for col in merged_df.columns if col.endswith('_pred')]
    differences_df = merged_df[key_columns + pred_columns].copy()
    differences_df.rename(columns=lambda x: x.replace('_pred', '') if x.endswith('_pred') else x, inplace=True)
    feature_cols = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())
    for feature in feature_cols:
        pred_col = f"{feature}_pred"
        actual_col = f"{feature}_actual"
        if pred_col in merged_df.columns and actual_col in merged_df.columns:
            differences_df[feature] = merged_df[actual_col] - merged_df[pred_col]
        else:
            logging.warning(f"Columns {pred_col} or {actual_col} not found in merged_df.")
    for col in differences_df.columns:
        if col in predictions_df.columns:
            differences_df[col] = differences_df[col].astype(predictions_df[col].dtype)
    combined_differences = pd.concat([existing_differences, differences_df], ignore_index=True)
    combined_differences = combined_differences[predictions_df.columns]
    combined_differences.sort_values(by='timestamp', ascending=False, inplace=True)
    combined_differences.to_csv(differences_path, index=False)
    logging.info(f"Differences updated and saved to {differences_path}")

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

def save_model(model, optimizer, filename: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename, _use_new_zipfile_serialization=True)
    logging.info(f"Model saved to {filename}")

def load_model(model, optimizer, filename: str, device: torch.device) -> None:
    if os.path.exists(filename):
        logging.info(f"Loading model from {filename}")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Model and optimizer state loaded.")
    else:
        logging.info(f"No model file found at {filename}. Starting from scratch.")