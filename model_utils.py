import logging
import os
from typing import Dict, Tuple, Sequence, Optional, List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.optimizer import Optimizer

from get_binance_data import GetBinanceData

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
    TRAINING_PARAMS,
    get_interval,
)
from data_utils import shared_data_processor, CustomLabelEncoder
from sklearn.preprocessing import MinMaxScaler

def create_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_PARAMS["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAINING_PARAMS["num_workers"],
    )
    return train_loader, val_loader

def fit_transform(real_data_df: pd.DataFrame) -> pd.DataFrame:
    shared_data_processor.is_fitted = True
    # Обучаем и применяем LabelEncoders для категориальных колонок
    for col in shared_data_processor.categorical_columns:
        encoder = shared_data_processor.label_encoders.get(col)
        if encoder is None:
            encoder = CustomLabelEncoder()
            shared_data_processor.label_encoders[col] = encoder
        real_data_df[col] = encoder.fit_transform(real_data_df[col])
    # Обучаем и применяем MinMaxScalers для численных колонок
    for col in shared_data_processor.scalable_columns:
        if col in real_data_df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            real_data_df[col] = scaler.fit_transform(real_data_df[[col]])
            shared_data_processor.scalers[col] = scaler
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    # Приводим остальные числовые колонки к нужному типу
    for col in shared_data_processor.numerical_columns:
        if col in real_data_df.columns:
            dtype = MODEL_FEATURES.get(col, np.float32)
            real_data_df[col] = real_data_df[col].astype(dtype)
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    # Приводим timestamp к типу int64
    if 'timestamp' in real_data_df.columns:
        real_data_df['timestamp'] = real_data_df['timestamp'].astype(np.int64)
    # Оставляем только необходимые колонки
    real_data_df = real_data_df[list(MODEL_FEATURES.keys())]
    return real_data_df

def transform(real_data_df: pd.DataFrame) -> pd.DataFrame:
    # Применяем ранее обученные LabelEncoders для категориальных колонок
    for col in shared_data_processor.categorical_columns:
        encoder = shared_data_processor.label_encoders.get(col)
        if encoder is None:
            raise ValueError(f"LabelEncoder not found for column {col}.")
        real_data_df[col] = encoder.transform(real_data_df[col])
    # Применяем ранее обученные MinMaxScalers для численных колонок
    for col in shared_data_processor.scalable_columns:
        if col in real_data_df.columns:
            scaler = shared_data_processor.scalers.get(col)
            if scaler is None:
                raise ValueError(f"Scaler not found for column {col}.")
            real_data_df[col] = scaler.transform(real_data_df[[col]])
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    # Приводим остальные числовые колонки к нужному типу
    for col in shared_data_processor.numerical_columns:
        if col in real_data_df.columns:
            dtype = MODEL_FEATURES.get(col, np.float32)
            real_data_df[col] = real_data_df[col].astype(dtype)
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    # Приводим timestamp к типу int64
    if 'timestamp' in real_data_df.columns:
        real_data_df['timestamp'] = real_data_df['timestamp'].astype(np.int64)
    # Оставляем только необходимые колонки
    real_data_df = real_data_df[list(MODEL_FEATURES.keys())]
    return real_data_df

def inverse_transform(real_data_df: pd.DataFrame) -> pd.DataFrame:
    df_inv = real_data_df.copy()
    for col in shared_data_processor.scalable_columns:
        if col in df_inv.columns:
            scaler = shared_data_processor.scalers.get(col)
            if scaler is None:
                raise ValueError(f"Scaler not found for column {col}.")
            df_inv[col] = scaler.inverse_transform(df_inv[[col]])
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    return df_inv

def predict_future_price(
    model: nn.Module,
    latest_real_data_df: pd.DataFrame,
    device: torch.device,
    prediction_minutes: int = PREDICTION_MINUTES,
    future_steps: int = 1,
    seq_length: int = SEQ_LENGTH,
    target_symbol: str = TARGET_SYMBOL
) -> pd.DataFrame:
    if latest_real_data_df.empty:
        logging.error("Latest DataFrame is empty.")
        return pd.DataFrame()
    latest_real_data_df = latest_real_data_df.sort_values(by="timestamp").reset_index(drop=True)
    if len(latest_real_data_df) < seq_length:
        logging.info("Insufficient data for prediction.")
        return pd.DataFrame()
    last_binance_timestamp = latest_real_data_df["timestamp"].iloc[-1]
    if pd.isna(last_binance_timestamp):
        logging.error("Invalid last Binance timestamp value.")
        return pd.DataFrame()
    interval = get_interval(prediction_minutes)
    if interval is None:
        logging.error("Invalid prediction interval.")
        return pd.DataFrame()
    model.eval()
    all_predicted_data = []
    with torch.no_grad():
        interval_ms = INTERVAL_MAPPING[interval]["milliseconds"]
        timestamps_to_predict = [
            last_binance_timestamp + interval_ms * i for i in range(1, future_steps + 1)
        ]
        for next_timestamp in timestamps_to_predict:
            current_df = latest_real_data_df.tail(seq_length).copy()
            if len(current_df) < seq_length:
                logging.info(f"Insufficient data to predict for timestamp {next_timestamp}.")
                continue
            try:
                inputs = torch.tensor(
                    current_df.values, dtype=torch.float32
                ).unsqueeze(0).to(device)
                logging.info(f"Inputs shape: {inputs.shape}")
                logging.info(f"Inputs data: {inputs}")
                predictions = model(inputs).cpu().detach().numpy()
            except Exception as e:
                logging.error(f"Error during prediction for timestamp {next_timestamp}: {e}")
                continue
    # Остальной код
            predicted_data_df = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))
            predicted_data_df_denormalized = inverse_transform(predicted_data_df)
            predicted_data_df_denormalized["symbol"] = target_symbol
            predicted_data_df_denormalized["interval"] = prediction_minutes
            predicted_data_df_denormalized["timestamp"] = int(next_timestamp)
            # Вычисляем временные признаки
            predicted_data_df_denormalized['hour'] = pd.to_datetime(
                predicted_data_df_denormalized['timestamp'], unit='ms').dt.hour
            predicted_data_df_denormalized['dayofweek'] = pd.to_datetime(
                predicted_data_df_denormalized['timestamp'], unit='ms').dt.dayofweek
            predicted_data_df_denormalized['sin_hour'] = np.sin(
                2 * np.pi * predicted_data_df_denormalized['hour'] / 24)
            predicted_data_df_denormalized['cos_hour'] = np.cos(
                2 * np.pi * predicted_data_df_denormalized['hour'] / 24)
            predicted_data_df_denormalized['sin_day'] = np.sin(
                2 * np.pi * predicted_data_df_denormalized['dayofweek'] / 7)
            predicted_data_df_denormalized['cos_day'] = np.cos(
                2 * np.pi * predicted_data_df_denormalized['dayofweek'] / 7)
            final_columns = list(MODEL_FEATURES.keys())
            predicted_data_df_denormalized = predicted_data_df_denormalized[final_columns]
            all_predicted_data.append(predicted_data_df_denormalized)
        if all_predicted_data:
            all_predictions = pd.concat(all_predicted_data, ignore_index=True)
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
    lock_path = f"{combined_dataset_path}.lock"
    with FileLock(lock_path):
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            real_combined_data_df = pd.read_csv(combined_dataset_path)
        else:
            logging.error("Combined dataset not found.")
            return
    if os.path.exists(differences_path) and os.path.getsize(differences_path) > 0:
        existing_differences_df = pd.read_csv(differences_path)
    else:
        existing_differences_df = pd.DataFrame(columns=predictions_df.columns)
    required_columns = predictions_df.columns.tolist()
    missing_columns_pred = set(required_columns) - set(predictions_df.columns)
    missing_columns_actual = set(required_columns) - set(real_combined_data_df.columns)
    if missing_columns_pred:
        logging.error(f"Missing columns in predictions DataFrame: {missing_columns_pred}")
        return
    if missing_columns_actual:
        logging.error(f"Missing columns in actual DataFrame: {missing_columns_actual}")
        return
    actual_data_df = real_combined_data_df[
        (real_combined_data_df['symbol'].isin(predictions_df['symbol'].unique())) &
        (real_combined_data_df['interval'].isin(predictions_df['interval'].unique())) &
        (real_combined_data_df['hour'].isin(predictions_df['hour'].unique())) &
        (real_combined_data_df['dayofweek'].isin(predictions_df['dayofweek'].unique())) &
        (real_combined_data_df['timestamp'].isin(predictions_df['timestamp'].unique()))
    ]
    if actual_data_df.empty:
        logging.info("No matching actual data found for predictions.")
        return
    merged_predictions_actual_df = pd.merge(
        predictions_df,
        actual_data_df,
        on=['symbol', 'interval', 'hour', 'dayofweek', 'timestamp'],
        suffixes=('_pred', '_actual')
    )
    if merged_predictions_actual_df.empty:
        logging.info("No matching timestamps between predictions and actual data.")
        return
    if not existing_differences_df.empty:
        merged_predictions_actual_df = pd.merge(
            merged_predictions_actual_df,
            existing_differences_df[['symbol', 'interval', 'hour', 'dayofweek', 'timestamp']],
            on=['symbol', 'interval', 'hour', 'dayofweek', 'timestamp'],
            how='left',
            indicator=True
        )
        merged_predictions_actual_df = merged_predictions_actual_df[merged_predictions_actual_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        if merged_predictions_actual_df.empty:
            logging.info("All differences have already been processed.")
            return
    key_columns = ['symbol', 'interval', 'hour', 'dayofweek', 'timestamp']
    pred_columns = [col for col in merged_predictions_actual_df.columns if col.endswith('_pred')]
    differences_data_df = merged_predictions_actual_df[key_columns + pred_columns].copy()
    differences_data_df.rename(columns=lambda x: x.replace('_pred', '') if x.endswith('_pred') else x, inplace=True)
    feature_cols = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())
    for feature in feature_cols:
        pred_col = f"{feature}_pred"
        actual_col = f"{feature}_actual"
        if pred_col in merged_predictions_actual_df.columns and actual_col in merged_predictions_actual_df.columns:
            differences_data_df[feature] = merged_predictions_actual_df[actual_col] - merged_predictions_actual_df[pred_col]
        else:
            logging.warning(f"Columns {pred_col} or {actual_col} not found in merged DataFrame.")
    for col in differences_data_df.columns:
        if col in predictions_df.columns:
            differences_data_df[col] = differences_data_df[col].astype(predictions_df[col].dtype)
    combined_differences_df = pd.concat([existing_differences_df, differences_data_df], ignore_index=True)
    combined_differences_df = combined_differences_df[predictions_df.columns]
    combined_differences_df.sort_values(by='timestamp', ascending=True, inplace=True)
    combined_differences_df.to_csv(differences_path, index=False)
    logging.info(f"Differences updated and saved to {differences_path}")

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

def save_model(model: nn.Module, optimizer: Optimizer, filepath: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(model: nn.Module, optimizer: Optimizer, filepath: str, device: torch.device):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Model loaded from {filepath}")
    else:
        logging.info(f"No saved model found at {filepath}. Starting from scratch.")

def load_and_prepare_data(
    data_fetcher: GetBinanceData,
    is_training: bool = False,
    latest_timestamp: Optional[int] = None,
    count: int = SEQ_LENGTH
) -> pd.DataFrame:
    if is_training:
        real_data = data_fetcher.fetch_combined_data()
    else:
        real_data = shared_data_processor.get_latest_dataset_prices(
            symbol=TARGET_SYMBOL,
            interval=PREDICTION_MINUTES,
            count=count,
            latest_timestamp=latest_timestamp
        )
    if real_data.empty:
        logging.error("Data is empty.")
        return pd.DataFrame()
    real_data = shared_data_processor.preprocess_binance_data(real_data)
    real_data = shared_data_processor.fill_missing_add_features(real_data)
    real_data = real_data.sort_values(by="timestamp").reset_index(drop=True)
    if is_training and not shared_data_processor.is_fitted:
        real_data = fit_transform(real_data)
        shared_data_processor.save(DATA_PROCESSOR_FILENAME)
    else:
        real_data = transform(real_data)
    return real_data
