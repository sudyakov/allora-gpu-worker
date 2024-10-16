import logging
import os
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split

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

def fit_transform(df: pd.DataFrame) -> pd.DataFrame:
    shared_data_processor.is_fitted = True
    for col in shared_data_processor.categorical_columns:
        encoder = shared_data_processor.label_encoders.get(col)
        if encoder is None:
            encoder = CustomLabelEncoder()
            shared_data_processor.label_encoders[col] = encoder
        df[col] = encoder.fit_transform(df[col])
    for col in shared_data_processor.scalable_columns:
        if col in df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[col] = scaler.fit_transform(df[[col]])
            shared_data_processor.scalers[col] = scaler
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    for col in shared_data_processor.numerical_columns:
        if col in df.columns:
            dtype = MODEL_FEATURES.get(col, np.float32)
            df[col] = df[col].astype(dtype)
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(np.int64)
    df = df[list(MODEL_FEATURES.keys())]
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    for col in shared_data_processor.categorical_columns:
        encoder = shared_data_processor.label_encoders.get(col)
        if encoder is None:
            raise ValueError(f"LabelEncoder not found for column {col}.")
        df[col] = encoder.transform(df[col])
    for col in shared_data_processor.scalable_columns:
        if col in df.columns:
            scaler = shared_data_processor.scalers.get(col)
            if scaler is None:
                raise ValueError(f"Scaler not found for column {col}.")
            df[col] = scaler.transform(df[[col]])
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    for col in shared_data_processor.numerical_columns:
        if col in df.columns:
            dtype = MODEL_FEATURES.get(col, np.float32)
            df[col] = df[col].astype(dtype)
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(np.int64)
    df = df[list(MODEL_FEATURES.keys())]
    return df

def inverse_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_inv = df.copy()
    for col in shared_data_processor.scalable_columns:
        if col in df_inv.columns:
            scaler = shared_data_processor.scalers.get(col)
            if scaler is None:
                raise ValueError(f"Scaler not found for column {col}.")
            df_inv[col] = scaler.inverse_transform(df_inv[[col]])
        else:
            raise KeyError(f"Column {col} is missing in DataFrame.")
    return df_inv

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

def predict_future_price(
    model: nn.Module,
    latest_df: pd.DataFrame,
    device: torch.device,
    prediction_minutes: int = PREDICTION_MINUTES,
    future_steps: int = 1,
    seq_length: int = SEQ_LENGTH,
    target_symbol: str = TARGET_SYMBOL
) -> pd.DataFrame:
    if latest_df.empty:
        logging.error("Latest DataFrame is empty.")
        return pd.DataFrame()
    latest_df = latest_df.sort_values(by="timestamp").reset_index(drop=True)
    if len(latest_df) < seq_length:
        logging.info("Insufficient data for prediction.")
        return pd.DataFrame()
    last_binance_timestamp = latest_df["timestamp"].iloc[-1]
    if pd.isna(last_binance_timestamp):
        logging.error("Invalid last Binance timestamp value.")
        return pd.DataFrame()
    interval = get_interval(prediction_minutes)
    if interval is None:
        logging.error("Invalid prediction interval.")
        return pd.DataFrame()
    model.eval()
    predictions_list = []
    with torch.no_grad():
        interval_ms = INTERVAL_MAPPING[interval]["milliseconds"]
        timestamps_to_predict = [
            last_binance_timestamp + interval_ms * i for i in range(1, future_steps + 1)
        ]
        for next_timestamp in timestamps_to_predict:
            current_df = latest_df.tail(seq_length).copy()
            if len(current_df) < seq_length:
                logging.info(f"Insufficient data to predict for timestamp {next_timestamp}.")
                continue
            try:
                current_df_transformed = transform(current_df)
                inputs = torch.tensor(
                    current_df_transformed.values, dtype=torch.float32
                ).unsqueeze(0).to(device)
                predictions = model(inputs).cpu().detach().numpy()  # Добавлено .detach()
            except Exception as e:
                logging.error(f"Error during prediction for timestamp {next_timestamp}: {e}")
                continue
            predictions_df = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))

        predictions_df_denormalized = inverse_transform(predictions_df)
        predictions_df_denormalized["symbol"] = target_symbol
        predictions_df_denormalized["interval"] = prediction_minutes
        predictions_df_denormalized["timestamp"] = int(next_timestamp)
        predictions_df_denormalized = shared_data_processor.fill_missing_add_features(predictions_df_denormalized)
        final_columns = list(MODEL_FEATURES.keys())
        predictions_df_denormalized = predictions_df_denormalized[final_columns]
        predictions_list.append(predictions_df_denormalized)
        latest_df = pd.concat([latest_df, predictions_df_denormalized], ignore_index=True)

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
    combined_differences.sort_values(by='timestamp', ascending=True, inplace=True)
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
