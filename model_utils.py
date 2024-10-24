import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from config import (
    ADD_FEATURES,
    DATA_PROCESSOR_FILENAME,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    MODEL_PARAMS,
    PATHS,
    PREDICTION_MINUTES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    TRAINING_PARAMS,
    get_interval,
)
from data_utils import shared_data_processor
from get_binance_data import GetBinanceData


def create_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Используем параметр persistent_workers для ускорения загрузки данных
    num_workers = TRAINING_PARAMS.get("num_workers", 0)
    persistent_workers = True if num_workers > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


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

    logging.info("Dataset after transformation:\n%s", latest_real_data_df.to_string())

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
        next_timestamp = last_binance_timestamp + interval_ms
        current_df = latest_real_data_df.tail(seq_length).copy()

        if len(current_df) < seq_length:
            logging.info(f"Insufficient data to predict for timestamp {next_timestamp}.")
            return pd.DataFrame()

        try:
            inputs = torch.tensor(current_df.values, dtype=torch.float32).unsqueeze(0).to(device)
            predictions = model(inputs).cpu().numpy()
            predicted_data_df = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))

            # Используем метод inverse_transform из DataProcessor
            predicted_data_df_denormalized = shared_data_processor.inverse_transform(predicted_data_df)

            predicted_data_df_denormalized["symbol"] = target_symbol
            predicted_data_df_denormalized["interval"] = prediction_minutes
            predicted_data_df_denormalized["timestamp"] = int(next_timestamp)
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
        except Exception as e:
            logging.error(f"Error during prediction for timestamp {next_timestamp}: {e}")

    if all_predicted_data:
        all_predictions = pd.concat(all_predicted_data, ignore_index=True)
        return all_predictions
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

    interval = get_interval(PREDICTION_MINUTES)
    if interval is None:
        logging.error("Invalid prediction interval.")
        return
    interval_ms = INTERVAL_MAPPING[interval]['milliseconds']

    # Сдвигаем метки времени предсказаний на один интервал назад
    predictions_df_adjusted = predictions_df.copy()
    predictions_df_adjusted['timestamp'] -= interval_ms

    actual_data_df = real_combined_data_df[
        (real_combined_data_df['symbol'].isin(predictions_df['symbol'].unique())) &
        (real_combined_data_df['interval'].isin(predictions_df['interval'].unique())) &
        (real_combined_data_df['timestamp'].isin(predictions_df['timestamp'].unique()))
    ]

    if actual_data_df.empty:
        logging.info("No matching actual data found for predictions to make differences.")
        return

    merged_predictions_actual_df = pd.merge(
        predictions_df_adjusted,
        actual_data_df,
        on=['symbol', 'interval', 'timestamp'],
        suffixes=('_pred', '_actual')
    )

    if merged_predictions_actual_df.empty:
        logging.info("No matching timestamps between predictions and actual data.")
        return

    if not existing_differences_df.empty:
        merged_predictions_actual_df = pd.merge(
            merged_predictions_actual_df,
            existing_differences_df[['symbol', 'interval', 'timestamp']],
            on=['symbol', 'interval', 'timestamp'],
            how='left',
            indicator=True
        )
        merged_predictions_actual_df = merged_predictions_actual_df[
            merged_predictions_actual_df['_merge'] == 'left_only'
        ].drop(columns=['_merge'])

        if merged_predictions_actual_df.empty:
            logging.info("All differences have already been processed.")
            return

    key_columns = ['symbol', 'interval', 'timestamp']
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

    dataframes_to_concat = [df for df in [existing_differences_df, differences_data_df] if not df.empty]

    if dataframes_to_concat:
        combined_differences_df = pd.concat(dataframes_to_concat, ignore_index=True)
    else:
        combined_differences_df = pd.DataFrame(columns=predictions_df.columns)
    combined_differences_df = combined_differences_df[predictions_df.columns]
    combined_differences_df.sort_values(by='timestamp', ascending=True, inplace=True)
    combined_differences_df.to_csv(differences_path, index=False)
    logging.info(f"Differences updated and saved to {differences_path}")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def save_model(model: nn.Module, optimizer: Optimizer, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(model: nn.Module, optimizer: Optimizer, filepath: str, device: torch.device):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Model loaded from {filepath}")
    else:
        logging.info(f"No saved model found at {filepath}. Starting from scratch.")


def load_and_prepare_data(
    data_fetcher: GetBinanceData,
    is_training: bool = False,
    latest_timestamp: Optional[int] = None,
    count: int = SEQ_LENGTH,
    external_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if external_data is not None:
        real_data = external_data
    elif is_training:
        real_data = data_fetcher.fetch_combined_data()
    else:
        real_data = shared_data_processor.get_latest_dataset_prices(
            symbol=None,
            interval=None,
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
        real_data = shared_data_processor.fit_transform(real_data)
        shared_data_processor.save(DATA_PROCESSOR_FILENAME)
    else:
        real_data = shared_data_processor.transform(real_data)

    return real_data
