import logging
import os
from typing import Dict, Optional, Tuple, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from get_binance_data import main as get_binance_data_main
from config import (
    ADD_FEATURES,
    DATA_PROCESSOR_FILENAME,
    INTERVAL_MAPPING,
    MODEL_FILENAME,
    MODEL_FEATURES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    PATHS,
    PREDICTION_MINUTES,
    IntervalKey,
    get_interval,
    TRAINING_PARAMS,
    MODEL_PARAMS
)
from data_utils import shared_data_processor
from get_binance_data import GetBinanceData

def predict_future_price(
    model: nn.Module,
    latest_df: pd.DataFrame,
    device: torch.device,
    prediction_minutes: int = PREDICTION_MINUTES,
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        if len(latest_df) < SEQ_LENGTH:
            logging.info("Insufficient data for prediction.")
            return pd.DataFrame()

        latest_df_transformed = shared_data_processor.transform(latest_df)

        inputs = torch.tensor(latest_df_transformed.values, dtype=torch.float32).unsqueeze(0).to(device)

        predictions = model(inputs).cpu().numpy()
        predictions_df = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))

        predictions_df_denormalized = shared_data_processor.inverse_transform(predictions_df)

        last_timestamp = latest_df["timestamp"].iloc[-1]
        if pd.isna(last_timestamp):
            logging.error("Invalid last timestamp value.")
            return pd.DataFrame()

        next_timestamp = np.int64(last_timestamp) + INTERVAL_MAPPING[get_interval(prediction_minutes)]["milliseconds"]

        predictions_df_denormalized["symbol"] = TARGET_SYMBOL
        predictions_df_denormalized["interval"] = prediction_minutes
        predictions_df_denormalized["timestamp"] = next_timestamp

        predictions_df_denormalized = shared_data_processor.fill_missing_add_features(predictions_df_denormalized)

        final_columns = list(MODEL_FEATURES.keys())
        predictions_df_denormalized = predictions_df_denormalized[final_columns]

    return predictions_df_denormalized

def compare_predictions_with_actual(
    predictions_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    differences_path: str
) -> None:
    # Убедиться, что столбцы имеют правильные типы данных
    for col in ['timestamp', 'symbol', 'interval']:
        if col in actual_df.columns and col in predictions_df.columns:
            predictions_df[col] = predictions_df[col].astype(np.int64)
            actual_df[col] = actual_df[col].astype(np.int64)
        else:
            logging.error(f"Missing column '{col}' in dataframes.")
            return

    # Применение inverse_transform только к actual_df
    actual_df = shared_data_processor.inverse_transform(actual_df)

    # Проверка наличия необходимых столбцов
    required_columns = ['timestamp', 'symbol', 'interval'] + list(SCALABLE_FEATURES.keys())
    missing_columns_pred = set(required_columns) - set(predictions_df.columns)
    missing_columns_actual = set(required_columns) - set(actual_df.columns)

    if missing_columns_pred:
        logging.error(f"Missing columns in predictions DataFrame: {missing_columns_pred}")
        return
    if missing_columns_actual:
        logging.error(f"Missing columns in actual DataFrame: {missing_columns_actual}")
        return

    # Merge predictions and actuals on timestamp, symbol, and interval
    merged_df = pd.merge(
        predictions_df,
        actual_df,
        on=['timestamp', 'symbol', 'interval'],
        suffixes=('_pred', '_actual')
    )

    if merged_df.empty:
        logging.info("No matching timestamps between predictions and actual data.")
        return

    # Compute differences for SCALABLE_FEATURES
    for feature in SCALABLE_FEATURES.keys():
        pred_col = f"{feature}_pred"
        actual_col = f"{feature}_actual"
        diff_col = f"{feature}_diff"
        merged_df[diff_col] = merged_df[actual_col] - merged_df[pred_col]

    # Select columns to save
    diff_columns = ['timestamp', 'symbol', 'interval'] + \
                    [f"{feature}_diff" for feature in SCALABLE_FEATURES.keys()]
    differences_df = merged_df[diff_columns]

    # Save differences to file
    shared_data_processor.ensure_file_exists(differences_path)
    if os.path.exists(differences_path) and os.path.getsize(differences_path) > 0:
        existing_diffs = pd.read_csv(differences_path)
        combined_diffs = pd.concat([existing_diffs, differences_df], ignore_index=True)
        combined_diffs.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], inplace=True)
    else:
        combined_diffs = differences_df

    combined_diffs.to_csv(differences_path, index=False)
    logging.info(f"Differences saved to {differences_path}")
