import logging
import os
from typing import Dict, Optional, Tuple, Sequence

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from get_binance_data import main as get_binance_data_main
from config import (
    ADD_FEATURES,
    DATA_PROCESSOR_FILENAME,
    INTERVAL_MAPPING,
    MODEL_FILENAME,
    MODEL_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    PATHS,
    PREDICTION_MINUTES,
    IntervalKey,
    get_interval,
    TRAINING_PARAMS,
    MODEL_PARAMS,
    SCALABLE_FEATURES,
    RAW_FEATURES,
    TIME_FEATURES,
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

def update_differences(
    differences_path: str,
    predictions_path: str,
    combined_dataset_path: str
) -> None:
    # Load predictions
    if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
        predictions_df = pd.read_csv(predictions_path)
    else:
        logging.info("No predictions available to process.")
        return

    # Load actual data
    if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
        combined_df = pd.read_csv(combined_dataset_path)
    else:
        logging.error("Combined dataset not found.")
        return

    # Load existing differences or initialize
    if os.path.exists(differences_path) and os.path.getsize(differences_path) > 0:
        existing_differences = pd.read_csv(differences_path)
    else:
        existing_differences = pd.DataFrame(columns=predictions_df.columns)

    # Ensure required columns exist
    required_columns = predictions_df.columns.tolist()
    missing_columns_pred = set(required_columns) - set(predictions_df.columns)
    missing_columns_actual = set(required_columns) - set(combined_df.columns)

    if missing_columns_pred:
        logging.error(f"Missing columns in predictions DataFrame: {missing_columns_pred}")
        return
    if missing_columns_actual:
        logging.error(f"Missing columns in actual DataFrame: {missing_columns_actual}")
        return

    # Filter actual data to match predictions
    actual_df = combined_df[
        combined_df['timestamp'].isin(predictions_df['timestamp'].unique()) &
        combined_df['symbol'].isin(predictions_df['symbol'].unique()) &
        combined_df['interval'].isin(predictions_df['interval'].unique())
    ]

    if actual_df.empty:
        logging.info("No matching actual data found for predictions.")
        return

    # Merge predictions and actual data
    merged_df = pd.merge(
        predictions_df,
        actual_df,
        on=['timestamp', 'symbol', 'interval'],
        suffixes=('_pred', '_actual')
    )

    if merged_df.empty:
        logging.info("No matching timestamps between predictions and actual data.")
        return

    # Exclude records already in differences
    if not existing_differences.empty:
        merged_df = pd.merge(
            merged_df,
            existing_differences[['timestamp', 'symbol', 'interval']],
            on=['timestamp', 'symbol', 'interval'],
            how='left',
            indicator=True
        )
        merged_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        if merged_df.empty:
            logging.info("All differences have already been processed.")
            return

    feature_cols = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())

    # Copy columns from predictions
    differences_df = merged_df[predictions_df.columns].copy()

    for feature in feature_cols:
        pred_col = f"{feature}_pred"
        actual_col = f"{feature}_actual"
        differences_df[feature] = merged_df[actual_col] - merged_df[pred_col]

    # Ensure data types match predictions DataFrame
    for col in differences_df.columns:
        if col in predictions_df.columns:
            differences_df[col] = differences_df[col].astype(predictions_df[col].dtype)

    # Combine with existing differences and save
    combined_differences = pd.concat([existing_differences, differences_df], ignore_index=True)
    combined_differences.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], inplace=True)
    combined_differences = combined_differences[predictions_df.columns]  # Ensure same column order
    combined_differences.to_csv(differences_path, index=False)
    logging.info(f"Differences updated and saved to {differences_path}")
