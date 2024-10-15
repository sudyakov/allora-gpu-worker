import logging
import os
from typing import Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from filelock import FileLock  # Добавляем импорт библиотеки для блокировки файлов
from tqdm import tqdm


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
    future_steps: int = 1  # Количество шагов в будущее
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
        interval = get_interval(prediction_minutes)
        if interval is None:
            logging.error("Invalid prediction interval.")
            return pd.DataFrame()
        interval_ms = INTERVAL_MAPPING[interval]["milliseconds"]
        # Генерируем метки времени для будущих предсказаний
        timestamps_to_predict = [
            last_binance_timestamp + interval_ms * i for i in range(1, future_steps + 1)
        ]
        for next_timestamp in timestamps_to_predict:
            # Подготовка данных для текущей последовательности
            current_df = latest_df.tail(SEQ_LENGTH)
            if len(current_df) < SEQ_LENGTH:
                logging.info(f"Insufficient data to predict for timestamp {next_timestamp}.")
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
            # Обновляем `latest_df` для следующей итерации, добавляя предсказание
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

def update_predictions(
    model: nn.Module,
    combined_dataset_path: str,
    predictions_path: str,
    device: torch.device,
    seq_length: int = SEQ_LENGTH,
    prediction_minutes: int = PREDICTION_MINUTES,
    target_symbol: str = TARGET_SYMBOL
) -> None:
    # Загружаем объединенный датасет
    if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
        combined_df = pd.read_csv(combined_dataset_path)
    else:
        logging.error(f"Combined dataset not found at {combined_dataset_path}.")
        return

    # Предобработка данных
    combined_df = shared_data_processor.preprocess_binance_data(combined_df)
    combined_df = shared_data_processor.fill_missing_add_features(combined_df)
    combined_df = combined_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)

    # Фильтрация данных по целевому символу и интервалу
    symbol_df = combined_df[
        (combined_df['symbol'] == target_symbol) &
        (combined_df['interval'] == prediction_minutes)
    ].copy()

    if symbol_df.empty:
        logging.error("No data available for the target symbol and interval.")
        return

    # Определяем последние предсказанные временные метки
    if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
        predictions_df = pd.read_csv(predictions_path)
        last_prediction_timestamp = predictions_df['timestamp'].max()
    else:
        predictions_df = pd.DataFrame(columns=combined_df.columns)
        last_prediction_timestamp = None

    # Формируем список временных меток для предсказания
    if last_prediction_timestamp is not None:
        data_to_predict = symbol_df[symbol_df['timestamp'] > last_prediction_timestamp]
    else:
        data_to_predict = symbol_df.copy()

    if data_to_predict.empty:
        logging.info("No new timestamps to predict.")
        return

    missing_predictions = []

    # Используем реальные данные для предсказаний
    interval = get_interval(prediction_minutes)
    if interval is None:
        logging.error(f"Invalid prediction_minutes: {prediction_minutes}")
        return
    interval_ms = INTERVAL_MAPPING[interval]['milliseconds']

    for idx in tqdm(range(len(data_to_predict)), desc="Predicting missing timestamps"):
        current_timestamp = data_to_predict.iloc[idx]['timestamp']
        prediction_timestamp = current_timestamp + interval_ms

        logging.info(f"Starting prediction for timestamp {prediction_timestamp}")

        # Извлекаем последовательность данных до текущей метки времени
        sequence_df = symbol_df[
            (symbol_df['timestamp'] < current_timestamp)
        ].tail(seq_length)

        if len(sequence_df) < seq_length:
            logging.warning(f"Not enough data to predict for timestamp {prediction_timestamp}.")
            continue

        # Преобразуем данные (масштабирование и кодирование)
        try:
            sequence_df_transformed = shared_data_processor.transform(sequence_df)
            logging.debug(f"Transformed sequence for {prediction_timestamp}:\n{sequence_df_transformed}")
        except Exception as e:
            logging.error(f"Error transforming data for timestamp {prediction_timestamp}: {e}")
            continue

        inputs = torch.tensor(sequence_df_transformed.values, dtype=torch.float32).unsqueeze(0).to(device)

        # Делаем предсказание
        model.eval()
        with torch.no_grad():
            predictions = model(inputs).cpu().numpy()

        # Преобразуем предсказания обратно (обратное масштабирование)
        try:
            predictions_df_single = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))
            predictions_df_denormalized = shared_data_processor.inverse_transform(predictions_df_single)
            logging.debug(f"Denormalized prediction for {prediction_timestamp}:\n{predictions_df_denormalized}")
        except Exception as e:
            logging.error(f"Error in inverse transform for timestamp {prediction_timestamp}: {e}")
            continue

        # Добавляем дополнительные данные
        predictions_df_denormalized['symbol'] = target_symbol
        predictions_df_denormalized['interval'] = prediction_minutes
        predictions_df_denormalized['timestamp'] = prediction_timestamp
        predictions_df_denormalized['hour'] = ((prediction_timestamp // (1000 * 60 * 60)) % 24).astype(np.int64)
        predictions_df_denormalized['dayofweek'] = ((prediction_timestamp // (1000 * 60 * 60 * 24)) % 7).astype(np.int64)

        # Дополняем недостающие признаки
        predictions_df_denormalized = shared_data_processor.fill_missing_add_features(predictions_df_denormalized)

        # Убедитесь, что все необходимые признаки обновлены
        logging.debug(f"Features after fill_missing_add_features for {prediction_timestamp}:\n{predictions_df_denormalized}")

        # Выберите только конечные признаки
        final_columns = list(MODEL_FEATURES.keys())
        predictions_df_denormalized = predictions_df_denormalized[final_columns]

        # Логирование предсказания
        logging.debug(f"Prediction DataFrame for timestamp {prediction_timestamp}:\n{predictions_df_denormalized}")

        missing_predictions.append(predictions_df_denormalized)

        # **Удаляем код объединения предсказаний с реальными данными**
        # new_row = predictions_df_denormalized.copy()
        # new_row['timestamp'] = prediction_timestamp
        # new_row['prediction_timestamp'] = prediction_timestamp + interval_ms
        # symbol_df = pd.concat([symbol_df, new_row], ignore_index=True)
        # logging.debug(f"Updated symbol_df after adding new prediction:\n{symbol_df.tail()}")

    if missing_predictions:
        # Объединяем новые предсказания с существующими
        new_predictions_df = pd.concat(missing_predictions, ignore_index=True)
        updated_predictions_df = pd.concat([predictions_df, new_predictions_df], ignore_index=True)
        updated_predictions_df = updated_predictions_df.drop_duplicates(subset=['timestamp', 'symbol', 'interval'])
        # Сортируем предсказания по возрастанию timestamp
        updated_predictions_df = updated_predictions_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)

        # Проверка на наличие дубликатов временных меток
        if updated_predictions_df['timestamp'].duplicated().any():
            logging.error("Duplicate timestamps found in updated_predictions_df.")
            # Можно удалить дубликаты или предпринять другие действия
            updated_predictions_df = updated_predictions_df.drop_duplicates(subset=['timestamp', 'symbol', 'interval'])

        logging.debug(f"Final updated_predictions_df:\n{updated_predictions_df.tail()}")

        # Сохраняем обновленный файл с предсказаниями
        updated_predictions_df.to_csv(predictions_path, index=False)
        logging.info(f"Updated predictions saved to {predictions_path}.")
    else:
        logging.info("No predictions were made due to insufficient data.")

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