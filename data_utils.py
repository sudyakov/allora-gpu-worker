import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union, List, Any, OrderedDict, Sequence, TypedDict, Literal
from datetime import datetime, timezone
import requests
import pickle
from torch.utils.data import DataLoader, TensorDataset

from config import (
    SEQ_LENGTH,
    TARGET_SYMBOL,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    ADD_FEATURES,
    MODEL_FILENAME,
    DATA_PROCESSOR_FILENAME,
    PATHS,
    SYMBOL_MAPPING,
    API_BASE_URL,
    IntervalConfig,
    IntervalKey
)

class CustomLabelEncoder:
    def __init__(self, predefined_mapping: Optional[Dict[Any, int]] = None):
        if predefined_mapping:
            self.classes_ = predefined_mapping
            self.classes_reverse = {v: k for k, v in predefined_mapping.items()}
        else:
            self.classes_: Dict[Any, int] = {}
            self.classes_reverse: Dict[int, Any] = {}

    def fit(self, data: pd.Series) -> None:
        if not self.classes_:
            unique_classes = data.dropna().unique()
            self.classes_ = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.classes_reverse = {idx: cls for cls, idx in self.classes_.items()}

    def transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_:
            raise ValueError("LabelEncoder не инициализирован маппингом.")
        return data.map(self.classes_).fillna(-1).astype(int)

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_reverse:
            raise ValueError("LabelEncoder не инициализирован маппингом.")
        return data.map(self.classes_reverse)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)

class DataProcessor:
    def __init__(self):
        self.label_encoders: Dict[str, CustomLabelEncoder] = {}

        # Следуем строгому порядку и названиям
        self.categorical_columns: Sequence[str] = list(RAW_FEATURES.keys())
        self.numerical_columns: Sequence[str] = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())

        self.symbol_mapping = SYMBOL_MAPPING
        self.interval_mapping = {k: idx for idx, k in enumerate(INTERVAL_MAPPING.keys())}

        self.label_encoders["symbol"] = CustomLabelEncoder(predefined_mapping=self.symbol_mapping)
        self.label_encoders["interval"] = CustomLabelEncoder(predefined_mapping=self.interval_mapping)

    def preprocess_binance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        for col, dtype in RAW_FEATURES.items():
            if col in df.columns and col != 'timestamp':
                df[col] = df[col].astype(dtype)
        return df

    def fill_missing_add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in df.columns:
            dt = df['timestamp']
            df['hour'] = dt.dt.hour.astype(np.int64)
            df['dayofweek'] = dt.dt.dayofweek.astype(np.int64)
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
            df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
            df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
        df = df.ffill().bfill()
        logging.debug(f"Заполненный DataFrame:\n{df.head()}")
        return df

    def sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        sorted_df = df.sort_values('timestamp', ascending=False)
        logging.debug(f"Отсортированный DataFrame:\n{sorted_df.head()}")
        return sorted_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                logging.error("Не найден кодировщик для столбца %s.", col)
                raise ValueError(f"Не найден кодировщик для столбца {col}.")
            df[col] = encoder.transform(df[col])

        for col, dtype in {**SCALABLE_FEATURES, **ADD_FEATURES}.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
            else:
                logging.error("Столбец %s отсутствует в DataFrame.", col)
                raise KeyError(f"Столбец {col} отсутствует в DataFrame.")

        # Преобразование 'timestamp' в числовой формат после всех операций с датой и временем
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'datetime64[ns]':
            df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6  # Преобразуем в миллисекунды

        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Порядок столбцов после fit_transform: %s", df.columns.tolist())
        logging.info("Типы данных после fit_transform:")
        logging.info(df.dtypes)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                logging.error("LabelEncoder для столбца %s не найден.", col)
                raise ValueError(f"LabelEncoder для столбца {col} не найден.")
            df[col] = encoder.transform(df[col])

        for col, dtype in {**SCALABLE_FEATURES, **ADD_FEATURES}.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
            else:
                logging.error("Столбец %s отсутствует в DataFrame.", col)
                raise KeyError(f"Столбец {col} отсутствует в DataFrame.")

        # Преобразование 'timestamp' в числовой формат после всех операций с датой и временем
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'datetime64[ns]':
            df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6  # Преобразуем в миллисекунды

        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Порядок столбцов после transform: %s", df.columns.tolist())
        logging.info("Типы данных после transform:")
        logging.info(df.dtypes)
        return df

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time
