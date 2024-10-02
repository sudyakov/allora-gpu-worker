import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union, List
from datetime import datetime, timezone
import requests
import pickle
from torch.utils.data import DataLoader, TensorDataset

from config import (
    INTERVAL_MAPPING,
    SYMBOL_MAPPING,
    PATHS,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    MODEL_FEATURES,
    MODEL_PARAMS,
    ADD_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    PREDICTION_MINUTES,
    TIME_FEATURES,
    API_BASE_URL
)


class CustomLabelEncoder:
    def __init__(self, predefined_mapping: Optional[Dict[Union[str, int, float], int]] = None):
        if predefined_mapping:
            self.classes_ = predefined_mapping
            self.classes_reverse = {v: k for k, v in predefined_mapping.items()}
        else:
            self.classes_: Dict[Union[str, int, float], int] = {}
            self.classes_reverse: Dict[int, Union[str, int, float]] = {}

    def fit(self, data: pd.Series) -> None:
        if not self.classes_:
            unique_classes = data.dropna().unique()
            self.classes_ = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.classes_reverse = {idx: cls for cls, idx in self.classes_.items()}

    def transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_:
            raise ValueError("LabelEncoder has not been fitted yet.")
        return data.map(self.classes_).fillna(-1).astype(int)

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_reverse:
            raise ValueError("LabelEncoder has not been fitted yet.")
        return data.map(self.classes_reverse)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)


class DataProcessor:
    def __init__(self):
        self.label_encoders: Dict[str, CustomLabelEncoder] = {}
        self.categorical_columns: List[str] = ["symbol", "interval_str"]
        self.numerical_columns: List[str] = list(SCALABLE_FEATURES.keys())

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            if col == "symbol":
                # Используем предопределённый SYMBOL_MAPPING
                le = CustomLabelEncoder(predefined_mapping=SYMBOL_MAPPING)
                df[col] = le.transform(df[col])
                self.label_encoders[col] = le
            else:
                # Стандартное кодирование для других категориальных столбцов
                le = CustomLabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        for col in self.numerical_columns:
            if col in SCALABLE_FEATURES:
                df[col] = df[col].astype(SCALABLE_FEATURES[col])
            elif col in RAW_FEATURES:
                df[col] = df[col].astype(RAW_FEATURES[col])
            else:
                logging.error("Column %s not found in feature definitions.", col)
                raise KeyError(f"Column {col} not defined in any feature dictionary.")

        df['timestamp'] = df['timestamp'].astype('int64')

        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Column order after fit_transform: %s", df.columns.tolist())
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            le = self.label_encoders.get(col)
            if le is None:
                logging.error("LabelEncoder for column %s is not fitted.", col)
                raise ValueError(f"LabelEncoder for column {col} is not fitted.")
            df[col] = le.transform(df[col])

        for col in self.numerical_columns:
            if col in SCALABLE_FEATURES:
                df[col] = df[col].astype(SCALABLE_FEATURES[col])
            elif col in RAW_FEATURES:
                df[col] = df[col].astype(RAW_FEATURES[col])
            else:
                logging.error("Column %s not found in feature definitions.", col)
                raise KeyError(f"Column {col} not defined in any feature dictionary.")

        df['timestamp'] = df['timestamp'].astype('int64')

        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Column order after transform: %s", df.columns.tolist())
        return df

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time
