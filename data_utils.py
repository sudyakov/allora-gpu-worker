import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union, List, Any
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
            raise ValueError("LabelEncoder has not been initialized with a mapping.")
        return data.map(self.classes_).fillna(-1).astype(int)

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_reverse:
            raise ValueError("LabelEncoder has not been initialized with a mapping.")
        return data.map(self.classes_reverse)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)

class DataProcessor:
    def __init__(self):
        self.label_encoders: Dict[str, CustomLabelEncoder] = {}
        self.categorical_columns: List[str] = ["symbol", "interval_str"]
        self.numerical_columns: List[str] = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())

        self.symbol_mapping = SYMBOL_MAPPING
        self.interval_str_mapping = {k: idx for idx, k in enumerate(INTERVAL_MAPPING.keys())}

        self.label_encoders["symbol"] = CustomLabelEncoder(predefined_mapping=self.symbol_mapping)
        self.label_encoders["interval_str"] = CustomLabelEncoder(predefined_mapping=self.interval_str_mapping)

    def preprocess_binance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
        for col, dtype in RAW_FEATURES.items():
            if col in df.columns:
                if col == 'timestamp':
                    df[col] = pd.to_datetime(df[col], unit='ms')
                else:
                    df[col] = df[col].astype(dtype)
        return df

    def fill_missing_add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], unit='ms')
            df['hour'] = dt.dt.hour
            df['dayofweek'] = dt.dt.dayofweek
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df = df.ffill().bfill()
        logging.debug(f"Filled DataFrame:\n{df.head()}")
        return df

    def sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        sorted_df = df.sort_values('timestamp', ascending=False)
        logging.debug(f"Sorted DataFrame:\n{sorted_df.head()}")
        return sorted_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                logging.error("No encoder found for column %s.", col)
                raise ValueError(f"No encoder found for column {col}.")
            df[col] = encoder.transform(df[col])

        for col in self.numerical_columns:
            if col in SCALABLE_FEATURES:
                df[col] = df[col].astype(SCALABLE_FEATURES[col])
            elif col in RAW_FEATURES:
                df[col] = df[col].astype(RAW_FEATURES[col])
            elif col in ADD_FEATURES:
                df[col] = df[col].astype(ADD_FEATURES[col])
            else:
                logging.error("Column %s not found in feature definitions.", col)
                raise KeyError(f"Column {col} not defined in any feature dictionary.")

        df['timestamp'] = df['timestamp'].astype('int64')
        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Column order after fit_transform: %s", df.columns.tolist())
        logging.info("Data types after fit_transform:")
        logging.info(df.dtypes)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                logging.error("LabelEncoder for column %s is not found.", col)
                raise ValueError(f"LabelEncoder for column {col} is not found.")
            df[col] = encoder.transform(df[col])

        for col in self.numerical_columns:
            if col in SCALABLE_FEATURES:
                df[col] = df[col].astype(SCALABLE_FEATURES[col])
            elif col in RAW_FEATURES:
                df[col] = df[col].astype(RAW_FEATURES[col])
            elif col in ADD_FEATURES:
                df[col] = df[col].astype(ADD_FEATURES[col])
            else:
                logging.error("Column %s not found in feature definitions.", col)
                raise KeyError(f"Column {col} not defined in any feature dictionary.")

        df['timestamp'] = df['timestamp'].astype('int64')
        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Column order after transform: %s", df.columns.tolist())
        logging.info("Data types after transform:")
        logging.info(df.dtypes)
        return df

    def prepare_dataset(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH) -> TensorDataset:
        features = list(MODEL_FEATURES.keys())
        target_columns = list(SCALABLE_FEATURES.keys())
        missing_columns = [col for col in target_columns if col not in df.columns]
        if missing_columns:
            logging.error("Missing target columns in DataFrame: %s", missing_columns)
            raise KeyError(f"Missing target columns in DataFrame: {missing_columns}")

        logging.info("Data types before tensor conversion:")
        logging.info(df[features].dtypes)

        object_columns = df[features].select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            logging.error(f"Columns with object dtype: {object_columns}")
            for col in object_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(float)
                    logging.info(f"Column {col} converted to float.")
                except Exception as e:
                    logging.error(f"Error converting column {col} to numeric: {e}")
                    raise

        logging.info("Data types after type conversion:")
        logging.info(df[features].dtypes)

        data_tensor = torch.tensor(df[features].values, dtype=torch.float32)

        target_indices = torch.tensor([df.columns.get_loc(col) for col in target_columns], dtype=torch.long)

        sequences = []
        targets = []
        for i in range(len(data_tensor) - seq_length + 1):  # Adjusted loop range
            sequences.append(data_tensor[i:i + seq_length])
            if i + seq_length < len(data_tensor):
                targets.append(data_tensor[i + seq_length].index_select(0, target_indices))
            else:
                # For the last sequence, duplicate the last available target
                targets.append(data_tensor[-1].index_select(0, target_indices))

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)

        return TensorDataset(sequences, targets)

    def save(self, filepath: str) -> None:
        self.ensure_file_exists(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info("DataProcessor saved: %s", filepath)

    @staticmethod
    def ensure_file_exists(filepath: str) -> None:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(filepath):
            df = pd.DataFrame(columns=list(MODEL_FEATURES.keys()))
            df.to_csv(filepath, index=False)

    @staticmethod
    def load(filepath: str) -> 'DataProcessor':
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        logging.info("DataProcessor loaded: %s", filepath)
        return processor

    def get_latest_dataset_prices(self, symbol: str, interval: int, count: int = SEQ_LENGTH) -> pd.DataFrame:
        combined_dataset_path = PATHS['combined_dataset']
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            df_combined = pd.read_csv(combined_dataset_path)
            df_filtered = df_combined[
                (df_combined['symbol'] == symbol) & (df_combined['interval'] == interval)
            ]
            if not df_filtered.empty:
                df_filtered = df_filtered.sort_values('timestamp', ascending=False).head(count)
                return df_filtered
            else:
                logging.debug(f"No data for symbol {symbol} and interval {interval} in combined_dataset.")
        else:
            logging.debug(f"combined_dataset.csv file not found at path {combined_dataset_path}")
        return pd.DataFrame(columns=list(MODEL_FEATURES.keys()))

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time
