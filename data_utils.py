import os
import pickle
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

from config import (
    ADD_FEATURES,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    PATHS,
    PREDICTION_MINUTES,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    SYMBOL_MAPPING,
    TARGET_SYMBOL,
)


class CustomLabelEncoder:
    def __init__(self, predefined_mapping: Optional[Dict[Any, int]] = None):
        if predefined_mapping:
            self.classes_ = predefined_mapping
            self.classes_reverse = {v: k for k, v in predefined_mapping.items()}
        else:
            self.classes_ = {}
            self.classes_reverse = {}

    def fit(self, data: pd.Series) -> None:
        if not self.classes_:
            unique_classes = data.dropna().unique()
            self.classes_ = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.classes_reverse = {idx: cls for cls, idx in self.classes_.items()}

    def transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_:
            raise ValueError("LabelEncoder is not initialized with mapping.")
        return data.map(self.classes_).fillna(-1).astype(int)

    def inverse_transform(self, data: pd.Series) -> pd.Series:
        if not self.classes_reverse:
            raise ValueError("LabelEncoder is not initialized with mapping.")
        return data.map(self.classes_reverse)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)


class DataProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, 'initialized', False):
            return
        self.column_name_to_index: Dict[str, int] = {}
        self.is_fitted = False
        self.label_encoders: Dict[str, CustomLabelEncoder] = {}
        self.categorical_columns: Sequence[str] = ['symbol', 'interval', 'hour', 'dayofweek']
        self.numerical_columns: Sequence[str] = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())
        self.scalable_columns: Sequence[str] = list(SCALABLE_FEATURES.keys())
        self.symbol_mapping = SYMBOL_MAPPING.copy()

        if TARGET_SYMBOL not in self.symbol_mapping:
            max_symbol_code = max(self.symbol_mapping.values(), default=-1)
            self.symbol_mapping[TARGET_SYMBOL] = max_symbol_code + 1

        self.label_encoders["symbol"] = CustomLabelEncoder(predefined_mapping=self.symbol_mapping)
        self.label_encoders['hour'] = CustomLabelEncoder()
        self.label_encoders['dayofweek'] = CustomLabelEncoder()
        self.interval_mapping = {k: idx for idx, k in enumerate(INTERVAL_MAPPING.keys())}

        if PREDICTION_MINUTES not in INTERVAL_MAPPING:
            max_interval_code = max(self.interval_mapping.values(), default=-1)
            self.interval_mapping[PREDICTION_MINUTES] = max_interval_code + 1

        self.label_encoders["interval"] = CustomLabelEncoder(predefined_mapping=self.interval_mapping)
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.initialized = True

    def sort_dataframe(self, data_df: pd.DataFrame) -> pd.DataFrame:
        return data_df.sort_values('timestamp', ascending=True)

    def set_column_name_to_index(self, columns: Sequence[str]) -> None:
        self.column_name_to_index = {col: idx for idx, col in enumerate(columns)}

    def preprocess_binance_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        data_df = data_df.replace([np.inf, -np.inf], pd.NA).dropna()
        for col, dtype in {**RAW_FEATURES, **SCALABLE_FEATURES}.items():
            if col in data_df.columns:
                data_df[col] = data_df[col].astype(dtype)
        return data_df

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'DataProcessor':
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        return processor

    def fill_missing_add_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in data_df.columns:
            data_df['hour'] = ((data_df['timestamp'] // (1000 * 60 * 60)) % 24).astype(np.int64)
            data_df['dayofweek'] = ((data_df['timestamp'] // (1000 * 60 * 60 * 24)) % 7).astype(np.int64)
            data_df['sin_hour'] = np.sin(2 * np.pi * data_df['hour'] / 24).astype(np.float32)
            data_df['cos_hour'] = np.cos(2 * np.pi * data_df['hour'] / 24).astype(np.float32)
            data_df['sin_day'] = np.sin(2 * np.pi * data_df['dayofweek'] / 7).astype(np.float32)
            data_df['cos_day'] = np.cos(2 * np.pi * data_df['dayofweek'] / 7).astype(np.float32)
        data_df = data_df.ffill().bfill()
        data_df = data_df[list(MODEL_FEATURES.keys())]
        return data_df

    def fit_transform(self, data_df: pd.DataFrame) -> pd.DataFrame:
        self.is_fitted = True
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                encoder = CustomLabelEncoder()
                self.label_encoders[col] = encoder
            data_df[col] = encoder.fit_transform(data_df[col])

        for col in self.scalable_columns:
            if col in data_df.columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_df[col] = scaler.fit_transform(data_df[[col]])
                self.scalers[col] = scaler
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")

        for col in self.numerical_columns:
            if col in data_df.columns:
                dtype = MODEL_FEATURES.get(col, np.float32)
                data_df[col] = data_df[col].astype(dtype)
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")

        if 'timestamp' in data_df.columns:
            data_df['timestamp'] = data_df['timestamp'].astype(np.int64)

        data_df = data_df[list(MODEL_FEATURES.keys())]
        return data_df

    def transform(self, data_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                raise ValueError(f"LabelEncoder not found for column {col}.")
            data_df[col] = encoder.transform(data_df[col])

        for col in self.scalable_columns:
            if col in data_df.columns:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError(f"Scaler not found for column {col}.")
                data_df[col] = scaler.transform(data_df[[col]])
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")

        for col in self.numerical_columns:
            if col in data_df.columns:
                dtype = MODEL_FEATURES.get(col, np.float32)
                data_df[col] = data_df[col].astype(dtype)
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")

        if 'timestamp' in data_df.columns:
            data_df['timestamp'] = data_df['timestamp'].astype(np.int64)

        data_df = data_df[list(MODEL_FEATURES.keys())]
        return data_df

    def inverse_transform(self, data_df: pd.DataFrame) -> pd.DataFrame:
        df_inv = data_df.copy()
        for col in self.scalable_columns:
            if col in df_inv.columns:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError(f"Scaler not found for column {col}.")
                df_inv[col] = scaler.inverse_transform(df_inv[[col]])
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")
        return df_inv

    def prepare_dataset(
        self,
        data_df: pd.DataFrame,
        seq_length: int = SEQ_LENGTH,
        target_symbols: Optional[List[str]] = None,
        target_intervals: Optional[List[int]] = None
    ) -> TensorDataset:
        if len(data_df) < seq_length:
            raise ValueError("Not enough data to create sequences.")

        features = list(MODEL_FEATURES.keys())
        target_columns = list(SCALABLE_FEATURES.keys())
        missing_columns = [col for col in target_columns if col not in data_df.columns]

        if missing_columns:
            raise KeyError(f"Missing target columns in DataFrame: {missing_columns}")

        data_df = data_df[features]
        if 'timestamp' in data_df.columns:
            data_df['timestamp'] = data_df['timestamp'].astype(np.float32)

        symbol_idx = features.index('symbol')
        interval_idx = features.index('interval')
        target_indices = torch.tensor([features.index(col) for col in target_columns])

        data_tensor = torch.tensor(data_df.values, dtype=torch.float32)
        sequences = []
        targets = []
        target_masks = []

        label_encoders = self.label_encoders
        target_symbol_codes = (
            [label_encoders['symbol'].classes_[sym] for sym in target_symbols]
            if target_symbols else list(label_encoders['symbol'].classes_.values())
        )
        target_interval_codes = (
            [label_encoders['interval'].classes_[interval] for interval in target_intervals]
            if target_intervals else list(label_encoders['interval'].classes_.values())
        )

        for i in range(len(data_tensor) - seq_length):
            sequence = data_tensor[i:i + seq_length]
            next_step = data_tensor[i + seq_length]

            sequences.append(sequence)
            target = next_step.index_select(0, target_indices)
            targets.append(target)

            mask = (
                int(next_step[symbol_idx].item() in target_symbol_codes) and
                int(next_step[interval_idx].item() in target_interval_codes)
            )
            target_masks.append(mask)

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        target_masks = torch.tensor(target_masks, dtype=torch.float32)

        return TensorDataset(sequences, targets, target_masks)

    def get_latest_dataset_prices(
        self,
        symbol: Optional[str] = None,
        interval: Optional[int] = None,
        count: int = SEQ_LENGTH,
        latest_timestamp: Optional[int] = None
    ) -> pd.DataFrame:
        combined_dataset_path = PATHS['combined_dataset']
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            combined_data_df = pd.read_csv(combined_dataset_path)
            df_filtered = combined_data_df.copy()

            if symbol is not None:
                df_filtered = df_filtered[df_filtered['symbol'] == symbol]
            if interval is not None:
                df_filtered = df_filtered[df_filtered['interval'] == interval]
            if latest_timestamp is not None:
                df_filtered = df_filtered[df_filtered['timestamp'] <= latest_timestamp]
            else:
                latest_timestamp = df_filtered['timestamp'].max()

            if not df_filtered.empty:
                df_filtered = df_filtered.sort_values('timestamp', ascending=True)
                df_filtered = df_filtered[df_filtered['timestamp'] <= latest_timestamp].tail(count)
                return df_filtered

        return pd.DataFrame(columns=list(MODEL_FEATURES.keys()))

shared_data_processor = DataProcessor()
