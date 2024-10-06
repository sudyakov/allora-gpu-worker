import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Dict, Any, Sequence
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    SEQ_LENGTH,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    ADD_FEATURES,
    PATHS,
    SYMBOL_MAPPING,
    TRAINING_PARAMS,
    TARGET_SYMBOL,
    PREDICTION_MINUTES
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

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
        self.is_fitted = False
        self.label_encoders: Dict[str, CustomLabelEncoder] = {}
        self.categorical_columns: Sequence[str] = ['symbol', 'interval']
        self.numerical_columns: Sequence[str] = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())
        self.scalable_columns: Sequence[str] = list(SCALABLE_FEATURES.keys())
        self.symbol_mapping = SYMBOL_MAPPING
        self.label_encoders["symbol"] = CustomLabelEncoder(predefined_mapping=self.symbol_mapping)
        self.interval_mapping = {k: idx for idx, k in enumerate(INTERVAL_MAPPING.keys())}
        self.label_encoders["interval"] = CustomLabelEncoder(predefined_mapping=self.interval_mapping)
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.initialized = True

    def preprocess_binance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
        for col, dtype in RAW_FEATURES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        for col, dtype in SCALABLE_FEATURES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def save(self, filepath: str) -> None:
        self.ensure_file_exists(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath: str) -> 'DataProcessor':
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        self.__dict__.update(processor.__dict__)
        return self

    def fill_missing_add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in df.columns:
            df['hour'] = ((df['timestamp'] // (1000 * 60 * 60)) % 24).astype(np.int64)
            df['dayofweek'] = ((df['timestamp'] // (1000 * 60 * 60 * 24)) % 7).astype(np.int64)
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
            df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
            df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
        df = df.ffill().bfill()
        df = df[list(MODEL_FEATURES.keys())]
        return df

    def sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        sorted_df = df.sort_values('timestamp', ascending=False)
        return sorted_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.is_fitted = True
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                raise ValueError(f"Encoder not found for column {col}.")
            df[col] = encoder.transform(df[col])
        for col in self.scalable_columns:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")
        for col in self.numerical_columns:
            if col in df.columns:
                dtype = MODEL_FEATURES.get(col, np.float32)
                df[col] = df[col].astype(dtype)
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(np.int64)
        df = df[list(MODEL_FEATURES.keys())]
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            encoder = self.label_encoders.get(col)
            if encoder is None:
                raise ValueError(f"LabelEncoder not found for column {col}.")
            df[col] = encoder.transform(df[col])
        for col in self.scalable_columns:
            if col in df.columns:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError(f"Scaler not found for column {col}.")
                df[col] = scaler.transform(df[[col]])
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")
        for col in self.numerical_columns:
            if col in df.columns:
                dtype = MODEL_FEATURES.get(col, np.float32)
                df[col] = df[col].astype(dtype)
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(np.int64)
        df = df[list(MODEL_FEATURES.keys())]
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_inv = df.copy()
        for col in self.scalable_columns:
            if col in df_inv.columns:
                scaler = self.scalers.get(col)
                if scaler is None:
                    raise ValueError(f"Scaler not found for column {col}.")
                df_inv[col] = scaler.inverse_transform(df_inv[[col]])
            else:
                raise KeyError(f"Column {col} is missing in DataFrame.")
        return df_inv

    def prepare_dataset(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH) -> TensorDataset:
        if len(df) < seq_length:
            raise ValueError("Not enough data to create sequences.")
        features = list(MODEL_FEATURES.keys())
        target_columns = [col for col in SCALABLE_FEATURES.keys()]
        missing_columns = [col for col in target_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing target columns in DataFrame: {missing_columns}")
        df = df[features]
        if 'timestamp' in df.columns:
            df.loc[:, 'timestamp'] = df['timestamp'].astype(np.float32)
        target_symbol_encoded = self.label_encoders['symbol'].transform(pd.Series([TARGET_SYMBOL]))[0]
        target_interval_encoded = self.label_encoders['interval'].transform(pd.Series([PREDICTION_MINUTES]))[0]
        data_tensor = torch.tensor(df[features].values, dtype=torch.float32)
        target_indices = torch.tensor([features.index(col) for col in target_columns])
        sequences = []
        targets = []
        for i in range(len(data_tensor) - seq_length):
            sequence = data_tensor[i:i + seq_length]
            next_step = data_tensor[i + seq_length]
            symbol_idx = features.index('symbol')
            interval_idx = features.index('interval')
            if next_step[symbol_idx].item() == target_symbol_encoded and next_step[interval_idx].item() == target_interval_encoded:
                sequences.append(sequence)
                target = next_step.index_select(0, target_indices)
                targets.append(target)
            else:
                continue
        if not sequences:
            raise ValueError("No sequences with the target symbol and interval were found.")
        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        return TensorDataset(sequences, targets)

    def create_dataloader(self, dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
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

    def ensure_file_exists(self, filepath: str) -> None:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(filepath):
            df = pd.DataFrame(columns=list(MODEL_FEATURES.keys()))
            df.to_csv(filepath, index=False)

    def get_latest_dataset_prices(
        self,
        symbol: Optional[str] = None,
        interval: Optional[int] = None,
        count: int = SEQ_LENGTH
    ) -> pd.DataFrame:
        combined_dataset_path = PATHS['combined_dataset']
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            df_combined = pd.read_csv(combined_dataset_path)
            df_filtered = df_combined.copy()
            if symbol is not None:
                df_filtered = df_filtered[df_filtered['symbol'] == symbol]
            if interval is not None:
                df_filtered = df_filtered[df_filtered['interval'] == interval]
            if not df_filtered.empty:
                df_filtered = df_filtered.sort_values('timestamp', ascending=False).head(count)
                return df_filtered
        return pd.DataFrame(columns=list(MODEL_FEATURES.keys()))


shared_data_processor = DataProcessor()
