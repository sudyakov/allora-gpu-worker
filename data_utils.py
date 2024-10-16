import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Dict, Any, Sequence, List
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
        self.categorical_columns: Sequence[str] = ['symbol', 'interval', 'hour', 'dayofweek']
        self.numerical_columns: Sequence[str] = list(SCALABLE_FEATURES.keys()) + list(ADD_FEATURES.keys())
        self.scalable_columns: Sequence[str] = list(SCALABLE_FEATURES.keys())
        self.symbol_mapping = SYMBOL_MAPPING

        if TARGET_SYMBOL not in self.symbol_mapping:
            max_symbol_code = max(self.symbol_mapping.values(), default=-1)
            self.symbol_mapping[TARGET_SYMBOL] = max_symbol_code + 1

        self.label_encoders["symbol"] = CustomLabelEncoder(predefined_mapping=self.symbol_mapping)
        self.label_encoders['hour'] = CustomLabelEncoder()
        self.label_encoders['dayofweek'] = CustomLabelEncoder()

        self.interval_mapping = {k: idx for idx, k in enumerate(INTERVAL_MAPPING.keys())}

        if PREDICTION_MINUTES not in INTERVAL_MAPPING.keys():
            max_interval_code = max(self.interval_mapping.values(), default=-1)
            self.interval_mapping[PREDICTION_MINUTES] = max_interval_code + 1

        self.label_encoders["interval"] = CustomLabelEncoder(predefined_mapping=self.interval_mapping)
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.initialized = True

    def preprocess_binance_data(self, real_data_df: pd.DataFrame) -> pd.DataFrame:
        real_data_df = real_data_df.replace([float("inf"), float("-inf")], pd.NA).dropna()
        for col, dtype in RAW_FEATURES.items():
            if col in real_data_df.columns:
                real_data_df[col] = real_data_df[col].astype(dtype)
        for col, dtype in SCALABLE_FEATURES.items():
            if col in real_data_df.columns:
                real_data_df[col] = real_data_df[col].astype(dtype)
        return real_data_df

    def save(self, filepath: str) -> None:
        self.ensure_file_exists(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath: str) -> 'DataProcessor':
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        self.__dict__.update(processor.__dict__)
        return self

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

    def sort_dataframe(self, data_df: pd.DataFrame) -> pd.DataFrame:
        sorted_df = data_df.sort_values('timestamp', ascending=True)
        return sorted_df

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
        target_columns = [col for col in SCALABLE_FEATURES.keys()]
        missing_columns = [col for col in target_columns if col not in data_df.columns]

        if missing_columns:
            raise KeyError(f"Missing target columns in DataFrame: {missing_columns}")

        data_df = data_df[features]

        if 'timestamp' in data_df.columns:
            data_df.loc[:, 'timestamp'] = data_df['timestamp'].astype(np.float32)

        symbol_idx = features.index('symbol')
        interval_idx = features.index('interval')
        target_indices = torch.tensor([features.index(col) for col in target_columns])

        data_tensor = torch.tensor(data_df[features].values, dtype=torch.float32)

        sequences = []
        targets = []
        target_masks = []

        if target_symbols:
            target_symbol_codes = [self.label_encoders['symbol'].classes_[sym] for sym in target_symbols]
        else:
            target_symbol_codes = list(self.label_encoders['symbol'].classes_.values())

        if target_intervals:
            target_interval_codes = [self.label_encoders['interval'].classes_[interval] for interval in target_intervals]
        else:
            target_interval_codes = list(self.label_encoders['interval'].classes_.values())

        for i in range(len(data_tensor) - seq_length):
            sequence = data_tensor[i:i + seq_length]
            next_step = data_tensor[i + seq_length]

            sequences.append(sequence)
            target = next_step.index_select(0, target_indices)
            targets.append(target)

            if (next_step[symbol_idx].item() in target_symbol_codes) and \
                (next_step[interval_idx].item() in target_interval_codes):
                target_masks.append(1)
            else:
                target_masks.append(0)

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        target_masks = torch.tensor(target_masks, dtype=torch.float32)

        return TensorDataset(sequences, targets, target_masks)

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
        count: int = SEQ_LENGTH,
        latest_timestamp: Optional[int] = None
    ) -> pd.DataFrame:
        combined_dataset_path = PATHS['combined_dataset']
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            combined_real_data_df = pd.read_csv(combined_dataset_path)
            df_filtered = combined_real_data_df.copy()
            
            if symbol is not None:
                df_filtered = df_filtered[df_filtered['symbol'] == symbol]
            
            if interval is not None:
                df_filtered = df_filtered[df_filtered['interval'] == interval]
            
            if latest_timestamp is not None:
                df_filtered = df_filtered[df_filtered['timestamp'] <= latest_timestamp]
            
            if not df_filtered.empty:
                df_filtered = df_filtered.sort_values('timestamp', ascending=True).tail(count)
                return df_filtered
            
        return pd.DataFrame(columns=list(MODEL_FEATURES.keys()))

shared_data_processor = DataProcessor()
