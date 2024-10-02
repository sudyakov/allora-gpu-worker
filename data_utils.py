import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Literal, Union, List
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

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.matmul(lstm_out, self.attention_weights).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector

class EnhancedBiLSTMModel(nn.Module):
    def __init__(
        self,
        numerical_columns: List[str],
        categorical_columns: List[str],
        column_name_to_index: Dict[str, int],
        input_size: int = MODEL_PARAMS.get("input_size", 64),
        hidden_layer_size: int = MODEL_PARAMS.get("hidden_layer_size", 128),
        num_layers: int = MODEL_PARAMS.get("num_layers", 2),
        dropout: float = MODEL_PARAMS.get("dropout", 0.5),
        embedding_dim: int = MODEL_PARAMS.get("embedding_dim", 32),
        num_symbols: int = len(SYMBOL_MAPPING),
        num_intervals: int = len(INTERVAL_MAPPING),
        timestamp_embedding_dim: int = MODEL_PARAMS.get("timestamp_embedding_dim", 16),
    ):
        super(EnhancedBiLSTMModel, self).__init__()
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.column_name_to_index = column_name_to_index

        self.symbol_embedding = nn.Embedding(num_embeddings=num_symbols, embedding_dim=embedding_dim)
        self.interval_embedding = nn.Embedding(num_embeddings=num_intervals, embedding_dim=embedding_dim)
        self.timestamp_embedding = nn.Linear(1, timestamp_embedding_dim)

        numerical_input_size = len(numerical_columns)
        self.lstm_input_size = numerical_input_size + (2 * embedding_dim) + timestamp_embedding_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, input_size)
        self.relu = nn.ReLU()
        self.apply(self._initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerical_indices = [self.column_name_to_index[col] for col in self.numerical_columns]
        numerical_data = x[:, :, numerical_indices]

        symbols = x[:, :, self.column_name_to_index['symbol']].long()
        intervals = x[:, :, self.column_name_to_index['interval_str']].long()

        symbol_embeddings = self.symbol_embedding(symbols)
        interval_embeddings = self.interval_embedding(intervals)

        timestamp = x[:, :, self.column_name_to_index['timestamp']].unsqueeze(-1)
        timestamp_embedded = self.timestamp_embedding(timestamp)

        lstm_input = torch.cat((numerical_data, symbol_embeddings, interval_embeddings, timestamp_embedded), dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)
        predictions = self.relu(predictions)
        return predictions

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

class CustomMinMaxScaler:
    def __init__(self, feature_range: tuple = (0, 1)):
        self.min: Optional[pd.Series] = None
        self.max: Optional[pd.Series] = None
        self.feature_range = feature_range

    def fit(self, data: pd.DataFrame) -> None:
        self.min = data.min()
        self.max = data.max()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.min) / (self.max - self.min) * (
            self.feature_range[1] - self.feature_range[0]
        ) + self.feature_range[0]

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (data - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        ) * (self.max - self.min) + self.min

class CustomLabelEncoder:
    def __init__(self):
        self.classes_: Dict[Union[str, int, float], int] = {}
        self.classes_reverse: Dict[int, Union[str, int, float]] = {}

    def fit(self, data: pd.Series) -> None:
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
        self.scaler = CustomMinMaxScaler(feature_range=(0, 1))
        self.label_encoders: Dict[str, CustomLabelEncoder] = {}
        self.categorical_columns: List[str] = ["symbol", "interval_str"]
        self.numerical_columns: List[str] = list(SCALABLE_FEATURES.keys())
        logging.info("Numerical columns for scaling: %s", self.numerical_columns)

    def preprocess_binance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
        for col, dtype in RAW_FEATURES.items():
            if col in df.columns:
                if col == 'timestamp':
                    df[col] = pd.to_datetime(df[col], unit='ms')
                else:
                    df[col] = df[col].astype(dtype)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
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

        self.scaler.fit(df[self.numerical_columns])
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
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

        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        df['timestamp'] = df['timestamp'].astype('int64')

        df = df[list(MODEL_FEATURES.keys())]
        logging.info("Column order after transform: %s", df.columns.tolist())
        return df

    def prepare_dataset(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH) -> TensorDataset:
        features = list(MODEL_FEATURES.keys())
        target_columns = [col for col in features if col != 'timestamp']
        missing_columns = [col for col in features if col not in df.columns]
        if missing_columns:
            logging.error("Missing columns in DataFrame: %s", missing_columns)
            raise KeyError(f"Missing columns in DataFrame: {missing_columns}")

        data_tensor = torch.tensor(df[features].values, dtype=torch.float32)
        target_indices = torch.tensor([df.columns.get_loc(col) for col in target_columns], dtype=torch.long)

        sequences = []
        targets = []
        for i in range(len(data_tensor) - seq_length):
            sequences.append(data_tensor[i:i + seq_length])
            targets.append(data_tensor[i + seq_length].index_select(0, target_indices))

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        return TensorDataset(sequences, targets)

    def save(self, filepath: str) -> None:
        ensure_directory_exists(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logging.info("DataProcessor saved: %s", filepath)

    @staticmethod
    def load(filepath: str) -> 'DataProcessor':
        with open(filepath, 'rb') as f:
            processor = pickle.load(f)
        logging.info("DataProcessor loaded: %s", filepath)
        return processor

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = df['timestamp'].astype(int)
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)

    for feature_dict in [RAW_FEATURES, TIME_FEATURES, SCALABLE_FEATURES, ADD_FEATURES]:
        for col, dtype in feature_dict.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

    logging.debug(f"Preprocessed DataFrame:\n{df.head()}")
    return df

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values('timestamp', ascending=False)
    logging.debug(f"Sorted DataFrame:\n{sorted_df.head()}")
    return sorted_df

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def fill_missing_add_features(df: pd.DataFrame) -> pd.DataFrame:
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

def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time

def ensure_file_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(MODEL_FEATURES.keys()))
        df.to_csv(filepath, index=False)

def ensure_directory_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def get_latest_dataset_prices(symbol: str, interval: int, count: int = SEQ_LENGTH) -> pd.DataFrame:
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
