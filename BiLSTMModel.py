import logging
import os
from typing import Optional, Dict, Literal, TypedDict, Any, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from get_binance_data import DownloadData
from config import (
    SEQ_LENGTH,
    PREDICTION_MINUTES,
    RAW_FEATURES,
    MODEL_FEATURES,
    PATHS,
)

# Конфигурации символов
SYMBOL_MAPPING: Dict[str, int] = {
    "BTCUSDT": 0,
    "ETHUSDT": 1,
}

TARGET_SYMBOL: str = "ETHUSDT"

IntervalKey = Literal["1m", "5m", "15m"]

class IntervalConfig(TypedDict):
    days: int
    minutes: int
    milliseconds: int

# Конфигурации интервалов
INTERVAL_MAPPING: Dict[IntervalKey, IntervalConfig] = {
    "1m": {"days": 7, "minutes": 1, "milliseconds": 1 * 60 * 1000},
    "5m": {"days": 14, "minutes": 5, "milliseconds": 5 * 60 * 1000},
    "15m": {"days": 28, "minutes": 15, "milliseconds": 15 * 60 * 1000},
}

# Определение типов данных для сырых и модельных признаков
RAW_FEATURES: Dict[str, type] = {
    'symbol': str,
    'interval': int,
    'timestamp': int,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float,
    'quote_asset_volume': float,
    'number_of_trades': int,
    'taker_buy_base_asset_volume': float,
    'taker_buy_quote_asset_volume': float
}

MODEL_FEATURES: Dict[str, type] = {
    **RAW_FEATURES,
    'hour': int,
    'dayofweek': int,
    'sin_hour': float,
    'cos_hour': float,
    'sin_day': float,
    'cos_day': float
}

MODEL_VERSION = "2.0"

MODEL_PARAMS: Dict[str, Any] = {
    'input_size': len(MODEL_FEATURES),
    'hidden_layer_size': 256,
    'num_layers': 4,
    'dropout': 0.2,
    'embedding_dim': 128,
    'num_symbols': len(SYMBOL_MAPPING)
}

TRAINING_PARAMS: Dict[str, Any] = {
    'batch_size': 512,
    'initial_epochs': 10,
    'initial_lr': 0.0005,
    'max_epochs': 100,
    'min_lr': 0.00001,
    'use_mixed_precision': True,
    'num_workers': 8
}

MODEL_FILENAME = os.path.join(
    PATHS['models_dir'],
    f'enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth'
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Attention(nn.Module):
    def __init__(self, hidden_layer_size: int):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_layer_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.matmul(lstm_out, self.attention_weights).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector

class EnhancedBiLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int,
        num_layers: int,
        dropout: float,
        embedding_dim: int,
        num_symbols: int
    ):
        super(EnhancedBiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_symbols, embedding_dim=embedding_dim)
        self.lstm_input_size = input_size - 1 + embedding_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, input_size)
        self.relu = nn.ReLU()
        self.apply(self._initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerical_data = x[:, :, :-1]
        symbols = x[:, :, -1].long()
        embeddings = self.embedding(symbols)
        lstm_input = torch.cat((numerical_data, embeddings), dim=2)
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
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataProcessor:
    def __init__(self):
        self.scaler = CustomMinMaxScaler(feature_range=(-1, 1))
        self.label_encoders: Dict[str, 'CustomLabelEncoder'] = {}
        self.categorical_columns = ['symbol']
        self.numerical_columns = [col for col in MODEL_FEATURES.keys() if col not in self.categorical_columns]

    def preprocess_binance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['timestamp'] = df['timestamp'].astype(int)
        df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
        for col, dtype in RAW_FEATURES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            le = CustomLabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        for col in self.numerical_columns:
            df[col] = df[col].astype(MODEL_FEATURES[col])

        self.scaler.fit(df[self.numerical_columns])
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        df = df[self.numerical_columns + self.categorical_columns]
        return df

    def prepare_dataset(self, df: pd.DataFrame, seq_length: int) -> TensorDataset:
        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        sequences = []
        targets = []
        timestamps = []
        for i in range(len(data_tensor) - seq_length):
            sequences.append(data_tensor[i:i + seq_length])
            targets.append(data_tensor[i + seq_length])
            timestamps.append(df.iloc[i + seq_length]['timestamp'])
        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        timestamps = torch.tensor(timestamps, dtype=torch.float32)
        tensor_dataset = TensorDataset(sequences, targets, timestamps)
        return tensor_dataset

def ensure_directory_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

class CustomMinMaxScaler:
    def __init__(self, feature_range: tuple = (-1, 1)):
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
        return (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * (
            self.max - self.min
        ) + self.min

class CustomLabelEncoder:
    def __init__(self):
        self.classes_: Dict[Any, int] = {}
        self.classes_reverse: Dict[int, Any] = {}

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

def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=TRAINING_PARAMS['num_workers'])

def train_and_save_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    differences_data: pd.DataFrame
) -> tuple:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = TRAINING_PARAMS['min_lr']

    for epoch in range(TRAINING_PARAMS['initial_epochs']):
        total_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}",
            unit="batch",
            leave=False
        )
        for inputs, targets, timestamps in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape != targets.shape:
                logging.error(f"Output shape {outputs.shape} does not match target shape {targets.shape}")
                continue

            loss = criterion(outputs, targets)

            if timestamps is not None:
                target_timestamps = timestamps.cpu().tolist()
                diff_data = differences_data[differences_data['timestamp'].isin(target_timestamps)]
                if not diff_data.empty:
                    diff_tensor = torch.tensor(
                        diff_data[list(MODEL_FEATURES.keys())].values,
                        dtype=torch.float32
                    ).to(device)
                    loss += criterion(outputs, diff_tensor)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']} - Loss: {avg_loss:.4f}")
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
            save_model(model, optimizer, MODEL_FILENAME)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= n_epochs_stop:
                logging.info("Early stopping.")
                break
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                logging.info(f"Learning rate decreased to: {param_group['lr']}")

    return model, optimizer

def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, filepath: str) -> None:
    try:
        ensure_directory_exists(filepath)
        # Сохранение state_dict модели и оптимизатора отдельно
        torch.save(model.state_dict(), filepath)
        optimizer_filepath = filepath.replace('.pth', '_optimizer.pth')
        torch.save(optimizer.state_dict(), optimizer_filepath)
        logging.info(f"Model and optimizer saved: {filepath}, {optimizer_filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device: torch.device) -> None:
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            # Загрузка state_dict модели
            state_dict = torch.load(filepath, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            # Загрузка state_dict оптимизатора
            optimizer_filepath = filepath.replace('.pth', '_optimizer.pth')
            if os.path.exists(optimizer_filepath):
                optimizer_state_dict = torch.load(optimizer_filepath, map_location=device)
                optimizer.load_state_dict(optimizer_state_dict)
            logging.info(f"Model and optimizer loaded: {filepath}, {optimizer_filepath}")
        else:
            logging.info(f"Model not found, создаётся новая модель: {filepath}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def predict_future_price(model: nn.Module, last_sequence: torch.Tensor, data_processor: DataProcessor) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        if last_sequence.dim() == 2:
            last_sequence = last_sequence.unsqueeze(0)
        input_sequence = last_sequence.to(next(model.parameters()).device)
        predictions = model(input_sequence).cpu().numpy()
        numeric_features = data_processor.numerical_columns
        categorical_features = data_processor.categorical_columns

        predictions_df = pd.DataFrame(predictions, columns=numeric_features + categorical_features)
        scaled_predictions = data_processor.scaler.inverse_transform(predictions_df[numeric_features])

        for col in categorical_features:
            scaled_predictions[col] = predictions_df[col].astype(int)
            le = data_processor.label_encoders.get(col)
            if le:
                scaled_predictions[col] = le.inverse_transform(scaled_predictions[col])

        scaled_predictions = scaled_predictions.clip(lower=0)
        scaled_predictions['symbol'] = TARGET_SYMBOL
        scaled_predictions['interval'] = PREDICTION_MINUTES

    return scaled_predictions

def fill_missing_predictions_to_csv(filename: str, model: nn.Module, data_processor: DataProcessor) -> Optional[pd.DataFrame]:
    data_fetcher = DataFetcher()
    latest_binance_timestamp = data_fetcher.get_latest_timestamp(TARGET_SYMBOL)
    if latest_binance_timestamp is None:
        logging.warning("Latest timestamp not found in Binance archive.")
        return None

    interval_key = get_interval(PREDICTION_MINUTES)
    if interval_key is None:
        logging.error(f"Interval for {PREDICTION_MINUTES} minutes not found.")
        return None

    prediction_milliseconds = INTERVAL_MAPPING[interval_key]['milliseconds']
    next_prediction_timestamp = latest_binance_timestamp + prediction_milliseconds

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        existing_df = pd.read_csv(filename)
        latest_prediction_time = existing_df['timestamp'].max()
        if latest_prediction_time >= next_prediction_timestamp:
            logging.info("No new predictions required.")
            return existing_df
    else:
        existing_df = pd.DataFrame(columns=list(MODEL_FEATURES.keys()) + ['timestamp', 'symbol', 'interval'])

    logging.info(f"Creating prediction for timestamp: {next_prediction_timestamp}")
    sequence = data_fetcher.get_sequence_for_timestamp(
        latest_binance_timestamp,
        TARGET_SYMBOL
    )
    if sequence is not None:
        predictions = predict_future_price(model, sequence, data_processor)
        predictions['timestamp'] = next_prediction_timestamp
        predictions['symbol'] = TARGET_SYMBOL
        predictions['interval'] = PREDICTION_MINUTES
        existing_df = pd.concat([predictions, existing_df], ignore_index=True)
        ensure_directory_exists(filename)
        existing_df.to_csv(filename, index=False)
        logging.info(f"Predictions saved: {filename}")
        return existing_df

    logging.warning("Data sequence for prediction not found.")
    return existing_df

def print_combined_row(current_row: pd.DataFrame, difference_row: pd.Series, predicted_next_row: pd.Series) -> None:
    print("Current vs Predicted")
    print("{:<20} {:<20} {:<20} {:<20}".format("Field", "Current Value", "Difference", "Next Predicted"))
    for col in MODEL_FEATURES:
        current_value = str(current_row[col].values[0]) if not current_row.empty else "N/A"
        difference_value = str(difference_row[col]) if col in difference_row else "N/A"
        predicted_value = str(predicted_next_row[col]) if col in predicted_next_row else "N/A"
        print("{:<20} {:<20} {:<20} {:<20}".format(col, current_value, difference_value, predicted_value))

def get_interval(minutes: int) -> Optional[str]:
    for key, config in INTERVAL_MAPPING.items():
        if config['minutes'] == minutes:
            return key
    logging.error(f"Interval for {minutes} minutes not found in INTERVAL_MAPPING.")
    return None

class DataFetcher:
    def __init__(self):
        self.download_data = DownloadData()
        self.combined_path = PATHS['combined_dataset']
        self.predictions_path = PATHS['predictions']
        self.differences_path = PATHS['differences']

    def load_data(self) -> tuple:
        data_processor = DataProcessor()
        combined_data = pd.read_csv(self.combined_path)
        combined_data = data_processor.preprocess_binance_data(combined_data)

        if os.path.exists(self.predictions_path):
            predictions_data = pd.read_csv(self.predictions_path)
            predictions_data = data_processor.preprocess_binance_data(predictions_data)
        else:
            predictions_data = pd.DataFrame(columns=RAW_FEATURES.keys())

        if os.path.exists(self.differences_path):
            differences_data = pd.read_csv(self.differences_path)
            differences_data = data_processor.preprocess_binance_data(differences_data)
        else:
            differences_data = pd.DataFrame(columns=RAW_FEATURES.keys())

        return combined_data, predictions_data, differences_data

    def get_latest_value(self, target_symbol: str) -> pd.DataFrame:
        return self.download_data.get_latest_price(target_symbol, PREDICTION_MINUTES)

    def get_latest_timestamp(self, target_symbol: str) -> Optional[int]:
        latest_row = self.download_data.get_latest_price(target_symbol, PREDICTION_MINUTES)
        if latest_row.empty:
            return None
        return int(latest_row['timestamp'].iloc[0])

    def get_sequence_for_timestamp(self, timestamp: int, target_symbol: str) -> Optional[torch.Tensor]:
        data_processor = DataProcessor()
        combined_data = pd.read_csv(self.combined_path)
        combined_data = data_processor.preprocess_binance_data(combined_data)
        combined_data = data_processor.prepare_data(combined_data)

        # Получаем числовой код символа
        if 'symbol' in data_processor.label_encoders:
            symbol_code = data_processor.label_encoders['symbol'].transform(pd.Series([target_symbol])).iloc[0]
        else:
            logging.error("Label encoder for 'symbol' not found.")
            return None

        interval = PREDICTION_MINUTES

        filtered_df = combined_data[
            (combined_data['symbol'] == symbol_code) &
            (combined_data['interval'] == interval) &
            (combined_data['timestamp'] <= timestamp)
        ].sort_values('timestamp')

        if len(filtered_df) >= SEQ_LENGTH:
            sequence = filtered_df.iloc[-SEQ_LENGTH:]
            sequence_values = sequence[data_processor.numerical_columns + data_processor.categorical_columns].values
            return torch.tensor(sequence_values, dtype=torch.float32)
        else:
            logging.warning("Недостаточно данных после фильтрации для создания последовательности.")
        return None

    def get_difference_row(self, current_time: int, symbol: str) -> pd.Series:
        if symbol not in SYMBOL_MAPPING:
            logging.error(f"Symbol {symbol} not found in SYMBOL_MAPPING")
            return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())
        if not os.path.exists(self.differences_path):
            return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())
        differences_data = pd.read_csv(self.differences_path)
        differences_data = DataProcessor().preprocess_binance_data(differences_data)
        difference_row = differences_data[
            (differences_data['symbol'] == symbol) &
            (differences_data['interval'] == PREDICTION_MINUTES) &
            (differences_data['timestamp'] == current_time)
        ]
        if not difference_row.empty:
            return difference_row.iloc[0]
        return pd.Series([None] * len(RAW_FEATURES), index=RAW_FEATURES.keys())

def main() -> None:
    device = get_device()
    model = EnhancedBiLSTMModel(
        input_size=MODEL_PARAMS['input_size'],
        hidden_layer_size=MODEL_PARAMS['hidden_layer_size'],
        num_layers=MODEL_PARAMS['num_layers'],
        dropout=MODEL_PARAMS['dropout'],
        embedding_dim=MODEL_PARAMS['embedding_dim'],
        num_symbols=MODEL_PARAMS['num_symbols']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])

    # Удалены вызовы ensure_file_exists(PATHS['predictions']) и ensure_file_exists(PATHS['differences'])

    load_model(model, optimizer, MODEL_FILENAME, device)

    data_fetcher = DataFetcher()
    combined_data, predictions_data, differences_data = data_fetcher.load_data()

    data_processor = DataProcessor()
    combined_data = data_processor.prepare_data(combined_data)

    if not predictions_data.empty:
        predictions_data = data_processor.prepare_data(predictions_data)
    if not differences_data.empty:
        differences_data = data_processor.prepare_data(differences_data)

    tensor_dataset = data_processor.prepare_dataset(combined_data, SEQ_LENGTH)
    dataloader = create_dataloader(tensor_dataset, TRAINING_PARAMS['batch_size'])

    model, optimizer = train_and_save_model(model, dataloader, device, differences_data)

    current_time = data_fetcher.get_latest_timestamp(TARGET_SYMBOL)
    if current_time is None:
        logging.error("No data found for the specified symbol and interval.")
        return

    saved_prediction = fill_missing_predictions_to_csv(PATHS['predictions'], model, data_processor)
    if saved_prediction is not None and not saved_prediction.empty:
        logging.info(f"Predictions updated: {PATHS['predictions']}")

    current_value_row = data_fetcher.get_latest_value(TARGET_SYMBOL)
    latest_prediction_row = saved_prediction.iloc[0] if saved_prediction is not None and not saved_prediction.empty else pd.Series({})
    difference_row = data_fetcher.get_difference_row(current_time, TARGET_SYMBOL)
    print_combined_row(current_value_row, difference_row, latest_prediction_row)

if __name__ == "__main__":
    while True:
        main()
