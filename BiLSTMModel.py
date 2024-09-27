import logging
import os
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    SEQ_LENGTH,             # Sequence length for the model
    PREDICTION_MINUTES,     # Prediction time in minutes
    TARGET_SYMBOL,          # Target symbol for prediction
    SYMBOL_MAPPING,         # Symbol mapping
    INTERVAL_MAPPING,       # Interval mapping
    RAW_FEATURES,           # Raw features from Binance API
    MODEL_FEATURES,         # Features used by the model, including additional ones
    PATHS,                  # Paths to necessary files and directories
)

from get_binance_data import DownloadData, sort_dataframe

MODEL_VERSION = "2.0"

MODEL_PARAMS = {
    'input_size': len(MODEL_FEATURES),
    'hidden_layer_size': 256,
    'num_layers': 4,
    'dropout': 0.2,
    'embedding_dim': 128,
    'num_symbols': len(SYMBOL_MAPPING)  # Number of unique symbols
}

TRAINING_PARAMS = {
    'batch_size': 512,
    'initial_epochs': 10,
    'initial_lr': 0.0005,
    'max_epochs': 100,
    'min_lr': 0.00001,
    'use_mixed_precision': True,
    'num_workers': 8
}

MODEL_FILENAME = os.path.join(PATHS['models_dir'], f'enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth')

console = Console()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Attention(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_layer_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_out):
        attention_scores = torch.matmul(lstm_out, self.attention_weights).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector


class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, dropout, embedding_dim, num_symbols):
        super(EnhancedBiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_symbols, embedding_dim=embedding_dim)
        self.lstm_input_size = input_size - 1 + embedding_dim  # Subtract 1 for the symbol and add embedding_dim
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            hidden_layer_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, input_size)
        self.relu = nn.ReLU()
        self.apply(self._initialize_weights)

    def forward(self, x):
        numerical_data = x[:, :, :-1]
        symbols = x[:, :, -1].long()
        embeddings = self.embedding(symbols)
        lstm_input = torch.cat((numerical_data, embeddings), dim=2)

        lstm_out, _ = self.lstm(lstm_input)
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)
        predictions = self.relu(predictions)
        return predictions

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataProcessor:
    def __init__(self):
        self.scaler = CustomMinMaxScaler(feature_range=(-1, 1))
        self.label_encoders = {}
        self.categorical_columns = ['symbol']
        self.numerical_columns = [col for col in MODEL_FEATURES.keys() if col not in self.categorical_columns]

    def preprocess_binance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['timestamp'] = df['timestamp'].astype(int)
        df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
        for col, dtype in RAW_FEATURES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        return df

    def prepare_data(self, df: pd.DataFrame):
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

    def prepare_dataset(self, df: pd.DataFrame, seq_length: int):
        data_tensor = torch.tensor(df.values, dtype=torch.float32)
        sequences = data_tensor.unfold(0, seq_length, 1)
        targets = data_tensor[seq_length:]
        sequences = sequences[:targets.size(0)]
        tensor_dataset = TensorDataset(sequences, targets)
        return tensor_dataset


def ensure_file_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(MODEL_FEATURES.keys()))
        df.to_csv(filepath, index=False)


class CustomMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, data: pd.DataFrame):
        self.min = data.min()
        self.max = data.max()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data - self.min) / (self.max - self.min) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * (self.max - self.min) + self.min


class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = {}

    def fit(self, data: pd.Series):
        unique_classes = data.dropna().unique()
        self.classes_ = {cls: idx for idx, cls in enumerate(unique_classes)}

    def transform(self, data: pd.Series) -> pd.Series:
        return data.map(self.classes_).fillna(-1).astype(int)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_and_save_model(model, dataloader, device, differences_data):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = TRAINING_PARAMS['min_lr']

    for epoch in range(TRAINING_PARAMS['initial_epochs']):
        total_loss = 0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}",
            unit="batch",
            leave=False
        )

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if outputs.shape != targets.shape:
                logging.error(f"Output shape {outputs.shape} does not match target shape {targets.shape}")
                continue

            loss = criterion(outputs, targets)

            target_timestamps = targets[:, -1].cpu().tolist()
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
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= n_epochs_stop:
                logging.info("Early stopping.")
                break
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                logging.info(f"Learning rate decreased to: {param_group['lr']}")

    save_model(model, optimizer, MODEL_FILENAME)
    return model, optimizer


def save_model(model, optimizer, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, filepath)
        console.print(f"Model saved: {filepath}", style="bold green")
    except Exception as e:
        console.print(f"Error saving model: {e}", style="bold red")
        raise


def load_model(model, optimizer, filepath, device):
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.to(device)
            console.print(f"Model loaded: {filepath}", style="bold green")
        else:
            console.print(f"Model not found, creating new one: {filepath}", style="bold yellow")
    except Exception as e:
        console.print(f"Error loading model: {e}", style="bold red")
        raise


def predict_future_price(model, last_sequence, data_processor: DataProcessor):
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
            scaled_predictions[col] = predictions_df[col]

        scaled_predictions = scaled_predictions.clip(lower=0)
        scaled_predictions['symbol'] = TARGET_SYMBOL
        scaled_predictions['interval'] = PREDICTION_MINUTES

    return scaled_predictions


def fill_missing_predictions_to_csv(filename, model, data_processor: DataProcessor):
    data_fetcher = DataFetcher()
    latest_binance_timestamp = data_fetcher.get_latest_timestamp(TARGET_SYMBOL)

    if latest_binance_timestamp is None:
        logging.warning("Latest timestamp not found in Binance archive.")
        return

    interval_key = get_interval(PREDICTION_MINUTES)

    if interval_key is None:
        logging.error(f"Interval for {PREDICTION_MINUTES} minutes not found.")
        return

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
        existing_df.to_csv(filename, index=False)
        logging.info(f"Predictions saved: {filename}")
        return existing_df

    logging.warning("Data sequence for prediction not found.")
    return existing_df


def print_combined_row(current_row, difference_row, predicted_next_row):
    table = Table(title="Current vs Predicted")

    table.add_column("Field", style="cyan")
    table.add_column("Current Value", style="magenta")
    table.add_column("Difference", style="yellow")
    table.add_column("Next Predicted", style="green")

    for col in MODEL_FEATURES:
        current_value = str(current_row[col].iloc[0]) if not current_row.empty else "N/A"
        difference_value = str(difference_row[col]) if col in difference_row else "N/A"
        predicted_value = str(predicted_next_row[col]) if col in predicted_next_row else "N/A"
        table.add_row(col, current_value, difference_value, predicted_value)

    console.print(table)


def get_interval(minutes: int) -> Optional[str]:
    for key, config in INTERVAL_MAPPING.items():
        if config['minutes'] == minutes:
            return key
    console.print(f"Interval for {minutes} minutes not found in INTERVAL_MAPPING.", style="red")
    return None


class DataFetcher:
    def __init__(self):
        self.download_data = DownloadData()
        self.combined_path = PATHS['combined_dataset']
        self.predictions_path = PATHS['predictions']
        self.differences_path = PATHS['differences']

    def load_data(self):
        combined_data = pd.read_csv(self.combined_path)
        combined_data = DataProcessor().preprocess_binance_data(combined_data)
        predictions_data = pd.read_csv(self.predictions_path) if os.path.exists(self.predictions_path) else pd.DataFrame(columns=RAW_FEATURES.keys())
        differences_data = pd.read_csv(self.differences_path) if os.path.exists(self.differences_path) else pd.DataFrame(columns=RAW_FEATURES.keys())
        return combined_data, predictions_data, differences_data

    def get_latest_value(self, target_symbol: str) -> pd.DataFrame:
        return self.download_data.get_latest_price(target_symbol, PREDICTION_MINUTES)

    def get_latest_timestamp(self, target_symbol: str) -> Optional[int]:
        latest_row = self.download_data.get_latest_price(target_symbol, PREDICTION_MINUTES)
        if latest_row.empty:
            return None
        return latest_row['timestamp'].iloc[0]

    def get_sequence_for_timestamp(self, timestamp: int, target_symbol: str) -> Optional[torch.Tensor]:
        combined_data = pd.read_csv(self.combined_path)
        combined_data = DataProcessor().preprocess_binance_data(combined_data)
        filtered_df = combined_data[
            (combined_data['symbol'] == target_symbol) &
            (combined_data['interval'] == PREDICTION_MINUTES) &
            (combined_data['timestamp'] <= timestamp)
        ]

        if len(filtered_df) >= SEQ_LENGTH:
            sequence = filtered_df.sort_values('timestamp', ascending=False).head(SEQ_LENGTH)
            sequence = sequence[list(RAW_FEATURES.keys())].values[::-1]
            return torch.tensor(sequence, dtype=torch.float32)

        return None

    def get_difference_row(self, current_time: int, symbol: str) -> pd.Series:
        if symbol not in SYMBOL_MAPPING:
            console.print(f"Symbol {symbol} not found in SYMBOL_MAPPING", style="red")
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


def main():
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

    ensure_file_exists(PATHS['predictions'])
    ensure_file_exists(PATHS['differences'])

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

    if saved_prediction is not None:
        logging.info(f"Predictions updated: {PATHS['predictions']}")

    current_value_row = data_fetcher.get_latest_value(TARGET_SYMBOL)
    latest_prediction_row = saved_prediction.iloc[0] if saved_prediction is not None else pd.Series({})
    difference_row = data_fetcher.get_difference_row(current_time, TARGET_SYMBOL)

    print_combined_row(current_value_row, difference_row, latest_prediction_row)


if __name__ == "__main__":
    while True:
        main()
