import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from rich.table import Table
from rich.console import Console

from config import *
from utils import preprocess_binance_data, get_current_time, select_scaler, timestamp_to_readable_time
from download_data import DownloadData
from gpu_util import get_device_info

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

console = Console()
MODEL_FILENAME = f'enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth'
download_data = DownloadData()

class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.price_linear = nn.Linear(hidden_layer_size * 2, 4)
        self.volume_linear = nn.Linear(hidden_layer_size * 2, 5)

    def forward(self, input_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        price_pred = self.price_linear(lstm_out[:, -1])
        volume_pred = self.volume_linear(lstm_out[:, -1])
        return price_pred, volume_pred

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataset(csv_file: str, seq_length: int = SEQ_LENGTH, target_symbol: str = TARGET_SYMBOL) -> Tuple[TensorDataset, MinMaxScaler, MinMaxScaler, int]:
    price_df = pd.read_csv(csv_file, dtype=DATA_TYPES)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
    price_df = price_df[price_df['symbol'] == target_symbol].sort_values('timestamp')
    price_df = preprocess_binance_data(price_df)

    if price_df.empty:
        raise ValueError("No valid data remaining after preprocessing")

    price_scaler = MinMaxScaler(feature_range=(-1, 1))
    volume_scaler = MinMaxScaler(feature_range=(-1, 1))

    price_features = ['open', 'high', 'low', 'close']
    volume_features = ['volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    price_scaler.fit(price_df[price_features])
    volume_scaler.fit(price_df[volume_features])

    scaled_prices = price_scaler.transform(price_df[price_features])
    scaled_volumes = volume_scaler.transform(price_df[volume_features])

    scaled_data = np.hstack((scaled_prices, scaled_volumes))

    sequences = np.array([scaled_data[i:i+seq_length] for i in range(len(scaled_data) - seq_length)])
    labels = scaled_data[seq_length:]

    if len(sequences) == 0 or len(labels) == 0:
        raise ValueError("No valid sequences or labels generated")

    return TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(labels)), price_scaler, volume_scaler, scaled_data.shape[1]

def load_model(model: nn.Module, path: str) -> Tuple[nn.Module, MinMaxScaler, MinMaxScaler]:
    full_path = os.path.join(path, MODEL_FILENAME)
    if os.path.exists(full_path):
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        price_scaler = checkpoint.get('price_scaler')
        volume_scaler = checkpoint.get('volume_scaler')
        if price_scaler is None or volume_scaler is None:
            raise ValueError("Scalers are missing in the checkpoint")
        console.print(f"Model loaded from {full_path}", style="bold green")
    else:
        raise FileNotFoundError(f"No existing model found at {full_path}. Training a new model.")
    return model, price_scaler, volume_scaler

def train_model(model: nn.Module, dataloader: DataLoader, device: torch.device, epochs: int = TRAINING_PARAMS['initial_epochs'], lr: float = TRAINING_PARAMS['initial_lr']) -> nn.Module:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    price_criterion = nn.MSELoss()
    volume_criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        total_loss = 0
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in epoch_iterator:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast():
                price_pred, volume_pred = model(inputs)
                price_loss = price_criterion(price_pred, targets[:, :4])
                volume_loss = volume_criterion(volume_pred, targets[:, 4:])
                loss = price_loss + volume_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()

            epoch_iterator.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        console.print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", style="bold green")
        scheduler.step(avg_loss)

    return model

def make_prediction(model: nn.Module, dataset: TensorDataset, price_scaler: MinMaxScaler, volume_scaler: MinMaxScaler, current_time: int):
    last_sequence = dataset[-1][0]
    predictions = predict_future_price(model, last_sequence, price_scaler, volume_scaler, current_time=current_time)
    if predictions.empty:
        raise ValueError("Predictions are empty")
    if PATHS['predictions']:
        save_predictions_to_csv(predictions, PATHS['predictions'])
        display_prediction(predictions, predictions)
    else:
        console.print("Путь к файлу предсказаний не задан", style="bold red")
    real_price = download_data.get_current_price(TARGET_SYMBOL, PREDICTION_MINUTES)
    if real_price.empty:
        raise ValueError("Не удалось получить текущую цену")
    compare_prediction_with_real(predictions, real_price)

def predict_future_price(model: nn.Module, last_sequence: torch.Tensor, price_scaler: MinMaxScaler, volume_scaler: MinMaxScaler, steps: int = 1, current_time: int = None) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        input_sequence = last_sequence.unsqueeze(0).to(model.price_linear.weight.device)
        price_pred, volume_pred = model(input_sequence)
        predicted_prices = price_scaler.inverse_transform(price_pred.cpu().numpy())
        predicted_volumes = volume_scaler.inverse_transform(volume_pred.cpu().numpy())

    predicted_data = np.hstack((predicted_prices, predicted_volumes))
    df = pd.DataFrame(predicted_data, columns=FEATURE_NAMES)

    archive_data = pd.read_csv(PATHS['combined_dataset'])
    archive_data['timestamp'] = pd.to_numeric(archive_data['timestamp'])
    latest_timestamp = archive_data['timestamp'].max()

    if current_time is None:
        if latest_timestamp is None:
            raise ValueError("Нет данных для вычисления временной метки прогноза")
        current_time = latest_timestamp
    df['timestamp'] = [current_time + 60000 * PREDICTION_MINUTES]
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = f'{PREDICTION_MINUTES}m'

    missing_columns = [col for col in DATASET_COLUMNS if col not in df.columns]
    if missing_columns:
        console.print(f"Missing columns in predictions: {missing_columns}", style="bold red")

    return df[DATASET_COLUMNS]

def compare_prediction_with_real(predictions: pd.DataFrame, real_price: pd.DataFrame):
    if predictions.empty or real_price.empty:
        console.print("Нет данных для сравнения", style="bold red")
        return

    prediction_time = predictions['timestamp'].iloc[0]
    real_price_time = real_price['timestamp'].iloc[0]
    if prediction_time == real_price_time + 60000 * PREDICTION_MINUTES:
        predicted_close = predictions['close'].iloc[0]
        real_close = real_price['close'].iloc[0]
        error = abs(predicted_close - real_close)
        console.print(f"Prediction time: {prediction_time}, Real time: {real_price_time}, Predicted close: {predicted_close}, Real close: {real_close}, Error: {error}", style="bold blue")
    else:
        console.print(f"Временные метки прогноза ({prediction_time}) и реальной цены ({real_price_time}) не совпадают", style="bold red")

def update_dataset(existing_dataset: TensorDataset, new_data: pd.DataFrame, price_scaler: MinMaxScaler, volume_scaler: MinMaxScaler, max_size: int = 10000) -> TensorDataset:
    old_sequences, old_labels = existing_dataset.tensors

    console.print("New data before preprocessing:")
    console.print(new_data.head())

    new_data = preprocess_binance_data(new_data)

    console.print("New data after preprocessing:")
    console.print(new_data.head())

    if new_data.empty:
        raise ValueError("New data is empty after preprocessing")

    price_features = ['open', 'high', 'low', 'close']
    volume_features = ['volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']

    scaled_prices = price_scaler.transform(new_data[price_features])
    scaled_volumes = volume_scaler.transform(new_data[volume_features])

    scaled_new_data = np.hstack((scaled_prices, scaled_volumes))
    new_sequences = np.array([scaled_new_data[i:i+SEQ_LENGTH] for i in range(max(0, len(scaled_new_data) - SEQ_LENGTH))])
    new_labels = scaled_new_data[SEQ_LENGTH:]

    if len(new_sequences) == 0 or len(new_labels) == 0:
        console.print("No valid sequences or labels generated from new data", style="bold red")
        return existing_dataset

    combined_sequences = torch.cat([old_sequences, torch.FloatTensor(new_sequences)], dim=0)
    combined_labels = torch.cat([old_labels, torch.FloatTensor(new_labels)], dim=0)
    if len(combined_sequences) > max_size:
        combined_sequences = combined_sequences[-max_size:]
        combined_labels = combined_labels[-max_size:]
    return TensorDataset(combined_sequences, combined_labels)

def display_prediction(current_price: pd.DataFrame, predictions: pd.DataFrame):
    if current_price.empty or predictions.empty:
        console.print("Нет данных для отображения", style="bold red")
        return

    table = Table(title=f"{TARGET_SYMBOL} Prediction", title_style="bold cyan", title_justify="left")
    table.add_column("Type", style="cyan")
    table.add_column("Timestamp", style="magenta")
    for feature in FEATURE_NAMES:
        table.add_column(feature.capitalize(), justify="right", style="green")

    current_row = current_price.iloc[0]
    prediction_row = predictions.iloc[0]

    table.add_row(
        "Current",
        datetime.fromtimestamp(current_row['timestamp'] / 1000).strftime(DATETIME_FORMAT),
        *[f"{float(current_row[feature]):.2f}" for feature in FEATURE_NAMES]
    )
    table.add_row(
        "Prediction",
        datetime.fromtimestamp(prediction_row['timestamp'] / 1000).strftime(DATETIME_FORMAT),
        *[f"{float(prediction_row[feature]):.2f}" for feature in FEATURE_NAMES]
    )
    console.print(table)

def save_predictions_to_csv(predictions: pd.DataFrame, filename: str):
    if not filename:
        raise ValueError("Filename for saving predictions is not provided")

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    missing_columns = [col for col in DATASET_COLUMNS if col not in predictions.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in predictions: {missing_columns}")

    predictions = predictions.reindex(columns=DATASET_COLUMNS)
    predictions = predictions.astype(DATA_TYPES)
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            predictions.to_csv(f, index=False, header=True, float_format='%.6f')
        else:
            predictions.to_csv(f, index=False, header=False, float_format='%.6f')

def check_previous_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PATHS['predictions'])
    actual_prices = pd.read_csv(PATHS['combined_dataset'])
    predictions['timestamp'] = pd.to_numeric(predictions['timestamp'])
    actual_prices['timestamp'] = pd.to_numeric(actual_prices['timestamp'])
    predictions = predictions.sort_values(['timestamp', 'symbol'])
    actual_prices = actual_prices.sort_values(['timestamp', 'symbol'])
    merged = pd.merge_asof(predictions, actual_prices, on='timestamp', by='symbol', suffixes=('_pred', '_real'))
    merged['error'] = abs(merged['close_pred'] - merged['close_real'])
    return merged

def update_training_params(epochs: int, lr: float) -> Tuple[int, float]:
    if os.path.exists(PATHS['predictions']):
        previous_predictions = check_previous_predictions()
        if len(previous_predictions) > PREDICTION_MINUTES:
            mean_error = previous_predictions['error'].mean()
            if mean_error > 0.05:
                epochs = min(TRAINING_PARAMS['max_epochs'], epochs + 5)
                lr = max(TRAINING_PARAMS['min_lr'], lr * 0.9)
            else:
                epochs = TRAINING_PARAMS['initial_epochs']
                lr = TRAINING_PARAMS['initial_lr']
    return epochs, lr

def save_model(model: nn.Module, price_scaler: MinMaxScaler, volume_scaler: MinMaxScaler, path: str):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, MODEL_FILENAME)
    torch.save({
        'model_state_dict': model.state_dict(),
        'price_scaler': price_scaler,
        'volume_scaler': volume_scaler
    }, full_path)
    console.print(f"Model saved as {full_path}", style="bold green")

def main():
    try:
        device = get_device()
        console.print(get_device_info(), style="bold green")
        os.makedirs(PATHS['visualization_dir'], exist_ok=True)
        console.print("Downloading latest data...", style="bold green")
        for symbol in SYMBOLS:
            for interval in INTERVALS_PERIODS.keys():
                download_data.update_data(symbol, INTERVALS_PERIODS[interval]['minutes'])
        console.print("Initializing model and dataset...", style="bold green")
        model = EnhancedBiLSTMModel(**MODEL_PARAMS).to(device)
        dataset, fitted_price_scaler, fitted_volume_scaler, _ = prepare_dataset(PATHS['combined_dataset'])
        model, loaded_price_scaler, loaded_volume_scaler = load_model(model, PATHS['models_dir'])
        price_scaler = select_scaler(fitted_price_scaler, loaded_price_scaler)
        volume_scaler = select_scaler(fitted_volume_scaler, loaded_volume_scaler)

        console.print("Starting training...", style="bold green")
        model = train_model(model, DataLoader(dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True, pin_memory=True, num_workers=TRAINING_PARAMS['num_workers']), device)

        console.print("Starting prediction loop...", style="bold green")
        while True:
            current_time, _ = get_current_time()
            make_prediction(model, dataset, price_scaler, volume_scaler, current_time)
            new_data = download_data.get_current_price(TARGET_SYMBOL, PREDICTION_MINUTES)
            if new_data.empty:
                raise ValueError("Не удалось получить новые данные")
            dataset = update_dataset(dataset, new_data, price_scaler, volume_scaler)
            time.sleep(PREDICTION_MINUTES * 60)
    except Exception as e:
        console.print(f"An error occurred: {str(e)}", style="bold red")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

