import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table

from config import *
from download_data import DownloadData
from utils import *
from visualize import *

console = Console()
MODEL_FILENAME = os.path.join(PATHS['models_dir'], f'enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth')
download_data = DownloadData()

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
    def __init__(self, input_size, hidden_layer_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, input_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)
        return predictions

def get_device():
    """Определяет и возвращает доступное устройство (CPU или GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_interval(minutes):
    """Возвращает интервал на основе заданного количества минут."""
    return next(v['interval'] for k, v in INTERVALS_PERIODS.items() if v['minutes'] == minutes)

def prepare_dataset(csv_file, seq_length=SEQ_LENGTH, target_symbol=TARGET_SYMBOL):
    df = pd.read_csv(csv_file)
    df = df[df['symbol'] == target_symbol].sort_values('timestamp')
    df = preprocess_binance_data(df)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(df[FEATURE_NAMES]), columns=FEATURE_NAMES, index=df.index)
    
    sequences, labels = [], []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data.values[i:i+seq_length].astype(np.float64))
        labels.append(scaled_data.values[i+seq_length].astype(np.float64))
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Проверка размерностей
    console.print(f"sequences shape: {sequences.shape}", style="bold blue")
    console.print(f"labels shape: {labels.shape}", style="bold blue")
    
    return TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(labels)), scaler, df

from tqdm import tqdm

def train_model(model, dataloader, device, differences_data):
    """Обучает модель на предоставленных данных."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    criterion = nn.MSELoss()

    for epoch in range(TRAINING_PARAMS['initial_epochs']):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAINING_PARAMS['initial_epochs']}", unit="batch")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            target_timestamps = targets[:, -1].cpu().numpy()
            differences = differences_data[differences_data['timestamp'].isin(target_timestamps)]
            if not differences.empty:
                differences = torch.FloatTensor(differences[FEATURE_NAMES].values).to(device)
                loss += criterion(outputs, differences)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{TRAINING_PARAMS['initial_epochs']}, Loss: {avg_loss:.4f}")
    
    return model

def predict_future_price(model, last_sequence, scaler, steps=1):
    model.eval()
    with torch.no_grad():
        # Ensure the input sequence has 3 dimensions: (batch_size, seq_length, input_size)
        if last_sequence.dim() == 2:
            last_sequence = last_sequence.unsqueeze(0)
        
        input_sequence = last_sequence.to(next(model.parameters()).device)
        predictions = model(input_sequence)
        predicted_data = scaler.inverse_transform(predictions.cpu().numpy().astype(np.float64))
        predicted_data = np.abs(predicted_data)
    return predicted_data

def save_predictions_to_csv(predictions, filename, current_time):
    """Сохраняет предсказания в CSV файл."""
    df = pd.DataFrame(predictions, columns=FEATURE_NAMES)
    prediction_milliseconds = next(v['milliseconds'] for k, v in INTERVALS_PERIODS.items() if v['minutes'] == PREDICTION_MINUTES)
    next_interval = (current_time // prediction_milliseconds + 1) * prediction_milliseconds
    df['timestamp'] = next_interval
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = get_interval(PREDICTION_MINUTES)
    df = df[DATASET_COLUMNS]
    
    if not pd.io.common.file_exists(filename):
        df.to_csv(filename, index=False, float_format='%.10f')
    else:
        existing_df = pd.read_csv(filename)
        # Исключаем пустые или все-NA записи
        existing_df = existing_df.dropna(how='all')
        combined_df = pd.concat([df, existing_df], ignore_index=True).sort_values('timestamp', ascending=False)
        combined_df.to_csv(filename, index=False, float_format='%.10f')
    
    return df.iloc[0]

def print_combined_row(current_row, difference_row, predicted_next_row):
    """Выводит сравнительную таблицу текущих значений, разницы предсказаний и предсказаний."""
    table = Table(title="Current vs Predicted")
    table.add_column("Field", style="cyan")
    table.add_column("Current Value", style="magenta")
    table.add_column("Difference", style="yellow")
    table.add_column("Predicted next", style="green")
    
    # Добавляем строку с удобным для чтения временем
    table.add_row(
        "Readable Time*",
        timestamp_to_readable_time(current_row['timestamp'].iloc[0]) if not current_row.empty and current_row['timestamp'].iloc[0] is not None else "N/A",
        timestamp_to_readable_time(difference_row['timestamp']) if difference_row['timestamp'] is not None else "N/A",
        timestamp_to_readable_time(predicted_next_row['timestamp']) if predicted_next_row['timestamp'] is not None else "N/A"
    )
    
    for col in DATASET_COLUMNS:
        table.add_row(
            col,
            f"{current_row[col].iloc[0]:.10f}" if not current_row.empty and isinstance(current_row[col].iloc[0], float) else str(current_row[col].iloc[0]) if not current_row.empty else "N/A",
            f"{difference_row[col]:.10f}" if isinstance(difference_row[col], float) else str(difference_row[col]) if difference_row[col] is not None else "N/A",
            f"{predicted_next_row[col]:.10f}" if isinstance(predicted_next_row[col], float) else str(predicted_next_row[col]) if predicted_next_row[col] is not None else "N/A"
        )
    
    console.print(table)

def get_predicted_prev_row(current_time: int, symbol: str) -> pd.Series:
    """Получает предыдущее предсказание для заданного времени и символа."""
    saved_predictions = pd.read_csv(PATHS['predictions']).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    predicted_prev_row = saved_predictions[
        (saved_predictions['symbol'] == symbol) &
        (saved_predictions['interval'] == interval) &
        (saved_predictions['timestamp'] == current_time)
    ]
    if not predicted_prev_row.empty:
        return predicted_prev_row.iloc[-1]
    else:
        differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
        if not differences_data.empty:
            return differences_data.iloc[-1]
        else:
            return pd.Series([None] * len(DATASET_COLUMNS), index=DATASET_COLUMNS)

def get_latest_value(data_file, target_symbol):
    """Получает последнее доступное значение для заданного символа из combined_dataset.csv."""
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    console.print(f"Filtering data for symbol: {target_symbol} and interval: {interval}", style="blue")
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        console.print(f"No data found for symbol: {target_symbol} and interval: {interval}", style="blue")
        return pd.DataFrame(columns=DATASET_COLUMNS)
    
    latest_value_row = filtered_df.iloc[0]
    return latest_value_row.to_frame().T

def get_latest_prediction(predictions_file, target_symbol):
    """Получает последнее предсказание для заданного символа."""
    df = pd.read_csv(predictions_file).sort_values('timestamp', ascending=False)
    
    interval = get_interval(PREDICTION_MINUTES)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        return pd.Series()
    
    return filtered_df.iloc[0]


def get_difference_row(current_time: int, symbol: str) -> pd.Series:
    """Получает строку разницы для заданного времени и символа из файла differences.csv."""
    differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    difference_row = differences_data[
        (differences_data['symbol'] == symbol) &
        (differences_data['interval'] == interval) &
        (differences_data['timestamp'] == current_time)
    ]
    if not difference_row.empty:
        return difference_row.iloc[0]
    else:
        return pd.Series([None] * len(DATASET_COLUMNS), index=DATASET_COLUMNS)

def save_difference_to_csv(predictions, actuals, filename, current_time):
    """Сохраняет разницу между предсказанными и фактическими значениями в CSV файл."""
    difference = actuals - predictions
    df = pd.DataFrame(difference.reshape(1, -1), columns=FEATURE_NAMES)
    df['timestamp'] = current_time
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = get_interval(PREDICTION_MINUTES)
    df = df[DATASET_COLUMNS]
    
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, float_format='%.10f')
    else:
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([df, existing_df], ignore_index=True).sort_values('timestamp', ascending=False)
        combined_df.to_csv(filename, index=False, float_format='%.10f')
    
    return df.iloc[0]

def load_training_data():
    """Загружает данные для обучения из всех трех источников."""
    binance_data = pd.read_csv(PATHS['combined_dataset']).sort_values('timestamp', ascending=False)
    
    if os.path.getsize(PATHS['predictions']) > 0:
        predictions_data = pd.read_csv(PATHS['predictions']).sort_values('timestamp', ascending=False)
    else:
        predictions_data = pd.DataFrame(columns=DATASET_COLUMNS)
    
    if os.path.getsize(PATHS['differences']) > 0:
        differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
    else:
        differences_data = pd.DataFrame(columns=DATASET_COLUMNS)
    
    return binance_data, predictions_data, differences_data

def ensure_file_exists(filepath):
    """Проверяет, существует ли файл, и создает его, если он отсутствует."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')  # Создаем пустой файл
        # Инициализация файла с правильной структурой
        if 'predictions' in filepath or 'differences' in filepath:
            df = pd.DataFrame(columns=DATASET_COLUMNS)
            df.to_csv(filepath, index=False)

def save_model(model, filepath):
    """Сохраняет модель в указанный файл."""
    torch.save(model.state_dict(), filepath)
    console.print(f"Model saved to {filepath}", style="bold green")

def load_model(model, filepath, device):
    """Загружает модель из указанного файла."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        model.load_state_dict(torch.load(filepath, map_location=device))
        console.print(f"Model loaded from {filepath}", style="bold green")
    else:
        console.print(f"No model found at {filepath}. Starting with a new model.", style="bold yellow")


def get_latest_timestamp(data_file, target_symbol, prediction_minutes):
    """Получает самый новый таймстамп для заданного символа и интервала."""
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    interval = get_interval(prediction_minutes)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        return None
    
    return filtered_df['timestamp'].max()


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levellevel)s - %(message)s')

def main():
    device = get_device()
    model = EnhancedBiLSTMModel(**MODEL_PARAMS).to(device)
    
    ensure_file_exists(PATHS['predictions'])
    ensure_file_exists(PATHS['differences'])
    ensure_file_exists(MODEL_FILENAME)
    
    load_model(model, MODEL_FILENAME, device)
    
    logging.info("Downloading latest data...")
    for symbol in SYMBOLS:
        for interval_name, interval_info in INTERVALS_PERIODS.items():
            logging.info(f"Updating data for {symbol} with interval {interval_name}...")
            download_data.update_data(symbol, interval_info['minutes'])
    
    logging.info("Preparing dataset...")
    dataset, scaler, df = prepare_dataset(PATHS['combined_dataset'])
    dataloader = DataLoader(dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
    
    logging.info("Loading differences data...")
    _, _, differences_data = load_training_data()
    
    logging.info("Training model...")
    model = train_model(model, dataloader, device, differences_data)
    save_model(model, MODEL_FILENAME)
    
    current_time = get_latest_timestamp(PATHS['combined_dataset'], TARGET_SYMBOL, PREDICTION_MINUTES)
    if current_time is None:
        logging.error("No data found for the specified symbol and interval.")
        return
    
    last_sequence = dataset[-1][0]
    predictions = predict_future_price(model, last_sequence, scaler)
    
    server_time, readable_server_time = get_current_time()
    logging.info(f"Current Binance server time: {readable_server_time}")
    
    prediction_milliseconds = next(v['milliseconds'] for k, v in INTERVALS_PERIODS.items() if v['minutes'] == PREDICTION_MINUTES)
    prediction_time = current_time + prediction_milliseconds
    
    saved_prediction = save_predictions_to_csv(predictions, PATHS['predictions'], current_time)
    
    # Загрузка самых свежих данных после сохранения предсказаний
    current_value_row = get_latest_value(PATHS['combined_dataset'], TARGET_SYMBOL)
    latest_prediction_row = get_latest_prediction(PATHS['predictions'], TARGET_SYMBOL)
    difference_row = get_difference_row(current_time, TARGET_SYMBOL)
    
    if not current_value_row.empty and not latest_prediction_row.empty:
        actuals = current_value_row[FEATURE_NAMES].values
        predictions = latest_prediction_row[FEATURE_NAMES].values
        save_difference_to_csv(predictions, actuals, PATHS['differences'], current_time)
        logging.info("Difference between current value and previous prediction saved.")
        
        # Обновляем difference_row после сохранения новых различий
        difference_row = get_difference_row(current_time, TARGET_SYMBOL)
    else:
        logging.warning("Unable to compare current value with previous prediction due to missing data.")
    
    # Выводим таблицу только после всех обновлений
    print_combined_row(current_value_row, difference_row, latest_prediction_row)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(30)
        create_visualization()
        time.sleep(30)
