import logging
import os
import pickle

import joblib
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import *
from download_data import DownloadData
from utils import *
from visualize import create_visualization

console = Console()

# Список символов и словарь для кодирования
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SYMBOL_MAPPING = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {idx: symbol for symbol, idx in SYMBOL_MAPPING.items()}

MODEL_FILENAME = os.path.join(PATHS['models_dir'], f'enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth')
download_data = DownloadData()

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
    def __init__(self, input_size, hidden_layer_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, input_size)
        self.apply(self._initialize_weights)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h_0 = torch.zeros(
            self.lstm.num_layers * 2,
            x.size(0),
            self.lstm.hidden_size
        ).to(x.device)
        c_0 = torch.zeros(
            self.lstm.num_layers * 2,
            x.size(0),
            self.lstm.hidden_size
        ).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)
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


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, dataloader, device, differences_data, val_dataloader=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = 1e-6

    for epoch in range(TRAINING_PARAMS['initial_epochs']):
        total_loss = 0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}",
            unit="batch"
        )
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            assert outputs.shape == targets.shape, (
                f"Output shape {outputs.shape} не совпадает с target shape {targets.shape}"
            )
            loss = criterion(outputs, targets)
            target_timestamps = targets[:, -1].cpu().tolist()
            differences = differences_data[differences_data['timestamp'].isin(target_timestamps)]
            if not differences.empty:
                differences = torch.tensor(
                    differences[list(FEATURE_NAMES.keys())].values.tolist(),
                    dtype=torch.float32
                ).to(device)
                loss += criterion(outputs, differences)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}, Loss: {avg_loss:.4f}")

        if val_dataloader:
            val_loss = validate_model(model, val_dataloader, device, criterion)
            model.train()
            logging.info(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_stop:
                    logging.info("Early stopping")
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                    logging.info(f"Learning rate reduced to {param_group['lr']}")

    return model, optimizer


def load_and_prepare_data():
    combined_data = pd.read_csv(PATHS['combined_dataset'])
    predictions_data = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
    differences_data = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))

    if os.path.exists(PATHS['predictions']):
        predictions_data = pd.read_csv(PATHS['predictions'])
    if os.path.exists(PATHS['differences']):
        differences_data = pd.read_csv(PATHS['differences'])

    for df in [combined_data, predictions_data, differences_data]:
        if not df.empty:
            df = preprocess_binance_data(df)

    logging.debug(f"Combined Data: {combined_data}")
    logging.debug(f"Predictions Data: {predictions_data}")
    logging.debug(f"Differences Data: {differences_data}")

    return combined_data, predictions_data, differences_data


class CustomMinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, data: pd.DataFrame):
        self.min = data.min()
        self.max = data.max()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_scaled = (data - self.min) / (self.max - self.min)
        data_scaled = data_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return data_scaled

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data_original = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        data_original = data_original * (self.max - self.min) + self.min
        return data_original


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


def prepare_dataset(df, seq_length, scaler: CustomMinMaxScaler):
    df['symbol'] = df['symbol'].map(SYMBOL_MAPPING).fillna(-1).astype(int)
    assert df['symbol'].dtype == int, "Столбец 'symbol' должен быть числовым"
    
    # Сохраняем столбец 'symbol'
    symbol_column = df['symbol']
    
    numeric_columns = [
        col for col, dtype in FEATURE_NAMES.items() if dtype in [float, int] and col != 'symbol'
    ]

    scaled_numeric_data = scaler.transform(df[numeric_columns])

    # Добавляем 'symbol' обратно к масштабированным данным
    scaled_data = pd.concat([scaled_numeric_data, symbol_column], axis=1)
    
    # Убедимся, что порядок столбцов соответствует FEATURE_NAMES
    scaled_data = scaled_data[list(FEATURE_NAMES.keys())]

    sequences = []
    targets = []
    for i in range(len(scaled_data) - seq_length):
        sequence = scaled_data.iloc[i:i + seq_length].values.tolist()
        target = scaled_data.iloc[i + seq_length].values.tolist()
        sequences.append(sequence)
        targets.append(target)

    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    return TensorDataset(sequences_tensor, targets_tensor)


def validate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_model(model, optimizer, filepath):
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, filepath)
        console.print(f"Модель успешно сохранена по пути: {filepath}", style="bold green")
    except Exception as e:
        console.print(f"Ошибка при сохранении модели: {e}", style="bold red")
        raise e


def load_model(model, optimizer, filepath, device):
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.to(device)
            console.print(f"Модель успешно загружена из: {filepath}", style="bold green")
        else:
            console.print(f"Файл модели не найден по пути: {filepath}. Создаётся новая модель.", style="bold yellow")
    except (AttributeError, pickle.UnpicklingError) as e:
        console.print(
            f"Ошибка при загрузке модели: {e}. Удаление поврежденного файла и создание новой модели.",
            style="bold red"
        )
        if os.path.exists(filepath):
            os.remove(filepath)
        model.apply(model._initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    except Exception as e:
        console.print(f"Ошибка при загрузке модели: {e}", style="bold red")
        raise e


def predict_future_price(model, last_sequence, scaler: CustomMinMaxScaler):
    model.eval()
    with torch.no_grad():
        if last_sequence.dim() == 2:
            last_sequence = last_sequence.unsqueeze(0)

        input_sequence = last_sequence.to(next(model.parameters()).device)
        predictions = model(input_sequence)

        predictions_np = predictions.cpu().numpy()
        predictions_df = pd.DataFrame(predictions_np, columns=list(FEATURE_NAMES.keys()))

        scaled_predictions = scaler.inverse_transform(predictions_df)
        predictions_df = pd.DataFrame(scaled_predictions, columns=list(FEATURE_NAMES.keys()))

        predictions_df['symbol'] = TARGET_SYMBOL
        predictions_df['interval'] = PREDICTION_MINUTES

    return predictions_df


def get_latest_prediction(predictions_file, target_symbol):
    if not os.path.exists(predictions_file):
        return pd.Series(dtype='float64')
    df = pd.read_csv(predictions_file)
    df = preprocess_binance_data(df)
    filtered_df = df[
        (df['symbol'] == target_symbol) &
        (df['interval'] == PREDICTION_MINUTES)
    ]
    if filtered_df.empty:
        return pd.Series(dtype='float64')
    latest_prediction = filtered_df.sort_values('timestamp', ascending=False).iloc[0]
    logging.debug(f"Latest Prediction: {latest_prediction}")
    return latest_prediction


def fill_missing_predictions_to_csv(filename, model, scaler: CustomMinMaxScaler, df=None):
    latest_binance_timestamp = get_latest_timestamp(
        PATHS['combined_dataset'],
        TARGET_SYMBOL,
        PREDICTION_MINUTES
    )
    if latest_binance_timestamp is None or pd.isna(latest_binance_timestamp):
        logging.warning("Не удалось получить последний timestamp из архива Binance.")
        return

    prediction_milliseconds = INTERVALS_PERIODS[get_interval(PREDICTION_MINUTES)]['milliseconds']
    next_prediction_timestamp = latest_binance_timestamp + prediction_milliseconds

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        existing_df = pd.read_csv(filename)
        existing_df['timestamp'] = existing_df['timestamp'].astype(int)
        latest_prediction_time = existing_df['timestamp'].max()

        if latest_prediction_time >= next_prediction_timestamp:
            logging.info("Нет необходимости в новых предсказаниях.")
            return existing_df
    else:
        # Определите все необходимые столбцы, включая 'symbol', 'timestamp' и 'interval'
        existing_df = pd.DataFrame(columns=list(FEATURE_NAMES.keys()) + ['timestamp', 'symbol', 'interval'])
        latest_prediction_time = None

    logging.info(f"Создание предсказания для timestamp: {next_prediction_timestamp}")

    sequence = get_sequence_for_timestamp(
        latest_binance_timestamp,
        TARGET_SYMBOL,
        PREDICTION_MINUTES
    )
    if sequence is not None:
        sequence_df = pd.DataFrame(sequence, columns=list(FEATURE_NAMES.keys()))
        sequence_df['symbol'] = SYMBOL_MAPPING.get(sequence_df['symbol'].iloc[0], 0)

        numerical_columns = [col for col in FEATURE_NAMES if FEATURE_NAMES[col] != str]
        sequence_df[numerical_columns] = sequence_df[numerical_columns].astype("float32")

        # Изменено: Преобразование в списки перед созданием тензора
        predictions = predict_future_price(model, torch.tensor(sequence_df.values.tolist(), dtype=torch.float32), scaler)
        df_predictions = pd.DataFrame(predictions, columns=list(FEATURE_NAMES.keys()))

        # Установите 'symbol' вручную, не используя предсказание модели
        df_predictions['timestamp'] = next_prediction_timestamp
        df_predictions['symbol'] = TARGET_SYMBOL  # Присваиваем строковое значение
        df_predictions['interval'] = PREDICTION_MINUTES

        for col, dtype in FEATURE_NAMES.items():
            if col in df_predictions.columns:
                if dtype == str:
                    df_predictions[col] = df_predictions[col].astype(dtype)
                else:
                    df_predictions[col] = df_predictions[col].astype(dtype)
                logging.debug(
                    f"Column {col} value: {df_predictions[col].iloc[0]}, "
                    f"type: {df_predictions[col].dtype}"
                )
                if dtype == str:
                    assert pd.api.types.is_string_dtype(df_predictions[col]), (
                        f"Column {col} имеет неверный тип {df_predictions[col].dtype}, ожидается {dtype}"
                    )
                else:
                    assert df_predictions[col].dtype == dtype, (
                        f"Column {col} имеет неверный тип {df_predictions[col].dtype}, ожидается {dtype}"
                    )

        # Обеспечьте наличие всех столбцов перед конкатенацией
        df_predictions = df_predictions.reindex(columns=existing_df.columns, fill_value=None)

        existing_df = pd.concat([existing_df, df_predictions], ignore_index=True)
        existing_df['symbol'] = existing_df['symbol'].map(ID_TO_SYMBOL).fillna(TARGET_SYMBOL)
        existing_df.to_csv(filename, index=False)
        logging.info(f"Предсказания сохранены в файл: {filename}")
        return existing_df

    existing_df['symbol'] = existing_df['symbol'].map(ID_TO_SYMBOL)
    existing_df.to_csv(filename, index=False)
    logging.info(f"Предсказания сохранены в файл: {filename}")
    return existing_df


def prepare_data(df):
    categorical_columns = [col for col, dtype in FEATURE_NAMES.items() if dtype == str]
    label_encoders = {}

    for col in categorical_columns:
        le = CustomLabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numerical_columns = [col for col in FEATURE_NAMES if FEATURE_NAMES[col] != str]
    df[numerical_columns] = df[numerical_columns].astype(float)

    return df, label_encoders


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def print_combined_row(current_row, difference_row, predicted_next_row):
    table = Table(title="Current vs Predicted")
    table.add_column("Field", style="cyan")
    table.add_column("Current Value", style="magenta")
    table.add_column("Difference", style="yellow")
    table.add_column("Predicted next", style="green")
    for col in FEATURE_NAMES:
        current_value = str(current_row[col].iloc[0]) if not current_row.empty else "N/A"
        difference_value = str(difference_row[col]) if 'col' in difference_row else "N/A"
        predicted_value = str(predicted_next_row[col]) if col in predicted_next_row else "N/A"
        table.add_row(col, current_value, difference_value, predicted_value)
    console.print(table)


def main():
    device = get_device()
    model = EnhancedBiLSTMModel(**MODEL_PARAMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])

    ensure_file_exists(PATHS['predictions'])
    ensure_file_exists(PATHS['differences'])
    ensure_file_exists(MODEL_FILENAME)

    load_model(model, optimizer, MODEL_FILENAME, device)

    logging.info("Downloading latest data...")
    for symbol in SYMBOLS:
        for interval_name, interval_info in INTERVALS_PERIODS.items():
            logging.info(f"Updating data for {symbol} with interval {interval_name}...")
            download_data.update_data(symbol, interval_info['minutes'])

    logging.info("Loading and preparing data...")
    combined_data, predictions_data, differences_data = load_and_prepare_data()

    logging.info("Encoding categorical data...")
    combined_data, label_encoders = prepare_data(combined_data)
    predictions_data, _ = prepare_data(predictions_data)
    differences_data, _ = prepare_data(differences_data)

    logging.info("Preparing scaler...")
    scaler = CustomMinMaxScaler(feature_range=(-1, 1))
    numeric_features = [col for col, dtype in FEATURE_NAMES.items() if dtype in [float, int]]
    scaler.fit(combined_data[numeric_features])

    logging.info("Preparing dataset...")
    tensor_dataset = prepare_dataset(combined_data, SEQ_LENGTH, scaler)

    dataloader = create_dataloader(tensor_dataset, TRAINING_PARAMS['batch_size'])

    logging.info("Training model...")
    model, optimizer = train_model(model, dataloader, device, differences_data)
    save_model(model, optimizer, MODEL_FILENAME)
    logging.info("Model training completed and saved.")

    current_time = get_latest_timestamp(PATHS['combined_dataset'], TARGET_SYMBOL, PREDICTION_MINUTES)
    if current_time is None:
        logging.error("No data found for the specified symbol and interval.")
        return

    readable_server_time = get_current_time()[1]
    logging.info(f"Current Binance server time: {readable_server_time}")

    saved_prediction = fill_missing_predictions_to_csv(PATHS['predictions'], model, scaler)
    logging.info(f"Saved predictions to CSV: {saved_prediction}")

    current_value_row = get_latest_value(PATHS['combined_dataset'], TARGET_SYMBOL)
    latest_prediction_row = get_latest_prediction(PATHS['predictions'], TARGET_SYMBOL)
    difference_row = get_difference_row(current_time, TARGET_SYMBOL)

    print_combined_row(current_value_row, difference_row, latest_prediction_row)
    logging.info("Завершено отображение объединённой строки.")

if __name__ == "__main__":
    while True:
        main()
