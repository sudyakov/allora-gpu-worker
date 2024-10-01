import os
import pickle
import logging
from typing import Optional, Dict, Literal, TypedDict, Tuple, Union, List
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from get_binance_data import DownloadData, main as binance_main
SEQ_LENGTH: int = 100

class IntervalConfig(TypedDict):
    days: int
    minutes: int
    milliseconds: int

IntervalKey = Literal["1m", "5m", "15m"]

SYMBOL_MAPPING: Dict[str, int] = {
    "ETHUSDT": 0,
    "BTCUSDT": 1,
}

TARGET_SYMBOL: str = "ETHUSDT"
PREDICTION_MINUTES: int = 5

INTERVAL_MAPPING: Dict[IntervalKey, IntervalConfig] = {
    "1m": {"days": 7, "minutes": 1, "milliseconds": 60000},
    "5m": {"days": 14, "minutes": 5, "milliseconds": 300000},
    "15m": {"days": 28, "minutes": 15, "milliseconds": 900000},
}

RAW_FEATURES: Dict[str, type] = {
    'symbol': str,
    'interval_str': str,
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
    "hour": int,
    "dayofweek": int,
    "sin_hour": float,
    "cos_hour": float,
    "sin_day": float,
    "cos_day": float,
}

PATHS: Dict[str, str] = {
    'combined_dataset': 'data/combined_dataset.csv',
    'predictions': 'data/predictions.csv',
    'differences': 'data/differences.csv',
    'models_dir': 'models',
    'visualization_dir': 'visualizations',
    'data_dir': 'data',
}

MODEL_VERSION = "2.0"

class ModelParams(TypedDict):
    input_size: int
    hidden_layer_size: int
    num_layers: int
    dropout: float
    embedding_dim: int
    num_symbols: int
    num_intervals: int
    timestamp_embedding_dim: int

MODEL_PARAMS: ModelParams = {
    "input_size": len(MODEL_FEATURES) - 1,
    "hidden_layer_size": 256,
    "num_layers": 4,
    "dropout": 0.2,
    "embedding_dim": 128,
    "num_symbols": len(SYMBOL_MAPPING),
    "num_intervals": 3,
    "timestamp_embedding_dim": 64,
}

class TrainingParams(TypedDict):
    batch_size: int
    initial_epochs: int
    initial_lr: float
    max_epochs: int
    min_lr: float
    use_mixed_precision: bool
    num_workers: int

TRAINING_PARAMS: TrainingParams = {
    "batch_size": 512,
    "initial_epochs": 5,
    "initial_lr": 0.0005,
    "max_epochs": 100,
    "min_lr": 0.00001,
    "use_mixed_precision": True,
    "num_workers": 8,
}

MODEL_FILENAME = os.path.join(PATHS["models_dir"], f"enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth")
DATA_PROCESSOR_FILENAME = os.path.join(PATHS["models_dir"], f"data_processor_{TARGET_SYMBOL}_v{MODEL_VERSION}.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        numerical_columns: List[str],
        categorical_columns: List[str],
        column_name_to_index: Dict[str, int],
        input_size: int = MODEL_PARAMS["input_size"],
        hidden_layer_size: int = MODEL_PARAMS["hidden_layer_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        embedding_dim: int = MODEL_PARAMS["embedding_dim"],
        num_symbols: int = MODEL_PARAMS["num_symbols"],
        num_intervals: int = MODEL_PARAMS["num_intervals"],
        timestamp_embedding_dim: int = MODEL_PARAMS["timestamp_embedding_dim"],
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

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataProcessor:
    def __init__(self):
        self.scaler = CustomMinMaxScaler(feature_range=(0, 1))
        self.label_encoders: Dict[str, 'CustomLabelEncoder'] = {}
        self.categorical_columns: List[str] = ["symbol", "interval_str"]
        self.numerical_columns: List[str] = [
            col for col in MODEL_FEATURES.keys() if col not in self.categorical_columns and col != 'timestamp'
        ]

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
            df[col] = df[col].astype(MODEL_FEATURES[col])

        self.scaler.fit(df[self.numerical_columns])
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])

        df['timestamp'] = df['timestamp'].astype('int64')

        additional_features = ["hour", "dayofweek", "sin_hour", "cos_hour", "sin_day", "cos_day"]
        df = df[list(RAW_FEATURES.keys()) + additional_features]
        logging.info(f"Порядок колонок после fit_transform: {df.columns.tolist()}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            le = self.label_encoders.get(col)
            if le is None:
                logging.error(f"LabelEncoder для колонки {col} не обучен.")
                raise ValueError(f"LabelEncoder для колонки {col} не обучен.")
            df[col] = le.transform(df[col])

        for col in self.numerical_columns:
            df[col] = df[col].astype(MODEL_FEATURES[col])

        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])

        df['timestamp'] = df['timestamp'].astype('int64')

        additional_features = ["hour", "dayofweek", "sin_hour", "cos_hour", "sin_day", "cos_day"]
        df = df[list(RAW_FEATURES.keys()) + additional_features]
        logging.info(f"Порядок колонок после transform: {df.columns.tolist()}")
        return df

    def prepare_dataset(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH) -> TensorDataset:
        features = list(MODEL_FEATURES.keys())
        target_columns = [col for col in features if col != 'timestamp']

        data_tensor = torch.tensor(df[features].values, dtype=torch.float32)
        target_indices = [df.columns.get_loc(col) for col in target_columns]

        sequences = []
        targets = []
        for i in range(len(data_tensor) - seq_length):
            sequences.append(data_tensor[i : i + seq_length])
            targets.append(data_tensor[i + seq_length, target_indices])

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)
        tensor_dataset = TensorDataset(sequences, targets)
        return tensor_dataset


    def save(self, filepath: str) -> None:
        try:
            ensure_directory_exists(filepath)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logging.info(f"DataProcessor сохранен: {filepath}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении DataProcessor: {e}")
            raise

    @staticmethod
    def load(filepath: str) -> 'DataProcessor':
        try:
            with open(filepath, 'rb') as f:
                processor = pickle.load(f)
            logging.info(f"DataProcessor загружен: {filepath}")
            return processor
        except Exception as e:
            logging.error(f"Ошибка при загрузке DataProcessor: {e}")
            raise

def ensure_directory_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

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

def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_PARAMS["num_workers"],
    )

def train_and_save_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = TRAINING_PARAMS["min_lr"]

    for epoch in range(TRAINING_PARAMS["initial_epochs"]):
        total_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}",
            unit="batch",
            leave=False,
        )
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                logging.error(
                    f"Output shape {outputs.shape} does not match target shape {targets.shape}"
                )
                continue
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        logging.info(
            f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']} - Loss: {avg_loss:.4f}"
        )
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
            save_model(model, optimizer, MODEL_FILENAME)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= n_epochs_stop:
                logging.info("Ранняя остановка.")
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"] * 0.5, min_lr)
                logging.info(f"Снижена скорость обучения до: {param_group['lr']}")
    return model, optimizer

def save_model(
    model: nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> None:
    try:
        ensure_directory_exists(filepath)
        torch.save(model.state_dict(), filepath)
        optimizer_filepath = filepath.replace(".pth", "_optimizer.pth")
        torch.save(optimizer.state_dict(), optimizer_filepath)
        logging.info(f"Модель и оптимизатор сохранены: {filepath}, {optimizer_filepath}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")
        raise

def load_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device,
) -> None:
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            state_dict = torch.load(filepath, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            optimizer_filepath = filepath.replace(".pth", "_optimizer.pth")
            if os.path.exists(optimizer_filepath):
                optimizer_state_dict = torch.load(
                    optimizer_filepath, map_location=device
                )
                optimizer.load_state_dict(optimizer_state_dict)
            logging.info(
                f"Модель и оптимизатор загружены: {filepath}, {optimizer_filepath}"
            )
        else:
            logging.info(f"Модель не найдена, создается новая модель: {filepath}")
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        raise

def predict_future_price(
    model: EnhancedBiLSTMModel,
    latest_df: pd.DataFrame,
    data_processor: DataProcessor,
    prediction_minutes: int = PREDICTION_MINUTES
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        try:
            processed_df = data_processor.transform(latest_df)
        except Exception as e:
            logging.error(f"Ошибка при обработке latest_df: {e}")
            return pd.DataFrame()

        if len(processed_df) < SEQ_LENGTH:
            logging.warning("Недостаточно данных для предсказания.")
            logging.info(f"Текущего количества строк: {len(processed_df)}, требуется: {SEQ_LENGTH}")
            return pd.DataFrame()

        last_sequence_df = processed_df.iloc[-SEQ_LENGTH:]
        sequence_values = last_sequence_df[
            data_processor.numerical_columns + data_processor.categorical_columns + ['timestamp']
        ].values

        last_sequence = torch.tensor(sequence_values, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        predictions = model(last_sequence).cpu().numpy()

        numeric_features = data_processor.numerical_columns
        categorical_features = data_processor.categorical_columns

        predictions_df = pd.DataFrame(
            predictions, columns=numeric_features + categorical_features
        )

        scaled_numeric = data_processor.scaler.inverse_transform(
            predictions_df[numeric_features]
        )

        for col in categorical_features:
            predictions_df[col] = predictions_df[col].astype(int)
            le = data_processor.label_encoders.get(col)
            if le:
                predictions_df[col] = le.inverse_transform(predictions_df[col])

        scaled_numeric = scaled_numeric.clip(lower=0)
        predictions_df[numeric_features] = scaled_numeric

        last_timestamp = latest_df["timestamp"].max()
        if pd.isna(last_timestamp):
            logging.error("Последний таймстамп отсутствует или некорректен.")
            return pd.DataFrame()

        interval_key = get_interval(prediction_minutes)
        if interval_key is None:
            logging.error(f"Неизвестный интервал: {prediction_minutes} минут.")
            return pd.DataFrame()

        next_timestamp = int(last_timestamp) + INTERVAL_MAPPING[interval_key]["milliseconds"]

        predictions_df["timestamp"] = next_timestamp
        predictions_df["symbol"] = TARGET_SYMBOL
        predictions_df["interval"] = prediction_minutes
        predictions_df["interval_str"] = interval_key

        try:
            predictions_df = predictions_df[list(RAW_FEATURES.keys())]
        except KeyError as e:
            logging.error(f"Отсутствуют необходимые столбцы: {e}")
            return pd.DataFrame()

    return predictions_df

def get_interval(minutes: int) -> Optional[str]:
    for key, config in INTERVAL_MAPPING.items():
        if config["minutes"] == minutes:
            return key
    logging.error(f"Интервал для {minutes} минут не найден в INTERVAL_MAPPING.")
    return None

class DataFetcher:
    def __init__(self):
        self.download_data = DownloadData()
        self.combined_path = PATHS["combined_dataset"]
        self.predictions_path = PATHS["predictions"]

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        combined_data = pd.read_csv(self.combined_path)
        predictions_data = pd.read_csv(self.predictions_path) if os.path.exists(self.predictions_path) else pd.DataFrame(columns=RAW_FEATURES.keys())
        return combined_data, predictions_data

    def get_latest_binances_value(self, target_symbol: str) -> pd.DataFrame:
        return self.download_data.get_latest_prices(target_symbol, PREDICTION_MINUTES, SEQ_LENGTH)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main() -> None:
    setup_logging()
    # binance_main()
    device = get_device()

    if os.path.exists(DATA_PROCESSOR_FILENAME):
        data_processor = DataProcessor.load(DATA_PROCESSOR_FILENAME)
    else:
        data_processor = DataProcessor()

    combined_data, _ = DataFetcher().load_data()
    combined_data = data_processor.preprocess_binance_data(combined_data)
    combined_data = combined_data.sort_values(by='timestamp').reset_index(drop=True)

    if not os.path.exists(DATA_PROCESSOR_FILENAME):
        combined_data = data_processor.fit_transform(combined_data)
        data_processor.save(DATA_PROCESSOR_FILENAME)
    else:
        combined_data = data_processor.transform(combined_data)

    column_name_to_index = {col: idx for idx, col in enumerate(combined_data.columns)}
    model = EnhancedBiLSTMModel(
        numerical_columns=data_processor.numerical_columns,
        categorical_columns=data_processor.categorical_columns,
        column_name_to_index=column_name_to_index
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
    load_model(model, optimizer, MODEL_FILENAME, device)
    tensor_dataset = data_processor.prepare_dataset(combined_data, SEQ_LENGTH)
    dataloader = create_dataloader(tensor_dataset, TRAINING_PARAMS["batch_size"])
    model, optimizer = train_and_save_model(model, dataloader, device)

    # binance_main()
    latest_df = DataFetcher().get_latest_binances_value(TARGET_SYMBOL)
    latest_df = latest_df.sort_values(by='timestamp').reset_index(drop=True)
    print(latest_df)
    predicted_df = predict_future_price(model, latest_df, data_processor)
    print(predicted_df)

if __name__ == "__main__":
    while True:
        main()
