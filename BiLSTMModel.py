import logging
import os
from typing import Dict, List, Optional, Tuple, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np

from get_binance_data import main as get_binance_data_main
from config import (
    ADD_FEATURES,
    DATA_PROCESSOR_FILENAME,
    INTERVAL_MAPPING,
    MODEL_FILENAME,
    MODEL_FEATURES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    PATHS,
    PREDICTION_MINUTES,
    IntervalConfig,
    IntervalKey,
    get_interval,
    TRAINING_PARAMS,
    MODEL_PARAMS
)
from data_utils import shared_data_processor
from get_binance_data import GetBinanceData

def log(message: str):
    logging.info(message)

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    return device

def save_model(model, optimizer, filename: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename, _use_new_zipfile_serialization=True)
    log(f"Model saved to {filename}")

def load_model(model, optimizer, filename: str, device: torch.device) -> None:
    if os.path.exists(filename):
        log(f"Loading model from {filename}")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log("Model and optimizer state loaded.")
    else:
        log(f"No model file found at {filename}. Starting from scratch.")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("BiLSTMModel.log"),
        ],
    )
    logging.info("Logging is set up.")


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
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
        numerical_columns: Sequence[str],
        categorical_columns: Sequence[str],
        column_name_to_index: Dict[str, int],
    ):
        super().__init__()
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.column_name_to_index = column_name_to_index

        self.symbol_embedding = nn.Embedding(
            num_embeddings=MODEL_PARAMS["num_symbols"],
            embedding_dim=MODEL_PARAMS["embedding_dim"],
        )
        self.interval_embedding = nn.Embedding(
            num_embeddings=MODEL_PARAMS["num_intervals"],
            embedding_dim=MODEL_PARAMS["embedding_dim"],
        )
        self.timestamp_embedding = nn.Linear(1, MODEL_PARAMS["timestamp_embedding_dim"])

        numerical_input_size = len(numerical_columns)
        self.lstm_input_size = (
            numerical_input_size
            + 2 * MODEL_PARAMS["embedding_dim"]
            + MODEL_PARAMS["timestamp_embedding_dim"]
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=MODEL_PARAMS["hidden_layer_size"],
            num_layers=MODEL_PARAMS["num_layers"],
            dropout=MODEL_PARAMS["dropout"],
            batch_first=True,
            bidirectional=True,
        )

        self.attention = Attention(MODEL_PARAMS["hidden_layer_size"])
        self.linear = nn.Linear(
            MODEL_PARAMS["hidden_layer_size"] * 2, len(SCALABLE_FEATURES)
        )

        self.apply(self._initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(next(self.parameters()).device)
        numerical_indices = [self.column_name_to_index[col] for col in self.numerical_columns]
        numerical_data = x[:, :, numerical_indices]

        symbols = x[:, :, self.column_name_to_index["symbol"]].long()
        intervals = x[:, :, self.column_name_to_index["interval"]].long()
        timestamp = x[:, :, self.column_name_to_index["timestamp"]].float().unsqueeze(-1)

        symbol_embeddings = self.symbol_embedding(symbols)
        interval_embeddings = self.interval_embedding(intervals)
        timestamp_embeddings = self.timestamp_embedding(timestamp)

        lstm_input = torch.cat(
            (numerical_data, symbol_embeddings, interval_embeddings, timestamp_embeddings), dim=2
        )

        lstm_out, _ = self.lstm(lstm_input)
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)

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


def train_and_save_model(
    model: EnhancedBiLSTMModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, Adam]:
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = TRAINING_PARAMS["min_lr"]

    for epoch in range(TRAINING_PARAMS["initial_epochs"]):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}",
            unit="batch",
            leave=False,
        )
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                logging.error("Input data contains NaN or infinite values. Stopping training.")
                return model, optimizer
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logging.error("Target data contains NaN or infinite values. Stopping training.")
                return model, optimizer

            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                logging.error(
                    "Output shape %s does not match target shape %s",
                    outputs.shape,
                    targets.shape,
                )
                continue
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logging.info(
            f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']} - Training Loss: {avg_loss:.4f}"
        )

        # Валидация модели
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(model, optimizer, MODEL_FILENAME)
            logging.info(f"Validation loss improved to {best_val_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            logging.info(
                f"No improvement in validation loss. ({epochs_no_improve}/{n_epochs_stop})"
            )
            if epochs_no_improve >= n_epochs_stop:
                logging.info("Early stopping triggered.")
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"] * 0.5, min_lr)
                logging.info(f"Reducing learning rate to: {param_group['lr']}")
    return model, optimizer


def predict_future_price(
    model: EnhancedBiLSTMModel,
    latest_df: pd.DataFrame,
    prediction_minutes: int = PREDICTION_MINUTES,
    device: torch.device = get_device(),
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        if len(latest_df) < SEQ_LENGTH:
            logging.warning("Insufficient data for prediction.")
            return pd.DataFrame()

        latest_df_transformed = shared_data_processor.transform(latest_df)
        tensor_dataset = shared_data_processor.prepare_dataset(latest_df_transformed, SEQ_LENGTH)
        if len(tensor_dataset) == 0:
            logging.warning("No data after dataset preparation.")
            return pd.DataFrame()

        inputs, _ = tensor_dataset[-1]
        inputs = inputs.unsqueeze(0).to(device)
        predictions = model(inputs).cpu().numpy()
        predictions_df = pd.DataFrame(predictions, columns=list(SCALABLE_FEATURES.keys()))

        predictions_df_denormalized = shared_data_processor.inverse_transform(predictions_df)

        last_timestamp = latest_df["timestamp"].iloc[-1]
        if pd.isna(last_timestamp):
            logging.error("Invalid last timestamp value.")
            return pd.DataFrame()

        interval_key: Optional[IntervalKey] = get_interval(prediction_minutes)
        if interval_key is None:
            logging.error("Invalid prediction interval.")
            return pd.DataFrame()

        next_timestamp = np.int64(last_timestamp) + INTERVAL_MAPPING[interval_key]["milliseconds"]

        predictions_df_denormalized["symbol"] = TARGET_SYMBOL
        predictions_df_denormalized["interval"] = prediction_minutes
        predictions_df_denormalized["timestamp"] = next_timestamp

        predictions_df_denormalized = predictions_df_denormalized[
            ["symbol", "interval", "timestamp"] + list(SCALABLE_FEATURES.keys())
        ]

        #predictions_df_prepared = shared_data_processor.fill_missing_add_features(predictions_df_denormalized)

    return predictions_df_denormalized


def main():
    device = get_device()

    if os.path.exists(DATA_PROCESSOR_FILENAME):
        shared_data_processor.load(DATA_PROCESSOR_FILENAME)
    else:
        logging.error("Data processor file not found. Exiting.")
        return

    get_binance_data = GetBinanceData()
    combined_data = get_binance_data.fetch_combined_data()
    if combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        return

    combined_data = shared_data_processor.preprocess_binance_data(combined_data)
    combined_data = shared_data_processor.fill_missing_add_features(combined_data)
    combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)

    if os.path.exists(DATA_PROCESSOR_FILENAME):
        combined_data = shared_data_processor.transform(combined_data)
    else:
        combined_data = shared_data_processor.fit_transform(combined_data)
        shared_data_processor.save(DATA_PROCESSOR_FILENAME)

    MODEL_PARAMS["num_symbols"] = max(
        combined_data["symbol"].max() + 1, MODEL_PARAMS.get("num_symbols", 0)
    )
    MODEL_PARAMS["num_intervals"] = max(
        combined_data["interval"].max() + 1, MODEL_PARAMS.get("num_intervals", 0)
    )
    logging.info(
        f"num_symbols: {MODEL_PARAMS['num_symbols']}, num_intervals: {MODEL_PARAMS['num_intervals']}"
    )
    if (combined_data['symbol'] < 0).any():
        raise ValueError("Negative indices found in 'symbol' column.")
    if (combined_data['interval'] < 0).any():
        raise ValueError("Negative indices found in 'interval' column.")
    if combined_data['symbol'].max() >= MODEL_PARAMS["num_symbols"]:
        raise ValueError("Symbol indices exceed the number of symbols in embedding.")
    if combined_data['interval'].max() >= MODEL_PARAMS["num_intervals"]:
        raise ValueError("Interval indices exceed the number of intervals in embedding.")

    logging.info(f"Columns after processing: {combined_data.columns.tolist()}")

    column_name_to_index = {col: idx for idx, col in enumerate(combined_data.columns)}

    model = EnhancedBiLSTMModel(
        categorical_columns=shared_data_processor.categorical_columns,
        numerical_columns=shared_data_processor.numerical_columns,
        column_name_to_index=column_name_to_index,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
    load_model(model, optimizer, MODEL_FILENAME, device)

    if combined_data.isnull().values.any():
        logging.warning("Data contains missing values.")
    if np.isinf(combined_data.values).any():
        logging.warning("Data contains infinite values.")

    tensor_dataset = shared_data_processor.prepare_dataset(combined_data, SEQ_LENGTH)

    # Создание загрузчиков данных
    train_loader, val_loader = shared_data_processor(tensor_dataset, TRAINING_PARAMS["batch_size"], shuffle=True)
    model, optimizer = train_and_save_model(model, train_loader, val_loader, optimizer, device)
    get_binance_data_main()
    latest_df = shared_data_processor.get_latest_dataset_prices(
        TARGET_SYMBOL, PREDICTION_MINUTES, SEQ_LENGTH
    )
    latest_df = latest_df.sort_values(by="timestamp").reset_index(drop=True)
    logging.info(f"Latest dataset loaded with {len(latest_df)} records.")
    if not latest_df.empty:
        predicted_df = predict_future_price(model, latest_df, PREDICTION_MINUTES, device)
        if not predicted_df.empty:
            predictions_path = PATHS["predictions"]
            shared_data_processor.ensure_file_exists(predictions_path)
            
            try:
                existing_predictions = pd.read_csv(predictions_path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                existing_predictions = pd.DataFrame()
            
            current_timestamp = predicted_df["timestamp"].iloc[0]
            if current_timestamp in existing_predictions["timestamp"].values:
                logging.info(f"Prediction for timestamp {current_timestamp} already exists. Skipping save.")
            else:
                combined_predictions = pd.concat([predicted_df, existing_predictions], ignore_index=True)
                combined_predictions.to_csv(
                    predictions_path,
                    index=False
                )
                logging.info(f"Predicted prices saved to {predictions_path}.")
        else:
            logging.warning("Predictions were not made due to previous errors.")
    else:
        logging.warning("Latest dataset is empty. Skipping prediction.")


if __name__ == "__main__":
    setup_logging()
    while True:
        main()
