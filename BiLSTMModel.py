import logging
import os
from typing import Dict, List, Optional, Tuple, TypedDict, Sequence
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
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
from data_utils import DataProcessor
from get_binance_data import GetBinanceData
from model_utils import create_dataloader, get_device, load_model, save_model

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
            MODEL_PARAMS["hidden_layer_size"] * 2, len(SCALABLE_FEATURES)-1
        )
        self.relu = nn.ReLU()
        self.apply(self._initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerical_indices = [self.column_name_to_index[col] for col in self.numerical_columns]
        numerical_data = x[:, :, numerical_indices]
        symbols = x[:, :, self.column_name_to_index["symbol"]].long()
        intervals = x[:, :, self.column_name_to_index["interval"]].long()
        timestamp = x[:, :, self.column_name_to_index["timestamp"]].float().unsqueeze(-1)
        if torch.isnan(timestamp).any() or torch.isinf(timestamp).any():
            logging.error("Timestamp contains NaN or infinite values.")
            raise ValueError("Timestamp tensor contains NaN or infinite values.")
        symbol_embeddings = self.symbol_embedding(symbols)
        interval_embeddings = self.interval_embedding(intervals)
        timestamp_embeddings = self.timestamp_embedding(timestamp)
        lstm_input = torch.cat(
            (numerical_data, symbol_embeddings, interval_embeddings, timestamp_embeddings), dim=2
        )
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

def train_and_save_model(
    model: EnhancedBiLSTMModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, Adam]:
    model.train()
    optimizer = Adam(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
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
        avg_loss = total_loss / len(dataloader)
        logging.info(
            f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']} - Loss: {avg_loss:.4f}"
        )
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
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
    data_processor: DataProcessor,
    prediction_minutes: int = PREDICTION_MINUTES,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        if len(latest_df) < SEQ_LENGTH:
            logging.warning("Insufficient data for prediction.")
            return pd.DataFrame()
        latest_df = data_processor.transform(latest_df)
        tensor_dataset = data_processor.prepare_dataset(latest_df, SEQ_LENGTH)
        if len(tensor_dataset) == 0:
            logging.warning("No data available after preparing dataset.")
            return pd.DataFrame()
        inputs, _ = tensor_dataset[-1]
        inputs = inputs.unsqueeze(0).to(device)
        predictions = model(inputs).cpu().numpy()
        predictions_df = pd.DataFrame(predictions, columns=[col for col in SCALABLE_FEATURES.keys() if col != 'timestamp'])
        last_timestamp = latest_df["timestamp"].iloc[-1]
        if pd.isna(last_timestamp):
            logging.error("Invalid last timestamp.")
            return pd.DataFrame()
        interval_key: Optional[IntervalKey] = get_interval(prediction_minutes)
        if interval_key is None:
            logging.error("Invalid prediction interval.")
            return pd.DataFrame()
        next_timestamp = int(last_timestamp) + INTERVAL_MAPPING[interval_key]["milliseconds"]

        predictions_df["symbol"] = TARGET_SYMBOL
        predictions_df["interval"] = prediction_minutes
        predictions_df["timestamp"] = next_timestamp

        predictions_df = predictions_df[
            ["symbol", "interval", "timestamp"] + list(SCALABLE_FEATURES.keys())
        ]
    return predictions_df

def main():
    setup_logging()
    device = get_device()
    if os.path.exists(DATA_PROCESSOR_FILENAME):
        data_processor = DataProcessor.load(DATA_PROCESSOR_FILENAME)
    else:
        data_processor = DataProcessor()
        logging.info(f"Numerical columns for scaling: {data_processor.numerical_columns}")
    get_binance_data = GetBinanceData()
    combined_data = get_binance_data.fetch_combined_data()
    if combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        return
    combined_data = data_processor.preprocess_binance_data(combined_data)
    combined_data = data_processor.fill_missing_add_features(combined_data)
    combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)
    if os.path.exists(DATA_PROCESSOR_FILENAME):
        combined_data = data_processor.transform(combined_data)
    else:
        combined_data = data_processor.fit_transform(combined_data)
        data_processor.save(DATA_PROCESSOR_FILENAME)
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
        categorical_columns=data_processor.categorical_columns,
        numerical_columns=data_processor.numerical_columns,
        column_name_to_index=column_name_to_index,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
    load_model(model, optimizer, MODEL_FILENAME, device)
    tensor_dataset = data_processor.prepare_dataset(combined_data, SEQ_LENGTH)
    dataloader = create_dataloader(tensor_dataset, TRAINING_PARAMS["batch_size"])
    model, optimizer = train_and_save_model(model, dataloader, device)
    get_binance_data_main()
    latest_df = data_processor.get_latest_dataset_prices(
        TARGET_SYMBOL, PREDICTION_MINUTES, SEQ_LENGTH
    )
    latest_df = latest_df.sort_values(by="timestamp").reset_index(drop=True)
    logging.info(f"Latest dataset loaded with {len(latest_df)} records.")
    if not latest_df.empty:
        predicted_df = predict_future_price(model, latest_df, data_processor, PREDICTION_MINUTES, device)
        if not predicted_df.empty:
            predictions_path = PATHS["predictions"]
            DataProcessor.ensure_file_exists(predictions_path)
            predicted_df.to_csv(
                predictions_path,
                mode="a",
                header=not os.path.exists(predictions_path),
                index=False,
            )
            logging.info(f"Predicted prices saved to {predictions_path}.")
        else:
            logging.warning("No predictions were made due to previous errors.")
    else:
        logging.warning("Latest dataset is empty. Skipping prediction.")

if __name__ == "__main__":
    while True:
        main()
