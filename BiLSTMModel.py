import logging
import os
from typing import Dict, Optional, Tuple, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
    IntervalKey,
    get_interval,
    TRAINING_PARAMS,
    MODEL_PARAMS
)
from data_utils import shared_data_processor
from get_binance_data import GetBinanceData
from model_utils import (
    predict_future_price
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
)

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device

def save_model(model, optimizer, filename: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename, _use_new_zipfile_serialization=True)
    logging.info(f"Model saved to {filename}")

def load_model(model, optimizer, filename: str, device: torch.device) -> None:
    if os.path.exists(filename):
        logging.info(f"Loading model from {filename}")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Model and optimizer state loaded.")
    else:
        logging.info(f"No model file found at {filename}. Starting from scratch.")

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

        predictions = torch.clamp(predictions, min=-10, max=10)
        predictions = torch.exp(predictions)

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
    optimizer: AdamW,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, AdamW]:
    criterion = nn.MSELoss(reduction='none')  # Изменяем reduction на 'none'
    best_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = TRAINING_PARAMS["min_lr"]

    for epoch in range(TRAINING_PARAMS["initial_epochs"]):
        model.train()
        total_loss = 0.0
        total_corr = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']}",
            unit="batch",
            leave=True,
        )
        for inputs, targets, masks in progress_bar:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

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

            # Применяем маску к функции потерь
            loss = criterion(outputs, targets)
            loss = (loss.mean(dim=1) * masks).sum() / masks.sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Вычисляем корреляцию только для тех примеров, где маска == 1
            if masks.sum() > 0:
                preds = outputs[masks == 1].detach().cpu().numpy()
                truths = targets[masks == 1].detach().cpu().numpy()
                corr = np.mean([
                    np.corrcoef(preds[i], truths[i])[0, 1] if not np.isnan(np.corrcoef(preds[i], truths[i])[0, 1]) else 0
                    for i in range(len(preds))
                ])
                total_corr += corr
            else:
                corr = 0

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", corr=f"{corr:.4f}")

        avg_loss = total_loss / len(train_loader)
        avg_corr = total_corr / len(train_loader)
        logging.info(
            f"Epoch {epoch + 1}/{TRAINING_PARAMS['initial_epochs']} - Training Loss: {avg_loss:.4f}, Correlation: {avg_corr:.4f}"
        )

        model.eval()
        val_loss = 0.0
        val_corr = 0.0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = (loss.mean(dim=1) * masks).sum() / masks.sum()
                val_loss += loss.item()

                if masks.sum() > 0:
                    preds = outputs[masks == 1].cpu().numpy()
                    truths = targets[masks == 1].cpu().numpy()
                    corr = np.mean([
                        np.corrcoef(preds[i], truths[i])[0, 1] if not np.isnan(np.corrcoef(preds[i], truths[i])[0, 1]) else 0
                        for i in range(len(preds))
                    ])
                    val_corr += corr

        avg_val_loss = val_loss / len(val_loader)
        avg_val_corr = val_corr / len(val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Correlation: {avg_val_corr:.4f}")

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

def main():
    device = get_device()
    get_binance_data = GetBinanceData()

    combined_data = get_binance_data.fetch_combined_data()
    if combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        return

    combined_data = shared_data_processor.preprocess_binance_data(combined_data)
    combined_data = shared_data_processor.fill_missing_add_features(combined_data)
    combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)

    if os.path.exists(DATA_PROCESSOR_FILENAME):
        shared_data_processor.load(DATA_PROCESSOR_FILENAME)
        shared_data_processor.is_fitted = True
        combined_data = shared_data_processor.transform(combined_data)
    else:
        logging.info("Data processor file not found. Fitting a new DataProcessor.")
        combined_data = shared_data_processor.fit_transform(combined_data)
        shared_data_processor.save(DATA_PROCESSOR_FILENAME)

    MODEL_PARAMS["num_symbols"] = len(shared_data_processor.symbol_mapping)
    MODEL_PARAMS["num_intervals"] = len(shared_data_processor.interval_mapping)
    
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
    optimizer = AdamW(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
    load_model(model, optimizer, MODEL_FILENAME, device)

    if combined_data.isnull().values.any():
        logging.info("Data contains missing values.")
    if np.isinf(combined_data.values).any():
        logging.info("Data contains infinite values.")

    try:
        tensor_dataset = shared_data_processor.prepare_dataset(
            combined_data,
            seq_length=SEQ_LENGTH,
            target_symbols=[TARGET_SYMBOL],
            target_intervals=[PREDICTION_MINUTES]
        )

    except Exception as e:
        logging.error(f"Error preparing dataset: {e}")
        return

    train_loader, val_loader = shared_data_processor.create_dataloader(
        tensor_dataset, TRAINING_PARAMS["batch_size"], shuffle=True
    )

    try:
        model, optimizer = train_and_save_model(model, train_loader, val_loader, optimizer, device)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return

    get_binance_data_main()
    
    latest_df = shared_data_processor.get_latest_dataset_prices(symbol=None, interval=PREDICTION_MINUTES, count=SEQ_LENGTH)
    latest_df = latest_df.sort_values(by="timestamp").reset_index(drop=True)
    logging.info(f"Latest dataset loaded with {len(latest_df)} records.")

    if not latest_df.empty:
        predicted_df = predict_future_price(model, latest_df, device, PREDICTION_MINUTES)
        if not predicted_df.empty:
            predictions_path = PATHS["predictions"]
            shared_data_processor.ensure_file_exists(predictions_path)

            try:
                existing_predictions = pd.read_csv(predictions_path)
                existing_predictions = existing_predictions[predicted_df.columns]
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
                existing_predictions = pd.DataFrame(columns=predicted_df.columns)

            current_timestamp = predicted_df["timestamp"].iloc[0]
            if current_timestamp in existing_predictions["timestamp"].values:
                logging.info(f"Prediction for timestamp {current_timestamp} already exists. Skipping save.")
            else:
                combined_predictions = pd.concat([predicted_df, existing_predictions], ignore_index=True)
                combined_predictions = combined_predictions[predicted_df.columns]
                combined_predictions.to_csv(
                    predictions_path,
                    index=False
                )
                logging.info(f"Predicted prices saved to {predictions_path}.")
        else:
            logging.info("Predictions were not made due to previous errors.")
    else:
        logging.info("Latest dataset is empty. Skipping prediction.")

if __name__ == "__main__":
    while True:
        main()
