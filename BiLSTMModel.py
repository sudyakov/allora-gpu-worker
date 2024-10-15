import logging
import os
from time import sleep
from typing import Dict, Tuple, Sequence
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from get_binance_data import main as get_binance_data_main
from filelock import FileLock
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
    predict_future_price,
    update_differences,
    get_device,
    save_model,
    load_model,
    fit_transform,
    transform,
    inverse_transform,
    create_dataloader
)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
)

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
        self.hour_embedding = nn.Embedding(
            num_embeddings=24,
            embedding_dim=MODEL_PARAMS["embedding_dim"],
        )
        self.dayofweek_embedding = nn.Embedding(
            num_embeddings=7,
            embedding_dim=MODEL_PARAMS["embedding_dim"],
        )
        self.timestamp_embedding = nn.Linear(1, MODEL_PARAMS["timestamp_embedding_dim"])
        numerical_input_size = len(numerical_columns)
        self.lstm_input_size = (
            numerical_input_size
            + 4 * MODEL_PARAMS["embedding_dim"]
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
        hours = x[:, :, self.column_name_to_index['hour']].long()
        days = x[:, :, self.column_name_to_index['dayofweek']].long()
        timestamp = x[:, :, self.column_name_to_index["timestamp"]].float().unsqueeze(-1)
        symbol_embeddings = self.symbol_embedding(symbols)
        interval_embeddings = self.interval_embedding(intervals)
        hour_embeddings = self.hour_embedding(hours)
        day_embeddings = self.dayofweek_embedding(days)
        timestamp_embeddings = self.timestamp_embedding(timestamp)
        lstm_input = torch.cat(
            (
                numerical_data,
                symbol_embeddings,
                interval_embeddings,
                timestamp_embeddings,
                hour_embeddings,
                day_embeddings
            ),
            dim=2
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

def _train_model(
    model: EnhancedBiLSTMModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epochs: int,
    desc: str
) -> Tuple[EnhancedBiLSTMModel, AdamW]:
    criterion = nn.MSELoss(reduction='none')
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_corr = 0.0
        progress_bar = tqdm(
            loader,
            desc=f"{desc} Epoch {epoch + 1}/{epochs}",
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
            loss = criterion(outputs, targets)
            loss = (loss.mean(dim=1) * masks).sum() / masks.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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
        avg_loss = total_loss / len(loader)
        avg_corr = total_corr / len(loader)
        logging.info(
            f"{desc} Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Correlation: {avg_corr:.4f}"
        )
    return model, optimizer

def train_and_save_model(
    model: EnhancedBiLSTMModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, AdamW]:
    return _train_model(model, train_loader, optimizer, device, TRAINING_PARAMS["initial_epochs"], "Training")

def fine_tune_model(
    model: EnhancedBiLSTMModel,
    optimizer: AdamW,
    fine_tune_loader: DataLoader,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, AdamW]:
    return _train_model(model, fine_tune_loader, optimizer, device, TRAINING_PARAMS["fine_tune_epochs"], "Fine-tuning")

def main():
    device = get_device()
    data_fetcher = GetBinanceData()
    combined_data = data_fetcher.fetch_combined_data()
    if combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        return
    combined_data = shared_data_processor.preprocess_binance_data(combined_data)
    combined_data = shared_data_processor.fill_missing_add_features(combined_data)
    combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)
    if not shared_data_processor.is_fitted:
        combined_data = fit_transform(combined_data)
        shared_data_processor.save(DATA_PROCESSOR_FILENAME)
    else:
        combined_data = transform(combined_data)
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
    train_loader, val_loader = create_dataloader(
        tensor_dataset, TRAINING_PARAMS["batch_size"], shuffle=True
    )
    try:
        model, optimizer = train_and_save_model(model, train_loader, val_loader, optimizer, device)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return

    get_binance_data_main()
    sleep(5)

    predictions_path = PATHS["predictions"]
    latest_df = shared_data_processor.get_latest_dataset_prices(
        symbol=TARGET_SYMBOL,
        interval=PREDICTION_MINUTES,
        count=SEQ_LENGTH,
        latest_timestamp=None
    )
    predicted_df = predict_future_price(
        model=model,
        latest_df=latest_df,
        device=device,
        prediction_minutes=PREDICTION_MINUTES,
        future_steps=1,
        seq_length=SEQ_LENGTH,
        target_symbol=TARGET_SYMBOL
    )
    if not predicted_df.empty:
        if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
            existing_predictions = pd.read_csv(predictions_path)
            combined_predictions = pd.concat([existing_predictions, predicted_df], ignore_index=True)
            combined_predictions.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], inplace=True)
            combined_predictions.sort_values(by='timestamp', ascending=False, inplace=True)
        else:
            combined_predictions = predicted_df

        shared_data_processor.ensure_file_exists(predictions_path)
        combined_predictions.to_csv(predictions_path, index=False)
        logging.info(f"Predicted prices saved to {predictions_path}.")
    else:
        logging.info("No predictions were made due to insufficient data.")
    
    
    combined_dataset_path = PATHS['combined_dataset']
    differences_path = PATHS['differences']
    
    update_differences(
        differences_path=differences_path,
        predictions_path=predictions_path,
        combined_dataset_path=combined_dataset_path
    )
    
    if os.path.exists(differences_path) and os.path.getsize(differences_path) > 0:
        differences_df = pd.read_csv(differences_path)
        processed_differences = shared_data_processor.preprocess_binance_data(differences_df)
        processed_differences = shared_data_processor.fill_missing_add_features(processed_differences)
        processed_differences = processed_differences.sort_values(by="timestamp").reset_index(drop=True)
        processed_differences = transform(processed_differences)
        if processed_differences.isnull().values.any():
            logging.error("Differences data contains missing values. Skipping fine-tuning.")
        elif np.isinf(processed_differences.values).any():
            logging.error("Differences data contains infinite values. Skipping fine-tuning.")
        else:
            try:
                fine_tune_dataset = shared_data_processor.prepare_dataset(
                    processed_differences,
                    seq_length=SEQ_LENGTH,
                    target_symbols=[TARGET_SYMBOL],
                    target_intervals=[PREDICTION_MINUTES]
                )
                fine_tune_loader, _ = create_dataloader(
                    fine_tune_dataset, TRAINING_PARAMS["batch_size"], shuffle=True
                )
                model, optimizer = fine_tune_model(model, optimizer, fine_tune_loader, device)
            except Exception as e:
                logging.error(f"Error during fine-tuning: {e}")
    else:
        logging.info("No new differences found for fine-tuning.")

if __name__ == "__main__":
    while True:
        main()
