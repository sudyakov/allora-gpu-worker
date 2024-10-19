import logging
import os
from typing import Dict, Sequence, Tuple
from time import sleep

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import traceback

from config import (
    INTERVAL_MAPPING,
    MODEL_FILENAME,
    MODEL_PARAMS,
    PATHS,
    PREDICTION_MINUTES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    TRAINING_PARAMS,
    get_interval,
)
from data_utils import shared_data_processor
from get_binance_data import GetBinanceData, main as get_binance_data_main
from model_utils import (
    create_dataloader,
    get_device,
    load_and_prepare_data,
    load_model,
    predict_future_price,
    save_model,
    update_differences,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
writer = SummaryWriter('runs/BiLSTMModel')


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)
        self.time_projection = nn.Linear(1, hidden_size * 2)

    def forward(self, lstm_out: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        time_embeddings = self.time_projection(timestamps.unsqueeze(-1))
        attention_scores = torch.matmul(lstm_out + time_embeddings, self.attention_weights).squeeze(-1)
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
        self.linear = nn.Linear(MODEL_PARAMS["hidden_layer_size"] * 2, len(SCALABLE_FEATURES))
        self.apply(self._initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(next(self.parameters()).device)
        numerical_indices = [self.column_name_to_index[col] for col in self.numerical_columns]
        numerical_data = x[:, :, numerical_indices]

        symbols = x[:, :, self.column_name_to_index["symbol"]].long()
        intervals = x[:, :, self.column_name_to_index["interval"]].long()
        hours = x[:, :, self.column_name_to_index['hour']].long()
        days = x[:, :, self.column_name_to_index['dayofweek']].long()
        timestamps = x[:, :, self.column_name_to_index["timestamp"]].float()

        symbol_embeddings = self.symbol_embedding(symbols)
        interval_embeddings = self.interval_embedding(intervals)
        hour_embeddings = self.hour_embedding(hours)
        day_embeddings = self.dayofweek_embedding(days)
        timestamp_embeddings = self.timestamp_embedding(timestamps.unsqueeze(-1))

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
        context_vector = self.attention(lstm_out, timestamps)
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


def compute_time_weights(timestamps: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    max_timestamp = timestamps.max()
    normalized_timestamps = (timestamps - timestamps.min()) / (max_timestamp - timestamps.min() + 1e-8)
    time_weights = alpha ** (1 - normalized_timestamps)
    return time_weights


def _train_model(
    model: EnhancedBiLSTMModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epochs: int,
    desc: str
) -> Tuple[EnhancedBiLSTMModel, AdamW]:
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_corr = 0.0
        progress_bar = tqdm(loader, desc=f"{desc} Epoch {epoch + 1}/{epochs}", unit="batch", leave=True)
        for inputs, targets, masks in progress_bar:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            timestamps = inputs[:, -1, shared_data_processor.column_name_to_index['timestamp']]
            time_weights = compute_time_weights(timestamps).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                logging.error(
                    "Output shape %s does not match target shape %s",
                    outputs.shape,
                    targets.shape,
                )
                continue

            loss = ((outputs - targets) ** 2) * time_weights.unsqueeze(1)
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
        logging.info(f"{desc} Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Correlation: {avg_corr:.4f}")
        save_model(model, optimizer, MODEL_FILENAME)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Correlation/train', avg_corr, epoch)
    return model, optimizer


def train_and_save_model(
    model: EnhancedBiLSTMModel,
    train_loader: DataLoader,
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


def main(model: EnhancedBiLSTMModel, optimizer: AdamW, data_fetcher: GetBinanceData):
    device = get_device()

    get_binance_data_main()
    sleep(5)

    real_combined_data = load_and_prepare_data(data_fetcher, is_training=True)
    if real_combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        return model, optimizer

    shared_data_processor.set_column_name_to_index(list(real_combined_data.columns))
    logging.info("Dataset after transformation:")
    logging.info(f"\n{real_combined_data.tail()}")

    if real_combined_data.isnull().values.any():
        logging.info("Data contains missing values.")
    if np.isinf(real_combined_data.values).any():
        logging.info("Data contains infinite values.")

    try:
        training_dataset = shared_data_processor.prepare_dataset(
            real_combined_data,
            seq_length=SEQ_LENGTH,
            target_symbols=[TARGET_SYMBOL],
            target_intervals=[PREDICTION_MINUTES]
        )
    except Exception as e:
        logging.error(f"Error preparing dataset: {e}")
        return model, optimizer

    train_loader, _ = create_dataloader(
        training_dataset, TRAINING_PARAMS["batch_size"], shuffle=True
    )

    try:
        model, optimizer = train_and_save_model(model, train_loader, optimizer, device)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return model, optimizer

    combined_dataset_path = PATHS["combined_dataset"]
    if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
        real_combined_data = pd.read_csv(combined_dataset_path)
        latest_data_timestamp = real_combined_data['timestamp'].max()
    else:
        logging.error("Combined dataset not found.")
        return model, optimizer

    predictions_path = PATHS["predictions"]
    if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
        existing_predictions_df = pd.read_csv(predictions_path)
        if not existing_predictions_df.empty:
            last_prediction_timestamp = existing_predictions_df['timestamp'].max()
        else:
            last_prediction_timestamp = None
    else:
        existing_predictions_df = pd.DataFrame()
        last_prediction_timestamp = None

    interval = get_interval(PREDICTION_MINUTES)
    if interval is None:
        logging.error("Invalid PREDICTION_MINUTES value.")
        return model, optimizer

    interval_ms = INTERVAL_MAPPING[interval]["milliseconds"]
    timestamps_to_predict = []

    if last_prediction_timestamp is not None:
        timestamps_to_predict = list(range(
            int(last_prediction_timestamp + interval_ms),
            int(latest_data_timestamp + 2 * interval_ms),
            int(interval_ms)
        ))
    else:
        timestamps_to_predict = [int(latest_data_timestamp + interval_ms)]

    predictions_list = []
    for next_timestamp in tqdm(timestamps_to_predict, desc="Generating Predictions"):
        latest_df = load_and_prepare_data(
            data_fetcher,
            is_training=False,
            latest_timestamp=next_timestamp - interval_ms,
            count=SEQ_LENGTH
        )
        if latest_df.empty:
            logging.warning(f"No data available for timestamp {next_timestamp}. Skipping prediction.")
            continue

        logging.debug(f"Input data for timestamp {next_timestamp}:\n{latest_df}")
        predicted_df = predict_future_price(
            model=model,
            latest_real_data_df=latest_df,
            device=device,
            prediction_minutes=PREDICTION_MINUTES,
            future_steps=1,
            seq_length=SEQ_LENGTH,
            target_symbol=TARGET_SYMBOL
        )
        logging.debug(f"Prediction for timestamp {next_timestamp}:\n{predicted_df}")
        if not predicted_df.empty:
            predictions_list.append(predicted_df)
        else:
            logging.info(f"No prediction made for timestamp {next_timestamp} due to insufficient data.")

    if predictions_list:
        all_predictions = pd.concat(predictions_list, ignore_index=True)
        combined_predictions = pd.concat([existing_predictions_df, all_predictions], ignore_index=True)
        combined_predictions.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], keep='last', inplace=True)
        combined_predictions.sort_values(by='timestamp', ascending=True, inplace=True)
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        combined_predictions.to_csv(predictions_path, index=False)
        logging.info(f"All predicted prices saved to {predictions_path}.")
    else:
        logging.info("No predictions were made during this run due to insufficient data.")

    differences_path = PATHS['differences']
    update_differences(
        differences_path=differences_path,
        predictions_path=predictions_path,
        combined_dataset_path=combined_dataset_path
    )

    if os.path.exists(differences_path) and os.path.getsize(differences_path) > 0:
        differences_data = pd.read_csv(differences_path)
        differences_processed_data = load_and_prepare_data(
            data_fetcher,
            is_training=False
        )
        if differences_processed_data.empty:
            logging.error("Differences data is empty. Skipping fine-tuning.")
        elif differences_processed_data.isnull().values.any():
            logging.error("Differences data contains missing values. Skipping fine-tuning.")
        elif np.isinf(differences_processed_data.values).any():
            logging.error("Differences data contains infinite values. Skipping fine-tuning.")
        else:
            try:
                fine_tuning_dataset = shared_data_processor.prepare_dataset(
                    differences_processed_data,
                    seq_length=SEQ_LENGTH,
                    target_symbols=[TARGET_SYMBOL],
                    target_intervals=[PREDICTION_MINUTES]
                )
                fine_tune_loader, _ = create_dataloader(
                    fine_tuning_dataset, TRAINING_PARAMS["batch_size"], shuffle=True
                )
                model, optimizer = fine_tune_model(model, optimizer, fine_tune_loader, device)
            except Exception as e:
                logging.error(f"Error during fine-tuning: {e}")
    else:
        logging.info("No new differences found for fine-tuning.")

    return model, optimizer


if __name__ == "__main__":
    device = get_device()
    data_fetcher = GetBinanceData()
    get_binance_data_main()
    sleep(5)

    real_combined_data = load_and_prepare_data(data_fetcher, is_training=True)
    if real_combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        exit()

    shared_data_processor.set_column_name_to_index(real_combined_data.columns.tolist())
    MODEL_PARAMS["num_symbols"] = len(shared_data_processor.symbol_mapping)
    MODEL_PARAMS["num_intervals"] = len(shared_data_processor.interval_mapping)
    logging.info(
        f"num_symbols: {MODEL_PARAMS['num_symbols']}, num_intervals: {MODEL_PARAMS['num_intervals']}"
    )

    model = EnhancedBiLSTMModel(
        categorical_columns=shared_data_processor.categorical_columns,
        numerical_columns=shared_data_processor.numerical_columns,
        column_name_to_index=shared_data_processor.column_name_to_index,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])

    load_model(model, optimizer, MODEL_FILENAME, device)

    while True:
        try:
            logging.info("Starting main loop iteration.")
            model, optimizer = main(model, optimizer, data_fetcher)
            logging.info("Main loop iteration completed successfully.")
            sleep(30)  # Задержка между итерациями цикла
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            traceback.print_exc()
            logging.info("Retrying after delay...")
            sleep(10)  # Задержка перед повторной попыткой