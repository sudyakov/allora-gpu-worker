import logging
import os
import time
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    get_interval,
    TRAINING_PARAMS,
    MODEL_PARAMS
)
from data_utils import shared_data_processor
from get_binance_data import GetBinanceData, main as get_binance_data_main
from model_utils import (
    predict_future_price,
    update_differences,
    get_device,
    save_model,
    load_model
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

        embedding_dim = MODEL_PARAMS["embedding_dim"]
        self.symbol_embedding = nn.Embedding(
            num_embeddings=MODEL_PARAMS["num_symbols"],
            embedding_dim=embedding_dim,
        )
        self.interval_embedding = nn.Embedding(
            num_embeddings=MODEL_PARAMS["num_intervals"],
            embedding_dim=embedding_dim,
        )
        self.hour_embedding = nn.Embedding(
            num_embeddings=24,
            embedding_dim=embedding_dim,
        )
        self.dayofweek_embedding = nn.Embedding(
            num_embeddings=7,
            embedding_dim=embedding_dim,
        )
        self.timestamp_embedding = nn.Linear(1, MODEL_PARAMS["timestamp_embedding_dim"])

        numerical_input_size = len(numerical_columns)
        self.lstm_input_size = (
            numerical_input_size
            + 4 * embedding_dim
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
                day_embeddings,
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
    model, optimizer = _train_model(
        model, train_loader, optimizer, device, TRAINING_PARAMS["initial_epochs"], "Training"
    )
    save_model(model, optimizer, MODEL_FILENAME)
    return model, optimizer


def fine_tune_model(
    model: EnhancedBiLSTMModel,
    optimizer: AdamW,
    fine_tune_loader: DataLoader,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, AdamW]:
    model, optimizer = _train_model(
        model, fine_tune_loader, optimizer, device, TRAINING_PARAMS["fine_tune_epochs"], "Fine-tuning"
    )
    save_model(model, optimizer, MODEL_FILENAME)
    return model, optimizer


def get_last_prediction_timestamp(predictions_path: str) -> Optional[int]:
    if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
        predictions_df = pd.read_csv(predictions_path)
        if not predictions_df.empty:
            return predictions_df['timestamp'].max()
    return None


def main():
    device = get_device()
    data_fetcher = GetBinanceData()

    # Обновляем данные с Binance
    get_binance_data_main()
    time.sleep(1)

    # Определяем пути к файлам вне условных блоков
    predictions_path = PATHS["predictions"]
    combined_dataset_path = PATHS['combined_dataset']
    differences_path = PATHS['differences']

    combined_data = data_fetcher.fetch_combined_data()
    if combined_data.empty:
        logging.error("Combined data is empty. Exiting.")
        return

    # Предобработка данных с использованием DataProcessor
    combined_data = shared_data_processor.preprocess_binance_data(combined_data)
    combined_data = shared_data_processor.fill_missing_add_features(combined_data)
    combined_data = combined_data.sort_values(by="timestamp").reset_index(drop=True)

    # Трансформация данных
    if not shared_data_processor.is_fitted:
        combined_data = shared_data_processor.fit_transform(combined_data)
        shared_data_processor.save(DATA_PROCESSOR_FILENAME)
    else:
        combined_data = shared_data_processor.transform(combined_data)

    MODEL_PARAMS["num_symbols"] = len(shared_data_processor.symbol_mapping)
    MODEL_PARAMS["num_intervals"] = len(shared_data_processor.interval_mapping)
    logging.info(
        f"num_symbols: {MODEL_PARAMS['num_symbols']}, num_intervals: {MODEL_PARAMS['num_intervals']}"
    )

    # Проверки на корректность индексов
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

    # Инициализация модели и оптимизатора
    model = EnhancedBiLSTMModel(
        categorical_columns=shared_data_processor.categorical_columns,
        numerical_columns=shared_data_processor.numerical_columns,
        column_name_to_index=column_name_to_index,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=TRAINING_PARAMS["initial_lr"])
    load_model(model, optimizer, MODEL_FILENAME, device)

    # Подготовка датасета для обучения
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

    # Обучение модели
    try:
        model, optimizer = train_and_save_model(model, train_loader, val_loader, optimizer, device)
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return

    # Получаем последний предсказанный timestamp
    last_prediction_timestamp = get_last_prediction_timestamp(PATHS['predictions'])

    # Загружаем обновленный комбинированный датасет
    combined_df = combined_data.copy()

    # Отфильтровываем данные после последнего предсказанного timestamp
    if last_prediction_timestamp is not None:
        new_data_df = combined_df[combined_df['timestamp'] > last_prediction_timestamp]
    else:
        new_data_df = combined_df

    # Проверяем, есть ли новые данные
    if not new_data_df.empty and len(new_data_df) >= SEQ_LENGTH:
        # Предобрабатываем новые данные
        new_data_df = new_data_df.sort_values(by="timestamp").reset_index(drop=True)

        # Генерируем последовательности для предсказаний
        sequences = []
        timestamps = []
        for i in range(len(new_data_df) - SEQ_LENGTH + 1):
            sequence = new_data_df.iloc[i:i + SEQ_LENGTH]
            sequences.append(sequence.values)
            next_timestamp = sequence['timestamp'].iloc[-1] + INTERVAL_MAPPING[get_interval(PREDICTION_MINUTES)]["milliseconds"]
            timestamps.append(next_timestamp)

        # Преобразуем в тензоры
        sequences = torch.tensor(sequences, dtype=torch.float32).to(device)

        # Делаем предсказания батчами
        batch_size = TRAINING_PARAMS['batch_size']
        predictions = []
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            with torch.no_grad():
                outputs = model(batch_sequences)
                outputs = outputs.cpu().numpy()
                # Инвертируем нормализацию
                outputs_df = pd.DataFrame(outputs, columns=SCALABLE_FEATURES.keys())
                outputs_df_denorm = shared_data_processor.inverse_transform(outputs_df)
                predictions.append(outputs_df_denorm)

        # Объединяем предсказания
        predictions_df = pd.concat(predictions, ignore_index=True)
        predictions_df['symbol'] = TARGET_SYMBOL
        predictions_df['interval'] = PREDICTION_MINUTES
        predictions_df['timestamp'] = timestamps
        predictions_df = shared_data_processor.fill_missing_add_features(predictions_df)
        predictions_df = predictions_df[list(MODEL_FEATURES.keys())]

        # Объединяем с предыдущими предсказаниями
        predictions_path = PATHS["predictions"]
        shared_data_processor.ensure_file_exists(predictions_path)
        try:
            existing_predictions = pd.read_csv(predictions_path)
            existing_predictions = existing_predictions[predictions_df.columns]
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
            existing_predictions = pd.DataFrame(columns=predictions_df.columns)
        combined_predictions = pd.concat([existing_predictions, predictions_df], ignore_index=True)
        combined_predictions.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], inplace=True)
        combined_predictions.sort_values(by='timestamp', ascending=False, inplace=True)
        combined_predictions.to_csv(predictions_path, index=False)
        logging.info(f"New predictions saved to {predictions_path}.")
        # Проверяем порядок данных в predictions.csv
        predictions_check = pd.read_csv(predictions_path)
        print(predictions_check[['timestamp', 'symbol', 'interval']].head())
    else:
        logging.info("No new data available for predictions.")


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
        processed_differences = shared_data_processor.transform(processed_differences)
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
                fine_tune_loader, _ = shared_data_processor.create_dataloader(
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
        time.sleep(3)  # Добавляем паузу между циклами
