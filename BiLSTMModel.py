import os
import pickle
import logging
from typing import Optional, Dict, Literal, Tuple, Union, List
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from get_binance_data import GetBinanceData

from config import (
    INTERVAL_MAPPING,
    SYMBOL_MAPPING,
    PATHS,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    MODEL_FEATURES,
    MODEL_PARAMS,
    ADD_FEATURES,
    SEQ_LENGTH,
    TARGET_SYMBOL,
    PREDICTION_MINUTES,
    MODEL_FILENAME,
    DATA_PROCESSOR_FILENAME,
    TRAINING_PARAMS,
)


IntervalKey = Literal['1m', '5m', '15m']


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("BiLSTMModel.log")
        ]
    )


def ensure_directory_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_PARAMS.get("num_workers", 4),
    )


def save_model(model: nn.Module, optimizer: Adam, filepath: str) -> None:
    ensure_directory_exists(filepath)
    torch.save(model.state_dict(), filepath)
    optimizer_filepath = filepath.replace(".pth", "_optimizer.pth")
    torch.save(optimizer.state_dict(), optimizer_filepath)
    logging.info("Model and optimizer saved: %s, %s", filepath, optimizer_filepath)


def load_model(
    model: EnhancedBiLSTMModel,
    optimizer: Adam,
    filepath: str,
    device: torch.device,
) -> None:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        optimizer_filepath = filepath.replace(".pth", "_optimizer.pth")
        if os.path.exists(optimizer_filepath):
            optimizer_state_dict = torch.load(optimizer_filepath, map_location=device)
            optimizer.load_state_dict(optimizer_state_dict)
        logging.info("Model and optimizer loaded: %s, %s", filepath, optimizer_filepath)
    else:
        logging.info("Model not found, creating new: %s", filepath)


def train_and_save_model(
    model: EnhancedBiLSTMModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[EnhancedBiLSTMModel, Adam]:
    model.train()
    optimizer = Adam(model.parameters(), lr=TRAINING_PARAMS.get("initial_lr", 0.001))
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = TRAINING_PARAMS.get("min_lr", 1e-6)
    for epoch in range(TRAINING_PARAMS.get("initial_epochs", 50)):
        total_loss = 0.0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{TRAINING_PARAMS.get('initial_epochs', 50)}",
            unit="batch",
            leave=False,
        )
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.shape != targets.shape:
                logging.error("Output shape %s does not match target shape %s", outputs.shape, targets.shape)
                continue
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        logging.info("Epoch %d/%d - Loss: %.4f", epoch + 1, TRAINING_PARAMS.get("initial_epochs", 50), avg_loss)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
            save_model(model, optimizer, MODEL_FILENAME)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= n_epochs_stop:
                logging.info("Early stopping.")
                break
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(param_group["lr"] * 0.5, min_lr)
                logging.info("Reducing learning rate to: %s", param_group["lr"])
    return model, optimizer


def get_interval(minutes: int) -> Optional[IntervalKey]:
    for key, config in INTERVAL_MAPPING.items():
        if config["minutes"] == minutes:
            return key
    logging.error("Interval for %d minutes not found in INTERVAL_MAPPING.", minutes)
    return None


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
            logging.error("Error processing latest_df: %s", e)
            return pd.DataFrame()

        # Ensure there are enough data points
        if len(processed_df) < SEQ_LENGTH:
            logging.warning("Not enough data for prediction.")
            logging.info("Current number of rows: %d, required: %d", len(processed_df), SEQ_LENGTH)
            return pd.DataFrame()

        last_sequence_df = processed_df.iloc[-SEQ_LENGTH:]
        sequence_values = last_sequence_df[
            data_processor.numerical_columns + data_processor.categorical_columns + ['timestamp']
        ].values
        last_sequence = torch.tensor(sequence_values, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        predictions = model(last_sequence).cpu().numpy()

        numeric_features = data_processor.numerical_columns
        categorical_features = data_processor.categorical_columns
        all_features: List[str] = numeric_features + categorical_features + list(ADD_FEATURES.keys())

        predictions_df = pd.DataFrame(predictions, columns=all_features)
        scaled_numeric = data_processor.scaler.inverse_transform(predictions_df[numeric_features])
        for col in categorical_features:
            predictions_df[col] = predictions_df[col].astype(int)
            le = data_processor.label_encoders.get(col)
            if le:
                predictions_df[col] = le.inverse_transform(predictions_df[col])
        scaled_numeric = scaled_numeric.clip(lower=0)
        predictions_df[numeric_features] = scaled_numeric

        last_timestamp = latest_df["timestamp"].max()
        if pd.isna(last_timestamp):
            logging.error("Last timestamp is missing or invalid.")
            return pd.DataFrame()

        interval_key = get_interval(prediction_minutes)
        if interval_key is None:
            logging.error("Unknown interval: %d minutes.", prediction_minutes)
            return pd.DataFrame()

        next_timestamp = int(last_timestamp) + INTERVAL_MAPPING[interval_key]["milliseconds"]
        predictions_df["timestamp"] = next_timestamp
        predictions_df["symbol"] = TARGET_SYMBOL
        predictions_df["interval"] = prediction_minutes
        predictions_df["interval_str"] = interval_key

        try:
            predictions_df = predictions_df[list(RAW_FEATURES.keys()) + list(ADD_FEATURES.keys())]
        except KeyError as e:
            logging.error("Missing required columns: %s", e)
            return pd.DataFrame()

    return predictions_df


def main() -> None:
    setup_logging()
    device = get_device()
    if os.path.exists(DATA_PROCESSOR_FILENAME):
        data_processor = DataProcessor.load(DATA_PROCESSOR_FILENAME)
    else:
        data_processor = DataProcessor()

    data_fetcher = DataFetcher()
    combined_data = data_fetcher.load_data()
    combined_data = data_processor.preprocess_binance_data(combined_data)
    combined_data = combined_data.sort_values(by='timestamp').reset_index(drop=True)

    if not os.path.exists(DATA_PROCESSOR_FILENAME):
        combined_data = data_processor.fit_transform(combined_data)
        data_processor.save(DATA_PROCESSOR_FILENAME)
    else:
        combined_data = data_processor.transform(combined_data)

    logging.info("Columns in combined_data: %s", combined_data.columns.tolist())

    column_name_to_index = {col: idx for idx, col in enumerate(combined_data.columns)}
    model = EnhancedBiLSTMModel(
        numerical_columns=data_processor.numerical_columns,
        categorical_columns=data_processor.categorical_columns,
        column_name_to_index=column_name_to_index
    ).to(device)
    optimizer = Adam(model.parameters(), lr=TRAINING_PARAMS.get("initial_lr", 0.001))
    load_model(model, optimizer, MODEL_FILENAME, device)
    tensor_dataset = data_processor.prepare_dataset(combined_data, SEQ_LENGTH)
    dataloader = create_dataloader(tensor_dataset, TRAINING_PARAMS.get("batch_size", 32))
    model, optimizer = train_and_save_model(model, dataloader, device)
    
    latest_df = GetBinanceData().get_latest_dataset_prices(TARGET_SYMBOL, PREDICTION_MINUTES, SEQ_LENGTH)
    latest_df = latest_df.sort_values(by='timestamp').reset_index(drop=True)
    logging.info("Latest dataset:\n%s", latest_df)
    predicted_df = predict_future_price(model, latest_df, data_processor)
    logging.info("Predicted future prices:\n%s", predicted_df)


if __name__ == "__main__":
    while True:
        main()
