import os

import torch
import logging
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Union, List, Any, OrderedDict, Sequence, TypedDict, Literal
from datetime import datetime, timezone
import requests
import pickle

from config import (
    SEQ_LENGTH,
    TARGET_SYMBOL,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    ADD_FEATURES,
    MODEL_FILENAME,
    DATA_PROCESSOR_FILENAME,
    PATHS,
    SYMBOL_MAPPING,
    API_BASE_URL,
    IntervalConfig,
    TRAINING_PARAMS
)
from data_utils import DataProcessor


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def create_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TRAINING_PARAMS["num_workers"],
    )


def save_model(model: nn.Module, optimizer: Adam, filepath: str) -> None:
    DataProcessor.ensure_file_exists(filepath)
    torch.save(model.state_dict(), filepath)
    optimizer_filepath = filepath.replace(".pth", "_optimizer.pth")
    torch.save(optimizer.state_dict(), optimizer_filepath)
    logging.info(f"Model saved to {filepath}, and optimizer to {optimizer_filepath}.")


def load_model(
    model: nn.Module,
    optimizer: Adam,
    filepath: str,
    device: torch.device,
) -> None:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        state_dict = torch.load(filepath, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        optimizer_filepath = filepath.replace(".pth", "_optimizer.pth")
        if os.path.exists(optimizer_filepath):
            optimizer_state_dict = torch.load(optimizer_filepath, map_location=device, weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            logging.info(f"Optimizer loaded from {optimizer_filepath}.")
        logging.info(f"Model loaded from {filepath}.")
    else:
        logging.warning(f"Model file {filepath} not found. A new model will be created.")
