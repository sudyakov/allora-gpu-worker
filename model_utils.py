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

from data_utils import shared_data_processor


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


def save_model(model, optimizer, filename: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    logging.info(f"Model saved to {filename}")


def load_model(model, optimizer, filename: str, device: torch.device) -> None:
    if os.path.exists(filename):
        logging.info(f"Loading model from {filename}")
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])  # Извлекаем состояние модели
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Извлекаем состояние оптимизатора
        logging.info("Model and optimizer state loaded.")
    else:
        logging.info(f"No model file found at {filename}. Starting from scratch.")
