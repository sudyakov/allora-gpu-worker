import pandas as pd
import os
from config import *
from utils import preprocess_binance_data
import logging
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_and_prepare_data():
    combined_data = pd.read_csv(PATHS['combined_dataset'])
    predictions_data = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
    differences_data = pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
    if os.path.exists(PATHS['predictions']):
        predictions_data = pd.read_csv(PATHS['predictions'])
    if os.path.exists(PATHS['differences']):
        differences_data = pd.read_csv(PATHS['differences'])
    for df in [combined_data, predictions_data, differences_data]:
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = preprocess_binance_data(df)
    logging.debug(f"Combined Data: {combined_data}")
    logging.debug(f"Predictions Data: {predictions_data}")
    logging.debug(f"Differences Data: {differences_data}")
    return combined_data, predictions_data, differences_data

def prepare_dataset(combined_dataset_path, seq_length=SEQ_LENGTH, target_symbol=TARGET_SYMBOL):
    df = pd.read_csv(combined_dataset_path)
    df = df[df['symbol'] == target_symbol].sort_values('timestamp')
    df = preprocess_binance_data(df)
    df['symbol'] = df['symbol'].astype('category')
    symbol_codes = df['symbol'].cat.codes
    df['interval'] = df['interval'].astype('category')
    numeric_columns = [col for col, dtype in FEATURE_NAMES.items() if dtype in [float, int]]
    categorical_columns = [col for col, dtype in FEATURE_NAMES.items() if dtype == str]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_numeric_data = pd.DataFrame(scaler.fit_transform(df[numeric_columns]), columns=numeric_columns, index=df.index)
    scaled_data = pd.concat([scaled_numeric_data, df[categorical_columns]], axis=1)
    scaler.scaled_columns = numeric_columns
    scaled_data['symbol'] = symbol_codes.values
    sequences = []
    labels = []
    for i in range(len(scaled_data) - seq_length):
        seq = scaled_data.iloc[i:i+seq_length].to_numpy().astype(float)
        label = scaled_data.iloc[i+seq_length].to_numpy().astype(float)
        sequences.append(seq)
        labels.append(label)
    sequences = np.array(sequences)
    labels = np.array(labels)
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    logging.debug(f"Prepared Dataset: sequences shape {sequences_tensor.shape}, labels shape {labels_tensor.shape}")
    return TensorDataset(sequences_tensor, labels_tensor), scaler, df
