import os
import logging
from time import sleep
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from config import *
from download_data import DownloadData
from utils import *
from visualize import *
from tqdm import tqdm

console = Console()
MODEL_FILENAME = os.path.join(PATHS['models_dir'], f'enhanced_bilstm_model_{TARGET_SYMBOL}_v{MODEL_VERSION}.pth')
download_data = DownloadData()

class Attention(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_layer_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_out):
        attention_scores = torch.matmul(lstm_out, self.attention_weights).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context_vector

class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size * 2, input_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        context_vector = self.attention(lstm_out)
        predictions = self.linear(context_vector)
        return predictions

def get_device():
    """Determines and returns the available device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_interval(minutes):
    """Returns the interval based on the given number of minutes."""
    return next(v['interval'] for k, v in INTERVALS_PERIODS.items() if v['minutes'] == minutes)

def prepare_dataset(csv_file, seq_length=SEQ_LENGTH, target_symbol=TARGET_SYMBOL):
    logging.info(f"Loading data from {csv_file} for symbol {target_symbol}")
    df = pd.read_csv(csv_file)
    df = df[df['symbol'] == target_symbol].sort_values('timestamp')
    
    if df.empty:
        logging.warning(f"No data found for symbol {target_symbol}")
        return None, None, None
    
    logging.info(f"Data loaded for symbol {target_symbol}, preprocessing data...")
    df = preprocess_binance_data(df)
    
    if df.empty:
        logging.warning(f"No data found for symbol {target_symbol} after preprocessing")
        return None, None, None
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    feature_columns = [col for col in FEATURE_NAMES.keys() if col not in ['timestamp', 'symbol', 'interval']]
    scaled_data = pd.DataFrame(scaler.fit_transform(df[feature_columns]), columns=feature_columns, index=df.index)
    
    if len(scaled_data) <= seq_length:
        logging.warning(f"Not enough data for symbol {target_symbol}. Need at least {seq_length + 1} rows, but got {len(scaled_data)}")
        return None, None, None
    
    sequences, labels = [], []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data.values[i:i+seq_length].astype(np.float64))
        labels.append(scaled_data.values[i+seq_length].astype(np.float64))
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    logging.info(f"Dataset prepared for symbol {target_symbol}")
    return TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(labels)), scaler, df

def load_training_data():
    """Loads training data from all three sources."""
    binance_data = pd.read_csv(PATHS['combined_dataset']).sort_values('timestamp', ascending=False)
    
    if os.path.getsize(PATHS['predictions']) > 0:
        predictions_data = pd.read_csv(PATHS['predictions']).sort_values('timestamp', ascending=False)
    else:
        predictions_data = pd.DataFrame(columns=FEATURE_NAMES.keys())
    
    if os.path.getsize(PATHS['differences']) > 0:
        differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
    else:
        differences_data = pd.DataFrame(columns=FEATURE_NAMES.keys())
    
    return binance_data, predictions_data, differences_data

def train_model(model, dataloader, device, differences_data, val_dataloader=None):
    """Trains the model on the provided data."""
    model.train()  # Ensure the model is in training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5  # Number of epochs with no improvement for early stopping
    min_lr = 1e-6  # Minimum learning rate

    for epoch in range(TRAINING_PARAMS['initial_epochs']):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAINING_PARAMS['initial_epochs']}", unit="batch")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            target_timestamps = targets[:, -1].cpu().numpy()
            differences = differences_data[differences_data['timestamp'].isin(target_timestamps)]
            if not differences.empty:
                differences = torch.FloatTensor(differences[FEATURE_NAMES].values).to(device)
                loss += criterion(outputs, differences)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{TRAINING_PARAMS['initial_epochs']}, Loss: {avg_loss:.4f}")

        # Validation check
        if val_dataloader:
            model.eval()  # Switch to evaluation mode for validation
            val_loss = validate_model(model, val_dataloader, device, criterion)
            model.train()  # Switch back to training mode after validation
            print(f"Validation Loss: {val_loss:.4f}")

            # Model adaptation
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_stop:
                    print("Early stopping")
                    break
                # Reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                    print(f"Learning rate reduced to {param_group['lr']}")

    return model

def validate_model(model, dataloader, device, criterion):
    """Validates the model on the validation dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def predict_future_price(model, last_sequence, scaler, steps=1):
    model.eval()
    with torch.no_grad():
        # Ensure the input sequence has 3 dimensions: (batch_size, seq_length, input_size)
        if last_sequence.dim() == 2:
            last_sequence = last_sequence.unsqueeze(0)
        
        input_sequence = last_sequence.to(next(model.parameters()).device)
        predictions = model(input_sequence)
        predicted_data = scaler.inverse_transform(predictions.cpu().numpy().astype(np.float64))
        predicted_data = np.abs(predicted_data)
    return predicted_data

def save_predictions_to_csv(predictions, filename, current_time):
    df = pd.DataFrame(predictions, columns=list(FEATURE_NAMES.keys()))
    df['timestamp'] = current_time
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = get_interval(PREDICTION_MINUTES)
    df.to_csv(filename, index=False, float_format='%.10f')
    return df.iloc[0]

def save_difference_to_csv(predictions, actuals, filename, current_time):
    difference = actuals - predictions
    df = pd.DataFrame(difference.reshape(1, -1), columns=list(FEATURE_NAMES.keys()))
    df['timestamp'] = current_time
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = get_interval(PREDICTION_MINUTES)
    df.to_csv(filename, index=False, float_format='%.10f')
    return df.iloc[0]

def ensure_file_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')
        if 'predictions' in filepath or 'differences' in filepath:
            columns = list(FEATURE_NAMES.keys())
            df = pd.DataFrame(columns=columns)
            df.to_csv(filepath, index=False)

def get_latest_value(data_file, target_symbol):
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES.keys())
    
    return filtered_df.iloc[0].to_frame().T

def print_combined_row(current_row, difference_row, predicted_next_row):
    table = Table(title="Current vs Predicted")
    table.add_column("Field", style="cyan")
    table.add_column("Current Value", style="magenta")
    table.add_column("Difference", style="yellow")
    table.add_column("Predicted next", style="green")
    
    for col in FEATURE_NAMES.keys():
        table.add_row(
            col,
            f"{current_row[col].iloc[0]:.10f}" if not current_row.empty and isinstance(current_row[col].iloc[0], float) else str(current_row[col].iloc[0]) if not current_row.empty else "N/A",
            f"{difference_row[col]:.10f}" if isinstance(difference_row[col], float) else str(difference_row[col]) if difference_row[col] is not None else "N/A",
            f"{predicted_next_row[col]:.10f}" if isinstance(predicted_next_row[col], float) else str(predicted_next_row[col]) if predicted_next_row[col] is not None else "N/A"
        )
    
    console.print(table)

def ensure_file_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')
        if 'predictions' in filepath or 'differences' in filepath:
            columns = list(FEATURE_NAMES.keys())
            df = pd.DataFrame(columns=columns)
            df.to_csv(filepath, index=False)

def get_predicted_prev_row(current_time: int, symbol: str) -> pd.Series:
    """Gets the previous prediction for the given time and symbol."""
    saved_predictions = pd.read_csv(PATHS['predictions']).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    predicted_prev_row = saved_predictions[
        (saved_predictions['symbol'] == symbol) &
        (saved_predictions['interval'] == interval) &
        (saved_predictions['timestamp'] == current_time)
    ]
    if not predicted_prev_row.empty:
        return predicted_prev_row.iloc[-1]
    else:
        differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
        if not differences_data.empty:
            return differences_data.iloc[-1]
        else:
            return pd.Series([None] * len(FEATURE_NAMES), index=FEATURE_NAMES.keys())

def get_latest_value(data_file, target_symbol):
    """Gets the latest available value for the given symbol from combined_dataset.csv."""
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        return pd.DataFrame(columns=list(FEATURE_NAMES.keys()))
    
    return filtered_df.iloc[0].to_frame().T

def get_latest_prediction(predictions_file, target_symbol):
    """Gets the latest prediction for the given symbol."""
    df = pd.read_csv(predictions_file).sort_values('timestamp', ascending=False)
    
    interval = get_interval(PREDICTION_MINUTES)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        return pd.Series()
    
    return filtered_df.iloc[0]

def get_difference_row(current_time: int, symbol: str) -> pd.Series:
    """Gets the difference row for the given time and symbol from differences.csv."""
    differences_data = pd.read_csv(PATHS['differences']).sort_values('timestamp', ascending=False)
    interval = get_interval(PREDICTION_MINUTES)
    difference_row = differences_data[
        (differences_data['symbol'] == symbol) &
        (differences_data['interval'] == interval) &
        (differences_data['timestamp'] == current_time)
    ]
    if not difference_row.empty:
        return difference_row.iloc[0]
    else:
        return pd.Series([None] * len(FEATURE_NAMES), index=FEATURE_NAMES.keys())

def save_difference_to_csv(predictions, actuals, filename, current_time):
    """Saves the difference between predicted and actual values to a CSV file."""
    difference = actuals - predictions
    df = pd.DataFrame(difference.reshape(1, -1), columns=FEATURE_NAMES)
    df['timestamp'] = current_time
    df['symbol'] = TARGET_SYMBOL
    df['interval'] = get_interval(PREDICTION_MINUTES)
    columns_order = FEATURE_NAMES + ['timestamp', 'symbol', 'interval']
    df = df[columns_order]
    
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, float_format='%.10f')
    else:
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([df, existing_df], ignore_index=True).sort_values('timestamp', ascending=False)
        combined_df.to_csv(filename, index=False, float_format='%.10f')
    
    return df.iloc[0]

def ensure_file_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write('')
        if 'predictions' in filepath or 'differences' in filepath:
            columns = list(FEATURE_NAMES.keys())
            df = pd.DataFrame(columns=columns)
            df.to_csv(filepath, index=False)

def save_model(model, filepath):
    """Saves the model to the specified file."""
    torch.save(model.state_dict(), filepath)
    console.print(f"Model saved to {filepath}", style="bold green")

def load_model(model, filepath, device):
    """Loads the model from the specified file."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        model.load_state_dict(torch.load(filepath, map_location=device))
        console.print(f"Model loaded from {filepath}", style="bold green")
    else:
        console.print(f"No model found at {filepath}. Starting with a new model.", style="bold yellow")

def get_latest_timestamp(data_file, target_symbol, prediction_minutes):
    """Gets the latest timestamp for the given symbol and interval."""
    df = pd.read_csv(data_file).sort_values('timestamp', ascending=False)
    interval = get_interval(prediction_minutes)
    filtered_df = df[(df['symbol'] == target_symbol) & (df['interval'] == interval)]
    
    if filtered_df.empty:
        return None
    
    return filtered_df['timestamp'].max()

def fill_missing_predictions(model, scaler, device, target_symbol, prediction_minutes):
    # Get the last saved prediction
    last_prediction = get_latest_prediction(PATHS['predictions'], target_symbol)
    last_prediction_time = last_prediction['timestamp'] if not last_prediction.empty else 0

    # Get the latest timestamp from the Binance archive
    latest_binance_time = get_latest_timestamp(PATHS['combined_dataset'], target_symbol, prediction_minutes)

    if latest_binance_time is None:
        logging.error(f"No latest timestamp found for {target_symbol} with interval {prediction_minutes} minutes.")
        return

    # Calculate the interval in milliseconds
    interval_ms = next(v['milliseconds'] for k, v in INTERVALS_PERIODS.items() if v['minutes'] == prediction_minutes)

    # Generate missing timestamps
    missing_timestamps = range(last_prediction_time + interval_ms, latest_binance_time + interval_ms, interval_ms)

    for timestamp in missing_timestamps:
        # Get the data sequence for prediction
        sequence = get_sequence_for_timestamp(timestamp - interval_ms, target_symbol, prediction_minutes)
        
        if sequence is not None:
                        # Convert the sequence to a tensor
            sequence_tensor = torch.FloatTensor(sequence.copy()).unsqueeze(0).to(device)
            
            # Make the prediction
            with torch.no_grad():
                prediction = model(sequence_tensor)
            
            # Inverse transform the scaling
            prediction = scaler.inverse_transform(prediction.cpu().numpy())
            
            # Save the prediction
            save_predictions_to_csv(prediction, PATHS['predictions'], timestamp)
            logging.info(f"Saved prediction for timestamp {timestamp}")

    logging.info(f"Filled {len(missing_timestamps)} missing predictions.")

def fill_missing_differences(model, scaler, device, target_symbol, prediction_minutes):
    # Get the last saved prediction
    last_prediction = get_latest_prediction(PATHS['predictions'], target_symbol)
    last_prediction_time = last_prediction['timestamp'] if not last_prediction.empty else 0

    # Get the latest timestamp from the Binance archive
    latest_binance_time = get_latest_timestamp(PATHS['combined_dataset'], target_symbol, prediction_minutes)

    if latest_binance_time is None:
        logging.error(f"No latest timestamp found for {target_symbol} with interval {prediction_minutes} minutes.")
        return

    # Calculate the interval in milliseconds
    interval_ms = next(v['milliseconds'] for k, v in INTERVALS_PERIODS.items() if v['minutes'] == prediction_minutes)

    # Generate missing timestamps
    missing_timestamps = range(last_prediction_time + interval_ms, latest_binance_time + interval_ms, interval_ms)

    for timestamp in missing_timestamps:
        # Get the data sequence for prediction
        sequence = get_sequence_for_timestamp(timestamp - interval_ms, target_symbol, prediction_minutes)
        
        if sequence is not None:
            # Convert the sequence to a tensor
            sequence_tensor = torch.FloatTensor(sequence.copy()).unsqueeze(0).to(device)
            
            # Make the prediction
            with torch.no_grad():
                prediction = model(sequence_tensor)
            
            # Inverse transform the scaling
            prediction = scaler.inverse_transform(prediction.cpu().numpy())
            
            # Get the actual data for the current time
            actuals = get_latest_value(PATHS['combined_dataset'], target_symbol)[FEATURE_NAMES].values
            
            # Save the difference
            save_difference_to_csv(prediction, actuals, PATHS['differences'], timestamp)
            logging.info(f"Saved difference for timestamp {timestamp}")

    logging.info(f"Filled {len(missing_timestamps)} missing differences.")

def get_sequence_for_timestamp(timestamp, target_symbol, prediction_minutes):
    df = pd.read_csv(PATHS['combined_dataset'])
    interval = get_interval(prediction_minutes)
    
    # Filter the data
    filtered_df = df[(df['symbol'] == target_symbol) & 
                    (df['interval'] == interval) & 
                    (df['timestamp'] <= timestamp)]
    
    if len(filtered_df) >= SEQ_LENGTH:
        sequence = filtered_df.sort_values('timestamp', ascending=False).head(SEQ_LENGTH)[FEATURE_NAMES].values
        return sequence[::-1]  # Return in the correct order
    return None

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    device = get_device()
    model = EnhancedBiLSTMModel(**MODEL_PARAMS).to(device)

    ensure_file_exists(PATHS['predictions'])
    ensure_file_exists(PATHS['differences'])
    ensure_file_exists(MODEL_FILENAME)

    load_model(model, MODEL_FILENAME, device)

    logging.info("Downloading latest data...")
    for symbol in SYMBOLS:
        for interval_name, interval_info in INTERVALS_PERIODS.items():
            logging.info(f"Updating data for {symbol} with interval {interval_name}...")
            download_data.update_data(symbol, interval_info['minutes'])

    logging.info("Preparing dataset...")
    dataset_result = prepare_dataset(PATHS['combined_dataset'])
    if dataset_result[0] is None:
        logging.error("Failed to prepare dataset. Skipping this iteration.")
        return

    dataset, scaler, df = dataset_result

    # Check for the presence of timestamp data in predictions.csv
    predictions_df = pd.read_csv(PATHS['predictions'])
    if predictions_df.empty or 'timestamp' not in predictions_df.columns or predictions_df['timestamp'].isnull().all():
        logging.info("Predictions file is empty or contains no timestamps. Saving the latest row from combined_dataset...")
        latest_row = df.iloc[-1]
        latest_row.to_frame().T.to_csv(PATHS['predictions'], index=False)
        logging.info(f"Saved the latest row to predictions: {latest_row}")
    else:
        # Continue with the normal prediction process
        dataloader = DataLoader(dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
        # Split the data into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=False)

        logging.info("Loading differences data...")
        _, _, differences_data = load_training_data()

        logging.info("Training model...")
        model = train_model(model, train_dataloader, device, differences_data, val_dataloader)
        save_model(model, MODEL_FILENAME)
        logging.info("Model training completed and saved.")

        logging.info("Predictions file is not empty and contains timestamps. Filling missing predictions...")
        fill_missing_predictions(model, scaler, device, TARGET_SYMBOL, PREDICTION_MINUTES)
        logging.info("Filled missing predictions.")

    # Load the most recent data after saving predictions
    current_time = get_latest_timestamp(PATHS['combined_dataset'], TARGET_SYMBOL, PREDICTION_MINUTES)
    current_value_row = get_latest_value(PATHS['combined_dataset'], TARGET_SYMBOL)
    latest_prediction_row = get_latest_prediction(PATHS['predictions'], TARGET_SYMBOL)

    # Check for the presence of actual data for the current time interval
    if not current_value_row.empty and current_value_row['timestamp'].iloc[0] == latest_prediction_row['timestamp']:
        actuals = current_value_row[FEATURE_NAMES].values
        predictions = latest_prediction_row[FEATURE_NAMES].values
        save_difference_to_csv(predictions, actuals, PATHS['differences'], current_time)
        logging.info("Difference between current value and previous prediction saved.")

    difference_row = get_difference_row(current_time, TARGET_SYMBOL)
    print_combined_row(current_value_row, difference_row, latest_prediction_row)
    logging.info("Completed printing combined row.")

if __name__ == "__main__":
    while True:
        logging.info("Starting main loop iteration...")
        main()
        sleep(3)
        logging.info("Creating visualization...")
        # create_visualization()
        logging.info("Visualization created.")
