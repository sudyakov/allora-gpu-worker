import os
import logging
from time import sleep
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from config import *
from download_data import DownloadData
from utils import *
from visualize import create_visualization
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
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataloader, device, differences_data, val_dataloader=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_PARAMS['initial_lr'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5
    min_lr = 1e-6

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
                differences = torch.FloatTensor(differences[list(FEATURE_NAMES.keys())].values).to(device)
                loss += criterion(outputs, differences)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{TRAINING_PARAMS['initial_epochs']}, Loss: {avg_loss:.4f}")

        if val_dataloader:
            model.eval()
            val_loss = validate_model(model, val_dataloader, device, criterion)
            model.train()
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_stop:
                    print("Early stopping")
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, min_lr)
                    print(f"Learning rate reduced to {param_group['lr']}")

    return model

def validate_model(model, dataloader, device, criterion):
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
        if last_sequence.dim() == 2:
            last_sequence = last_sequence.unsqueeze(0)
        
        input_sequence = last_sequence.to(next(model.parameters()).device)
        predictions = model(input_sequence)
        
        numeric_columns = scaler.feature_names_in_
        scaled_predictions = predictions.cpu().numpy()[:, [list(FEATURE_NAMES.keys()).index(col) for col in numeric_columns]]
        inverse_scaled = scaler.inverse_transform(scaled_predictions)
        
        predicted_data = predictions.cpu().numpy()
        for i, col in enumerate(numeric_columns):
            predicted_data[:, list(FEATURE_NAMES.keys()).index(col)] = inverse_scaled[:, i]
        
        predicted_data = np.abs(predicted_data)
        
        symbol_index = list(FEATURE_NAMES.keys()).index('symbol')
        predicted_data[:, symbol_index] = np.round(predicted_data[:, symbol_index]).astype(int)
    
    return predicted_data

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    console.print(f"Model saved to {filepath}", style="bold green")

def load_model(model, filepath, device):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        model.load_state_dict(torch.load(filepath, map_location=device))
        console.print(f"Model loaded from {filepath}", style="bold green")
    else:
        console.print(f"No model found at {filepath}. Starting with a new model.", style="bold yellow")

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
    dataset, scaler, df = prepare_dataset(PATHS['combined_dataset'])
    dataloader = DataLoader(dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)

    logging.info("Loading differences data...")
    _, _, differences_data = load_and_prepare_data()

    logging.info("Training model...")
    model = train_model(model, dataloader, device, differences_data)
    save_model(model, MODEL_FILENAME)
    logging.info("Model training completed and saved.")

    current_time = get_latest_timestamp(PATHS['combined_dataset'], TARGET_SYMBOL, PREDICTION_MINUTES)
    if current_time is None:
        logging.error("No data found for the specified symbol and interval.")
        return

    last_sequence = dataset[-1][0]
    predictions = predict_future_price(model, last_sequence, scaler)
    logging.info(f"Predictions: {predictions}")

    readable_server_time = get_current_time()
    logging.info(f"Current Binance server time: {readable_server_time}")

    saved_prediction = save_predictions_to_csv(predictions, PATHS['predictions'], current_time)
    logging.info(f"Saved predictions to CSV: {saved_prediction}")

    current_value_row = get_latest_value(PATHS['combined_dataset'], TARGET_SYMBOL)
    latest_prediction_row = get_latest_prediction(PATHS['predictions'], TARGET_SYMBOL)
    difference_row = get_difference_row(current_time, TARGET_SYMBOL)

    if not current_value_row.empty and not latest_prediction_row.empty:
        actuals = current_value_row[list(FEATURE_NAMES.keys())].values
        predictions = latest_prediction_row[list(FEATURE_NAMES.keys())].values
        save_difference_to_csv(predictions, actuals, PATHS['differences'], current_time)
        logging.info("Difference between current value and previous prediction saved.")

        difference_row = get_difference_row(current_time, TARGET_SYMBOL)
    else:
        logging.warning("Unable to compare current value with previous prediction due to missing data.")

    print_combined_row(current_value_row, difference_row, latest_prediction_row)
    logging.info("Completed printing combined row.")

if __name__ == "__main__":
    while True:
        logging.info("Starting main loop iteration...")
        main()
        sleep(3)
        logging.info("Creating visualization...")
        create_visualization()
        logging.info("Visualization created.")
