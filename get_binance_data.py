import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Union, Tuple, TypedDict

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException

from config import (
    API_BASE_URL,
    BINANCE_LIMIT_STRING,
    INTERVAL_MAPPING,
    SYMBOL_MAPPING,
    PATHS,
    RAW_FEATURES,
    SCALED_FEATURES,
    ADD_FEATURES,
    MODEL_FEATURES
)

LOG_FILE = 'get_binance_data.log'

BINANCE_API_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]

class FeatureConfig(TypedDict, total=False):
    type: type
    precision: Optional[int]
    rounding: Optional[str]  # Например, 'up', 'down', 'nearest'

class GetBinanceData:
    def __init__(self):
        self.API_BASE_URL = API_BASE_URL
        self.BINANCE_LIMIT_STRING = BINANCE_LIMIT_STRING
        self.BINANCE_API_COLUMNS = BINANCE_API_COLUMNS
        self.INTERVAL_MAPPING = INTERVAL_MAPPING
        self.SYMBOL_MAPPING = SYMBOL_MAPPING
        self.PATHS = PATHS
        self.BINANCE_FEATURES = MODEL_FEATURES
        self.logger = logging.getLogger("GetBinanceData")
        self.configure_logging()

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_interval_info(self, interval: int) -> Dict[str, Union[str, int]]:
        for key, value in self.INTERVAL_MAPPING.items():
            if value['minutes'] == interval:
                return value
        raise KeyError(f"Invalid interval value: {interval}")

    def get_binance_data(
        self,
        symbol: str,
        interval: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        interval_info = self.get_interval_info(interval)
        interval_str = next(
            key for key, value in self.INTERVAL_MAPPING.items() if value['minutes'] == interval
        )
        all_data = []
        current_start = start_time

        while current_start is None or (end_time is not None and current_start < end_time):
            url = f"{self.API_BASE_URL}/klines?symbol={symbol}&interval={interval_str}&limit={self.BINANCE_LIMIT_STRING}"
            if current_start:
                url += f"&startTime={current_start}"
            if end_time:
                url += f"&endTime={end_time}"

            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                if not data:
                    break

                df = pd.DataFrame(data, columns=self.BINANCE_API_COLUMNS)
                df = df.drop(columns=['close_time', 'ignore'])
                df['symbol'] = symbol
                df['interval'] = interval_info['minutes']
                df['interval_str'] = interval_str
                self.logger.debug(f"Raw DataFrame: {df.head()}")
                df = preprocess_binance_data(df)
                df = fill_missing_model_features(df)
                all_data.append(df)
                current_start = int(df['timestamp'].iloc[-1]) + 1
            except RequestException as e:
                self.logger.warning(f"Error loading data for pair {symbol} and interval {interval}: {e}")
                break

            if start_time is None:
                break

        if not all_data:
            return pd.DataFrame(columns=list(self.BINANCE_FEATURES.keys()))

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = fill_missing_model_features(combined_df)
        return combined_df[list(self.BINANCE_FEATURES.keys())]

    def get_current_price(self, symbol: str, interval: int) -> pd.DataFrame:
        interval_info = self.get_interval_info(interval)
        interval_str = next(
            key for key, value in self.INTERVAL_MAPPING.items() if value['minutes'] == interval
        )
        url = f"{self.API_BASE_URL}/klines?symbol={symbol}&interval={interval_str}&limit=1"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame(columns=list(self.BINANCE_FEATURES.keys()))

            df = pd.DataFrame(data, columns=self.BINANCE_API_COLUMNS)
            df = df.drop(columns=['close_time', 'ignore'])
            df['symbol'] = symbol
            df['interval'] = interval_info['minutes']
            df['interval_str'] = interval_str
            self.logger.debug(f"Raw DataFrame: {df.head()}")
            df = preprocess_binance_data(df)
            df = fill_missing_model_features(df)
            return df[list(self.BINANCE_FEATURES.keys())]
        except RequestException as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return pd.DataFrame(columns=list(self.BINANCE_FEATURES.keys()))

    def prepare_dataframe_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        current_time, _ = get_current_time()
        df = preprocess_binance_data(df[df['timestamp'] <= current_time])
        df = fill_missing_model_features(df)
        return sort_dataframe(df)

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        ensure_file_exists(filename)
        prepared_df = self.prepare_dataframe_for_save(df)
        if not prepared_df.empty:
            prepared_df.to_csv(filename, index=False, float_format='%.6f')
        self.logger.info(f"Data saved to {filename}")

    def save_combined_dataset(self, data: Dict[str, pd.DataFrame], filename: str):
        if data:
            ensure_file_exists(filename)
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                existing_data = pd.read_csv(filename)
                for key, df in data.items():
                    symbol, interval = key.split('_')
                    mask = (existing_data['symbol'] == symbol) & (existing_data['interval'] == int(interval))
                    existing_data = existing_data[~mask]
                combined_data = pd.concat(
                    [existing_data.dropna(axis=1, how='all')] +
                    [df.dropna(axis=1, how='all') for df in data.values()],
                    ignore_index=True
                )
            else:
                combined_data = pd.concat(data.values(), ignore_index=True)

            combined_data = fill_missing_model_features(combined_data)
            prepared_df = self.prepare_dataframe_for_save(combined_data)
            prepared_df.to_csv(filename, index=False, float_format='%.6f')
            self.logger.info(f"Combined dataset updated: {filename}")
        else:
            self.logger.warning("No data to save to combined dataset.")

    def print_data_summary(self, df: pd.DataFrame, symbol: str, interval: int):
        summary = f"Data summary for {symbol} ({interval} minutes):\n"
        summary += f"{'Timestamp':<20} {' '.join([f'{feature.capitalize():<10}' for feature in self.BINANCE_FEATURES])}\n"
        rows_to_display = [df.iloc[0], df.iloc[-1]] if len(df) > 1 else [df.iloc[0]]
        for i, row in enumerate(rows_to_display):
            label = "First" if i == 0 else "Last"
            timestamp = row['timestamp']
            summary += (
                f"{label:<20} {timestamp:<20} "
                f"{' '.join([f'{row[feature]:<10.6f}' if isinstance(row[feature], float) else str(row[feature]) for feature in self.BINANCE_FEATURES])}\n"
            )
        self.logger.info(summary)

    def update_data(self, symbol: str, interval: int) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
        interval_info = self.get_interval_info(interval)
        filename = os.path.join(self.PATHS['data_dir'], f"{symbol}_{interval_info['minutes']}_data.csv")
        server_time, _ = get_current_time()
        df_existing = pd.DataFrame(columns=list(self.BINANCE_FEATURES.keys()))

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            df_existing = pd.read_csv(filename, dtype=self.BINANCE_FEATURES)
            df_existing = fill_missing_model_features(df_existing)

        last_timestamp = (
            int(df_existing['timestamp'].max())
            if not df_existing.empty
            else server_time - (interval_info['days'] * 24 * 60 * 60 * 1000)
        )
        time_difference = server_time - last_timestamp

        if time_difference > interval_info['minutes'] * 60 * 1000:
            start_time = last_timestamp + 1
            end_time = server_time
            df_new = self.get_binance_data(symbol, interval, start_time, end_time)
            if df_new is not None and not df_new.empty:
                newest_timestamp = df_new['timestamp'].max()
                self.logger.info(
                    f"Updating data for {symbol} from {start_time} to {newest_timestamp} "
                    f"({timestamp_to_readable_time(start_time)} to {timestamp_to_readable_time(newest_timestamp)})"
                )
                df_updated = pd.concat(
                    [df_new.dropna(axis=1, how='all'), df_existing.dropna(axis=1, how='all')],
                    ignore_index=True
                )
                df_updated = df_updated.drop_duplicates(subset=['timestamp'], keep='first')
                df_updated = sort_dataframe(df_updated)
                df_updated = fill_missing_model_features(df_updated)
                self.save_to_csv(df_updated, filename)
                self.save_combined_dataset(
                    {f"{symbol}_{interval_info['minutes']}": df_updated},
                    self.PATHS['combined_dataset']
                )
                update_start_time = df_new['timestamp'].min()
                update_end_time = df_new['timestamp'].max()
                return df_updated, update_start_time, update_end_time
            else:
                self.logger.warning(f"Failed to retrieve new data for {symbol}.")
                return df_existing, None, None
        else:
            self.logger.info(f"Data for {symbol} does not require updating. Using current data.")
            return df_existing, None, None

    def get_latest_prices(self, symbol: str, interval: int, count: int = 1) -> pd.DataFrame:
        combined_dataset_path = self.PATHS['combined_dataset']
        if os.path.exists(combined_dataset_path) and os.path.getsize(combined_dataset_path) > 0:
            df_combined = pd.read_csv(combined_dataset_path)
            df_filtered = df_combined[(df_combined['symbol'] == symbol) & (df_combined['interval'] == interval)]
            if not df_filtered.empty:
                df_filtered = df_filtered.sort_values('timestamp', ascending=False).head(count)
                return df_filtered
            else:
                self.logger.warning(f"Нет данных для символа {symbol} и интервала {interval} в combined_dataset.")
                return pd.DataFrame(columns=list(self.BINANCE_FEATURES.keys()))
        else:
            self.logger.warning(f"Файл combined_dataset.csv не найден по пути {combined_dataset_path}")
            return pd.DataFrame(columns=list(self.BINANCE_FEATURES.keys()))

def preprocess_binance_data(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = df['timestamp'].astype(int)
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if 'interval' in df.columns:
        df['interval'] = df['interval'].astype(int)
    for col, config in RAW_FEATURES.items():
        if col in df.columns:
            df[col] = df[col].astype(config['type'])
    float_cols = df.select_dtypes(include=['float', 'float64']).columns
    df[float_cols] = df[float_cols].round(6)
    logging.debug(f"Preprocessed DataFrame: {df.head()}")
    return df

def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values('timestamp', ascending=False)
    logging.debug(f"Sorted DataFrame: {sorted_df.head()}")
    return sorted_df

def timestamp_to_readable_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def get_current_time() -> Tuple[int, str]:
    response = requests.get(f"{API_BASE_URL}/time")
    response.raise_for_status()
    server_time = response.json().get('serverTime')
    readable_time = timestamp_to_readable_time(server_time)
    return server_time, readable_time

def ensure_file_exists(filepath: str) -> None:
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(RAW_FEATURES.keys()))
        df.to_csv(filepath, index=False)

def fill_missing_model_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df = df.ffill().bfill()
    for col, config in MODEL_FEATURES.items():
        if col in df.columns and 'precision' in config:
            precision = config['precision']
            rounding = config.get('rounding', 'nearest')
            if rounding == 'up':
                df[col] = np.ceil(df[col] * 10**precision) / 10**precision
            elif rounding == 'down':
                df[col] = np.floor(df[col] * 10**precision) / 10**precision
            else:
                df[col] = df[col].round(precision)
    logging.debug(f"Filled DataFrame: {df.head()}")
    return df

def main():
    download_data = GetBinanceData()
    download_data.logger.info("Script started")

    try:
        response = requests.get(f"{download_data.API_BASE_URL}/time")
        response.raise_for_status()
        server_time = response.json()['serverTime']
        readable_time = timestamp_to_readable_time(server_time)
        download_data.logger.info(f"Binance API is available. Server time: {readable_time}")
    except Exception as e:
        download_data.logger.error(f"Failed to access Binance API: {e}")
        return

    binance_data = {}
    symbols = list(download_data.SYMBOL_MAPPING.keys())
    intervals = [value['minutes'] for value in download_data.INTERVAL_MAPPING.values()]

    for symbol in symbols:
        for interval in intervals:
            try:
                updated_data, start_time, end_time = download_data.update_data(symbol, interval)
                if updated_data is not None and not updated_data.empty:
                    key = f"{symbol}_{interval}"
                    binance_data[key] = updated_data
                    download_data.print_data_summary(updated_data, symbol, interval)
                else:
                    download_data.logger.error(f"Failed to update data for pair {symbol} and interval {interval}")

                current_price_df = download_data.get_current_price(symbol, interval)
                download_data.logger.info(f"Current price for {symbol} ({interval} minutes):\n{current_price_df}")
                download_data.logger.info("------------------------")
                time.sleep(1)
            except Exception as e:
                download_data.logger.error(f"Error updating data for {symbol} with interval {interval}: {e}")

    if binance_data:
        download_data.save_combined_dataset(binance_data, download_data.PATHS['combined_dataset'])
        download_data.logger.info("All files updated with the latest prices.")
    else:
        download_data.logger.warning("No data to update.")

    download_data.logger.info("Executing get_current_price and get_latest_prices methods for all symbols and intervals.")
    for symbol in symbols:
        for interval in intervals:
            try:
                current_price_df = download_data.get_current_price(symbol, interval)
                download_data.logger.info(f"Current price for {symbol} ({interval} minutes):\n{current_price_df}")

                latest_price_df = download_data.get_latest_prices(symbol, interval)
                download_data.logger.info(f"Latest price for {symbol} ({interval} minutes):\n{latest_price_df}")

                time.sleep(1)
            except Exception as e:
                download_data.logger.error(f"Error fetching prices for {symbol} with interval {interval}: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info("Completed.")
