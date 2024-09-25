import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Union
import pandas as pd
import requests
from requests.exceptions import RequestException
from config import *
from utils import *
LOG_FILE = 'download_data.log'
class DownloadData:
    def __init__(self):
        self.API_BASE_URL = API_BASE_URL
        self.BINANCE_LIMIT_STRING = BINANCE_LIMIT_STRING
        self.BINANCE_API_COLUMNS = BINANCE_API_COLUMNS
        self.INTERVALS_PERIODS = INTERVALS_PERIODS
        self.SYMBOLS = SYMBOLS
        self.PATHS = PATHS
        self.FEATURE_NAMES = FEATURE_NAMES
        self.logger = logging.getLogger("DownloadData")
        self.configure_logging()
    def configure_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    def get_interval_info(self, interval: int) -> Dict[str, Union[str, int]]:
        for key, value in self.INTERVALS_PERIODS.items():
            if value['minutes'] == interval:
                return value
        raise KeyError(f"Invalid interval value: {interval}")
    def get_binance_data(self, symbol: str, interval: int, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: int = BINANCE_LIMIT_STRING) -> pd.DataFrame:
        interval_info = self.get_interval_info(interval)
        interval_str = next(key for key, value in self.INTERVALS_PERIODS.items() if value['minutes'] == interval)
        all_data = []
        current_start = start_time
        while current_start is None or current_start < end_time:
            url = f"{self.API_BASE_URL}/klines?symbol={symbol}&interval={interval_str}&limit={limit}"
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
                df = df.rename(columns={
                    'quote_asset_volume': 'quote_asset_volume',
                    'taker_buy_base_asset_volume': 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume': 'taker_buy_quote_asset_volume'
                })
                df = preprocess_binance_data(df)
                all_data.append(df)
                current_start = int(df['timestamp'].iloc[-1]) + 1
            except RequestException as e:
                self.logger.warning(f"Error loading data for pair {symbol} and interval: {e}")
                break
            if start_time is None:
                break
        if not all_data:
            return pd.DataFrame(columns=list(self.FEATURE_NAMES.keys()))
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df[list(self.FEATURE_NAMES.keys())]
    def get_current_price(self, symbol: str, interval: int) -> pd.DataFrame:
        interval_info = self.get_interval_info(interval)
        interval_str = next(key for key, value in self.INTERVALS_PERIODS.items() if value['minutes'] == interval)
        url = f"{self.API_BASE_URL}/klines?symbol={symbol}&interval={interval_str}&limit=1"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame(columns=list(self.FEATURE_NAMES.keys()))
            df = pd.DataFrame(data, columns=self.BINANCE_API_COLUMNS)
            df = df.drop(columns=['close_time', 'ignore'])
            df['symbol'] = symbol
            df['interval'] = interval_info['minutes']
            df = df.rename(columns={
                'quote_asset_volume': 'quote_asset_volume',
                'taker_buy_base_asset_volume': 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume': 'taker_buy_quote_asset_volume'
            })
            return preprocess_binance_data(df)
        except RequestException as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return pd.DataFrame(columns=list(self.FEATURE_NAMES.keys()))
    def prepare_dataframe_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        current_time, _ = get_current_time()
        df = preprocess_binance_data(df[df['timestamp'] <= current_time])
        return sort_dataframe(df)
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        ensure_file_exists(filename)
        prepared_df = self.prepare_dataframe_for_save(df)
        if not prepared_df.empty:
            prepared_df.to_csv(filename, index=False)
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
                combined_data = pd.concat([existing_data, *data.values()], ignore_index=True)
            else:
                combined_data = pd.concat(data.values(), ignore_index=True)
            prepared_df = self.prepare_dataframe_for_save(combined_data)
            prepared_df.to_csv(filename, index=False)
            self.logger.info(f"Combined dataset updated: {filename}")
        else:
            self.logger.warning("No data to save to combined dataset.")
    def print_data_summary(self, df: pd.DataFrame, symbol: str, interval: int):
        summary = f"Data summary for {symbol} ({interval} minutes):\n"
        summary += f"{'Timestamp':<20} {' '.join([f'{feature.capitalize():<10}' for feature in self.FEATURE_NAMES])}\n"
        rows_to_display = [df.iloc[0], df.iloc[-1]] if len(df) > 1 else [df.iloc[0]]
        for i, row in enumerate(rows_to_display):
            label = "First" if i == 0 else "Last"
            timestamp = row['timestamp']
            summary += f"{label:<20} {timestamp:<20} {' '.join([f'{row[feature]:<10.2f}' if isinstance(row[feature], float) else str(row[feature]) for feature in self.FEATURE_NAMES])}\n"
        self.logger.info(summary)
    def update_data(self, symbol: str, interval: int):
        interval_info = self.get_interval_info(interval)
        filename = f"{self.PATHS['data_dir']}/{symbol}_{interval_info['minutes']}_data.csv"
        server_time, _ = get_current_time()
        df_existing = pd.DataFrame(columns=list(self.FEATURE_NAMES.keys()))
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            df_existing = pd.read_csv(filename, dtype=self.FEATURE_NAMES)
        last_timestamp = df_existing['timestamp'].max() if not df_existing.empty else server_time - (interval_info['days'] * 24 * 60 * 60 * 1000)
        last_timestamp = int(last_timestamp)
        time_difference = server_time - last_timestamp
        if time_difference > interval_info['minutes'] * 60 * 1000:
            start_time = last_timestamp + 1
            end_time = server_time
            df_new = self.get_binance_data(symbol, interval, start_time, end_time)
            if df_new is not None and not df_new.empty:
                newest_timestamp = df_new['timestamp'].max()
                self.logger.info(
                    f"Updating data for {symbol} from {start_time} to {newest_timestamp} ({timestamp_to_readable_time(start_time)} to {timestamp_to_readable_time(newest_timestamp)})"
                )
                df_updated = pd.concat([df_new, df_existing], ignore_index=True)
                df_updated = df_updated.drop_duplicates(subset=['timestamp'], keep='first')
                df_updated = sort_dataframe(df_updated)
                self.save_to_csv(df_updated, filename)
                self.save_combined_dataset({f"{symbol}_{interval_info['minutes']}": df_updated}, self.PATHS['combined_dataset'])
                df_existing = df_updated
                update_start_time = df_new['timestamp'].min()
                update_end_time = df_new['timestamp'].max()
                return df_existing, update_start_time, update_end_time
            else:
                self.logger.warning(f"Failed to fetch new data for {symbol}.")
                return df_existing, None, None
        else:
            self.logger.info(f"Data for {symbol} does not require updating. Using current data.")
            return df_existing, None, None
def main():
    download_data = DownloadData()
    download_data.logger.info("Script started")
    try:
        response = requests.get(f"{download_data.API_BASE_URL}/time")
        response.raise_for_status()
        server_time = response.json()['serverTime']
        readable_time = datetime.fromtimestamp(server_time / 1000, timezone.utc).strftime(DATETIME_FORMAT)
        download_data.logger.info(f"Binance API is accessible. Server time: {readable_time}")
    except Exception as e:
        download_data.logger.error(f"Unable to access Binance API: {e}")
        return
    binance_data = {}
    for symbol in download_data.SYMBOLS:
        for interval in [value['minutes'] for value in download_data.INTERVALS_PERIODS.values()]:
            try:
                updated_data, start_time, end_time = download_data.update_data(symbol, interval)
                if updated_data is not None and not updated_data.empty:
                    binance_data[f"{symbol}_{interval}"] = updated_data
                    download_data.print_data_summary(updated_data, symbol, interval)
                else:
                    download_data.logger.error(f"Failed to update data for pair {symbol} and interval {interval}")
                current_price_df = download_data.get_current_price(symbol, interval)
                download_data.logger.info(current_price_df)
                download_data.logger.info("------------------------")
                time.sleep(1)
            except Exception as e:
                download_data.logger.error(f"Error updating data for {symbol} at interval {interval}: {e}")
    if binance_data:
        download_data.save_combined_dataset(binance_data, download_data.PATHS['combined_dataset'])
        download_data.logger.info("All files updated with the latest prices.")
    else:
        download_data.logger.warning("No data to update.")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info("Done.")
