import os
import logging
import time
from typing import Dict, Optional, Tuple, TypedDict, Literal, OrderedDict

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from collections import OrderedDict

from config import (
    API_BASE_URL,
    BINANCE_LIMIT_STRING,
    INTERVAL_MAPPING,
    SYMBOL_MAPPING,
    TARGET_SYMBOL,
    PREDICTION_MINUTES,
    MODEL_FEATURES,
    PATHS,
    IntervalConfig,
    IntervalKey,
    RAW_FEATURES,
    TIME_FEATURES,
    SCALABLE_FEATURES,
    ADD_FEATURES,
)
from data_utils import (
    DataProcessor,
    get_current_time,
    timestamp_to_readable_time,
)


LOG_FILE = 'get_binance_data.log'

BINANCE_API_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]


class GetBinanceData:
    def __init__(self):
        self.API_BASE_URL = API_BASE_URL
        self.BINANCE_LIMIT_STRING = BINANCE_LIMIT_STRING
        self.BINANCE_API_COLUMNS = BINANCE_API_COLUMNS
        self.INTERVAL_MAPPING = INTERVAL_MAPPING
        self.SYMBOL_MAPPING = SYMBOL_MAPPING
        self.TARGET_SYMBOL = TARGET_SYMBOL
        self.PREDICTION_MINUTES = PREDICTION_MINUTES
        self.MODEL_FEATURES = MODEL_FEATURES
        self.PATHS = PATHS

        self.logger = logging.getLogger("GetBinanceData")
        self.configure_logging()
        self.data_processor = DataProcessor()

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_interval_info(self, interval_key: IntervalKey) -> IntervalConfig:
        if interval_key in self.INTERVAL_MAPPING:
            return self.INTERVAL_MAPPING[interval_key]
        else:
            raise KeyError(f"Invalid interval key: {interval_key}")

    def _fetch_data(self, url: str) -> Optional[pd.DataFrame]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data:
                return None
            df = pd.DataFrame(data, columns=self.BINANCE_API_COLUMNS)
            df = df.drop(columns=['close_time', 'ignore'])
            return df
        except RequestException as e:
            self.logger.warning(f"Request error: {e}")
            return None

    def get_binance_data(
        self,
        symbol: str,
        interval_key: IntervalKey,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        interval_info = self.get_interval_info(interval_key)
        all_data = []
        current_start = start_time

        while current_start is None or (end_time is not None and current_start < end_time):
            url = f"{self.API_BASE_URL}/klines?symbol={symbol}&interval={interval_key}&limit={self.BINANCE_LIMIT_STRING}"
            if current_start:
                url += f"&startTime={current_start}"
            if end_time:
                url += f"&endTime={end_time}"

            df = self._fetch_data(url)
            if df is None:
                break

            df['symbol'] = symbol
            df['interval'] = interval_info['minutes']
            df['interval_str'] = interval_key
            self.logger.debug(f"Raw DataFrame: {df.head()}")

            df = self.data_processor.preprocess_binance_data(df)
            df = self.data_processor.fill_missing_add_features(df)
            all_data.append(df)
            current_start = int(df['timestamp'].iloc[-1]) + 1

            if start_time is None:
                break

        if not all_data:
            return pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = self.data_processor.fill_missing_add_features(combined_df)
        return combined_df[list(self.MODEL_FEATURES.keys())]

    def get_current_price(self, symbol: str, interval_key: IntervalKey) -> pd.DataFrame:
        interval_info = self.get_interval_info(interval_key)
        url = f"{self.API_BASE_URL}/klines?symbol={symbol}&interval={interval_key}&limit=1"

        df = self._fetch_data(url)
        if df is None:
            return pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))

        df['symbol'] = symbol
        df['interval'] = interval_info['minutes']
        df['interval_str'] = interval_key
        self.logger.debug(f"Raw DataFrame: {df.head()}")

        df = self.data_processor.preprocess_binance_data(df)
        df = self.data_processor.fill_missing_add_features(df)
        return df[list(self.MODEL_FEATURES.keys())]

    def prepare_dataframe_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        current_time, _ = get_current_time()
        df = self.data_processor.preprocess_binance_data(df[df['timestamp'] <= current_time])
        df = self.data_processor.fill_missing_add_features(df)
        df = self.data_processor.sort_dataframe(df)
        
        # Приведение числовых колонок к np.float32
        numeric_columns = df.select_dtypes(include=['float32', 'float64']).columns
        df[numeric_columns] = df[numeric_columns].astype(np.float32)

        return df


    def save_to_csv(self, df: pd.DataFrame, filename: str):
        self.data_processor.ensure_file_exists(filename)
        prepared_df = self.prepare_dataframe_for_save(df)
        if not prepared_df.empty:
            prepared_df.to_csv(filename, index=False)
            self.logger.info(f"Data saved to {filename}")

    def save_combined_dataset(self, data: Dict[str, pd.DataFrame], filename: str):
        if not data:
            self.logger.warning("No data to save to combined dataset.")
            return

        self.data_processor.ensure_file_exists(filename)
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            existing_data = pd.read_csv(filename)
            combined_data = pd.concat(
                [existing_data] +
                [df.dropna(axis=1, how='all') for df in data.values() if not df.empty],
                ignore_index=True
            )
        else:
            combined_data = pd.concat(
                [df.dropna(axis=1, how='all') for df in data.values() if not df.empty],
                ignore_index=True
            )

        # Удаляем дубликаты по 'timestamp', 'symbol' и 'interval'
        combined_data = combined_data.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], keep='first')
        combined_data = combined_data.sort_values(['symbol', 'interval', 'timestamp'])


        combined_data = self.data_processor.fill_missing_add_features(combined_data)
        prepared_df = self.prepare_dataframe_for_save(combined_data)
        prepared_df.to_csv(filename, index=False)
        self.logger.info(f"Combined dataset updated: {filename}")


    def fetch_combined_data(self) -> pd.DataFrame:
        combined_path = self.PATHS['combined_dataset']
        if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
            try:
                df = pd.read_csv(combined_path)
                self.logger.info(f"Combined data loaded from {combined_path}")
                return df
            except Exception as e:
                self.logger.error(f"Error loading combined data: {e}")
        else:
            self.logger.warning(f"Combined data file not found or empty: {combined_path}")
        return pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))

    def print_data_summary(self, df: pd.DataFrame, symbol: str, interval_key: IntervalKey):
        summary = f"Data summary for {symbol} ({interval_key}):\n"
        feature_headers = ' '.join([f'{feature.capitalize():<10}' for feature in self.MODEL_FEATURES.keys()])
        summary += f"{'Timestamp':<20} {feature_headers}\n"
        rows_to_display = [df.iloc[0], df.iloc[-1]] if len(df) > 1 else [df.iloc[0]]

        for i, row in enumerate(rows_to_display):
            label = "First" if i == 0 else "Last"
            timestamp = row['timestamp']
            feature_values = ' '.join([
                f'{row[feature]:<10.6f}' if isinstance(row[feature], float) else f"{row[feature]:<10}"
                for feature in self.MODEL_FEATURES.keys()
            ])
            summary += f"{label:<20} {timestamp:<20} {feature_values}\n"

        self.logger.info(summary)

    def update_data(self, symbol: str, interval_key: IntervalKey) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
        interval_info = self.get_interval_info(interval_key)
        filename = os.path.join(self.PATHS['data_dir'], f"{symbol}_{interval_info['minutes']}_data.csv")
        server_time, _ = get_current_time()

        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            df_existing = pd.read_csv(filename, dtype=self.MODEL_FEATURES)
            df_existing = self.data_processor.fill_missing_add_features(df_existing)
            last_timestamp = int(df_existing['timestamp'].max())
        else:
            df_existing = pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))
            last_timestamp = server_time - (interval_info['days'] * 24 * 60 * 60 * 1000)

        time_difference = server_time - last_timestamp

        if time_difference > interval_info['milliseconds']:
            start_time = last_timestamp + 1
            end_time = server_time
            df_new = self.get_binance_data(symbol, interval_key, start_time, end_time)

            if df_new is not None and not df_new.empty:
                readable_start = timestamp_to_readable_time(start_time)
                readable_newest = timestamp_to_readable_time(df_new['timestamp'].max())
                self.logger.info(
                    f"Updating data for {symbol} from {start_time} to {df_new['timestamp'].max()} "
                    f"({readable_start} to {readable_newest})"
                )
                df_updated = pd.concat(
                    [df_existing, df_new],
                    ignore_index=True
                ).drop_duplicates(subset=['timestamp'], keep='first')
                df_updated = self.data_processor.sort_dataframe(df_updated)
                df_updated = self.data_processor.fill_missing_add_features(df_updated)
                self.save_to_csv(df_updated, filename)
                self.save_combined_dataset(
                    {f"{symbol}_{interval_info['minutes']}": df_updated},
                    self.PATHS['combined_dataset']
                )
                return df_updated, df_new['timestamp'].min(), df_new['timestamp'].max()
            else:
                self.logger.warning(f"Failed to retrieve new data for {symbol}.")
                return df_existing, None, None
        else:
            self.logger.info(f"Data for {symbol} does not require updating. Using current data.")
            return df_existing, None, None


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

    symbols = list(download_data.SYMBOL_MAPPING.keys())
    intervals = list(download_data.INTERVAL_MAPPING.keys())

    for symbol in symbols:
        for interval_key in intervals:
            try:
                updated_data, start_time, end_time = download_data.update_data(symbol, interval_key)
                if updated_data is not None and not updated_data.empty:
                    download_data.print_data_summary(updated_data, symbol, interval_key)
                    
                    # Получаем ключ для текущего символа и интервала
                    key = f"{symbol}_{download_data.INTERVAL_MAPPING[interval_key]['minutes']}"
                    
                    # Сохраняем данные сразу после обновления
                    download_data.save_combined_dataset(
                        {key: updated_data},
                        download_data.PATHS['combined_dataset']
                    )
                else:
                    download_data.logger.error(f"Failed to update data for pair {symbol} and interval {interval_key}")
                download_data.logger.info("------------------------")
                time.sleep(1)
            except Exception as e:
                download_data.logger.error(f"Error updating data for {symbol} with interval {interval_key}: {e}")

    download_data.logger.info("All files updated with the latest prices.")

    download_data.logger.info("Executing get_current_price and get_latest_prices methods for all symbols and intervals.")
    for symbol in symbols:
        for interval_key in intervals:
            try:
                current_price_df = download_data.get_current_price(symbol, interval_key)
                download_data.logger.info(f"Current price for {symbol} ({interval_key}):\n{current_price_df}")
                
                interval_minutes = download_data.INTERVAL_MAPPING[interval_key]['minutes']
                latest_price_df = download_data.data_processor.get_latest_dataset_prices(symbol, interval_minutes)

                download_data.logger.info(f"Latest price for {symbol} ({interval_key}):\n{latest_price_df}")

                time.sleep(1)
            except Exception as e:
                download_data.logger.error(f"Error fetching prices for {symbol} with interval {interval_key}: {e}")

    download_data.logger.info("Completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
