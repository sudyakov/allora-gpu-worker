import os
import logging
import time
from typing import Dict, Optional, Tuple, List, Union

import pandas as pd
import requests
from requests.exceptions import RequestException

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
    timestamp_to_readable_time,
    get_current_time,
)
from data_utils import (
    shared_data_processor
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

        self.data_processor = shared_data_processor

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
            df = pd.DataFrame(data)
            df.columns = self.BINANCE_API_COLUMNS
            df = df.drop(columns=['close_time', 'ignore'])
            df = self.data_processor.preprocess_binance_data(df)
            return df
        except RequestException as e:
            logging.warning(f"Request error: {e}")
            return None
        finally:
            time.sleep(1)

    def get_binance_data(
        self,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        symbols = [symbol] if symbol else list(self.SYMBOL_MAPPING.keys())
        intervals = [interval_key] if interval_key else list(self.INTERVAL_MAPPING.keys())
        all_data = []

        for sym in symbols:
            for interval in intervals:
                interval_info = self.get_interval_info(interval)
                current_start = start_time

                while current_start is None or (end_time is not None and current_start < end_time):
                    interval_str = f"{interval_info['minutes']}m"
                    url = f"{self.API_BASE_URL}/klines?symbol={sym}&interval={interval_str}&limit={self.BINANCE_LIMIT_STRING}"
                    if current_start:
                        url += f"&startTime={current_start}"
                    if end_time:
                        url += f"&endTime={end_time}"

                    df = self._fetch_data(url)
                    if df is None or df.empty:
                        break

                    df['symbol'] = sym
                    df['interval'] = interval_info['minutes']
                    df['interval_str'] = interval_str
                    logging.debug(f"Raw DataFrame: {df.head()}")

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

    def get_current_price(
        self,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None
    ) -> pd.DataFrame:
        symbols = [symbol] if symbol else list(self.SYMBOL_MAPPING.keys())
        intervals = [interval_key] if interval_key else list(self.INTERVAL_MAPPING.keys())
        all_data = []

        for sym in symbols:
            for interval in intervals:
                interval_info = self.get_interval_info(interval)
                interval_str = f"{interval_info['minutes']}m"
                url = f"{self.API_BASE_URL}/klines?symbol={sym}&interval={interval_str}&limit=1"

                df = self._fetch_data(url)
                if df is None or df.empty:
                    continue

                df['symbol'] = sym
                df['interval'] = interval_info['minutes']
                df['interval_str'] = interval_str
                logging.debug(f"Raw DataFrame: {df.head()}")

                df = self.data_processor.preprocess_binance_data(df)
                df = self.data_processor.fill_missing_add_features(df)
                all_data.append(df[list(self.MODEL_FEATURES.keys())])

        if not all_data:
            return pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))

        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    def prepare_dataframe_for_save(self, df: pd.DataFrame) -> pd.DataFrame:
        current_time, _ = get_current_time()
        df = self.data_processor.preprocess_binance_data(df[df['timestamp'] <= current_time])
        df = self.data_processor.fill_missing_add_features(df)
        df = self.data_processor.sort_dataframe(df)
        df = df.astype(self.MODEL_FEATURES)
        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        self.data_processor.ensure_file_exists(filename)
        prepared_df = self.prepare_dataframe_for_save(df)
        if not prepared_df.empty:
            prepared_df.to_csv(filename, index=False)
            logging.info(f"Data saved to {filename}")

    def save_combined_dataset(self, data: Dict[str, pd.DataFrame], filename: str):
        if not data:
            logging.warning("No data to save to the combined dataset.")
            return

        self.data_processor.ensure_file_exists(filename)
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            existing_data = pd.read_csv(filename, dtype=self.MODEL_FEATURES)
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

        combined_data = combined_data.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], keep='first')
        combined_data = combined_data.sort_values(['symbol', 'interval', 'timestamp'])

        combined_data = self.data_processor.fill_missing_add_features(combined_data)
        prepared_df = self.prepare_dataframe_for_save(combined_data)
        prepared_df.to_csv(filename, index=False)
        logging.info(f"Combined dataset updated: {filename}")
        logging.info("------------------------")

    def fetch_combined_data(self) -> pd.DataFrame:
        combined_path = self.PATHS['combined_dataset']
        if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
            try:
                df = pd.read_csv(combined_path, dtype=self.MODEL_FEATURES)
                logging.info(f"Combined data loaded from {combined_path}")
                return df
            except Exception as e:
                logging.error(f"Error loading combined data: {e}")
        else:
            logging.warning(f"Combined data file not found or is empty: {combined_path}")
        return pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))

    def print_data_summary(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None
    ):
        if symbol and interval_key:
            summary = f"Data summary for {symbol} ({interval_key}):\n"
        else:
            summary = "Data summary for all symbols and intervals:\n"
        feature_headers = ' '.join([f'{feature.capitalize():<10}' for feature in self.MODEL_FEATURES.keys()])
        summary += f"{'Timestamp':<20} {feature_headers}\n"

        if df.empty:
            logging.info("No data to display.")
            return

        rows_to_display = [df.iloc[0], df.iloc[-1]] if len(df) > 1 else [df.iloc[0]]

        for i, row in enumerate(rows_to_display):
            label = "First" if i == 0 else "Last"
            timestamp = row['timestamp']
            feature_values = ' '.join([
                f'{row[feature]:<10.6f}' if isinstance(row[feature], float) else f"{row[feature]:<10}"
                for feature in self.MODEL_FEATURES.keys()
            ])
            summary += f"{label:<20} {timestamp:<20} {feature_values}\n"

        logging.info(summary)


    def update_data(
        self,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None
    ) -> pd.DataFrame:
        symbols = [symbol] if symbol else list(self.SYMBOL_MAPPING.keys())
        intervals = [interval_key] if interval_key else list(self.INTERVAL_MAPPING.keys())
        all_updated_data = []

        for sym in symbols:
            for interval in intervals:
                interval_info = self.get_interval_info(interval)
                filename = os.path.join(self.PATHS['data_dir'], f"{sym}_{interval_info['minutes']}_data.csv")
                server_time, _ = get_current_time()

                dtype_dict = self.MODEL_FEATURES

                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    df_existing = pd.read_csv(filename, dtype=dtype_dict)
                    if not df_existing.empty:
                        df_existing = self.data_processor.fill_missing_add_features(df_existing)
                        last_timestamp = int(df_existing['timestamp'].max())
                    else:
                        last_timestamp = server_time - (interval_info['days'] * 24 * 60 * 60 * 1000)
                else:
                    df_existing = pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))
                    last_timestamp = server_time - (interval_info['days'] * 24 * 60 * 60 * 1000)

                start_time = last_timestamp + 1
                end_time = server_time

                # Проверяем, есть ли новые данные для загрузки
                if start_time >= end_time:
                    logging.info(f"No new data available for symbol {sym} and interval {interval_info['minutes']}m.")
                    all_updated_data.append(df_existing)
                    continue

                df_new = self.get_binance_data(sym, interval, start_time, end_time)

                if df_new is not None and not df_new.empty:
                    df_new = df_new.astype(dtype_dict)

                    readable_start = timestamp_to_readable_time(start_time)
                    readable_newest = timestamp_to_readable_time(df_new['timestamp'].max())
                    logging.info(
                        f"Updating data for {sym} from {readable_start} to {readable_newest}"
                    )

                    df_existing = df_existing.astype(dtype_dict)

                    df_updated = pd.concat(
                        [df_existing, df_new],
                        ignore_index=True
                    ).drop_duplicates(subset=['timestamp'], keep='first')
                    df_updated = self.data_processor.sort_dataframe(df_updated)
                    df_updated = self.data_processor.fill_missing_add_features(df_updated)

                    df_updated = df_updated.astype(dtype_dict)

                    self.save_to_csv(df_updated, filename)
                    all_updated_data.append(df_updated)
                else:
                    logging.warning(f"Failed to retrieve new data for {sym}.")
                    all_updated_data.append(df_existing)

        if all_updated_data:
            combined_df = pd.concat(all_updated_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame(columns=list(self.MODEL_FEATURES.keys()))


def main():
    logging.info("Script started")

    download_data = GetBinanceData()

    try:
        response = requests.get(f"{download_data.API_BASE_URL}/time")
        response.raise_for_status()
        server_time = response.json()['serverTime']
        readable_time = timestamp_to_readable_time(server_time)
        logging.info(f"Binance API is available. Server time: {readable_time}")
    except Exception as e:
        logging.error(f"Failed to access Binance API: {e}")
        return

    symbols = [None]
    intervals = [None]

    for symbol in symbols:
        for interval_key in intervals:
            try:
                updated_data = download_data.update_data(symbol, interval_key)
                if updated_data is not None and not updated_data.empty:
                    download_data.print_data_summary(updated_data, symbol, interval_key)

                    data_dict = {}
                    for sym in download_data.SYMBOL_MAPPING.keys():
                        for interval in download_data.INTERVAL_MAPPING.keys():
                            key = f"{sym}_{download_data.INTERVAL_MAPPING[interval]['minutes']}"
                            df_subset = updated_data[
                                (updated_data['symbol'] == sym) &
                                (updated_data['interval'] == download_data.INTERVAL_MAPPING[interval]['minutes'])
                            ]
                            if not df_subset.empty:
                                data_dict[key] = df_subset

                    download_data.save_combined_dataset(
                        data_dict,
                        download_data.PATHS['combined_dataset']
                    )
                else:
                    logging.error(f"Failed to update data for symbol {symbol} and interval {interval_key}")
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error updating data for symbol {symbol} with interval {interval_key}: {e}")

    logging.info("All files have been updated with the latest prices.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
    )
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
