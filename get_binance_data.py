import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.exceptions import RequestException

from config import (
    ADD_FEATURES,
    API_BASE_URL,
    BINANCE_LIMIT_STRING,
    INTERVAL_MAPPING,
    MODEL_FEATURES,
    PATHS,
    PREDICTION_MINUTES,
    RAW_FEATURES,
    SCALABLE_FEATURES,
    SEQ_LENGTH,
    SYMBOL_MAPPING,
    TARGET_SYMBOL,
    get_current_time,
    timestamp_to_readable_time,
    IntervalConfig,
    IntervalKey,
)
from data_utils import shared_data_processor

BINANCE_API_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
]


class GetBinanceData:
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.binance_limit_string = BINANCE_LIMIT_STRING
        self.binance_api_columns = BINANCE_API_COLUMNS
        self.interval_mapping = INTERVAL_MAPPING
        self.symbol_mapping = SYMBOL_MAPPING
        self.target_symbol = TARGET_SYMBOL
        self.prediction_minutes = PREDICTION_MINUTES
        self.model_features = MODEL_FEATURES
        self.paths = PATHS
        self.data_processor = shared_data_processor

    def get_interval_info(self, interval_key: IntervalKey) -> IntervalConfig:
        if interval_key in self.interval_mapping:
            return self.interval_mapping[interval_key]
        else:
            raise KeyError(f"Invalid interval key: {interval_key}")

    def _fetch_data(self, url: str) -> Optional[pd.DataFrame]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if not data:
                return None
            raw_data_df = pd.DataFrame(data)
            raw_data_df.columns = self.binance_api_columns
            raw_data_df = raw_data_df.drop(columns=['close_time', 'ignore'])
            raw_data_df = self.data_processor.preprocess_binance_data(raw_data_df)
            return raw_data_df
        except RequestException as e:
            logging.warning(f"Request error: {e}")
            return None

    def get_binance_data(
        self,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        symbols = [symbol] if symbol else list(self.symbol_mapping.keys())
        intervals = [interval_key] if interval_key else list(self.interval_mapping.keys())
        all_fetched_data = []

        for sym in symbols:
            for interval in intervals:
                interval_info = self.get_interval_info(interval)
                current_start_time = start_time

                while current_start_time is None or (end_time is not None and current_start_time < end_time):
                    interval_str = f"{interval_info['minutes']}m"
                    url = f"{self.api_base_url}/klines?symbol={sym}&interval={interval_str}&limit={self.binance_limit_string}"
                    if current_start_time:
                        url += f"&startTime={current_start_time}"
                    if end_time:
                        url += f"&endTime={end_time}"

                    fetched_data_df = self._fetch_data(url)
                    if fetched_data_df is None or fetched_data_df.empty:
                        break

                    fetched_data_df['symbol'] = sym
                    fetched_data_df['interval'] = interval_info['minutes']
                    logging.debug(f"Fetched DataFrame: {fetched_data_df.tail()}")

                    fetched_data_df = self.data_processor.fill_missing_add_features(fetched_data_df)
                    all_fetched_data.append(fetched_data_df)
                    current_start_time = int(fetched_data_df['timestamp'].iloc[-1]) + 1

                    if start_time is None:
                        break

        if not all_fetched_data:
            return pd.DataFrame(columns=list(self.model_features.keys()))

        combined_real_data_df = pd.concat(all_fetched_data, ignore_index=True)
        combined_real_data_df = self.data_processor.fill_missing_add_features(combined_real_data_df)
        return combined_real_data_df[list(self.model_features.keys())]

    def get_current_price(
        self,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None
    ) -> pd.DataFrame:
        symbols = [symbol] if symbol else list(self.symbol_mapping.keys())
        intervals = [interval_key] if interval_key else list(self.interval_mapping.keys())
        all_current_prices = []

        for sym in symbols:
            for interval in intervals:
                interval_info = self.get_interval_info(interval)
                interval_str = f"{interval_info['minutes']}m"
                url = f"{self.api_base_url}/klines?symbol={sym}&interval={interval_str}&limit=1"

                fetched_current_price_df = self._fetch_data(url)
                if fetched_current_price_df is None or fetched_current_price_df.empty:
                    continue

                fetched_current_price_df['symbol'] = sym
                fetched_current_price_df['interval'] = interval_info['minutes']
                logging.debug(f"Current Price DataFrame: {fetched_current_price_df.tail()}")

                fetched_current_price_df = self.data_processor.fill_missing_add_features(fetched_current_price_df)
                all_current_prices.append(fetched_current_price_df[list(self.model_features.keys())])

        if not all_current_prices:
            return pd.DataFrame(columns=list(self.model_features.keys()))

        combined_current_prices_df = pd.concat(all_current_prices, ignore_index=True)
        return combined_current_prices_df

    def prepare_dataframe_for_save(self, data_df: pd.DataFrame) -> pd.DataFrame:
        current_time, _ = get_current_time()
        data_df = data_df[data_df['timestamp'] <= current_time]
        data_df = self.data_processor.preprocess_binance_data(data_df)
        data_df = self.data_processor.fill_missing_add_features(data_df)
        data_df = self.data_processor.sort_dataframe(data_df)
        data_df = data_df.astype(self.model_features)
        return data_df

    def save_to_csv(self, data_df: pd.DataFrame, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        prepared_data_df = self.prepare_dataframe_for_save(data_df)
        if not prepared_data_df.empty:
            prepared_data_df.to_csv(filename, index=False)
            logging.info(f"Data saved to {filename}")

    def save_combined_dataset(self, data_dict: Dict[str, pd.DataFrame], filename: str):
        if not data_dict:
            logging.warning("No data to save to the combined dataset.")
            return

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        existing_data_df = pd.DataFrame()
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            existing_data_df = pd.read_csv(filename, dtype=self.model_features)

        combined_real_data_df = pd.concat(
            [existing_data_df] + [df for df in data_dict.values() if not df.empty],
            ignore_index=True
        )
        combined_real_data_df = combined_real_data_df.drop_duplicates(subset=['timestamp', 'symbol', 'interval'], keep='first')
        combined_real_data_df = self.data_processor.sort_dataframe(combined_real_data_df)
        combined_real_data_df = self.data_processor.fill_missing_add_features(combined_real_data_df)
        prepared_combined_data_df = self.prepare_dataframe_for_save(combined_real_data_df)
        prepared_combined_data_df.to_csv(filename, index=False)
        logging.info(f"Combined dataset updated: {filename}")

    def fetch_combined_data(self) -> pd.DataFrame:
        combined_real_data_path = self.paths['combined_dataset']
        if os.path.exists(combined_real_data_path) and os.path.getsize(combined_real_data_path) > 0:
            try:
                combined_real_data_df = pd.read_csv(combined_real_data_path, dtype=self.model_features)
                logging.info(f"Combined data loaded from {combined_real_data_path}")
                return combined_real_data_df
            except Exception as e:
                logging.error(f"Error loading combined data: {e}")
        else:
            logging.warning(f"Combined data file not found or is empty: {combined_real_data_path}")
        return pd.DataFrame(columns=list(self.model_features.keys()))

    def print_data_summary(
        self,
        data_df: pd.DataFrame,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None
    ):
        if symbol and interval_key:
            summary = f"Data summary for {symbol} ({interval_key}):\n"
        else:
            summary = "Data summary for all symbols and intervals:\n"
        feature_headers = ' '.join([f'{feature.capitalize():<10}' for feature in self.model_features.keys()])
        summary += f"{'Timestamp':<20} {feature_headers}\n"

        if data_df.empty:
            logging.info("No data to display.")
            return

        rows_to_display = [data_df.iloc[0], data_df.iloc[-1]] if len(data_df) > 1 else [data_df.iloc[0]]

        for i, row in enumerate(rows_to_display):
            label = "First" if i == 0 else "Last"
            timestamp = row['timestamp']
            feature_values = ' '.join([
                f'{row[feature]:<10.6f}' if isinstance(row[feature], float) else f"{row[feature]:<10}"
                for feature in self.model_features.keys()
            ])
            summary += f"{label:<20} {timestamp:<20} {feature_values}\n"

        logging.info(summary)

    def update_data(
        self,
        symbol: Optional[str] = None,
        interval_key: Optional[IntervalKey] = None
    ) -> Tuple[pd.DataFrame, Optional[int], Optional[int]]:
        symbols = [symbol] if symbol else list(self.symbol_mapping.keys())
        intervals = [interval_key] if interval_key else list(self.interval_mapping.keys())
        all_updated_data = []
        update_start_time = None
        update_end_time = None

        for sym in symbols:
            for interval in intervals:
                interval_info = self.get_interval_info(interval)
                filename = os.path.join(self.paths['data_dir'], f"{sym}_{interval_info['minutes']}_data.csv")
                server_time, _ = get_current_time()

                if os.path.exists(filename) and os.path.getsize(filename) > 0:
                    existing_data_df = pd.read_csv(filename, dtype=self.model_features)
                    existing_data_df = self.data_processor.fill_missing_add_features(existing_data_df)
                    last_timestamp = int(existing_data_df['timestamp'].max())
                else:
                    existing_data_df = pd.DataFrame(columns=list(self.model_features.keys()))
                    last_timestamp = server_time - (interval_info['days'] * 24 * 60 * 60 * 1000)

                time_difference = server_time - last_timestamp

                if time_difference > interval_info['milliseconds']:
                    update_start_time = last_timestamp + 1
                    update_end_time = server_time
                    new_data_df = self.get_binance_data(sym, interval, update_start_time, update_end_time)

                    if new_data_df is not None and not new_data_df.empty:
                        new_data_df = new_data_df.astype(self.model_features)

                        readable_start = timestamp_to_readable_time(update_start_time)
                        readable_newest = timestamp_to_readable_time(new_data_df['timestamp'].max())
                        logging.info(
                            f"Updating data for {sym} from {update_start_time} to {new_data_df['timestamp'].max()} "
                            f"({readable_start} to {readable_newest})"
                        )

                        existing_data_df = existing_data_df.astype(self.model_features)

                        updated_data_df = pd.concat(
                            [existing_data_df, new_data_df],
                            ignore_index=True
                        ).drop_duplicates(subset=['timestamp'], keep='first')
                        updated_data_df = self.data_processor.sort_dataframe(updated_data_df)
                        updated_data_df = self.data_processor.fill_missing_add_features(updated_data_df)
                        updated_data_df = updated_data_df.astype(self.model_features)

                        self.save_to_csv(updated_data_df, filename)
                        all_updated_data.append(updated_data_df)
                    else:
                        logging.warning(f"Failed to retrieve new data for {sym}.")
                else:
                    logging.info(f"Data for {sym} does not require updating. Using current data.")
                    all_updated_data.append(existing_data_df)

        if all_updated_data:
            combined_data_df = pd.concat(all_updated_data, ignore_index=True)
            return combined_data_df, update_start_time, update_end_time
        else:
            return pd.DataFrame(columns=list(self.model_features.keys())), None, None


def main():
    logging.info("Script started")
    data_fetcher = GetBinanceData()

    try:
        response = requests.get(f"{data_fetcher.api_base_url}/time")
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
                updated_data_df, start_time, end_time = data_fetcher.update_data(symbol, interval_key)
                if updated_data_df is not None and not updated_data_df.empty:
                    data_fetcher.print_data_summary(updated_data_df, symbol, interval_key)

                    data_dict = {}
                    for sym in data_fetcher.symbol_mapping.keys():
                        for interval in data_fetcher.interval_mapping.keys():
                            key = f"{sym}_{data_fetcher.interval_mapping[interval]['minutes']}"
                            subset_df = updated_data_df[
                                (updated_data_df['symbol'] == sym) &
                                (updated_data_df['interval'] == data_fetcher.interval_mapping[interval]['minutes'])
                            ]
                            if not subset_df.empty:
                                data_dict[key] = subset_df

                    data_fetcher.save_combined_dataset(
                        data_dict,
                        data_fetcher.paths['combined_dataset']
                    )
                else:
                    logging.error(f"Failed to update data for symbol {symbol} and interval {interval_key}")
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error updating data for symbol {symbol} with interval {interval_key}: {e}")

    logging.info("All files have been updated with the latest prices.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    def sort_dataframe(self, data_df: pd.DataFrame) -> pd.DataFrame:
        return data_df.sort_values('timestamp', ascending=True)