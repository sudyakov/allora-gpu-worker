import os
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import schedule
from config import SYMBOL, PREDICTION_MINUTES, PATHS, DATETIME_FORMAT
from download_data import get_current_price

def load_data(file_path):
    return pd.read_csv(file_path)

def update_real_prices():
    current_price = get_current_price(SYMBOL)
    if not current_price.empty:
        current_price.to_csv(PATHS['real_prices'], mode='a', header=False, index=False)

def create_visualization(predictions_df, real_prices_df):
    merged_df = pd.merge_asof(
        predictions_df, 
        real_prices_df, 
        on='timestamp', 
        by='symbol', 
        suffixes=('_pred', '_real')
    )
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], unit='ms')
    merged_df = merged_df.sort_values('timestamp')

    plt.figure(figsize=(16, 8))
    plt.plot(merged_df['timestamp'], merged_df['close_real'], label='Реальная цена', color='#00FF00')
    plt.plot(merged_df['timestamp'], merged_df['close_pred'], label='Прогноз', color='#FF00FF', linestyle='--')

    plt.title(f'{SYMBOL} - Прогноз vs Реальная цена', fontsize=20)
    plt.xlabel('Время', fontsize=14)
    plt.ylabel('Цена', fontsize=14)
    plt.legend(fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(DATETIME_FORMAT))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.gcf().autofmt_xdate()

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs(PATHS['visualization_dir'], exist_ok=True)
    plt.savefig(os.path.join(PATHS['visualization_dir'], 'prediction_visualization.png'), dpi=300)
    plt.close()

def run_visualization():
    update_real_prices()
    predictions_df = load_data(PATHS['predictions'])
    real_prices_df = load_data(PATHS['combined_dataset'])
    create_visualization(predictions_df, real_prices_df)
    print("График обновлен")

def main():
    schedule.every(1).minutes.do(run_visualization)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
