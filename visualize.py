import pandas as pd
import matplotlib.pyplot as plt
from config import PREDICTION_MINUTES, TARGET_SYMBOL

# Пути к файлам
combined_dataset_path = 'data/combined_dataset.csv'
predictions_path = 'data/predictions.csv'
differences_path = 'data/differences.csv'

def load_and_filter_data():
    # Загрузка данных
    combined_data = pd.read_csv(combined_dataset_path)
    predictions_data = pd.read_csv(predictions_path)
    differences_data = pd.read_csv(differences_path)

    # Фильтрация данных по TARGET_SYMBOL и интервалу
    interval = f'{PREDICTION_MINUTES}m'
    combined_data = combined_data[(combined_data['symbol'] == TARGET_SYMBOL) & (combined_data['interval'] == interval)]
    predictions_data = predictions_data[(predictions_data['symbol'] == TARGET_SYMBOL) & (predictions_data['interval'] == interval)]
    differences_data = differences_data[(differences_data['symbol'] == TARGET_SYMBOL) & (differences_data['interval'] == interval)]

    # Преобразование timestamp в datetime
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'], unit='ms')
    predictions_data['timestamp'] = pd.to_datetime(predictions_data['timestamp'], unit='ms')
    differences_data['timestamp'] = pd.to_datetime(differences_data['timestamp'], unit='ms')

    # Устреднение данных в differences_data по timestamp, исключая нечисловые столбцы
    numeric_columns = differences_data.select_dtypes(include=[float, int]).columns
    differences_data = differences_data.groupby('timestamp')[numeric_columns].mean().reset_index()

    return combined_data, predictions_data, differences_data

def plot_data(combined_data, predictions_data, differences_data):
    # Настройка темного фона
    plt.style.use('dark_background')

    # Создание фигуры и осей
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # График фактических данных
    ax1.plot(combined_data['timestamp'], combined_data['close'], label='Actual Prices', color='cyan')
    ax1.plot(predictions_data['timestamp'], predictions_data['close'], label='Predicted Prices', color='lime')
    ax1.plot(differences_data['timestamp'], differences_data['close'], label='Prediction Differences', color='red')

    # Настройки осей
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Price Prediction Visualization for {TARGET_SYMBOL}')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Создание второго y-axes для объемов
    ax2 = ax1.twinx()
    ax2.bar(combined_data['timestamp'], combined_data['volume'], alpha=0.3, color='yellow', label='Volume')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper right')

    # Показ графика
    plt.show()

def update_visualization():
    combined_data, predictions_data, differences_data = load_and_filter_data()
    
    # Проверка наличия новых данных
    if not combined_data.empty and not predictions_data.empty and not differences_data.empty:
        plot_data(combined_data, predictions_data, differences_data)
    else:
        print("Нет новых данных для обновления визуализации.")

if __name__ == "__main__":
    update_visualization()
