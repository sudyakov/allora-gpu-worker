import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import PREDICTION_MINUTES, TARGET_SYMBOL, PATHS

def load_and_prepare_data():
    combined_data = pd.read_csv(PATHS['combined_dataset'])
    predictions_data = pd.read_csv(PATHS['predictions'])
    
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'], unit='ms')
    predictions_data['timestamp'] = pd.to_datetime(predictions_data['timestamp'], unit='ms')
    
    # Фильтруем данные по TARGET_SYMBOL и интервалу PREDICTION_MINUTES
    interval = f"{PREDICTION_MINUTES}m"
    combined_data = combined_data[(combined_data['symbol'] == TARGET_SYMBOL) & (combined_data['interval'] == interval)]
    predictions_data = predictions_data[(predictions_data['symbol'] == TARGET_SYMBOL) & (predictions_data['interval'] == interval)]
    
    return combined_data, predictions_data

def create_candlestick_trace(df, name, color):
    return go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=name,
        increasing_line_color=color,
        decreasing_line_color=color
    )

def create_volume_trace(df, name, color):
    return go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name=name,
        marker_color=color
    )

def create_visualization():
    combined_data, predictions_data = load_and_prepare_data()
    
    os.makedirs(PATHS['visualization_dir'], exist_ok=True)
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,  # Уменьшаем вертикальные отступы между подграфиками
        row_heights=[0.6, 0.3, 0.1],  # Устанавливаем высоту строк: 60% для цен, 30% для объема, 10% для ошибок
        subplot_titles=(f'{TARGET_SYMBOL} Price', 'Volume', 'Prediction Error')
    )
    
    # Основной график цен
    fig.add_trace(create_candlestick_trace(combined_data, 'Actual', '#00FFFF'), row=1, col=1)  # Бирюзовый
    fig.add_trace(create_candlestick_trace(predictions_data, 'Predicted', '#FFFF00'), row=1, col=1)  # Желтый

    
    # Линии между предсказанными и реальными точками
    for _, row in predictions_data.iterrows():
        actual_data = combined_data[combined_data['timestamp'] == row['timestamp'] + pd.Timedelta(minutes=PREDICTION_MINUTES)]
        if not actual_data.empty:
            fig.add_trace(go.Scatter(
                x=[row['timestamp'], actual_data['timestamp'].iloc[0]],
                y=[row['close'], actual_data['close'].iloc[0]],
                mode='lines',
                line=dict(color='#FFFFFF', width=1),  # Белый
                showlegend=False
            ), row=1, col=1)
    
    # График объема торгов
    fig.add_trace(create_volume_trace(combined_data, 'Actual Volume', '#00FFFF'), row=2, col=1)  # Бирюзовый
    fig.add_trace(create_volume_trace(predictions_data, 'Predicted Volume', '#FFFF00'), row=2, col=1)  # Желтый
    
    # График ошибок предсказания
    error_data = combined_data.set_index('timestamp')['close'] - predictions_data.set_index('timestamp')['close']
    fig.add_trace(go.Scatter(
        x=error_data.index,
        y=error_data,
        mode='lines',
        name='Prediction Error',
        line=dict(color='#90EE90')  # Светло-зеленый
    ), row=3, col=1)
    
    fig.update_layout(
        title=f'{TARGET_SYMBOL} Price Prediction vs Actual',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1440,
        width=2560,
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(20,20,20)',
        font=dict(color='white')
    )
    
    # Устанавливаем диапазон оси X на основе временного интервала
    end_time = combined_data['timestamp'].max()
    start_time = end_time - pd.Timedelta(hours=6)
    full_range = pd.date_range(start=start_time, end=end_time, freq='5T')
    
    combined_data = combined_data.set_index('timestamp').reindex(full_range).reset_index()
    combined_data['timestamp'] = combined_data.index
    predictions_data = predictions_data.set_index('timestamp').reindex(full_range).reset_index()
    predictions_data['timestamp'] = predictions_data.index
    
    fig.update_xaxes(
        range=[start_time, end_time], 
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(255, 255, 255, 0.2)',
        dtick=300000  # Устанавливаем интервал сетки на 5 минут (300000 миллисекунд)
    )
    
    # Устанавливаем диапазон оси Y для графика цен
    min_price = combined_data['low'].min()
    max_price = combined_data['high'].max()
    fig.update_yaxes(
        range=[min_price, max_price], 
        row=1, col=1, 
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(255, 255, 255, 0.2)',
        dtick=(max_price - min_price) / 10  # Устанавливаем интервал сетки на 10 частей диапазона цен
    )
    
    # Устанавливаем диапазон оси Y для графика объема
    min_volume = combined_data['volume'].min()
    max_volume = combined_data['volume'].max()
    fig.update_yaxes(
        range=[min_volume, max_volume], 
        row=2, col=1, 
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(255, 255, 255, 0.2)',
        dtick=(max_volume - min_volume) / 10  # Устанавливаем интервал сетки на 10 частей диапазона объема
    )
    
    # Устанавливаем диапазон оси Y для графика ошибок предсказания
    min_error = error_data.min()
    max_error = error_data.max()
    fig.update_yaxes(
        range=[min_error, max_error], 
        row=3, col=1, 
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(255, 255, 255, 0.2)',
        dtick=(max_error - min_error) / 10  # Устанавливаем интервал сетки на 10 частей диапазона ошибок
    )
    
    fig.write_html(f"{PATHS['visualization_dir']}/price_prediction_{TARGET_SYMBOL}.html")
    fig.write_image(f"{PATHS['visualization_dir']}/price_prediction_{TARGET_SYMBOL}.png")
    
    # Убираем вызов fig.show(), чтобы не открывать график в браузере
    # fig.show()

if __name__ == "__main__":
    create_visualization()
