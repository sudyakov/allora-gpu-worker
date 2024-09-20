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
        vertical_spacing=0.1,  # Увеличиваем вертикальные отступы между подграфиками
        subplot_titles=(f'{TARGET_SYMBOL} Price', 'Volume', 'Prediction Error')
    )
    
    # Основной график цен
    fig.add_trace(create_candlestick_trace(combined_data, 'Actual', 'cyan'), row=1, col=1)
    fig.add_trace(create_candlestick_trace(predictions_data, 'Predicted', 'magenta'), row=1, col=1)
    
    # Линии между предсказанными и реальными точками
    for _, row in predictions_data.iterrows():
        actual_data = combined_data[combined_data['timestamp'] == row['timestamp'] + pd.Timedelta(minutes=PREDICTION_MINUTES)]
        if not actual_data.empty:
            fig.add_trace(go.Scatter(
                x=[row['timestamp'], actual_data['timestamp'].iloc[0]],
                y=[row['close'], actual_data['close'].iloc[0]],
                mode='lines',
                line=dict(color='lime', width=1),
                showlegend=False
            ), row=1, col=1)
    
    # График объема торгов
    fig.add_trace(create_volume_trace(combined_data, 'Actual Volume', 'blue'), row=2, col=1)
    fig.add_trace(create_volume_trace(predictions_data, 'Predicted Volume', 'magenta'), row=2, col=1)
    
    # График ошибок предсказания
    error_data = combined_data.set_index('timestamp')['close'] - predictions_data.set_index('timestamp')['close']
    fig.add_trace(go.Scatter(
        x=error_data.index,
        y=error_data,
        mode='lines',
        name='Prediction Error',
        line=dict(color='red')
    ), row=3, col=1)
    
    fig.update_layout(
        title=f'{TARGET_SYMBOL} Price Prediction vs Actual',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1500,  # Увеличиваем высоту графика для лучшей читаемости
        width=1920,
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(20,20,20)',
        font=dict(color='white')
    )
    
    # Устанавливаем диапазон оси X на основе PREDICTION_MINUTES
    end_time = combined_data['timestamp'].max()
    start_time = end_time - pd.Timedelta(minutes=PREDICTION_MINUTES)
    fig.update_xaxes(range=[start_time, end_time])
    
    # Логарифмическая шкала для объема
    fig.update_yaxes(type="log", row=2, col=1)
    
    fig.write_html(f"{PATHS['visualization_dir']}/price_prediction_{TARGET_SYMBOL}.html")
    fig.write_image(f"{PATHS['visualization_dir']}/price_prediction_{TARGET_SYMBOL}.png")
    
    fig.show()

if __name__ == "__main__":
    create_visualization()
