import os
from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import PREDICTION_MINUTES, TARGET_SYMBOL, PATHS, INTERVALS_PERIODS

def load_and_prepare_data():
    combined_data = pd.read_csv(PATHS['combined_dataset'])
    predictions_data = pd.read_csv(PATHS['predictions']) if os.path.exists(PATHS['predictions']) else pd.DataFrame()
    differences_data = pd.read_csv(PATHS['differences']) if os.path.exists(PATHS['differences']) else pd.DataFrame()
    
    for df in [combined_data, predictions_data, differences_data]:
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    if not predictions_data.empty:
        end_time = predictions_data['timestamp'].max()
    else:
        end_time = combined_data['timestamp'].max()
    
    start_time = end_time - timedelta(hours=12)
    start_time = start_time.replace(minute=1, second=0, microsecond=0)
    
    combined_data = filter_data(combined_data, start_time)
    predictions_data = filter_data(predictions_data, start_time)
    differences_data = filter_data(differences_data, start_time)
    
    return combined_data, predictions_data, differences_data, start_time, end_time


def filter_data(df, start_time):
    return df[(df['symbol'] == TARGET_SYMBOL) & 
            (df['interval'] == f"{PREDICTION_MINUTES}m") & 
            (df['timestamp'] >= start_time)] if not df.empty else pd.DataFrame()

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

def create_error_candlestick(df):
    return go.Candlestick(
        x=df['timestamp'],
        open=df['close'],
        high=df['high'],
        low=df['low'],
        close=df['open'],
        name='Prediction Error',
        increasing_line_color='green',
        decreasing_line_color='red'
    )

def create_visualization():
    combined_data, predictions_data, differences_data, start_time, end_time = load_and_prepare_data()
    
    os.makedirs(PATHS['visualization_dir'], exist_ok=True)
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(f'{TARGET_SYMBOL} Price', 'Volume', 'Differences')
    )
    
    add_empty_traces(fig)
    
    add_traces(fig, combined_data, predictions_data, differences_data)
    
    update_layout(fig, start_time, end_time)
    
    save_visualization(fig)

def create_dummy_trace(name, color, timestamp, row):
    if row == 1:  # График цены
        return go.Scatter(x=[timestamp], y=[None], mode='lines', name=name, line=dict(color=color))
    elif row == 2:  # График объёма
        return go.Bar(x=[timestamp], y=[None], name=name, marker_color=color)
    else:  # График разницы
        return go.Scatter(x=[timestamp], y=[None], name=name, line=dict(color=color))

def add_empty_traces(fig):
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual Line', line=dict(color='#00FFFF')), row=1, col=1)
    fig.add_trace(go.Bar(x=[], y=[], name='Actual Volume', marker_color='#00FFFF'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Predicted Line', line=dict(color='#FFFF00')), row=1, col=1)
    fig.add_trace(go.Bar(x=[], y=[], name='Predicted Volume', marker_color='#FFFF00'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], name='Differences'), row=3, col=1)

def add_traces(fig, combined_data, predictions_data, differences_data):
    if not combined_data.empty:
        fig.add_trace(create_candlestick_trace(combined_data, 'Actual', '#00FFFF'), row=1, col=1)
        fig.add_trace(go.Scatter(x=combined_data['timestamp'], y=combined_data['close'], mode='lines', name='Actual Line', line=dict(color='#00FFFF')), row=1, col=1)
        fig.add_trace(create_volume_trace(combined_data, 'Actual Volume', '#00FFFF'), row=2, col=1)
        
    if predictions_data.empty:
        fig.add_trace(create_dummy_trace('Predicted Line', '#FFFF00', combined_data['timestamp'].min() if not combined_data.empty else pd.Timestamp.now(), 1), row=1, col=1)
        fig.add_trace(create_dummy_trace('Predicted Volume', '#FFFF00', combined_data['timestamp'].min() if not combined_data.empty else pd.Timestamp.now(), 2), row=2, col=1)
    else:
        fig.add_trace(create_candlestick_trace(predictions_data, 'Predicted', '#FFFF00'), row=1, col=1)
        fig.add_trace(go.Scatter(x=predictions_data['timestamp'], y=predictions_data['close'], mode='lines', name='Predicted Line', line=dict(color='#FFFF00')), row=1, col=1)
        fig.add_trace(create_volume_trace(predictions_data, 'Predicted Volume', '#FFFF00'), row=2, col=1)

    if differences_data.empty:
        fig.add_trace(create_dummy_trace('Differences', '#FFFFFF', combined_data['timestamp'].min() if not combined_data.empty else pd.Timestamp.now(), 3), row=3, col=1)
    else:
        fig.add_trace(create_error_candlestick(differences_data), row=3, col=1)
        fig.add_trace(go.Scatter(x=differences_data['timestamp'], y=differences_data['close'], mode='lines', name='Differences Line', line=dict(color='#FFFFFF')), row=3, col=1)

def update_layout(fig, start_time, end_time):
    fig.update_layout(
        title=f'{TARGET_SYMBOL} Price Prediction vs Actual',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1200,
        width=1920,
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(20,20,20)',
        font=dict(color='white')
    )
    
    hourly_ticks = pd.date_range(start=start_time, end=end_time, freq='H')
    
    for i in range(1, 4):
        fig.update_xaxes(
            row=i, col=1,
            range=[start_time, end_time], 
            showgrid=True, 
            gridwidth=0.5, 
            gridcolor='rgba(255, 255, 255, 0.2)',
            tickmode='array',
            tickvals=hourly_ticks,
            ticktext=[tick.strftime('%H:00') for tick in hourly_ticks],
            tickangle=0,
            tickfont=dict(size=10)
        )
        update_yaxis(fig, i)

def update_yaxis(fig, row):
    y_data = get_y_data(fig, row)
    
    y_data = [y for y in y_data if y is not None]
    
    if y_data:
        min_y, max_y = min(y_data), max(y_data)
    else:
        min_y, max_y = (-10, 10) if row == 3 else (0, 1)
    
    fig.update_yaxes(
        row=row, col=1, 
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(255, 255, 255, 0.2)',
        dtick=(max_y - min_y) / 10 if max_y != min_y else 0.1,
        range=[min_y, max_y]
    )

def get_y_data(fig, row):
    y_data = []
    for trace in fig.select_traces(row=row, col=1):
        if hasattr(trace, 'y'):
            y_data.extend(trace.y)
        elif hasattr(trace, 'close'):
            y_data.extend(trace.close)
    return y_data

def save_visualization(fig):
    fig.write_html(f"{PATHS['visualization_dir']}/price_prediction_{TARGET_SYMBOL}.html")
    fig.write_image(f"{PATHS['visualization_dir']}/price_prediction_{TARGET_SYMBOL}.png")

if __name__ == "__main__":
    create_visualization()
