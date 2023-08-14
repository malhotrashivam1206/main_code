
import json
from datetime import datetime, timedelta
from pytz import timezone
from time import sleep
import pandas as pd
import dash
import os
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
import streamlit as st
import plotly.graph_objects as go
from pya3 import *
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tabulate
from dash import dash_table
import pymongo
from pymongo import MongoClient



# Replace these with your actual MongoDB connection details
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/banknifty47353CE"
DB_NAME = "crudeoil"
COLLECTION_NAME = "crudeoil253460"

client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

# Initialize variables for WebSocket communication
lp = 0
socket_opened = False
subscribe_flag = False
subscribe_list = []
unsubscribe_list = []
data_list = []  # List to store the received data
df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data



# File paths for saving data and graph
data_file_path = "crudeoil.csv"

graph_file_path = "crudeoil.html"

# Check if the data file exists
if os.path.exists(data_file_path):
    # Load existing data from the CSV file
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
else:
    df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data


all_trend_lines = []
trend_line_visibility = []


# Callback functions for WebSocket connection
def socket_open():
    print("Connected")
    global socket_opened
    socket_opened = True
    if subscribe_flag:
        alice.subscribe(subscribe_list)


def socket_close():
    global socket_opened, lp
    socket_opened = False
    lp = 0
    print("Closed")


def socket_error(message):
    global lp
    lp = 0
    print("Error:", message)


# Callback function for receiving data from WebSocket
def feed_data(message):
    global lp, subscribe_flag, data_list
    feed_message = json.loads(message)
    if feed_message["t"] == "ck":
        print("Connection Acknowledgement status: %s (Websocket Connected)" % feed_message["s"])
        subscribe_flag = True
        print("subscribe_flag:", subscribe_flag)
        print("-------------------------------------------------------------------------------")
        pass
    elif feed_message["t"] == "tk":
        print("Token Acknowledgement status: %s" % feed_message)
        print("-------------------------------------------------------------------------------")
        pass
    else:
        print("Feed:", feed_message)
        if 'lp' in feed_message:
            timestamp = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S.%f')
            feed_message['timestamp'] = timestamp
            lp = feed_message['lp']
            data_list.append(feed_message)  # Append the received data to the list
            # Insert the data into MongoDB
            collection.insert_one(feed_message)

            # Update marking information only for Heikin Ashi candles
            if len(df) >= 2 and df['mark'].iloc[-1] == '' and feed_message['t'] == 'c':
                if (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1]):
                    df.at[df.index[-1], 'mark'] = 'YES'
                elif (df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1]):
                    df.at[df.index[-1], 'mark'] = 'NO'

        else:
            print("'lp' key not found in feed message.")


# Connect to AliceBlue

# Socket Connection Request
alice.start_websocket(socket_open_callback=socket_open, socket_close_callback=socket_close,
                      socket_error_callback=socket_error, subscription_callback=feed_data, run_in_background=True,
                      market_depth=True)

while not socket_opened:
    pass

# Subscribe to Tata Motors
subscribe_list = [alice.get_instrument_by_token('MCX', 253460)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(5)
print(datetime.now())

def calculate_heikin_ashi(data):
    ha_open = (data['open'].shift() + data['close'].shift()) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = data[['high', 'open', 'close']].max(axis=1)
    ha_low = data[['low', 'open', 'close']].min(axis=1)

    ha_data = pd.DataFrame({'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})
    ha_data['open'] = ha_data['open'].combine_first(data['open'].shift())
    ha_data['high'] = ha_data['high'].combine_first(data['high'].shift())
    ha_data['low'] = ha_data['low'].combine_first(data['low'].shift())
    ha_data['close'] = ha_data['close'].combine_first(data['close'].shift())

    # Add the "mark" column based on Heikin Ashi candle conditions
    ha_data['mark'] = ''
    for i in range(1, len(ha_data)):
        if (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
            ha_data['close'].iloc[i] > ha_data['open'].iloc[i]):
            ha_data.at[ha_data.index[i], 'mark'] = 'YES'
        elif (ha_data['close'].iloc[i - 1] > ha_data['open'].iloc[i - 1] and
              ha_data['close'].iloc[i] < ha_data['open'].iloc[i]):
            ha_data.at[ha_data.index[i], 'mark'] = 'NO'

    # Print the "mark" column for debugging purposes
    print(ha_data['mark'])

    return ha_data


def calculate_supertrend(data, atr_period=2, factor=2.0, multiplier=2.0):
    data = data.copy()  # Create a copy of the data DataFrame

    close = data['close']
    high = data['high']
    low = data['low']

    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)

    atr = tr['tr'].rolling(atr_period).mean()

    median_price = (high + low) / 2
    data['upper_band'] = median_price + (multiplier * atr)
    data['lower_band'] = median_price - (multiplier * atr)

    supertrend = pd.Series(index=data.index)
    direction = pd.Series(index=data.index)

    supertrend.iloc[0] = data['upper_band'].iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(data)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(data['lower_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(data['upper_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1

        # Start uptrend calculation anew whenever a new uptrend begins
        if direction.iloc[i] == 1 and direction.iloc[i - 1] != 1:
            supertrend.iloc[i] = data['lower_band'].iloc[i]

        # Start downtrend calculation anew whenever a new downtrend begins
        if direction.iloc[i] == -1 and direction.iloc[i - 1] != -1:
            supertrend.iloc[i] = data['upper_band'].iloc[i]

    data['supertrend'] = supertrend  # Add the 'supertrend' column to the data DataFrame
    data['direction'] = direction  # Add the 'direction' column to the data DataFrame

    return data[['open', 'high', 'low', 'close', 'supertrend', 'direction', 'lower_band', 'upper_band']]

def calculate_trend_lines(data):
    current_trend = None
    trend_start = None
    trend_lines = []

    for i in range(len(data)):
        current_signal = data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            trend_start = current_signal.name

        if current_trend != current_signal['direction']:
            if trend_start is not None:
                trend_data = data.loc[trend_start:data.index[i - 1]]
                if len(trend_data) > 1:
                    trend_lines.append((current_trend, trend_data))

            current_trend = current_signal['direction']
            trend_start = current_signal.name

    # Handle the last trend if it's still ongoing
    if trend_start is not None and trend_start != data.index[-1]:
        trend_data = data.loc[trend_start:]
        if len(trend_data) > 1:
            trend_lines.append((current_trend, trend_data))

    return trend_lines


all_trend_lines = []



# Function to update the graph

def calculate_current_trend_lines(data):
    current_trend = None
    in_trend = False
    trend_start = None
    trend_lines = []
    buy_signals = pd.DataFrame(columns=['supertrend', 'direction'])
    sell_signals = pd.DataFrame(columns=['supertrend', 'direction'])

    for i in range(len(data)):
        current_signal = data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            in_trend = True
            trend_start = current_signal.name

        if current_trend != current_signal['direction']:
            if current_signal['direction'] == 1:
                sell_signals = pd.concat([sell_signals, current_signal])
            else:
                buy_signals = pd.concat([buy_signals, current_signal])

            if in_trend:
                trend_data = data.loc[trend_start:data.index[i - 1]]

                # Calculate the difference between first candle's high and last candle's close for the previous trend
                if current_trend == 1:
                    trend_data['difference'] = trend_data['high'] - trend_data['close'].iloc[-1]
                else:
                    trend_data['difference'] = trend_data['high'].iloc[-1] - trend_data['close']

                if len(trend_data) > 1:
                    trend_lines.append((current_trend, trend_data))

            else:
                if current_signal['direction'] == 1 and current_trend == -1:
                    updated_trend_data = data.loc[trend_start:data.index[i]]
                    updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
                    current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
                    current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]
                elif current_signal['direction'] == -1 and current_trend == 1:
                    updated_trend_data = data.loc[trend_start:data.index[i]]
                    updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
                    current_signal['supertrend'] = updated_supertrend_data['supertrend'].iloc[-1]
                    current_signal['direction'] = updated_supertrend_data['direction'].iloc[-1]

            current_trend = current_signal['direction']
            in_trend = False

        if not in_trend:
            if current_trend == 1:
                if not np.isnan(current_signal['upper_band']):
                    trend_start = current_signal.name
                    in_trend = True
            else:
                if not np.isnan(current_signal['lower_band']):
                    trend_start = current_signal.name
                    in_trend = True

    if in_trend:
        trend_data = data.loc[trend_start:]

        # Calculate the difference between first candle's high and last candle's close for the previous trend
        if current_trend == 1:
            trend_data['difference'] = trend_data['high'].iloc[0] - trend_data['close'].iloc[-1]
        else:
            trend_data['difference'] = trend_data['high'].iloc[-1] - trend_data['close'].iloc[0]

        if len(trend_data) > 1:
            first_high = trend_data['high'].iloc[0]
            last_close = trend_data['close'].iloc[-1]
            trend_data['difference'] = first_high - last_close

    # Handle the continuation of uptrend without a change in direction
    if len(trend_lines) > 0 and data.index[-1] not in trend_lines[-1][1].index and trend_lines[-1][0] == 1:
        last_trend_type, last_trend_data = trend_lines[-1]
        continuation_data = data.loc[data.index > last_trend_data.index[-1]]
        if len(continuation_data) > 1:
            updated_trend_data = pd.concat([last_trend_data.iloc[:-1], continuation_data])
            updated_supertrend_data = calculate_supertrend(updated_trend_data, factor=2.0)
            continuation_data['supertrend'] = updated_supertrend_data['supertrend'].values[-len(continuation_data):]
            continuation_data['direction'] = updated_supertrend_data['direction'].values[-len(continuation_data):]
            trend_lines[-1] = (last_trend_type, updated_trend_data)

    return trend_lines, buy_signals, sell_signals


# Function to update the graph
def update_graph(n, interval, chart_type):
    global df, data_list, all_trend_lines

    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)

    # Convert 'timestamp' column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.set_index('timestamp', inplace=True)

    # Check if there is new data
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        new_df.set_index('timestamp', inplace=True)
        df = pd.concat([df, new_df])

        df.to_csv(data_file_path)
        data_list = []

    df["lp"] = pd.to_numeric(df["lp"], errors="coerce")

    # Filter out data points outside of trading hours
    trading_start_time = pd.Timestamp(df.index[0].date()) + pd.Timedelta(hours=9)
    trading_end_time = pd.Timestamp(df.index[0].date()) + pd.Timedelta(hours=16)
    trading_hours_mask = (df.index.time >= trading_start_time.time()) & (df.index.time <= trading_end_time.time())
    df = df[trading_hours_mask]

    # Resample the data for the desired interval
    resampled_data = df["lp"].resample(f'{interval}T').ohlc()
    resampled_data = resampled_data.dropna()

    # Create a datetime index for the x-axis, starting from the first data point and ending at the last data point
    x = pd.date_range(start=df.index[0], end=df.index[-1], freq=f'{interval}T')

    # Plot the data using plotly
    fig = go.Figure(data=[go.Candlestick(x=x,
                open=resampled_data['open'],
                high=resampled_data['high'],
                low=resampled_data['low'],
                close=resampled_data['close'])])

    # Set x-axis label to show only the time
    fig.update_xaxes(type='category', tickformat='%H:%M')

    # Update the layout and display the figure
    fig.update_layout(title=f'Real-Time {chart_type} Chart',
                      xaxis_title='Time',
                      yaxis_title='Price',
                      template='plotly_dark')

    st.plotly_chart(fig, use_container_width=True)

    if chart_type == 'heikin_ashi':
        resampled_data = calculate_heikin_ashi(resampled_data)

        fig = go.Figure()

        # Add Heikin Ashi candlesticks to the figure
        last_timestamp = None  # To track the last timestamp of previous day's data


        # Add Heikin Ashi candlesticks to the figure
        if len(resampled_data) > 0:
            time_difference = (resampled_data.index[0] - resampled_data.index[-1]).total_seconds()
        else:
            time_difference = 0

        # Add Heikin Ashi candlesticks to the figure
        for i, row in enumerate(resampled_data.iterrows()):
            timestamp, candle = row
            candle_color = 'green' if candle['close'] > candle['open'] else 'red'

            # Adjust timestamp by adding the time difference
            timestamp += pd.Timedelta(seconds=time_difference)
            
            fig.add_trace(go.Candlestick(x=[timestamp],
                                        open=[candle['open']],
                                        high=[candle['high']],
                                        low=[candle['low']],
                                        close=[candle['close']],
                                        increasing_line_color=candle_color,
                                        decreasing_line_color=candle_color,
                                        name=f'Candle {i + 1}'))
            
            last_timestamp = timestamp
            # Add "yes" or "no" label above the candle
            label_y = None
            label_text = None
            if candle['mark'] == 'YES':
                label_y = candle['high'] + 5  # Adjust this value for proper positioning
                label_text = 'Yes'
            elif candle['mark'] == 'NO':
                label_y = candle['low'] - 15  # Adjust this value for proper positioning
                label_text = 'No'

            if label_y is not None:
                fig.add_annotation(
                    go.layout.Annotation(
                        x=timestamp,
                        y=label_y,
                        text=label_text,
                        showarrow=False,
                        font=dict(color='black', size=12),
                    )
                )

    # Calculate the Supertrend and get the direction from the result
    supertrend_data = calculate_supertrend(resampled_data, factor=2.0)  # Use the new factor parameter
    resampled_data = supertrend_data  # Update resampled_data with the DataFrame returned from calculate_supertrend

    resampled_data['difference'] = resampled_data['high'].iloc[0] - resampled_data['close'].iloc[-1]

    # Add 'volume' column with default value if it doesn't exist in resampled_data
    if 'volume' not in resampled_data:
        resampled_data['volume'] = 0

    # Create a new figure (initialize or update existing figure)
    if 'fig' not in globals():
        fig = plot_candlestick(resampled_data)
        all_trend_lines = []  # Initialize the list of trend lines for the new figure
    else:
        fig = go.Figure()
        all_trend_lines = []

        
    # Calculate the current trend lines using the updated Supertrend data
    trend_lines, buy_signals, sell_signals = calculate_current_trend_lines(resampled_data)
    trend_lines = calculate_trend_lines(resampled_data)

    for i, (trend_type, trend_data) in enumerate(trend_lines):
        color = 'green' if trend_type == 1 else 'red'
        trend_trace = go.Scatter(
            x=trend_data.index,
            y=trend_data['supertrend'],
            mode='lines',
            name=f'{"Uptrend" if trend_type == 1 else "Downtrend"} Line',
            line=dict(color=color, width=2),
            text=trend_data['difference'].apply(lambda x: f'Difference: {x:.2f}')
        )

        if i == 0:
            fig.add_trace(trend_trace)  # Add only the first trend line of each type
        else:
            all_trend_lines.append(trend_trace)  # Store the additional trend lines for future updates

    # Update the trend lines directly in the figure
    for trend_line in all_trend_lines:
        fig.add_trace(trend_line)

    for trend_line in trend_lines:
        trend_type, trend_data = trend_line
        color = 'green' if trend_type == 1 else 'red'
        trend_trace = go.Scatter(
        x=trend_data.index,
        y=trend_data['supertrend'],
        mode='lines',
        name=f'Uptrend Line {len(all_trend_lines) + 1}',
        line=dict(color=color, width=2),
    )

        fig.add_trace(trend_trace)

    # Save the current trend lines for future updates
    all_trend_lines = [
    go.Scatter(
        x=trend_data.index,
        y=trend_data['supertrend'],
        mode='lines',
        name='Uptrend Line' if trend_type == 1 else 'Downtrend Line',
        line=dict(color=color, width=2),
        text=trend_data['difference'].apply(lambda x: f'Difference: {x:.2f}')  # Include difference as text on the graph
    ) for trend_type, trend_data in trend_lines
]

    # Initialize trend_start and current_trend
    trend_start = None
    current_trend = None

    # Add the new trend lines as separate traces (without direct connections between trends)
    for trend_data in trend_lines:
        trend_type, data = trend_data
        color = 'green' if trend_type == 1 else 'red'
        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['supertrend'],
                                 mode='lines',
                                 name='Uptrend Line' if trend_type == 1 else 'Downtrend Line',
                                 line=dict(color=color, width=2)))
        # Update trend_start and current_trend
        trend_start = data.index[0]
        current_trend = trend_type

    # Check if there is a trend line left after the loop ends
    last_index = supertrend_data.index[-1]
    if trend_start is not None and trend_start != last_index:
        trend_data = supertrend_data.loc[trend_start:last_index]
        if len(trend_data) > 1:
            trend_lines.append((current_trend, trend_data))
            color = 'green' if current_trend == 1 else 'red'
            fig.add_trace(go.Scatter(x=trend_data.index,
                                     y=trend_data['supertrend'],
                                     mode='lines',
                                     name='Uptrend Line' if current_trend == 1 else 'Downtrend Line',
                                     line=dict(color=color, width=2)))

    # Add the sell signals above the candlesticks
    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['supertrend'],
                             mode='markers',
                             name='Sell Signal',
                             marker=dict(color='green', symbol='triangle-up', size=10)))

    # Add the buy signals below the candlesticks
    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['supertrend'],
                             mode='markers',
                             name='Buy Signal',
                             marker=dict(color='red', symbol='triangle-down', size=10)))
    
    fig.write_html(graph_file_path)

    return fig, resampled_data.to_dict('records')

# Function to plot candlestick graph with custom colors
candlestick_color = 'rgba(30, 138, 0, 1)'  # green
uptrend_color = 'rgba(50, 205, 50, 0.8)'  # Green
downtrend_color = 'rgba(220, 20, 60, 0.8)'  # Red
buy_signal_color = 'green'
sell_signal_color = 'red'

# Function to plot candlestick graph with custom colors
def plot_candlestick(data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index,
                       open=data['open'],
                       high=data['high'],
                       low=data['low'],
                       close=data['close'],
                       increasing_line_color='green',  # Customize colors here
                       decreasing_line_color='red',   # Customize colors here
                       line=dict(width=1))
    ])

    # Add a subtle background image
    fig.update_layout(images=[dict(
        source='url(https://example.com/background_image.jpg)',
        xref="paper", yref="paper",
        x=0, y=0,
        sizex=1, sizey=1,
        sizing="contain",
        opacity=0.3,
        layer="below")])

    # Customizing the layout of the graph
    fig.update_layout(
        title="Live Candlestick Graph",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14),
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
    )


    # Add secondary y-axis for price
    fig.update_layout(yaxis2=dict(overlaying='y', side='left', showgrid=False))

    return fig
trend_line_visibility = [False] * len(all_trend_lines)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# MongoDB setup
client = MongoClient(MONGO_CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define the callback to display the appropriate page content based on the URL pathname
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-2':
        # Fetch data from MongoDB
        data = collection.find({}, {'_id': 0}).sort('timestamp')
        df = pd.DataFrame(data)

        # Convert 'timestamp' column to datetime and set it as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        df.set_index('timestamp', inplace=True)

        # Create and return the content for the second page (Market Depth Table)
        return html.Div([
            html.H3('Market Depth Table'),
            dash_table.DataTable(
                id='data-table',
                columns=[{'name': col, 'id': col} for col in df.columns],
                data=df.to_dict('records'),
                style_table={'height': '1000px', 'overflowY': 'auto'}
            )
        ])
    else:
        # Default to the first page (Candlestick Chart)
        return html.Div([
    dcc.Graph(id='live-candlestick-graph', config={'displayModeBar': False, 'scrollZoom': False}),
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'Normal', 'value': 'normal'},
            {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
        ],
        value='normal',
        clearable=False,
        style={'width': '150px'}
    ),
    dcc.Dropdown(
        id='interval-dropdown',
        options=[
            {'label': '1 Min', 'value': 1},
            {'label': '3 Min', 'value': 3},
            {'label': '5 Min', 'value': 5},
            {'label': '10 Min', 'value': 10},
            {'label': '30 Min', 'value': 30},
            {'label': '60 Min', 'value': 60},
            {'label': '1 Day', 'value': 1440}
        ],
        value=1,
        clearable=False,
        style={'width': '150px'}
    ),
    dcc.Interval(id='graph-update-interval', interval=2000, n_intervals=0),
    html.Button('Show/Hide Trend Lines', id='toggle-trend-lines-button', n_clicks=0),
], style={'height': '100vh', 'width': '100vw'})
# Layout of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div([
        html.H1("Your Dashboard Title", style={'textAlign': 'center'}),
        html.Nav([
            dcc.Link('Candlestick Chart', href='/', className='nav-link'),
            dcc.Link('Market Depth Table', href='/page-2', className='nav-link'),
            # Add more navigation links as needed
        ], className='nav'),
    ], className='header'),
    html.Div([
        # Add other components specific to your layout
        # For example, filters, buttons, additional graphs, etc.
    ], className='content'),
    html.Div([
        html.P("Your Footer Information", style={'textAlign': 'center'}),
    ], className='footer'),
])
visible_trend_lines = []

# Define the callback to update the data for the market depth table on page two
@app.callback(
    Output('live-candlestick-graph', 'figure'),
    [
        Input('interval-dropdown', 'value'),
        Input('chart-type-dropdown', 'value'),
        Input('live-candlestick-graph', 'relayoutData'),
        Input('toggle-trend-lines-button', 'n_clicks')
    ],
    [
        State('graph-update-interval', 'n_intervals')
    ]
)
def update_graph_callback(interval, chart_type, relayoutData, n_clicks, n):
    fig = go.Figure()

    # Calculate new x-axis range based on user interaction
    if 'xaxis.range' in relayoutData:
        xaxis_range = relayoutData['xaxis.range']
    else:
        xaxis_range = [df.index[-1] - pd.Timedelta(hours=4), df.index[-1]]

    filtered_data = df[(df.index >= xaxis_range[0]) & (df.index <= xaxis_range[1])]

    fig = go.Figure()
    fig, _ = update_graph(n, interval, chart_type)

    global all_trend_lines, trend_line_visibility

    # Toggle visibility of trend lines based on the button click count
    show_trend_lines = n_clicks % 2 == 1

    for trend_line, is_visible in zip(all_trend_lines, trend_line_visibility):
        trend_line_idx = all_trend_lines.index(trend_line)
        fig.update_traces(
            visible=show_trend_lines if is_visible else 'legendonly',
            selector=dict(name=f'Uptrend Line {trend_line_idx + 1}')
        )

        if show_trend_lines:
            trend_line_visibility[trend_line_idx] = True
        else:
            trend_line_visibility[trend_line_idx] = False


    return fig




# Define the callback to update the data for the data table on page two
@app.callback(
    Output('data-table', 'data'),
    [Input('interval-dropdown', 'value')],
    [dash.dependencies.State('graph-update-interval', 'n_intervals')]
)
def update_data_table(interval, n):
    global df, data_list

    # Append new data to DataFrame
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df = new_df[["timestamp", "lp"]]
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], format='%Y-%m-%d %H:%M:%S.%f')
        new_df.set_index("timestamp", inplace=True)
        df = pd.concat([df, new_df])

        df.to_csv(data_file_path)
        data_list = []

    # Fetch data from MongoDB
    data = collection.find({}, {'_id': 0}).sort('timestamp')
    df = pd.DataFrame(data)

    # Convert 'timestamp' column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.set_index('timestamp', inplace=True)

    # Convert DataFrame to dictionary format for DataTable
    data_table_data = df.to_dict('records')

    return data_table_data
# Run the Dash app
if __name__ == '__main__':
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
    app.run_server(debug=True)