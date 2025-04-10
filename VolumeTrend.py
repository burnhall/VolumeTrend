#server = app.server

from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import datetime

import yfinance as yf

def get_minute_stock_data(ticker, time_duration):
    days = str(time_duration) + "d"
    stock_data = yf.download(ticker, period=days, interval='1m')

    return stock_data

def get_5minute_stock_data(ticker, time_duration):
    days = str(time_duration) + "d"
    stock_data = yf.download(ticker, period=days, interval='5m')

    return stock_data

def get_daily_stock_data(ticker, time_duration):
    days = str(time_duration) + "y"
    stock_data = yf.download(ticker, period=days, interval='1d')

    return stock_data

def get_oi_shorti (ticker):
    # call/put open interest, short interest, implied volatility
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if (len(expirations) > 0):
        df_calls = stock.option_chain(expirations[0]).calls
        df_puts = stock.option_chain(expirations[0]).puts
        put_oi = df_puts["openInterest"].sum()
        call_oi = df_calls["openInterest"].sum()
        pcr_oi = put_oi / call_oi if call_oi != 0 else -1
    
        short_interest = stock.info.get("sharesShort", 1)
        avg_volume = stock.info.get("averageVolume", None)
        si_ratio = short_interest / avg_volume if avg_volume else -1
        return f"{pcr_oi:.2f}", f"{si_ratio:.2f}"
    else:
        return -1, -1

def find_peaks_and_valleys (data, sigma):
    filtered_data = apply_gaussian_filter(data, sigma, "nearest")
    peaks, _ = find_peaks(filtered_data)
    filtered_data = [-num for num in filtered_data]
    valleys, _ = find_peaks(filtered_data)
    return peaks, valleys
    
def calculate_differences(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Calculate differences
    differences = [a - b for a, b in zip(list1, list2)]
    
    return differences

def average_min_max_open_close(list1, list2, list3, list4):
    # Calculate differences
    average = [(a + b + c + d)/4 for a, b, c, d in zip(list1, list2, list3, list4)]
    
    return average

def average_open_close(list1, list2):
    # Calculate differences
    average = [(a + b)/2 for a, b in zip(list1, list2)]
    
    return average

def accumulate_values(data):
    # Accumulate values
    for i in range(1, len(data)):
        data[i] += data[i - 1]

def average_of_lists(data, number_of_lists):
    if len(data) % 390 != 3.0:
        raise ValueError("List length not correct")
    
    for i in range(390):
        data[i] = int((data[i] + data[i+390] + data[i+780]) / 3)
    

def apply_gaussian_filter(data, sigma, m):
    if not isinstance(data, list):
        raise ValueError("Input data must be a list.")
    
    # Apply Gaussian filter
    data_Filtered = gaussian_filter1d(data, sigma, mode=m)
    return data_Filtered

def moving_average_end_at_current(data, window_size):
    # Calculate the moving average where the window ends at the current number.

    moving_avg = []
    for i in range(len(data)):
        # Define the window based on the current index
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i + 1]
        avg = sum(window) / len(window)
        moving_avg.append(avg)

    return moving_avg

def second_order_difference(data, n):
    # Calculate the second-order difference (second derivative) of a list.

    second_diff = []
    for i in range(0, n-1):
        second_diff.append(data[n] - data[1])
    for i in range(n, len(data)):
        diff = data[i] - data[i-n]
        second_diff.append(diff)
        
    return second_diff

def one_conversion(list, n):
    # converts a list such that all positive values are 1, all negatives are -1 and 0 is 0
    for i in range(len(list)):
        if list[i] > 0:
            list[i] = n
        elif list[i] < 0:
            list[i] = -n

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    "Calculates MACD and signal line."
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def percent_difference_from_first_value_in_list(data):
    for i in range(1, len(data)):
        if (data[0] + data[i]) == 0:
            data[i] = 0
        else:
            data[i] = ((data[i] - data[0]) / ((data[i] + data[0]) / 2)) * 100

    data[0] = 0

def get_tick_data(time_duration):
    custom_tick_vals = []
    custom_tick_text = []
    time = 0
    for i in range(1, time_duration):
        custom_tick_vals.append(time)
        custom_tick_text.append("D" + str(i))
        time += 390
    custom_tick_vals.extend([time, time+30, time+90, time+150, time+210, time+270, time+330])
    custom_tick_text.extend(["Open", "10", "11", "12", "1", "2", "3"])

    return custom_tick_text, custom_tick_vals

def get_open_close_min_max (data):
    return data[0], data[-1], max(data), min(data)

def get_list_of_tickers_from_keyword(list_of_tickers, keyword):
    if keyword in ["Movers", "Chip", "EV", "Ecom", "Crypto", "Social Media", "Oil and Gas", "ETFs", "Quantum", "Misc group"]:
        if keyword == "Movers":
            list_of_tickers.extend(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'INTC', 'TSLA', 'META', 'BABA', 'SHOP', 'ARM', 'NVDA'])
        if keyword == "Chip":
            list_of_tickers.extend(['NVDA', 'AMD', 'SMCI', 'AVGO', 'INTC', 'QCOM', 'ARM', 'MU'])
        if keyword == "EV":
            list_of_tickers.extend(['TSLA', 'RIVN', 'GM', 'LCID', 'NIO', 'XPEV', 'BLNK'])
        if keyword == "Ecom":
            list_of_tickers.extend(['AMZN', 'SHOP', 'BABA', 'ABNB'])
        if keyword == "Crypto":
            list_of_tickers.extend(['IBIT', 'HUT', 'MARA', "BITF"])
        if keyword == "Social Media":
            list_of_tickers.extend(['META', 'SNAP', 'PINS', 'RDDT'])
        if keyword == "Oil and Gas":
            list_of_tickers.extend(['PBR', 'XOM', 'KMI', 'ET', 'OXY', 'SLB', 'HAL', 'CVX', 'BKR', 'CVE'])
        if keyword == "ETFs":
            list_of_tickers.extend(["TQQQ", "SOXL", "TNA", "IBIT"])
        if keyword == "Quantum":
            list_of_tickers.extend(['IONQ', "QBTS"])
        if keyword == "Misc group":
            list_of_tickers.extend(['PLTR', 'NKE', 'HIMS', "ELF", "FUBO", "ROKU"])
    else:
        list_of_tickers.append(keyword)


list_of_tickers = ["^VIX", "QQQ", "SPY", "RIVN", "HUT", "SMCI", "GOOG", "SNAP", "AMD", "TSLA", "NVDA", "AAPL", "ARM", "PLTR", "AMZN", "SHOP", "NKE", "IONQ", 
                   "META", "MSFT", "QBTS", "SOXL", "TNA", "IBIT", "HIMS", "PINS", "RDDT", "ELF", "FUBO", "ROKU", "CVNA", "BABA", "BTC-USD", "ETH-USD", "COST", "WMT",
                  "ACHR", "OKLO", "RGTI", "JOBY", "RBLX", "MRVL", "KO", "DELL", "MU", "ADOBE", "AI", "CRM", "EA", "EXPE", "F", "SOFI", "U", "UPST", "VFS",
                  "ABNB", "UCO", "XOM"]

list_of_tickers_3 = ["^VIX", "QQQ", "Movers", "Chip", "EV", "Ecom", "Crypto", "Social Media", "Oil and Gas", "ETFs", "Quantum" "Misc group", "RIVN", "HUT", "SMCI", "GOOG", 
                     "SNAP", "AMD", "TSLA", "NVDA", "AAPL", "ARM", "PLTR", "AMZN", "SHOP", "NKE", "IONQ", "META", "MSFT", "QBTS", "SOXL", "TNA", "IBIT", "HIMS", 
                     "PINS", "RDDT", "ELF", "FUBO", "ROKU", "CVNA", "BABA", "BTC-USD", "ETH-USD", "COST", "WMT", "ACHR", "OKLO", "RGTI", "JOBY", "RBLX", "MRVL", 
                     "KO", "DELL", "MU", "ADOBE", "AI", "CRM", "EA", "EXPE", "F", "SOFI", "U", "UPST", "VFS", "ABNB", "UCO", "XOM"]

app = Dash()
server = app.server

app.layout = html.Div([
    html.H1('Stock Analysis Board', style={'textAlign': 'center', 'color': '#2c3e50', 'margin-bottom': '30px'}),
    
    html.Div([
        html.Div([
            dcc.Dropdown( id='ticker-list-id', options=list_of_tickers, value='QQQ', style={'margin': '5px 0'} ),
            html.Br(),
            dcc.Slider( id='time-duration', min=2, max=6, step=1, marks={i: str(i) for i in range(2, 7, 1)}, value=6 ),
            dcc.Checklist(options=[{ "label": "Full Time-Period Min, Max", "value": "FullTimePeriod",},
                                  ], id='check-list-id', labelStyle={"display": "flex", "align-items": "center", 'font-size': 15, "color": "green"})
        ], style={'width':'50%', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '10px', 'margin-right': '20px'}),
        
        html.Div([
            dcc.Dropdown( id='ticker-list-id-3', options=list_of_tickers_3, multi=True, value='QQQ', style={'margin': '5px 0'} ),
            html.Br(),
            dcc.Slider( id='mov-av-graph3-id', min=5, max=50, step=5, marks={i: str(i) for i in range(5, 51, 5)}, value=20 ),
            html.Button('Refresh Graph', id='refresh-button-id', style={"width": "100%", "height":"20px", "color":"green"})
        ], style={'width': '50%', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '10px', 'margin-right': '20px'}),
    ], style={'display': 'flex', 'margin': '20px'}),

    html.Div([
        dcc.Graph(id='graph-mini1-id', style={'padding': '0px', 'margin':'0px'}),            
        dcc.Graph(id='graph-mini2-id', style={'padding': '0px', 'margin':'0px'}),
    ], style={'display': 'flex', 'width':'100%'}),

    html.Div([
        dcc.Graph(id='graph-mini3-id', style={'padding': '0px', 'margin':'0px'}),            
        dcc.Graph(id='graph-mini4-id', style={'padding': '0px', 'margin':'0px'}),
    ], style={'display': 'flex', 'width':'100%'}),
    
    html.Div([
        dcc.Graph(id='graph-1-id', style={'padding': '0px', 'margin':'0px'}),            
        dcc.Graph(id='graph-2-id', style={'padding': '0px', 'margin':'0px'}),
        dcc.Graph(id='graph-3-id', style={'padding': '0px', 'margin':'0px'})
    ], style={'width':'100%', 'padding': '0px', 'margin':'0px', 'gap': '0px'})

], style={'padding': '0px'})


@callback(
    [Output('graph-mini1-id', 'figure'),
     Output('graph-mini2-id', 'figure'),
     Output('graph-mini3-id', 'figure'),
     Output('graph-mini4-id', 'figure'),
     Output('graph-1-id', 'figure'),
     Output('graph-2-id', 'figure'),
     Output('graph-3-id', 'figure')],
    [Input('ticker-list-id', 'value'),
     Input('time-duration', 'value'),
     Input('check-list-id', 'value'),
     Input('ticker-list-id-3', 'value'),
     Input('mov-av-graph3-id', 'value'),
     Input('refresh-button-id', 'n_clicks')]
)
def update_graph(ticker_list_id, time_duration, check_list_id, ticker_list_id_3, mov_av_graph3_id, refresh_button_id):
    graph_width = 1500
    custom_tick_text, custom_tick_vals = get_tick_data(time_duration)

    # Downloading data of last 12 months for minifig 1, 2, 3
    stock_data = get_daily_stock_data(ticker_list_id, 1) 
    # ------- FIGURE 1mini1mini1mini1mini1mini1mini1mini1mini1mini1mini1mini1mini1mini
    fig1mini = go.Figure()
    fig1mini.add_trace(go.Scatter(x = stock_data.index, y=stock_data['Close'][ticker_list_id], mode='lines', name='Year', line=dict(color='rgb(101, 110, 242)')))
    fig1mini.add_trace(go.Scatter(x = stock_data.index, y=stock_data['Close'][ticker_list_id].rolling(window=50).mean(), mode='lines', name='Year', line=dict(color='red')))
    fig1mini.add_trace(go.Bar(x = stock_data.index, y=stock_data['Volume'][ticker_list_id], name='Price', yaxis='y2', marker_color='rgba(242, 110, 10, 0.3)'))
    fig1mini.update_layout(xaxis_title='Year (50 MA)', width=graph_width//2, height=300, showlegend=False, template='plotly',
                           xaxis=dict(), yaxis=dict(title='Price'), yaxis2=dict(title='Volume', overlaying='y', side='right'))

    # ------- FIGURE 2mini2mini2mini2mini2mini2mini2mini2mini2mini2mini2mini2mini2mini
    fig2mini = go.Figure()
    days = 60
    fig2mini.add_trace(go.Scatter(x = stock_data.tail(days).index, y=stock_data.tail(days)['Close'][ticker_list_id], mode='lines', name='Month', line=dict(color='rgb(101, 110, 242)')))
    fig2mini.add_trace(go.Scatter(x = stock_data.tail(days).index, y=stock_data.tail(days)['Close'][ticker_list_id].rolling(window=20).mean(), mode='lines', name='Year', line=dict(color='red')))
    fig2mini.add_trace(go.Bar(x = stock_data.tail(days).index, y=stock_data.tail(days)['Volume'][ticker_list_id], name='Price', yaxis='y2', marker_color='rgba(242, 110, 10, 0.3)'))
    fig2mini.update_layout(xaxis_title='Month (20 MA)', width=graph_width//2, height=300, showlegend=False, template='plotly', 
                           xaxis=dict(), yaxis=dict(title='Price'), yaxis2=dict(title='Volume', overlaying='y', side='right'))
    
    # ------- FIGURE 3mini3mini3mini3mini3mini3mini3mini3mini3mini3mini3mini3mini3mini
    fig3mini = go.Figure()
    stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)
    stock_data = stock_data.reset_index()
    #histogram = stock_data['MACD'] - stock_data['Signal']
    
    fig3mini.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'].rolling(window=mov_av_graph3_id).mean(), name='MACD'))
    fig3mini.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal'].rolling(window=mov_av_graph3_id).mean(), name='Signal Line'))
    #fig3mini.add_trace(go.Bar(x=stock_data.index, y=histogram, yaxis='y2', marker_color='rgba(10, 220, 10, 0.5)'))
    fig3mini.add_hline(y = 0, line_width=3, line_color="rgba(101, 110, 242, 0.5)")
    fig3mini.update_layout(title=None, width=graph_width//2, height=300, yaxis=dict(title='MACD Daily'), yaxis2=dict(overlaying='y', side='right'), xaxis=dict(showticklabels=False), showlegend = False)

    # ------- FIGURE 4mini4mini4mini4mini4mini4mini4mini4mini4mini4mini4mini4mini4mini 
    # Downloading data of last 5 days of 5 min time frame for minifig 4
    stock_data = get_5minute_stock_data(ticker_list_id, time_duration)
    fig4mini = go.Figure()
    stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)
    stock_data = stock_data.reset_index()
    fig4mini.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'].rolling(window=mov_av_graph3_id).mean(), name='MACD'))
    fig4mini.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal'].rolling(window=mov_av_graph3_id).mean(), name='Signal Line'))
    fig4mini.add_hline(y = 0, line_width=3, line_color="rgba(101, 110, 242, 0.5)")
    fig4mini.update_layout(title=None, width=graph_width//2, height=300, yaxis=dict(title='MACD 5 min', overlaying='y'), xaxis=dict(showticklabels=False), showlegend = False)

    # Downloading data for minifig fig 1, 2
    stock_data = get_minute_stock_data(ticker_list_id, time_duration)
    openI, shortI = get_oi_shorti(ticker_list_id)
    title_graph = ticker_list_id + " P/C: " + openI + " Short Interest: " + shortI
    price = stock_data["Close"][ticker_list_id].tolist()
    x = [i for i in range(len(price))] # range for x-axis for all the three main graphs
    
    # ------- FIGURE 1111111111111111111111111111111111111111111111111111111111111111111111
    if check_list_id != None and "FullTimePeriod" in check_list_id:
        op, cl, mx, mn = get_open_close_min_max(price[:(time_duration-1) * 390])
    else: # yesterday's
        op, cl, mx, mn = get_open_close_min_max(price[(time_duration-2) * 390:(time_duration-1) * 390])
        
    price30 = apply_gaussian_filter(price, 30, "nearest")
    price100 = apply_gaussian_filter(price, 100, "nearest")
    price200 = apply_gaussian_filter(price, 200, "nearest")
    
    fig1 = go.Figure()
    #fig1.add_trace(go.Candlestick(x=x, open=open_prices, high=max_prices, low=min_prices, close=close_prices, increasing_line_color= 'rgba(10, 200, 10, 0.5)', decreasing_line_color= 'rgba(200, 10, 10, 0.5)', name = ticker_list_id))
    fig1.add_trace(go.Scatter(x = x, y=price, mode='lines', name='Price', line=dict(color='rgba(101, 110, 242, 0.5)')))
    fig1.add_trace(go.Scatter(x = x, y=price30[:-30], mode='lines', name='Gaussian 30',  line=dict(color='red')))
    fig1.add_trace(go.Scatter(x = x, y=price100[:-100], mode='lines', name='Gaussian 100', line=dict(color='light green')))
    fig1.add_trace(go.Scatter(x = x, y=price200[:-200], mode='lines', name='Gaussian 200', line=dict(color='black')))
    fig1.add_hline(y = cl, line_width=3, line_color="orange")
    fig1.add_hline(y = mx, line_width=3, line_dash="dash", line_color="blue")
    fig1.add_hline(y = mn, line_width=3, line_dash="dash", line_color="blue")
    
    fig1.update_layout(title = title_graph, xaxis_title='Time', yaxis_title='Price', legend_title='Gaussian Trend', 
                       width=graph_width+10, height=550, xaxis_rangeslider_visible=False, template='plotly', spikedistance=-1, 
                       xaxis=dict(tickvals = custom_tick_vals, ticktext = custom_tick_text, tickangle=-90, showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1  ),
                       yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1))
    # ------- FIGURE 22222222222222222222222222222222222222222222222222222222222222222222
    
    fig2 = go.Figure()
    second_diff15 = second_order_difference(moving_average_end_at_current(price, 15), 15)
    fig2.add_trace(go.Scatter(x = x, y=second_diff15, mode='lines', name='15 minutes', line=dict(color='orange'), visible='legendonly'))
    second_diff30 = second_order_difference(moving_average_end_at_current(price, 30), 30)
    fig2.add_trace(go.Scatter(x = x, y=second_diff30, mode='lines', name='30 minutes', line=dict(color='red')))
    second_diff60 = second_order_difference(moving_average_end_at_current(price, 60), 60)
    fig2.add_trace(go.Scatter(x = x, y=second_diff60, mode='lines', name='1 hour', line=dict(color='rgb(77, 163, 126)')))
    second_diff120 = second_order_difference(moving_average_end_at_current(price, 120), 120)
    fig2.add_trace(go.Scatter(x = x, y=second_diff120, mode='lines', name='2 hours', line=dict(color='black')))
    fig2.add_hline(y = 0, line_width=3, line_color="rgba(101, 110, 242, 0.5)")
    
    fig2.update_layout( title = ticker_list_id, xaxis_title='Time', yaxis_title='Rate of Change', legend_title='Time Periods', template='plotly', width=graph_width, height=450, 
                        xaxis=dict(tickvals = custom_tick_vals, ticktext = custom_tick_text,tickangle=-90))
    
    # ------- FIGURE 333333333333333333333333333333333333333333333333333333333333333333333
    fig3 = go.Figure()
    
    tickers_to_plot = ['QQQ']
    if isinstance(ticker_list_id_3, str) and len(ticker_list_id_3) > 0:
        tickers_to_plot = [ticker_list_id_3]
    else:
        tickers_to_plot = []
        for ticker in ticker_list_id_3:
            get_list_of_tickers_from_keyword(tickers_to_plot, ticker)
    
    if ticker_list_id_3 != None:
        for ticker in tickers_to_plot:
            stock_data = get_minute_stock_data(ticker, time_duration)
            price = stock_data["Close"][ticker].tolist()
            price = moving_average_end_at_current(price, mov_av_graph3_id)
            percent_difference_from_first_value_in_list(price)
            if ticker == "QQQ":
                fig3.add_trace(go.Scatter(x = x, y=price, mode='lines', name=ticker, line=dict(color='black')))
            else:
                fig3.add_trace(go.Scatter(x = x, y=price, mode='lines', name=ticker, visible='legendonly'))
        
        fig3.update_layout( xaxis_title='Time', yaxis_title='Price', legend_title='Multiple Tickers', template='plotly', width=graph_width, height=500,
                        xaxis=dict(tickvals = custom_tick_vals, ticktext = custom_tick_text, tickangle=-90 ))
    
    return fig1mini, fig2mini, fig3mini, fig4mini, fig1, fig2, fig3
    
if __name__ == '__main__':
    app.run()






