# app.py
import os
import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

try:
    import ccxt
except Exception as e:
    ccxt = None

# ----------------------
# Helpers
# ----------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def find_pivots(series: pd.Series, left: int = 5, right: int = 5) -> Tuple[pd.Series, pd.Series]:
    ph = pd.Series(np.nan, index=series.index)
    pl = pd.Series(np.nan, index=series.index)
    values = series.values
    n = len(series)
    for i in range(left, n-right):
        window = values[i-left:i+right+1]
        center = values[i]
        if np.argmax(window) == left and np.sum(window == window.max()) == 1:
            ph.iloc[i] = center
        if np.argmin(window) == left and np.sum(window == window.min()) == 1:
            pl.iloc[i] = center
    return ph, pl

def nearest_pivot_index(pivot_idx_list: List[int], target_idx: int) -> Optional[int]:
    if not pivot_idx_list:
        return None
    arr = np.array(pivot_idx_list)
    j = int(np.argmin(np.abs(arr - target_idx)))
    return int(arr[j])

def detect_divergences(df: pd.DataFrame,
                       rsi_len: int = 14,
                       left: int = 5,
                       right: int = 5,
                       lookback_bars: int = 150,
                       include_hidden: bool = True) -> Dict[str, List[Dict]]:
    close = df['close']
    rs = rsi(close, rsi_len)
    ph_price, pl_price = find_pivots(df['high'], left, right)[0], find_pivots(df['low'], left, right)[1]
    ph_rsi, pl_rsi     = find_pivots(rs, left, right)
    price_high_idx = np.where(~np.isnan(ph_price.values))[0].tolist()
    price_low_idx  = np.where(~np.isnan(pl_price.values))[0].tolist()
    rsi_high_idx   = np.where(~np.isnan(ph_rsi.values))[0].tolist()
    rsi_low_idx    = np.where(~np.isnan(pl_rsi.values))[0].tolist()

    results = {'bullish': [], 'bearish': [], 'hidden_bullish': [], 'hidden_bearish': []}

    for i2_pos in range(1, len(price_low_idx)):
        i1 = price_low_idx[i2_pos-1]
        i2 = price_low_idx[i2_pos]
        if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
            r1 = nearest_pivot_index(rsi_low_idx, i1)
            r2 = nearest_pivot_index(rsi_low_idx, i2)
            if r1 is None or r2 is None:
                continue
            price_ll = df['low'].iloc[i2] < df['low'].iloc[i1]
            rsi_hl   = rs.iloc[r2] > rs.iloc[r1]
            if price_ll and rsi_hl:
                results['bullish'].append({'bar_index': i2, 'time': df.index[i2],
                                           'i1': i1, 'i2': i2, 'r1': r1, 'r2': r2})

    for i2_pos in range(1, len(price_high_idx)):
        i1 = price_high_idx[i2_pos-1]
        i2 = price_high_idx[i2_pos]
        if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
            r1 = nearest_pivot_index(rsi_high_idx, i1)
            r2 = nearest_pivot_index(rsi_high_idx, i2)
            if r1 is None or r2 is None:
                continue
            price_hh = df['high'].iloc[i2] > df['high'].iloc[i1]
            rsi_lh   = rs.iloc[r2] < rs.iloc[r1]
            if price_hh and rsi_lh:
                results['bearish'].append({'bar_index': i2, 'time': df.index[i2],
                                           'i1': i1, 'i2': i2, 'r1': r1, 'r2': r2})

    if include_hidden:
        for i2_pos in range(1, len(price_low_idx)):
            i1 = price_low_idx[i2_pos-1]
            i2 = price_low_idx[i2_pos]
            if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
                r1 = nearest_pivot_index(rsi_low_idx, i1)
                r2 = nearest_pivot_index(rsi_low_idx, i2)
                if r1 is None or r2 is None:
                    continue
                price_hl = df['low'].iloc[i2] > df['low'].iloc[i1]
                rsi_ll   = rs.iloc[r2] < rs.iloc[r1]
                if price_hl and rsi_ll:
                    results['hidden_bullish'].append({'bar_index': i2, 'time': df.index[i2],
                                                      'i1': i1, 'i2': i2, 'r1': r1, 'r2': r2})

        for i2_pos in range(1, len(price_high_idx)):
            i1 = price_high_idx[i2_pos-1]
            i2 = price_high_idx[i2_pos]
            if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
                r1 = nearest_pivot_index(rsi_high_idx, i1)
                r2 = nearest_pivot_index(rsi_high_idx, i2)
                if r1 is None or r2 is None:
                    continue
                price_lh = df['high'].iloc[i2] < df['high'].iloc[i1]
                rsi_hh   = rs.iloc[r2] > rs.iloc[r1]
                if price_lh and rsi_hh:
                    results['hidden_bearish'].append({'bar_index': i2, 'time': df.index[i2],
                                                      'i1': i1, 'i2': i2, 'r1': r1, 'r2': r2})
    return results

def plot_chart(df, rsi_len, last_signal=None):
    rs = rsi(df['close'], rsi_len)
    fig_price = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])
    fig_rsi = go.Figure(data=[go.Scatter(x=df.index, y=rs, mode='lines')])
    fig_rsi.add_hline(y=70, line_dash='dot')
    fig_rsi.add_hline(y=30, line_dash='dot')
    return fig_price, fig_rsi

@st.cache_data(ttl=300)
def load_symbols(exchange_name: str, quote: str = 'USDT', max_symbols: int = 200) -> List[str]:
    if ccxt is None:
        return []
    ex = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    markets = ex.load_markets()
    syms = []
    for m in markets.values():
        if m.get('type') in ('swap', 'future', 'futures') and m.get('quote') == quote and m.get('active'):
            syms.append(m['symbol'])
    return sorted(syms)[:max_symbols]

@st.cache_data(ttl=300)
def fetch_ohlcv(exchange_name, symbol, timeframe='1h', limit=400):
    if ccxt is None:
        return None
    ex = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    df = df.set_index('time')
    return df

st.set_page_config(page_title="RSI Divergence Screener", layout="wide")
st.title("🔎 RSI Divergence Screener")

with st.sidebar:
    exchange_name = st.selectbox("Exchange", ["bitget", "binance", "bybit"], 0)
    quote = st.selectbox("Quote", ["USDT","USDC","USD"], 0)
    timeframes = st.multiselect("Timeframes", ["15m","30m","1h","4h","1d"], ["1h","4h","1d"])
    rsi_len = st.number_input("RSI length", 2, 100, 14)
    left = st.number_input("Pivot left", 1, 20, 5)
    right = st.number_input("Pivot right", 1, 20, 5)
    lookback = st.number_input("Lookback bars", 20, 2000, 250)
    include_hidden = st.checkbox("Include hidden", True)
    max_symbols = st.number_input("Max symbols", 5, 1000, 100)
    limit_bars = st.number_input("Bars per TF", 150, 2000, 500)
    run_btn = st.button("Run Scan", type="primary")

if run_btn:
    symbols = load_symbols(exchange_name, quote, max_symbols)
    st.write(f"Scanning {len(symbols)} symbols...")
    rows = []
    for sym in symbols:
        for tf in timeframes:
            df = fetch_ohlcv(exchange_name, sym, tf, limit_bars)
            if df is None:
                continue
            res = detect_divergences(df, rsi_len, left, right, lookback, include_hidden)
            if any(res.values()):
                rows.append({"symbol": sym, "timeframe": tf, **{k: bool(v) for k,v in res.items()}})
    if rows:
        df_res = pd.DataFrame(rows)
        st.dataframe(df_res)
    else:
        st.info("No divergences found.")
