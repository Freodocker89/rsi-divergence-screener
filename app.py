# app.py — RSI Divergence Screener (fast + market-cap ranking)
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objs as go

try:
    import ccxt
except Exception:
    ccxt = None

# ----------------------
# RSI + Divergence logic
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
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def find_pivots(series: pd.Series, left: int = 5, right: int = 5) -> Tuple[pd.Series, pd.Series]:
    ph = pd.Series(np.nan, index=series.index)
    pl = pd.Series(np.nan, index=series.index)
    values = series.values
    n = len(series)
    for i in range(left, n - right):
        window = values[i - left:i + right + 1]
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

def detect_divergences(
    df: pd.DataFrame,
    rsi_len: int = 14,
    left: int = 5,
    right: int = 5,
    lookback_bars: int = 150,
    include_hidden: bool = True
) -> Dict[str, List[Dict]]:
    close = df["close"]
    rs = rsi(close, rsi_len)

    ph_price, _ = find_pivots(df["high"], left, right)
    _, pl_price = find_pivots(df["low"], left, right)
    ph_rsi, pl_rsi = find_pivots(rs, left, right)

    price_high_idx = np.where(~np.isnan(ph_price.values))[0].tolist()
    price_low_idx  = np.where(~np.isnan(pl_price.values))[0].tolist()
    rsi_high_idx   = np.where(~np.isnan(ph_rsi.values))[0].tolist()
    rsi_low_idx    = np.where(~np.isnan(pl_rsi.values))[0].tolist()

    results = {"bullish": [], "bearish": [], "hidden_bullish": [], "hidden_bearish": []}

    # Regular bullish: Price LL, RSI HL (compare consecutive price lows)
    for i2_pos in range(1, len(price_low_idx)):
        i1 = price_low_idx[i2_pos - 1]
        i2 = price_low_idx[i2_pos]
        if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
            r1 = nearest_pivot_index(rsi_low_idx, i1)
            r2 = nearest_pivot_index(rsi_low_idx, i2)
            if r1 is None or r2 is None:
                continue
            if df["low"].iloc[i2] < df["low"].iloc[i1] and rs.iloc[r2] > rs.iloc[r1]:
                results["bullish"].append({"bar_index": i2, "time": df.index[i2], "i1": i1, "i2": i2, "r1": r1, "r2": r2})

    # Regular bearish: Price HH, RSI LH (compare consecutive price highs)
    for i2_pos in range(1, len(price_high_idx)):
        i1 = price_high_idx[i2_pos - 1]
        i2 = price_high_idx[i2_pos]
        if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
            r1 = nearest_pivot_index(rsi_high_idx, i1)
            r2 = nearest_pivot_index(rsi_high_idx, i2)
            if r1 is None or r2 is None:
                continue
            if df["high"].iloc[i2] > df["high"].iloc[i1] and rs.iloc[r2] < rs.iloc[r1]:
                results["bearish"].append({"bar_index": i2, "time": df.index[i2], "i1": i1, "i2": i2, "r1": r1, "r2": r2})

    if include_hidden:
        # Hidden bullish: Price HL, RSI LL (lows)
        for i2_pos in range(1, len(price_low_idx)):
            i1 = price_low_idx[i2_pos - 1]
            i2 = price_low_idx[i2_pos]
            if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
                r1 = nearest_pivot_index(rsi_low_idx, i1)
                r2 = nearest_pivot_index(rsi_low_idx, i2)
                if r1 is None or r2 is None:
                    continue
                if df["low"].iloc[i2] > df["low"].iloc[i1] and rs.iloc[r2] < rs.iloc[r1]:
                    results["hidden_bullish"].append({"bar_index": i2, "time": df.index[i2], "i1": i1, "i2": i2, "r1": r1, "r2": r2})
        # Hidden bearish: Price LH, RSI HH (highs)
        for i2_pos in range(1, len(price_high_idx)):
            i1 = price_high_idx[i2_pos - 1]
            i2 = price_high_idx[i2_pos]
            if i2 < len(df) - right and i2 >= len(df) - lookback_bars:
                r1 = nearest_pivot_index(rsi_high_idx, i1)
                r2 = nearest_pivot_index(rsi_high_idx, i2)
                if r1 is None or r2 is None:
                    continue
                if df["high"].iloc[i2] < df["high"].iloc[i1] and rs.iloc[r2] > rs.iloc[r1]:
                    results["hidden_bearish"].append({"bar_index": i2, "time": df.index[i2], "i1": i1, "i2": i2, "r1": r1, "r2": r2})

    return results

def summarize_results(symbol: str, tf: str, res: Dict[str, List[Dict]]) -> Dict:
    recent = {}
    for k, lst in res.items():
        if lst:
            recent[k] = lst[-1]
    return {
        "symbol": symbol,
        "timeframe": tf,
        "bullish": "✅" if "bullish" in recent else "",
        "bearish": "✅" if "bearish" in recent else "",
        "hidden_bullish": "✅" if "hidden_bullish" in recent else "",
        "hidden_bearish": "✅" if "hidden_bearish" in recent else "",
        "last_signal_time": str(max([v["time"] for v in recent.values()], default="")),
        "details": recent,
    }

def base_from_symbol(sym: str) -> str:
    left = sym.split("/")[0]
    base = left.split(":")[0]
    return base.upper()

# ----------------------
# Market-cap ranks (CoinGecko)
# ----------------------
@st.cache_data(show_spinner=False, ttl=1800)
def get_coingecko_ranks(pages: int = 4) -> dict:
    """Fetch ~1000 top coins by market cap. Returns SYMBOL (uppercase) -> rank (int)."""
    ranks = {}
    url = "https://api.coingecko.com/api/v3/coins/markets"
    for page in range(1, pages + 1):
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false",
            "locale": "en",
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break
        if not data:
            break
        for coin in data:
            sym = (coin.get("symbol") or "").upper()
            rank = coin.get("market_cap_rank")
            if sym and rank:
                ranks[sym] = int(rank)
    return ranks

# ----------------------
# CCXT exchange handling
# ----------------------
@st.cache_resource(show_spinner=False)
def get_exchange(exchange_name: str):
    if ccxt is None:
        return None
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    ex.load_markets()
    return ex

@st.cache_data(show_spinner=False, ttl=300)
def load_symbols(exchange_name: str, quote: str = "USDT", linear_only: bool = True, max_symbols: int = 500) -> List[str]:
    ex = get_exchange(exchange_name)
    if ex is None:
        return []
    markets = ex.markets
    syms = []
    for m in markets.values():
        if m.get("type") not in ("swap", "future", "futures"):
            continue
        if quote and m.get("quote") != quote:
            continue
        if linear_only and m.get("linear") is False:
            continue
        if m.get("active") is False:
            continue
        syms.append(m["symbol"])
    return sorted(list(dict.fromkeys(syms)))[:max_symbols]

@st.cache_data(show_spinner=False, ttl=300)
def fetch_ohlcv(exchange_name: str, symbol: str, timeframe: str = "1h", limit: int = 400) -> Optional[pd.DataFrame]:
    ex = get_exchange(exchange_name)
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return None
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time")
    return df

# ----------------------
# Async fast scan (optional)
# ----------------------
async def scan_async(exchange_name, symbols, timeframes, limit, rsi_len, left, right, lookback, include_hidden, concurrency):
    try:
        from ccxt import async_support as ccxt_async
    except Exception:
        # Fallback: no async support available
        return []

    ex = getattr(ccxt_async, exchange_name)({"enableRateLimit": True})
    await ex.load_markets()

    sem = asyncio.Semaphore(concurrency)

    async def fetch_one(sym, tf):
        async with sem:
            try:
                ohlcv = await ex.fetch_ohlcv(sym, timeframe=tf, limit=limit)
            except Exception:
                return sym, tf, None
            if not ohlcv:
                return sym, tf, None
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df = df.set_index("time")
            return sym, tf, df

    tasks = [fetch_one(sym, tf) for sym in symbols for tf in timeframes]
    results = []
    for coro in asyncio.as_completed(tasks):
        sym, tf, df = await coro
        if df is None or len(df) < (left + right + 10):
            continue
        res = detect_divergences(df, rsi_len, left, right, lookback, include_hidden)
        if any(len(v) > 0 for v in res.values()):
            results.append(summarize_results(sym, tf, res))

    try:
        await ex.close()
    except Exception:
        pass
    return results

# ----------------------
# Plotting
# ----------------------
def plot_chart(df: pd.DataFrame, symbol: str, rsi_len: int, last_signal: Optional[Dict] = None):
    rs = rsi(df["close"], rsi_len)

    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=symbol
    ))
    fig_price.update_layout(title=f"{symbol}", xaxis_rangeslider_visible=False)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=rs, mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dot")
    fig_rsi.add_hline(y=30, line_dash="dot")
    fig_rsi.update_layout(title="RSI")

    if last_signal:
        i1, i2 = last_signal.get("i1"), last_signal.get("i2")
        r1, r2 = last_signal.get("r1"), last_signal.get("r2")
        if i1 is not None and i2 is not None:
            fig_price.add_trace(go.Scatter(
                x=[df.index[i1], df.index[i2]],
                y=[df["close"].iloc[i1], df["close"].iloc[i2]],
                mode="lines+markers",
                name="Price Swing (Div)"
            ))
        if r1 is not None and r2 is not None:
            fig_rsi.add_trace(go.Scatter(
                x=[df.index[r1], df.index[r2]],
                y=[rs.iloc[r1], rs.iloc[r2]],
                mode="lines+markers",
                name="RSI Swing (Div)"
            ))

    return fig_price, fig_rsi

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="RSI Divergence Screener", layout="wide")
st.title("🔎 RSI Divergence Screener (Fast + Market-Cap Sorted)")

with st.sidebar:
    st.header("Settings")
    exchange_name = st.selectbox("Exchange", ["bitget", "binance", "bybit"], index=0)
    quote = st.selectbox("Quote asset", ["USDT", "USDC", "USD", "BTC"], index=0)
    timeframes = st.multiselect("Timeframes", ["15m", "30m", "1h", "4h", "1d"], default=["1h", "4h", "1d"])

    rsi_len = st.number_input("RSI length", 2, 100, 14, step=1)
    left = st.number_input("Pivot left bars", 1, 20, 5, step=1)
    right = st.number_input("Pivot right bars", 1, 20, 5, step=1)
    lookback_bars = st.number_input("Lookback bars", 20, 2000, 250, step=10)
    include_hidden = st.checkbox("Include hidden divergences", value=True)
    confirm_candle = st.checkbox("Require confirmation candle (pivot completed)", value=True)

    max_symbols = st.number_input("Max symbols to load from exchange", 10, 1000, 400, step=10)
    limit_per_tf = st.number_input("Bars per timeframe (fetch limit)", 150, 2000, 600, step=50)

    # Speed helpers
    top_n = st.number_input("Limit to Top-N by market cap (0 = all)", 0, 1000, 200, step=25)
    fast_mode = st.checkbox("Fast mode (concurrent fetch)", value=False)
    concurrency = st.slider("Max concurrent requests", 2, 12, 6) if fast_mode else 0

    run_btn = st.button("Run Scan", type="primary")

def attach_rank_rows(rows: List[Dict]) -> List[Dict]:
    ranks = get_coingecko_ranks()
    for r in rows:
        base = base_from_symbol(r["symbol"])
        r["rank"] = ranks.get(base, 999999)
    return rows

if run_btn:
    if ccxt is None:
        st.error("ccxt is not installed. Add it to requirements.txt and redeploy.")
        st.stop()

    with st.spinner("Loading symbols…"):
        symbols = load_symbols(exchange_name, quote, True, int(max_symbols))

    if not symbols:
        st.warning("No symbols found with current filters.")
        st.stop()

    # Prefilter by market cap
    if top_n and top_n > 0:
        ranks = get_coingecko_ranks()
        symbols = sorted(symbols, key=lambda s: ranks.get(base_from_symbol(s), 999999))[: int(top_n)]

    st.write(f"Scanning **{len(symbols)}** symbols on **{exchange_name}** quoted in **{quote}**…")

    results_rows: List[Dict] = []

    if fast_mode:
        # Concurrent path
        results_rows = asyncio.run(
            scan_async(
                exchange_name, symbols, timeframes, int(limit_per_tf),
                int(rsi_len), int(left), int(right), int(lookback_bars),
                include_hidden, int(concurrency)
            )
        )
    else:
        # Sequential but with single shared client (much faster than re-creating each time)
        progress = st.progress(0.0)
        total = len(symbols) * len(timeframes)
        done = 0
        for sym in symbols:
            for tf in timeframes:
                df = fetch_ohlcv(exchange_name, sym, tf, int(limit_per_tf))
                done += 1
                progress.progress(min(done / total, 1.0))
                if df is None or len(df) < (left + right + 10):
                    continue
                res = detect_divergences(df, int(rsi_len), int(left), int(right), int(lookback_bars), include_hidden)
                if confirm_candle:
                    def filter_confirmed(lst):
                        return [x for x in lst if x["bar_index"] <= len(df) - right - 1]
                    for k in list(res.keys()):
                        res[k] = filter_confirmed(res[k])
                if any(len(v) > 0 for v in res.values()):
                    results_rows.append(summarize_results(sym, tf, res))

    if not results_rows:
        st.info("No recent divergences found under the current settings.")
        st.stop()

    # Attach market-cap ranks and sort largest first, then most recent
    results_rows = attach_rank_rows(results_rows)
    table = (
        pd.DataFrame(results_rows)
        .sort_values(["rank", "last_signal_time"], ascending=[True, False])
        .reset_index(drop=True)
    )

    st.subheader("Results (sorted by market-cap rank)")
    st.dataframe(table[["rank", "symbol", "timeframe", "bullish", "bearish", "hidden_bullish", "hidden_bearish", "last_signal_time"]], use_container_width=True)

    # Chart selection
    st.subheader("Chart")
    sel_idx = st.number_input("Row index to chart", 0, len(table) - 1, 0, step=1)
    sel = table.iloc[int(sel_idx)]
    st.write(f"Showing: **{sel['symbol']}** on **{sel['timeframe']}**")

    df_plot = fetch_ohlcv(exchange_name, sel["symbol"], sel["timeframe"], int(limit_per_tf))
    last_sig = None
    details = sel["details"]
    # Prefer regular over hidden if multiple
    for key in ["bullish", "bearish", "hidden_bullish", "hidden_bearish"]:
        if key in details:
            last_sig = details[key]
    fig_price, fig_rsi = plot_chart(df_plot, sel["symbol"], int(rsi_len), last_sig)
    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)

    # CSV export
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", data=csv, file_name="rsi_divergence_scan.csv", mime="text/csv")

else:
    st.info("Configure settings in the sidebar and click **Run Scan**.")

