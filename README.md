# RSI Divergence Screener

Scans Bitget/Binance/Bybit perps for RSI bullish/bearish and hidden divergences.

## Local run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
