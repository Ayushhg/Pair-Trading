# 📊 Pair Trading Dashboard using Streamlit

This interactive dashboard analyzes the cointegration and statistical arbitrage opportunities between two financial instruments (stocks) using the **Pair Trading strategy**. It is built with Python and Streamlit, using real-time data from Yahoo Finance.

## 🚀 Features

- 📈 Real-time price data fetching using `yfinance`
- ⚖️ Cointegration test (Engle-Granger) to assess pair relationship
- 📉 Augmented Dickey-Fuller (ADF) test for spread stationarity
- 🧮 Hedge ratio estimation via OLS regression
- 🔁 Spread and Z-score calculation
- 📍 Dynamic pair trading signals based on Z-score thresholds
- 🧠 Simple backtest logic with cumulative strategy returns
- 🧾 Trade log generation (entry/exit points)
- 📤 Data export to CSV
- 🔧 Configurable parameters (tickers, period, interval, thresholds)

## 🛠️ Tech Stack

- Python
- [Streamlit](https://streamlit.io/)
- [yfinance](https://pypi.org/project/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [statsmodels](https://www.statsmodels.org/)
- [numpy](https://numpy.org/)

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pair-trading-dashboard.git
cd pair-trading-dashboard


pip install streamlit yfinance pandas numpy statsmodels
```

### 2. Run The App
streamlit run app.py

