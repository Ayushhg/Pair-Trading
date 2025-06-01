import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from datetime import datetime, timedelta

st.title("Pair Trading - Cointegration and Spread Analysis")

# Sidebar inputs
st.sidebar.header("Input Parameters")

stock1 = st.sidebar.text_input("Enter first stock symbol", value="MSFT")
stock2 = st.sidebar.text_input("Enter second stock symbol", value="AAPL")

interval = st.sidebar.selectbox("Select data interval", ['1m', '5m', '15m', '30m', '60m', '1d'], index=1)

period = st.sidebar.selectbox("Select data period", ['7d', '30d', '60d', '90d', '180d', '1y'], index=2)

run_analysis = st.sidebar.button("Run Analysis")

with st.sidebar.expander("Set Z-Score Thresholds for Trading"):
    entry_threshold = st.number_input("Entry Threshold (Z-Score)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    exit_threshold = st.number_input("Exit Threshold (Z-Score)", min_value=0.0, max_value=5.0, value=0.0, step=0.1,
                                     help="Position closes when absolute Z-Score falls below this value")


if run_analysis:
    if not stock1 or not stock2:
        st.error("Please enter both stock symbols.")
    else:
        with st.spinner("Fetching data..."):
            try:
                data1 = yf.download(stock1, interval=interval, period=period)
                data2 = yf.download(stock2, interval=interval, period=period)

                if data1.empty or data2.empty:
                    st.error("One or both symbols returned no data. Try different tickers or longer period.")
                    st.stop()

                data1.reset_index(inplace=True)
                data2.reset_index(inplace=True)

                data1.insert(0, "Price", stock1)
                data2.insert(0, "Price", stock2)

                data1 = data1.rename(columns={"Datetime": "Timestamp", "index": "Timestamp"})
                data2 = data2.rename(columns={"Datetime": "Timestamp", "index": "Timestamp"})

                data1.columns = ['Price', 'Timestamp', 'Close', 'High', 'Low', 'Open', 'Volume']
                data2.columns = ['Price', 'Timestamp', 'Close', 'High', 'Low', 'Open', 'Volume']

                data1 = data1.iloc[1:].reset_index(drop=True)
                data2 = data2.iloc[1:].reset_index(drop=True)

                # Align on timestamps by merging on Timestamp column
                df = pd.merge(data1[['Timestamp', 'Close']], data2[['Timestamp', 'Close']], on='Timestamp', suffixes=(f'_{stock1}', f'_{stock2}'))
                #df.rename(columns={'Datetime': 'Timestamp'}, inplace=True)

                # Check minimum data points
                if len(df) < 30:
                    st.warning(f"Warning: Only {len(df)} data points found. Results may be unreliable.")

                st.success("Data fetched successfully!")
                st.subheader("Price Data")
                st.line_chart(df.set_index('Timestamp')[[f'Close_{stock1}', f'Close_{stock2}']])

                # Calculate hedge ratio by OLS regression (stock1 on stock2)
                X = sm.add_constant(df[f'Close_{stock2}'])
                model = sm.OLS(df[f'Close_{stock1}'], X).fit()
                hedge_ratio = model.params[f'Close_{stock2}']
                st.write(f"**Hedge Ratio (β):** {hedge_ratio:.4f}")

                # Calculate spread
                df['Spread'] = df[f'Close_{stock1}'] - hedge_ratio * df[f'Close_{stock2}']

                st.subheader("Spread Plot")
                st.line_chart(df.set_index('Timestamp')['Spread'])

                # Engle-Granger Cointegration Test
                coint_score, coint_pvalue, _ = coint(df[f'Close_{stock1}'], df[f'Close_{stock2}'])
                st.subheader("Engle-Granger Cointegration Test")
                st.write(f"Test Statistic: **{coint_score:.4f}**")
                st.write(f"P-value: **{coint_pvalue:.4f}**")
                if coint_pvalue < 0.05:
                    st.success("Likely cointegrated (p < 0.05)")
                else:
                    st.warning("Likely not cointegrated (p ≥ 0.05)")

                # Augmented Dickey-Fuller test on spread for stationarity
                adf_result = adfuller(df['Spread'])
                st.subheader("Augmented Dickey-Fuller Test on Spread")
                st.write(f"ADF Statistic: **{adf_result[0]:.4f}**")
                st.write(f"P-value: **{adf_result[1]:.4f}**")
                if adf_result[1] < 0.05:
                    st.success("Spread is stationary (p < 0.05)")
                else:
                    st.warning("Spread is not stationary (p ≥ 0.05)")

                # Summary Statistics
                st.subheader("Summary Statistics")
                stats = {
                    f"{stock1} Mean": df[f'Close_{stock1}'].mean(),
                    f"{stock1} Std Dev": df[f'Close_{stock1}'].std(),
                    f"{stock2} Mean": df[f'Close_{stock2}'].mean(),
                    f"{stock2} Std Dev": df[f'Close_{stock2}'].std(),
                    "Correlation": df[[f'Close_{stock1}', f'Close_{stock2}']].corr().iloc[0, 1],
                    "Spread Mean": df['Spread'].mean(),
                    "Spread Std Dev": df['Spread'].std()
                }
                st.write(pd.Series(stats))

                # Calculate half-life of mean reversion
                spread_lag = df['Spread'].shift(1).fillna(method='bfill')
                spread_ret = df['Spread'] - spread_lag
                spread_lag_const = sm.add_constant(spread_lag)
                model_hl = sm.OLS(spread_ret, spread_lag_const).fit()
                halflife = -np.log(2) / model_hl.params[1] if model_hl.params[1] != 0 else np.nan
                st.write(f"Half-life of mean reversion: **{halflife:.2f} periods**")

                # Calculate Z-score of spread
                df['Z-Score'] = (df['Spread'] - df['Spread'].mean()) / df['Spread'].std()
                st.subheader("Z-Score of Spread")
                st.line_chart(df.set_index('Timestamp')['Z-Score'])

                # Optional: simple backtest signals
                st.subheader("Simple Pairs Trading Signals and Returns (Backtest)")
                # entry_threshold = 1
                # exit_threshold = -1

                df['Position'] = 0
                df.loc[df['Z-Score'] > entry_threshold, 'Position'] = -1  # short spread
                df.loc[df['Z-Score'] < -entry_threshold, 'Position'] = 1  # long spread

                # Close positions when Z-score reverts close to 0
                df.loc[df['Z-Score'].abs() < exit_threshold, 'Position'] = 0

                df['Position'] = df['Position'].ffill().fillna(0)

                # Calculate daily returns of spread
                df['Spread_Returns'] = df['Spread'].diff()
                df['Strategy_Returns'] = df['Position'].shift(1) * df['Spread_Returns']

                df['Cumulative_Strategy_Returns'] = df['Strategy_Returns'].cumsum()
                st.line_chart(df.set_index('Timestamp')['Cumulative_Strategy_Returns'])
                st.write("Note: This is a simple backtest example. Real trading requires transaction costs, slippage, and risk management.")
                
                # Identify trades (entries and exits)
                df['Position_Change'] = df['Position'].diff()

                trades = []

                for i, row in df.iterrows():
                    if row['Position_Change'] == 1:  # Entry Long
                        trades.append({
                            'Timestamp': row['Timestamp'],
                            'Trade': 'Entry Long',
                            'Price_Spread': row['Spread'],
                            'Z-Score': row['Z-Score']
                        })
                    elif row['Position_Change'] == -1:  # Exit Long
                        trades.append({
                            'Timestamp': row['Timestamp'],
                            'Trade': 'Exit Long',
                            'Price_Spread': row['Spread'],
                            'Z-Score': row['Z-Score']
                        })
                    elif row['Position_Change'] == -1:  # Entry Short (if from 0 to -1)
                        trades.append({
                            'Timestamp': row['Timestamp'],
                            'Trade': 'Entry Short',
                            'Price_Spread': row['Spread'],
                            'Z-Score': row['Z-Score']
                        })
                    elif row['Position_Change'] == 1:  # Exit Short (if from -1 to 0)
                        trades.append({
                            'Timestamp': row['Timestamp'],
                            'Trade': 'Exit Short',
                            'Price_Spread': row['Spread'],
                            'Z-Score': row['Z-Score']
                        })

                # Simplify logic to cover short and long trades:
                # Actually, better to check precise change values:

                def get_trade_type(change):
                    if change == 1:
                        return 'Entry Long'  # Position went from 0 to 1
                    elif change == -1:
                        return 'Exit Long'   # Position went from 1 to 0
                    elif change == -2:
                        return 'Entry Short' # Position went from 0 to -1 (or from 1 to -1)
                    elif change == 2:
                        return 'Exit Short'  # Position went from -1 to 0 (or -1 to 1)
                    else:
                        return None

                trades = []
                prev_pos = 0
                for i, row in df.iterrows():
                    change = row['Position'] - prev_pos
                    trade_type = get_trade_type(change)
                    if trade_type:
                        trades.append({
                            'Timestamp': row['Timestamp'],
                            'Trade': trade_type,
                            'Price_Spread': row['Spread'],
                            'Z-Score': row['Z-Score']
                        })
                    prev_pos = row['Position']

                trade_log = pd.DataFrame(trades)

                st.subheader("Trade Log")
                st.dataframe(trade_log)


                # Download data option
                csv = df.to_csv(index=False)
                st.download_button("Download Analysis Data as CSV", csv, "pair_trading_analysis.csv")

            except Exception as e:
                st.error(f"Error fetching data or performing analysis: {e}")
