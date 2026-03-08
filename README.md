# Extract-ETF-Data-50-Year-World-Wide
Extract ETF Data >50 Year World Wide

Main Overview of the Python Script

import pandas as pd
import yfinance as yf
import os

tickers = [
    "^GSPC", "^NDX", "^STOXX", "^GDAXI", "EWG", 
    "MCHI", "EWJ", "INDA", "EWT", "EWY", 
    "EWH", "EWP", "EWZ", "EWW", "PAK"
]

start = "1976-01-01"
end = "2026-02-21"

print("Downloading daily data... This will be a large file.")

# 1. Download data (Daily is the default)
data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

# 2. SKIP RESAMPLING: We jump straight to the transpose
# This keeps every single trading day as a column
df_wide = data.transpose()

# 3. Clean Date headers to 'YYYY-MM-DD'
# We must include the day (%d) so headers are unique
df_wide.columns = df_wide.columns.strftime('%Y-%m-%d')

# 4. Save to CSV
file_name = "etf_historical_DAILY_WIDE.csv"
df_wide.to_csv(file_name)

print("-" * 30)
print(f"Success! Daily file saved at: {os.getcwd()}/{file_name}")
print(f"Total Columns (Days): {len(df_wide.columns)}")
print("Preview (First 5 tickers, first 5 days):")
print(df_wide.iloc[:5, :5])
