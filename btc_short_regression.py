import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_date_price(dates, prices, title="BTC Price Over Time", xlabel="Date", ylabel="BTC Price"):

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(dates, prices, color='orange', linewidth=1, label='BTC Price')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Grid on y-axis added
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.7, color='gray')
    ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray', alpha=0.9)

    # Grid on x-axis added
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=0.7, color='gray')
    ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray', alpha=0.9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax.set_yscale("log")

    # Linear Regression
    dates_num = mdates.date2num(dates)
    dates_num = dates_num.reshape(-1, 1)

    model = LinearRegression()
    model.fit(dates_num, np.log(prices))

    predicted_prices = np.exp(model.predict(dates_num))
    plt.plot(dates, predicted_prices, color='blue', linestyle='--', label='Linear Regression')

    # Calculate and plot standard deviation lines
    log_prices = np.log(prices)
    residuals = log_prices - model.predict(dates_num)
    std_dev = np.std(residuals)

    for i in range(-2, 0):
        plt.plot(dates, np.exp(model.predict(dates_num) + i * std_dev),
                 linestyle='-', color='darkred', alpha=0.5,
                 label=f'{i} STD')

    for i in range(0, 3):
        plt.plot(dates, np.exp(model.predict(dates_num) + i * std_dev),
                 linestyle='-', color='green', alpha=0.5,
                 label=f'{i} STD')

    # Extend the prediction to April 2026
    # Use .iloc to access the last element by position, and remove timezone information
    future_dates = pd.date_range(start=dates.iloc[-1].tz_localize(None), end=pd.to_datetime('2026-04-30').tz_localize(None), freq='D')
    future_dates_num = mdates.date2num(future_dates)
    future_dates_num = future_dates_num.reshape(-1, 1)

    future_predicted_prices = np.exp(model.predict(future_dates_num))
    plt.plot(future_dates, future_predicted_prices, color='blue', linestyle='--', label='Future Prediction')

    # Extend the standard deviation lines
    for i in range(-2, 3):
        plt.plot(future_dates, np.exp(model.predict(future_dates_num) + i * std_dev),
                 linestyle='--', color='purple', alpha=0.5)  # No need to add labels again

    # Add vertical lines for specific dates
    important_dates = ['2025-10-20', '2025-12-20', '2026-03-20']
    for date in important_dates:
        ax.axvline(pd.to_datetime(date), color='red', linestyle='-', linewidth=1.5)

    # Format x-axis to show year and month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Set minor locator to show months
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Place a tick for each month

    # Add horizontal line at 150,000 for important level
    ax.axhline(y=150000, color='black', linestyle='--', label='150,000 Price Level')

    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.savefig("btc_price_2022_log_regression_std_future.png", dpi=300)
    plt.show()

df = pd.read_csv(r'c:\Users\Julia\_PW_PYTH_Proj\btc_regression\content\btc-usd-max.csv', parse_dates=['snapped_at'])

# Filter data from November 1, 2022, onwards
# Make start_date timezone-aware by setting it to UTC
# or the same timezone as your 'snapped_at' column.
start_date = pd.to_datetime('2022-11-01').tz_localize('UTC')
filtered_df = df[df['snapped_at'] >= start_date]

# Extract dates and prices from the filtered data
dates = filtered_df['snapped_at']
prices = filtered_df['price']

# Call the plotting function with the filtered data
plot_date_price(dates, prices, title="BTC Price Over Time (from Nov 2022)", xlabel="Date", ylabel="BTC Price")