import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_date_price(dates, prices, title, xlabel, ylabel):

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

    for i in range(1, 3):
        plt.plot(dates, np.exp(model.predict(dates_num) + i * std_dev),
                 linestyle='-', color='green', alpha=0.5,
                 label=f'{i} STD')

    # Extend the prediction to April 2026
    # Use .iloc to access the last element by position, and remove timezone information
    future_dates = pd.date_range(start=dates.iloc[-1].tz_localize(None), end=pd.to_datetime('2026-04-30').tz_localize(None), freq='D')
    future_dates_num = mdates.date2num(future_dates)

    print(future_dates_num)

    future_dates_num = future_dates_num.reshape(-1, 1)

    print(future_dates_num)

    future_predicted_prices = np.exp(model.predict(future_dates_num))
    plt.plot(future_dates, future_predicted_prices, color='blue', linestyle='--', label='Future Prediction')

    # Extend the standard deviation lines
    for i in range(-2, 3):
        plt.plot(future_dates, np.exp(model.predict(future_dates_num) + i * std_dev),
                 linestyle='--', color='purple', alpha=0.5)  # No need to add labels again

    # Add vertical lines for specific dates
    important_dates = ['2025-10-20', '2025-12-20', '2026-03-20']
    for date in important_dates:
        ax.axvline(pd.to_datetime(date), color='darkblue', linestyle='-', linewidth=0.8)

    # Format x-axis to show year and month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Set minor locator to show months
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Place a tick for each month

    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.savefig("btc_price_log_regression_forecast.png", dpi=300)
    plt.show()

df = pd.read_csv(r'c:\Users\Julia\_PW_PYTH_Proj\btc_regression\content\btc-usd-max.csv', parse_dates=['snapped_at'])
dates = df['snapped_at']
prices = df['price']

plot_date_price(dates, prices, title="BTC Price Over Time", xlabel="Date", ylabel="BTC Price")