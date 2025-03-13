#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import subprocess
import time  # Import the time module for time-related functions
import argparse

def main(stock_symbol):
    # Your main logic goes here
    print(f"Fetching and processing data for {stock_symbol}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock prediction script.")
    parser.add_argument('stock_symbol', type=str, help="Stock symbol to predict")
    args = parser.parse_args()

    main(args.stock_symbol)

# Define file paths
DATA_FETCHER_PATH = "../data-fetcher/stock_data.json"
FETCHER_LOG_PATH = "../data-fetcher/fetcher.log"

# Fetch the latest stock data using the C fetcher program
def fetch_stock_data():
    try:
        print("Fetching live stock data...")
        subprocess.run(["./data-fetcher/fetcher"], check=True)
        print("Stock data fetched successfully.")
    except subprocess.CalledProcessError:
        print("Error occurred while fetching stock data.")
        exit(1)

# Check if the stock data file exists and is updated
def is_data_updated():
    if not os.path.exists(DATA_FETCHER_PATH):
        print(f"Error: {DATA_FETCHER_PATH} not found!")
        return False

    # Check the last modification time of stock data
    last_modified_time = os.path.getmtime(DATA_FETCHER_PATH)
    current_time = time.time()  # Use time.time() instead of os.time()

    # Consider the data as outdated if it hasn't been updated for more than a day (24 hours)
    if current_time - last_modified_time > 86400:
        print("Stock data is outdated, updating...")
        return False
    return True

# Check if data is updated; if not, fetch new data
if not is_data_updated():
    fetch_stock_data()

# Load stock data
with open(DATA_FETCHER_PATH, "r") as file:
    data = json.load(file)

# Extract stock data
try:
    quote = data["Global Quote"]
    stock_symbol = quote["01. symbol"]
    stock_price = float(quote["05. price"])
except KeyError:
    print("Error: Invalid data format in stock_data.json")
    exit()

# Simulate past stock prices (for demo, we generate synthetic data)
np.random.seed(42)
days = np.array(range(1, 101)).reshape(-1, 1)  # Days 1 to 100
prices = stock_price + np.random.normal(0, 5, size=(100,))  # Simulated prices

# Convert to DataFrame
df = pd.DataFrame({"Day": days.flatten(), "Price": prices})

# Train a simple Linear Regression model
X = df[["Day"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict future stock price for day 101
future_day = np.array([[101]])
future_day_df = pd.DataFrame(future_day, columns=["Day"])
predicted_price = model.predict(future_day_df)[0]

print(f"\nStock Symbol: {stock_symbol}")
print(f"Latest Price: ${stock_price:.2f}")
print(f"Predicted Price for Day 101: ${predicted_price:.2f}\n")

# Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(df["Day"], df["Price"], label="Actual Prices", color="blue")
plt.plot(df["Day"], model.predict(X), label="Predicted Prices", color="red", linestyle="dashed")
plt.axvline(x=101, color="green", linestyle="dotted", label="Prediction for Day 101")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title(f"Stock Price Prediction for {stock_symbol}")
plt.legend()
plt.savefig("stock_prediction.png")  # Saves the plot as an image
print("Prediction chart saved as stock_prediction.png")


# Save the prediction result to a CSV file
result_data = {
    'Date': pd.to_datetime('today').strftime('%Y-%m-%d'),
    'Stock Symbol': stock_symbol,
    'Latest Price': stock_price,
    'Predicted Price': predicted_price
}

df_results = pd.DataFrame([result_data])
df_results.to_csv("predictions.csv", mode='a', header=False, index=False)
print("Prediction saved to predictions.csv")