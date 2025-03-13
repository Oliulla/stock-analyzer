import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define file paths
DATA_FETCHER_PATH = "../data-fetcher/stock_data.json"

# Load stock data
if not os.path.exists(DATA_FETCHER_PATH):
    print(f"Error: {DATA_FETCHER_PATH} not found!")
    exit()

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

