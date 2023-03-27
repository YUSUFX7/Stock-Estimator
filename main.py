import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the datasets
AAPL = pd.read_csv("AAPL.csv")
AMZN = pd.read_csv("AMZN.csv")
TSLA = pd.read_csv("TSLA.csv")
MSFT = pd.read_csv("MSFT.csv")

# Combine the datasets
stocks = pd.concat([AAPL['Close'], AMZN['Close'], TSLA['Close'], MSFT['Close']], axis=1)
stocks.columns = ['AAPL', 'AMZN', 'TSLA', 'MSFT']

# Split the data into training and testing sets
X = stocks.iloc[:, :-1].values
y = stocks.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the model to predict the stock prices for the testing data
y_pred = regressor.predict(X_test)

# Compare the predicted stock prices with the actual stock prices
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)





