# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:53:34 2024

Pulling Yahoo Finance Data

@author: anany
"""

import yfinance as yf
import pandas as pd
import os
import numpy as np

# Set working directory
os.chdir("C:\\Users\\anany\\OneDrive\\Documents\\Manchester\\Books\\Dissertation\\Codes")

# Function to calculate volatility at 30-day intervals without overlap
def calculate_interval_volatility(data, interval=30):
    volatilities = []
    for i in range(0, len(data) - interval + 1, interval):
        chunk = data['Daily Return'].iloc[i:i+interval]
        volatility = chunk.std()
        volatilities.append(volatility)
    return volatilities

# Define the ticker symbol
tickers = ['AAPL', 'MSFT', 'GOOGL']
# Get data for the tickers
for ticker in tickers:    
    # Get the historical prices for this ticker
    data = yf.download(ticker,period='max')
    # Convert Date from row index to column
    data.index.name = 'Date'
    data.reset_index(inplace=True)
    # Ensure the data is sorted by date
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    # Calculate daily returns
    data['Daily Return'] = data['Adj Close'].pct_change()
        
    # Save the data to a CSV file
    data.to_csv(f'{ticker}_historical_data.csv')
    
    
# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})



#%% Barclays Data Pull
###-------------For BCS	Barclays PLC Common Stock
ticker = 'BCS'

# Get the historical prices for this ticker
data = yf.download(ticker,period='max')
# Convert Date from row index to column
data.index.name = 'Date'
data.reset_index(inplace=True)
# Ensure the data is sorted by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
    
# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
        
# Save the data to a CSV file
data.to_csv(f'{ticker}_historical_data.csv')
    
data.head()

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})

#%% Random Foreest

# Define function to create features
def create_features(data, L):
    features = pd.DataFrame()
    for i in range(L):
        features[f'Volatility_t-{i+1}'] = data['Volatility'].shift(i+1)
    features['Volatility_t'] = data['Volatility']
    features = features.dropna()
    return features

# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())
 
# Train Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())


#%% Telefónica Data Pull
###-------------For BCS	Barclays PLC Common Stock
ticker = 'TEF'

# Get the historical prices for this ticker
data = yf.download(ticker,period='max')
# Convert Date from row index to column
data.index.name = 'Date'
data.reset_index(inplace=True)
# Ensure the data is sorted by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
    
# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
        
# Save the data to a CSV file
data.to_csv(f'{ticker}_historical_data.csv')
    
data.head()

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})


###--------------Random forest
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

#%% Verizon Data Pull
###-------------For BCS	Barclays PLC Common Stock
ticker = 'VZ'

# Get the historical prices for this ticker
data = yf.download(ticker,period='max')
# Convert Date from row index to column
data.index.name = 'Date'
data.reset_index(inplace=True)
# Ensure the data is sorted by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
    
# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
        
# Save the data to a CSV file
data.to_csv(f'{ticker}_historical_data.csv')
    
data.head()

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})


###--------------Random forest
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())


#%%
'''
tickerDf = yf.download('AAPL', period='max')

tickerDf.head(5)
## Use adjusted close

## Calculate volatility
# Calculate daily returns
tickerDf['Daily Return'] = tickerDf['Adj Close'].pct_change()
'''

'''
###----------------------------------------
# Ensure the data is sorted by date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()

# Calculate rolling volatilities for 30 days
data['Volatility_30'] = data['Daily Return'].rolling(window=30).std()
# Calculate rolling volatilities for 30 days
data['Volatility_60'] = data['Daily Return'].rolling(window=60).std()
# Calculate rolling volatilities for 30 days
data['Volatility_120'] = data['Daily Return'].rolling(window=120).std()
data = data.dropna()  # Drop rows with NaN values

# Display the first few rows with calculated volatility
print(data.head())
'''