# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:24:43 2024

@author: anany
"""
#%% Package import
import yfinance as yf
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

#%% Function to calculate volatility at 30-day intervals without overlap
def calculate_interval_volatility(data, interval=30):
    volatilities = []
    for i in range(0, len(data) - interval + 1, interval):
        chunk = data['Daily Return'].iloc[i:i+interval]
        volatility = chunk.std()
        volatilities.append(volatility)
    return volatilities

#%% Function to create features

def create_features(data, L):
    features = pd.DataFrame()
    for i in range(L):
        features[f'Volatility_t-{i+1}'] = data['Volatility'].shift(i+1)
    features['Volatility_t'] = data['Volatility']
    features = features.dropna()
    return features

#%% Function to calculate MAPE

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#%%1. Ford Motor Company
data = pd.read_csv("F.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Ford - Random Forest - L=5
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

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=L, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)

# Make predictions
y_pred = model.predict(X_test)


## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%2. Toyota Motor Corporation
data = pd.read_csv("TM.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Toyota - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%3. Amazon.com, Inc.
data = pd.read_csv("AMZN.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Amazon - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%4. Walmart Inc.
data = pd.read_csv("WMT.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Walmart - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%5. Telefónica
data = pd.read_csv("TEF.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Telefónica - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%6. Verizon Communications Inc. Common Stock
data = pd.read_csv("VZ.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Verizon - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%7. Barclays PLC Common Stock
data = pd.read_csv("BCS.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Barclays - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')


#%%8. Bank of America Corporation Common Stock
data = pd.read_csv("BAC.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=30)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::30].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Bank of America - Random Forest - L=5
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

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape}%')