# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:53:120 2024

@author: anany
"""


#%% Package import
import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

os.chdir("C:/Users/anany/OneDrive/Documents/Manchester/Books/Dissertation/Codes")

#%% Function to calculate volatility at 120-day intervals without overlap
def calculate_interval_volatility(data, interval=120):
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

#%% Error matrices

valid_error_ANN5 = pd.DataFrame(columns=['L','Company','MSE','RMSE','R2','MAPE'], index=range(1,8,1))
test_error_ANN5 = pd.DataFrame(columns=['L','Company','MSE','RMSE','R2','MAPE'], index=range(1,8,1))

#%%1. Ford Motor Company
data = pd.read_csv("F.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Ford - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[1] = [5,'F', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[1] = [5,'F', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('F (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%% Repeat for other companies

#%%2. Toyota Motor Corporation
data = pd.read_csv("TM.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Toyota - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[2] = [5,'TM', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[2] = [5,'TM', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('TM (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%%3. Amazon.com, Inc.
data = pd.read_csv("AMZN.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Amazon - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[3] = [5,'AMZN', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[3] = [5,'AMZN', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('AMZN (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%%4. Walmart Inc.
data = pd.read_csv("WMT.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Walmart - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[4] = [5,'WMT', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[4] = [5,'WMT', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('WMT (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%%5. Telefónica
data = pd.read_csv("TEF.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Telefónica - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[5] = [5,'TEF', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[5] = [5,'TEF', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('TEF (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%%6. Verizon Communications Inc. Common Stock
data = pd.read_csv("VZ.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Verizon - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[6] = [5,'VZ', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[6] = [5,'VZ', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('VZ (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%%7. Barclays PLC Common Stock
data = pd.read_csv("BCS.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Barclays - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[7] = [5,'BCS', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[7] = [5,'BCS', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('BCS (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%%8. Bank of America Corporation Common Stock
data = pd.read_csv("BAC.csv")

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()
data.head()

# Drop rows with NaN values
data = data.dropna()  
    
# Calculate the volatilities
volatilities = calculate_interval_volatility(data, interval=120)

# Create a DataFrame for the volatilities
vol_df = pd.DataFrame({
    'Interval Start Date': data['Date'].iloc[::120].reset_index(drop=True)[:len(volatilities)],
    'Volatility': volatilities
})
#%%% Bank of America - ANN - L = 5
# Define the length L for previous volatilities
L = 5  # Change this to test different lengths
features = create_features(vol_df, L)

# Split into features and target
X = features.drop(columns=['Volatility_t'])
y = features['Volatility_t']

print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

#%%%% Training and Validation Prediction

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_valid, y_valid), verbose=0)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
print(results.head())

## Error metrics
# Calculate MSE
mse = mean_squared_error(y_valid, valid_pred)
print(f'Mean Squared Error: {mse}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print(f'RMSE: {rmse}')

# Calculate R-squared
r2 = r2_score(y_valid, valid_pred)
print(f'R²: {r2}')

# Calculate MAPE
mape = mean_absolute_percentage_error(y_valid, valid_pred)
print(f'MAPE: {mape}%')

# Add to error matrix
valid_error_ANN5.loc[8] = [5,'BAC', mse, rmse, r2, mape]

#%%%% Test predictions
# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
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

# Add to error matrix
test_error_ANN5.loc[8] = [5,'BAC', mse, rmse, r2, mape]

#%%%% Plotting

valid_results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred.flatten()})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(y_train.index, y_train.values, label='Training Volatility', color='blue')
plt.plot(y_valid.index, y_valid.values, label='Given Validation Volatility', color='cyan')
plt.plot(y_test.index, y_test.values, label='Given Test Volatility', color='magenta')
plt.plot(valid_results.index, valid_results.Predicted, label='Predicted Validation Volatility', color='red')
plt.plot(test_results.index, test_results.Predicted, label='Predicted Test Volatility', color='green')
plt.xlabel('')
plt.ylabel('Volatility')
plt.title('BAC (N=120, L=5) Volatilities')
plt.legend()
plt.show()

#%% Print error matrices

print(valid_error_ANN5)
print(test_error_ANN5)
