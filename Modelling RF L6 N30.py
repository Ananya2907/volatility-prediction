# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:02:04 2024

@author: anany
"""

#%% Package import
import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.chdir("C:/Users/anany/OneDrive/Documents/Manchester/Books/Dissertation/Codes")

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

#%% Error matrices

valid_error_RF6 = pd.DataFrame(columns=['L','Company','MSE','RMSE','R2','MAPE'], index=range(1,8,1))
test_error_RF6 = pd.DataFrame(columns=['L','Company','MSE','RMSE','R2','MAPE'], index=range(1,8,1))

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
#%%% Ford - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[1] = [6,'F', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[1] = [6,'F', mse, rmse, r2, mape]


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
#%%% Toyota - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[2] = [6,'TM', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[2] = [6,'TM', mse, rmse, r2, mape]


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
#%%% Amazon - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[3] = [6,'AMZN', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[3] = [6,'AMZN', mse, rmse, r2, mape]


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
#%%% Walmart - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[4] = [6,'WMT', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[4] = [6,'WMT', mse, rmse, r2, mape]


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
#%%% Telefónica - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[5] = [6,'TEF', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[5] = [6,'TEF', mse, rmse, r2, mape]


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
#%%% Verizon - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[6] = [6,'VZ', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[6] = [6,'VZ', mse, rmse, r2, mape]


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
#%%% Barclays - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[7] = [6,'BCS', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[7] = [6,'BCS', mse, rmse, r2, mape]


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
#%%% Bank of America - Random Forest - L = 6
# Define the length L for previous volatilities
L = 6  # Change this to test different lengths
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

# Train a Random Forest model (you can replace this with other models)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Make predictions
valid_pred = model.predict(X_valid)

# Display the first few predictions
results = pd.DataFrame({'Actual': y_valid, 'Predicted': valid_pred})
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
valid_error_RF6.loc[8] = [6,'BAC', mse, rmse, r2, mape]
#%%%% Test predictions
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

# Add to error matrix
test_error_RF6.loc[8] = [6,'BAC', mse, rmse, r2, mape]

#%% Print error matrices

print(valid_error_RF6)
print(test_error_RF6)