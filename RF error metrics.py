# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:12:05 2024

@author: anany
"""


import matplotlib.dates as mdates
#Plotting vol_df
# Plotting volatilities vs Interval start date
plt.figure(figsize=(10, 6))
plt.plot(vol_df['Interval Start Date'], vol_df['Volatility'], linestyle='-', color='b')
plt.title('Volatility vs Interval Start Date')
plt.xlabel('Interval Start Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.xticks(rotation=90)
# Set x-axis to show only yearly dates
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()

#%% PLotting error matrix

#N=30
valid_error_RF_N30 = pd.concat([valid_error_RF3, valid_error_RF4, valid_error_RF5, valid_error_RF6, valid_error_RF7], ignore_index = True)
valid_error_RF_N30

test_error_RF_N30 = pd.concat([test_error_RF3, test_error_RF4, test_error_RF5, test_error_RF6, test_error_RF7], ignore_index = True)
test_error_RF_N30

#N=60
valid_error_RF_N60 = pd.concat([valid_error_RF3, valid_error_RF4, valid_error_RF5, valid_error_RF6, valid_error_RF7], ignore_index = True)
valid_error_RF_N60

test_error_RF_N60 = pd.concat([test_error_RF3, test_error_RF4, test_error_RF5, test_error_RF6, test_error_RF7], ignore_index = True)
test_error_RF_N60

#N=120
valid_error_RF_N120 = pd.concat([valid_error_RF3, valid_error_RF4, valid_error_RF5, valid_error_RF6, valid_error_RF7], ignore_index = True)
valid_error_RF_N120

test_error_RF_N120 = pd.concat([test_error_RF3, test_error_RF4, test_error_RF5, test_error_RF6, test_error_RF7], ignore_index = True)
test_error_RF_N120


# Plotting N30
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in test_error_RF_N30.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Test (N=30) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in test_error_RF_N30.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Test (N=30) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting N30
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in valid_error_RF_N30.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=30) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in valid_error_RF_N30.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=30) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()


# Plotting N60
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in test_error_RF_N60.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Test (N=60) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in test_error_RF_N60.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Test (N=60) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting N60
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in valid_error_RF_N60.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=60) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in valid_error_RF_N60.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=60) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()



# Plotting N120
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in test_error_RF_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Test (N=120) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in test_error_RF_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Test (N=120) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting N120
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in valid_error_RF_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=120) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in valid_error_RF_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=120) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

