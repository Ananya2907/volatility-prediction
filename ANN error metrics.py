# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:10:06 2024

@author: anany
"""


import pandas as pd
import matplotlib.pyplot as plt


#%% PLotting error matrix

#N=30
valid_error_ANN_N30 = pd.concat([valid_error_ANN3, valid_error_ANN4, valid_error_ANN5, valid_error_ANN6, valid_error_ANN7], ignore_index = True)
valid_error_ANN_N30

test_error_ANN_N30 = pd.concat([test_error_ANN3, test_error_ANN4, test_error_ANN5, test_error_ANN6, test_error_ANN7], ignore_index = True)
test_error_ANN_N30

#N=60
valid_error_ANN_N60 = pd.concat([valid_error_ANN3, valid_error_ANN4, valid_error_ANN5, valid_error_ANN6, valid_error_ANN7], ignore_index = True)
valid_error_ANN_N60

test_error_ANN_N60 = pd.concat([test_error_ANN3, test_error_ANN4, test_error_ANN5, test_error_ANN6, test_error_ANN7], ignore_index = True)
test_error_ANN_N60

#N=120
valid_error_ANN_N120 = pd.concat([valid_error_ANN3, valid_error_ANN4, valid_error_ANN5, valid_error_ANN6, valid_error_ANN7], ignore_index = True)
valid_error_ANN_N120

test_error_ANN_N120 = pd.concat([test_error_ANN3, test_error_ANN4, test_error_ANN5, test_error_ANN6, test_error_ANN7], ignore_index = True)
test_error_ANN_N120

# Plotting N30
plt.figure(figsize=(14, 7))

# RMSE plot
plt.subplot(2, 1, 1)
for key, grp in test_error_ANN_N30.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Test (N=30) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in test_error_ANN_N30.groupby(['Company']):
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
for key, grp in valid_error_ANN_N30.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=30) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in valid_error_ANN_N30.groupby(['Company']):
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
for key, grp in test_error_ANN_N60.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Test (N=60) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in test_error_ANN_N60.groupby(['Company']):
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
for key, grp in valid_error_ANN_N60.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=60) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in valid_error_ANN_N60.groupby(['Company']):
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
for key, grp in test_error_ANN_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Test (N=120) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in test_error_ANN_N120.groupby(['Company']):
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
for key, grp in valid_error_ANN_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['RMSE'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=120) : RMSE by Company and L')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# R2 plot
plt.subplot(2, 1, 2)
for key, grp in valid_error_ANN_N120.groupby(['Company']):
    plt.plot(grp['L'], grp['R2'], marker='o', linestyle='-', label=key)
plt.title('Validation (N=120) : R2 by Company and L')
plt.xlabel('L')
plt.ylabel('R2')
plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()

