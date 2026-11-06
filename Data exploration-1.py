# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:02:27 2024

@author: ananya
"""

import os  #provides a way to interact with the operating system
import pandas as pd #data analysis and manipulation tool
import matplotlib.pyplot as plt #creating static, animated, and interactive visualizations

cwd = os.getcwd()

print(cwd)

os.chdir("C:\\Users\\anany\\OneDrive\\Documents\\Manchester\\Books\\02 Data Analytics for Business Decision Making - BMAN60422\\Group project")


stores_df = pd.read_csv("DA2024_stores.csv")
test_df = pd.read_csv("DA2024_test.csv")
train_df = pd.read_csv("DA2024_train.csv")
print(stores_df.head())
stores_df
stores_df.describe()
stores_df.shape


