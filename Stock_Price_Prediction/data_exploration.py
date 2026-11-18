# ==========================================
# STOCK PRICE PREDICTION
# DATASET EXPLORATION
# ==========================================

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

files = os.listdir("data")

print("Available Dataset Files:")

for file in files:
    print(file)

df = pd.read_csv("/Users/apple/projects/Datascience/Stock_Price_Prediction/data/HistoricalQuotes.csv")

# First 5 records
print("\nFirst 5 Records:")
print(df.head())

# Last 5 records
print("\nLast 5 Records:")
print(df.tail())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Display original column names
print("Original Columns:")
print(df.columns.tolist())

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())

print("\nDataset Columns:")

for column in df.columns:
    print(column)

# Convert Date column into datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check datatype
print(df['Date'].dtype)

# Count missing dates
print("Missing dates:", df['Date'].isnull().sum())

# Remove rows with invalid dates
df = df.dropna(subset=['Date'])

# Sort dataset by Date
df = df.sort_values(by='Date')

# Reset index
df = df.reset_index(drop=True)

print(df.head())

# Check duplicates
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Reset index
df = df.reset_index(drop=True)

# Missing values in each column
print(df.isnull().sum())

plt.figure(figsize=(14, 6))

plt.plot(df['Date'], df['Close/Last'], label='Closing Price')

plt.title('Apple Stock Closing Price History')
plt.xlabel('Date')
plt.ylabel('Stock Price')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(df['Date'], df['Open'], label='Opening Price')
plt.plot(df['Date'], df['Close/Last'], label='Closing Price')

plt.title('Opening vs Closing Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

plt.plot(df['Date'], df['Volume'])

plt.title('Stock Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')

plt.grid()
plt.tight_layout()
plt.show()

# Select numerical columns
numeric_df = df.select_dtypes(include=np.number)

# Calculate correlation
correlation = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Stock Market Feature Correlation')

plt.tight_layout()
plt.show()

# Save cleaned dataset
df.to_csv('data/cleaned_AAPL.csv', index=False)

print("Cleaned dataset saved successfully!")

