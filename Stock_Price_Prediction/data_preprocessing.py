
# ==========================================
# STOCK PRICE PREDICTION
# COMMON DATA PREPROCESSING
# ==========================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------

df = pd.read_csv("data/cleaned_AAPL.csv")

print("Dataset loaded successfully!")

print("\nOriginal Columns:")
print(df.columns.tolist())


# ------------------------------------------
# STEP 2: STANDARDIZE COLUMN NAMES
# ------------------------------------------

df.columns = df.columns.str.strip()

df = df.rename(columns={
    'Close/Last': 'Close'
})

print("\nStandardized Columns:")
print(df.columns.tolist())


# ------------------------------------------
# STEP 3: CONVERT CURRENCY VALUES
# ------------------------------------------

features = ['Open', 'High', 'Low', 'Close', 'Volume']

for col in features:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
    )

    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )


# ------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# ------------------------------------------

print("\nMissing Values Before Cleaning:")
print(df[features].isnull().sum())

df = df.dropna(subset=features)

df = df.reset_index(drop=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)


# ------------------------------------------
# STEP 5: SELECT FEATURES
# ------------------------------------------

data = df[features].copy()

print("\nSelected Features:")
print(data.head())


# ------------------------------------------
# STEP 6: DEFINE INPUT AND TARGET
# ------------------------------------------

X = data[features].values

y = data['Close'].values


# ------------------------------------------
# STEP 7: CHRONOLOGICAL TRAIN/TEST SPLIT
# ------------------------------------------

split_index = int(len(data) * 0.8)

X_train_raw = X[:split_index]
X_test_raw = X[split_index:]

y_train_raw = y[:split_index]
y_test_raw = y[split_index:]


# ------------------------------------------
# STEP 8: FEATURE SCALING
# ------------------------------------------

feature_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(
    X_train_raw
)

X_test_scaled = feature_scaler.transform(
    X_test_raw
)

target_scaler = MinMaxScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train_raw.reshape(-1, 1)
)

y_test_scaled = target_scaler.transform(
    y_test_raw.reshape(-1, 1)
)


# ------------------------------------------
# STEP 9: CREATE SEQUENCES
# ------------------------------------------

LOOKBACK = 30


def create_sequences(X, y, lookback):

    X_sequences = []
    y_sequences = []

    for i in range(lookback, len(X)):

        X_sequences.append(
            X[i-lookback:i]
        )

        y_sequences.append(
            y[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# Training sequences

X_train, y_train = create_sequences(
    X_train_scaled,
    y_train_scaled,
    LOOKBACK
)


# Testing sequences
# Use last 30 training days as context

X_test_context = np.concatenate(
    [
        X_train_scaled[-LOOKBACK:],
        X_test_scaled
    ],
    axis=0
)

y_test_context = np.concatenate(
    [
        y_train_scaled[-LOOKBACK:],
        y_test_scaled
    ],
    axis=0
)

X_test, y_test = create_sequences(
    X_test_context,
    y_test_context,
    LOOKBACK
)


# ------------------------------------------
# STEP 10: DISPLAY FINAL SHAPES
# ------------------------------------------

print("\n========== FINAL DATASET SHAPES ==========")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nPreprocessing completed successfully!")

# ==========================================
# STEP 11: SAVE PROCESSED DATA
# ==========================================

import joblib

# Save training and testing arrays
np.savez_compressed(
    'data/processed_data.npz',
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# Save fitted scalers
joblib.dump(
    feature_scaler,
    'data/feature_scaler.pkl'
)

joblib.dump(
    target_scaler,
    'data/target_scaler.pkl'
)

print("\nProcessed data and scalers saved successfully!")
