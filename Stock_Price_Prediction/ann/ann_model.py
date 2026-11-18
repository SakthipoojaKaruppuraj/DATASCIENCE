
# ============================================================
# STOCK PRICE PREDICTION USING ARTIFICIAL NEURAL NETWORK (ANN)
# ============================================================

# STEP 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import time
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

print("\n======================================")
print(" ANN STOCK PRICE PREDICTION")
print("======================================")

print("TensorFlow Version:", tf.__version__)


# STEP 2: LOAD PROCESSED DATASET

dataset = np.load('data/processed_data.npz')

X_train = dataset['X_train']
X_test = dataset['X_test']

y_train = dataset['y_train']
y_test = dataset['y_test']

# Load target scaler
target_scaler = joblib.load(
    'data/target_scaler.pkl'
)

print("\nDataset Loaded Successfully!")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# STEP 3: FLATTEN INPUT DATA FOR ANN

# Original shape: (samples, 30, 5)
# ANN shape:      (samples, 150)

X_train_ann = X_train.reshape(
    X_train.shape[0],
    -1
)

X_test_ann = X_test.reshape(
    X_test.shape[0],
    -1
)

print("\nANN Input Shapes:")

print("X_train_ann:", X_train_ann.shape)
print("X_test_ann:", X_test_ann.shape)


# STEP 4: BUILD ANN MODEL

model = Sequential([

    Input(shape=(X_train_ann.shape[1],)),

    Dense(
        128,
        activation='relu'
    ),

    Dense(
        64,
        activation='relu'
    ),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        1,
        activation='linear'
    )

])

print("\nANN MODEL SUMMARY")

model.summary()


# STEP 5: COMPILE ANN MODEL

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss='mean_squared_error',

    metrics=['mae']

)

print("\nANN Model Compiled Successfully!")


# STEP 6: CALLBACKS

early_stopping = EarlyStopping(

    monitor='val_loss',

    patience=10,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=5,

    min_lr=0.00001

)


# STEP 7: TRAIN ANN MODEL

print("\n======================================")
print(" ANN MODEL TRAINING STARTED")
print("======================================")

start_time = time.time()

history = model.fit(

    X_train_ann,

    y_train,

    epochs=100,

    batch_size=32,

    validation_split=0.2,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    shuffle=False,

    verbose=1

)

training_time = time.time() - start_time

print("\nANN Training Completed!")

print("Training Time:", round(training_time, 2), "seconds")

print("Epochs Completed:", len(history.history['loss']))


# STEP 8: EVALUATE MODEL

test_loss, test_mae_scaled = model.evaluate(

    X_test_ann,

    y_test,

    verbose=0

)

print("\nANN Scaled Test Results:")

print("Test Loss:", test_loss)
print("Test MAE:", test_mae_scaled)


# STEP 9: GENERATE PREDICTIONS

predictions_scaled = model.predict(

    X_test_ann,

    verbose=0

)

# Convert scaled predictions into actual stock prices

y_pred = target_scaler.inverse_transform(
    predictions_scaled
)

y_actual = target_scaler.inverse_transform(
    y_test
)

print("\nSample Predictions:")

for i in range(5):

    print(
        f"Actual: ${y_actual[i][0]:.2f} | "
        f"Predicted: ${y_pred[i][0]:.2f}"
    )


# STEP 10: CALCULATE PERFORMANCE METRICS

mae = mean_absolute_error(
    y_actual,
    y_pred
)

mse = mean_squared_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(mse)

mape = mean_absolute_percentage_error(
    y_actual,
    y_pred
) * 100


print("\n======================================")
print(" ANN PERFORMANCE RESULTS")
print("======================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")

print("Training Time:", round(training_time, 2), "seconds")


# STEP 11: PLOT TRAINING AND VALIDATION LOSS

plt.figure(figsize=(10, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('ANN Training and Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('MSE Loss')

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    'ann/ann_training_loss.png',
    dpi=300
)

plt.show()


# STEP 12: ACTUAL VS PREDICTED STOCK PRICES

plt.figure(figsize=(14, 6))

plt.plot(
    y_actual.flatten(),
    label='Actual Stock Price',
    linewidth=2
)

plt.plot(
    y_pred.flatten(),
    label='ANN Predicted Price',
    linewidth=2
)

plt.title('ANN - Actual vs Predicted AAPL Stock Prices')

plt.xlabel('Test Samples')
plt.ylabel('Stock Price ($)')

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    'ann/ann_predictions.png',
    dpi=300
)

plt.show()


# STEP 13: SAVE TRAINED MODEL

os.makedirs('ann', exist_ok=True)

model.save(
    'ann/ann_model.keras'
)

print("\nANN Model Saved Successfully!")


# STEP 14: SAVE PREDICTIONS

predictions_df = pd.DataFrame({

    'Actual_Price': y_actual.flatten(),

    'Predicted_Price': y_pred.flatten()

})

predictions_df.to_csv(

    'ann/ann_predictions.csv',

    index=False

)

print("ANN Predictions Saved Successfully!")


# STEP 15: SAVE EVALUATION METRICS

metrics_df = pd.DataFrame({

    'Model': ['ANN'],

    'MAE': [mae],

    'MSE': [mse],

    'RMSE': [rmse],

    'MAPE': [mape],

    'Training_Time': [training_time]

})

metrics_df.to_csv(

    'ann/ann_metrics.csv',

    index=False

)

print("ANN Metrics Saved Successfully!")


# STEP 16: FINAL SUMMARY

print("\n======================================")
print(" ANN PROJECT SUMMARY")
print("======================================")

print("Architecture: Artificial Neural Network")

print("Hidden Layers: 128 -> 64 -> 32")

print("Optimizer: Adam")

print("Loss Function: Mean Squared Error")

print("Lookback Window: 30 Days")

print("Input Features: 5")

print("Output: Next-Day Closing Price")

print("--------------------------------------")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")

print("--------------------------------------")

print("ANN MODEL EXECUTION COMPLETED!")

print("======================================")
