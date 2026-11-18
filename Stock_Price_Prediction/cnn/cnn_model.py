
# ============================================
# STOCK PRICE PREDICTION USING CNN
# ============================================

# STEP 1: Import Required Libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Input
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================
# STEP 2: Define File Paths
# ============================================

DATA_PATH = "data/processed_data.npz"

TARGET_SCALER_PATH = "data/target_scaler.pkl"

OUTPUT_DIR = "cnn"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nCNN STOCK PRICE PREDICTION")
print("=" * 50)


# ============================================
# STEP 3: Load Preprocessed Dataset
# ============================================

data = np.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("\nDataset loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================
# STEP 4: Load Target Scaler
# ============================================

target_scaler = joblib.load(TARGET_SCALER_PATH)

print("\nTarget scaler loaded successfully!")


# ============================================
# STEP 5: Build CNN Architecture
# ============================================

model = Sequential([

    # Input Layer
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    # First Convolution Layer
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    # First Pooling Layer
    MaxPooling1D(pool_size=2),

    # Second Convolution Layer
    Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu"
    ),

    # Flatten Layer
    Flatten(),

    # Fully Connected Layer
    Dense(64, activation="relu"),

    # Dropout Layer
    Dropout(0.2),

    # Output Layer
    Dense(1, activation="linear")

])

print("\nCNN Architecture:")
model.summary()


# ============================================
# STEP 6: Compile CNN Model
# ============================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

print("\nCNN model compiled successfully!")


# ============================================
# STEP 7: Train CNN Model
# ============================================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    verbose=1
)

print("\nCNN model training completed!")


# ============================================
# STEP 8: Evaluate CNN Model
# ============================================

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nScaled Test Performance")
print("Test Loss:", test_loss)
print("Test MAE:", test_mae)


# ============================================
# STEP 9: Generate Predictions
# ============================================

y_pred = model.predict(X_test)

# Convert scaled values into original prices

y_test_actual = target_scaler.inverse_transform(y_test)

y_pred_actual = target_scaler.inverse_transform(y_pred)

print("\nActual Prices:")
print(y_test_actual[:5].flatten())

print("\nPredicted Prices:")
print(y_pred_actual[:5].flatten())


# ============================================
# STEP 10: Calculate Performance Metrics
# ============================================

mae = mean_absolute_error(
    y_test_actual,
    y_pred_actual
)

mse = mean_squared_error(
    y_test_actual,
    y_pred_actual
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test_actual,
    y_pred_actual
)

print("\nCNN MODEL PERFORMANCE")
print("=" * 40)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================
# STEP 11: Save Performance Metrics
# ============================================

metrics_df = pd.DataFrame({
    "Model": ["CNN"],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2_Score": [r2]
})

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "cnn_metrics.csv"),
    index=False
)

print("\nCNN metrics saved successfully!")


# ============================================
# STEP 12: Save Actual vs Predicted Prices
# ============================================

predictions_df = pd.DataFrame({
    "Actual_Price": y_test_actual.flatten(),
    "Predicted_Price": y_pred_actual.flatten()
})

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "cnn_predictions.csv"),
    index=False
)

print("CNN predictions saved successfully!")


# ============================================
# STEP 13: Visualize Actual vs Predicted Prices
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test_actual,
    label="Actual Prices"
)

plt.plot(
    y_pred_actual,
    label="CNN Predicted Prices"
)

plt.title("CNN Stock Price Prediction")

plt.xlabel("Days")

plt.ylabel("Stock Closing Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cnn_predictions.png")
)

plt.show()


# ============================================
# STEP 14: Visualize Training and Validation Loss
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")

plt.xlabel("Epochs")

plt.ylabel("Mean Squared Error")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cnn_training_loss.png")
)

plt.show()


# ============================================
# STEP 15: Save Trained CNN Model
# ============================================

model.save(
    os.path.join(OUTPUT_DIR, "cnn_model.keras")
)

print("\nCNN model saved successfully!")


# ============================================
# FINAL OUTPUT
# ============================================

print("\n" + "=" * 50)
print("CNN EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("\nGenerated Files:")

print("1. cnn/cnn_model.keras")
print("2. cnn/cnn_metrics.csv")
print("3. cnn/cnn_predictions.csv")
print("4. cnn/cnn_predictions.png")
print("5. cnn/cnn_training_loss.png")
