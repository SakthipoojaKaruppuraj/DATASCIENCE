
# ==================================================
# STOCK PRICE PREDICTION - MODEL COMPARISON
# ANN vs CNN vs RNN
# ==================================================

# STEP 1: Import Required Libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# STEP 2: Define File Paths
# ==================================================

ANN_METRICS = "ann/ann_metrics.csv"
CNN_METRICS = "cnn/cnn_metrics.csv"
RNN_METRICS = "rnn/rnn_metrics.csv"

ANN_PREDICTIONS = "ann/ann_predictions.csv"
CNN_PREDICTIONS = "cnn/cnn_predictions.csv"
RNN_PREDICTIONS = "rnn/rnn_predictions.csv"

OUTPUT_DIR = "comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nSTOCK PRICE PREDICTION - MODEL COMPARISON")
print("=" * 60)


# ==================================================
# STEP 3: Load Model Performance Metrics
# ==================================================

ann_df = pd.read_csv(ANN_METRICS)
cnn_df = pd.read_csv(CNN_METRICS)
rnn_df = pd.read_csv(RNN_METRICS)

# Combine all model metrics

metrics_df = pd.concat(
    [ann_df, cnn_df, rnn_df],
    ignore_index=True
)

# Standardize column names

metrics_df.columns = [
    col.strip().lower().replace(" ", "_")
    for col in metrics_df.columns
]

# Handle different possible R2 column names

if "r2_score" not in metrics_df.columns:
    for col in metrics_df.columns:
        if col in ["r2", "r2score"]:
            metrics_df.rename(
                columns={col: "r2_score"},
                inplace=True
            )
            break

required_columns = [
    "model",
    "mae",
    "mse",
    "rmse",
    "r2_score"
]

missing_columns = [
    col for col in required_columns
    if col not in metrics_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing metric columns: {missing_columns}"
    )

metrics_df = metrics_df[required_columns]

print("\nAll model metrics loaded successfully!")


# ==================================================
# STEP 4: Display Model Performance Comparison
# ==================================================

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(metrics_df.to_string(index=False))


# ==================================================
# STEP 5: Identify Best Performing Models
# ==================================================

best_mae = metrics_df.loc[
    metrics_df["mae"].idxmin()
]

best_mse = metrics_df.loc[
    metrics_df["mse"].idxmin()
]

best_rmse = metrics_df.loc[
    metrics_df["rmse"].idxmin()
]

best_r2 = metrics_df.loc[
    metrics_df["r2_score"].idxmax()
]

print("\nBEST MODEL BY EACH METRIC")
print("=" * 60)

print("Best MAE  :", best_mae["model"])
print("Best MSE  :", best_mse["model"])
print("Best RMSE :", best_rmse["model"])
print("Best R2   :", best_r2["model"])


# ==================================================
# STEP 6: Save Combined Metrics
# ==================================================

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison_metrics.csv"
    ),
    index=False
)

print("\nCombined metrics saved successfully!")


# ==================================================
# STEP 7: Visualize Performance Metrics
# ==================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 9)
)

metric_columns = [
    ("mae", "MAE Comparison"),
    ("mse", "MSE Comparison"),
    ("rmse", "RMSE Comparison"),
    ("r2_score", "R2 Score Comparison")
]

for ax, (column, title) in zip(
    axes.flatten(),
    metric_columns
):

    ax.bar(
        metrics_df["model"],
        metrics_df[column]
    )

    ax.set_title(title)

    ax.set_xlabel("Models")

    ax.set_ylabel(column.upper())

    ax.grid(
        axis="y",
        alpha=0.3
    )

plt.suptitle(
    "ANN vs CNN vs RNN - Performance Comparison",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_metrics_comparison.png"
    )
)

plt.show()


# ==================================================
# STEP 8: Load Individual Model Predictions
# ==================================================

ann_predictions = pd.read_csv(ANN_PREDICTIONS)
cnn_predictions = pd.read_csv(CNN_PREDICTIONS)
rnn_predictions = pd.read_csv(RNN_PREDICTIONS)


# ==================================================
# STEP 9: Validate Prediction Data
# ==================================================

prediction_files = {
    "ANN": ann_predictions,
    "CNN": cnn_predictions,
    "RNN": rnn_predictions
}

for model_name, df in prediction_files.items():

    if not all(
        col in df.columns
        for col in ["Actual_Price", "Predicted_Price"]
    ):

        raise ValueError(
            f"Invalid prediction columns in {model_name} file"
        )

    if df.empty:
        raise ValueError(
            f"{model_name} predictions file is empty"
        )

    if df.isnull().any().any():
        raise ValueError(
            f"Missing values found in {model_name} predictions"
        )

# Ensure all models use the same test samples

lengths = [
    len(df) for df in prediction_files.values()
]

if len(set(lengths)) != 1:
    raise ValueError(
        "Prediction files have different numbers of test samples"
    )

actual_prices = ann_predictions["Actual_Price"].to_numpy()

for model_name, df in [
    ("CNN", cnn_predictions),
    ("RNN", rnn_predictions)
]:

    if not np.allclose(
        actual_prices,
        df["Actual_Price"].to_numpy(),
        rtol=1e-5,
        atol=1e-5
    ):

        raise ValueError(
            f"Actual test prices do not match for {model_name}"
        )

print("\nAll prediction files validated successfully!")


# ==================================================
# STEP 10: Compare Actual vs Predicted Prices
# ==================================================

plt.figure(figsize=(14, 7))

plt.plot(
    actual_prices,
    label="Actual Prices",
    linewidth=2
)

plt.plot(
    ann_predictions["Predicted_Price"],
    label="ANN Predictions"
)

plt.plot(
    cnn_predictions["Predicted_Price"],
    label="CNN Predictions"
)

plt.plot(
    rnn_predictions["Predicted_Price"],
    label="RNN Predictions"
)

plt.title(
    "ANN vs CNN vs RNN - Stock Price Prediction"
)

plt.xlabel("Test Days")

plt.ylabel("Stock Closing Price")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "all_model_predictions.png"
    )
)

plt.show()


# ==================================================
# STEP 11: Create Prediction Comparison Dataset
# ==================================================

comparison_predictions = pd.DataFrame({

    "Actual_Price": actual_prices,

    "ANN_Predicted": ann_predictions[
        "Predicted_Price"
    ],

    "CNN_Predicted": cnn_predictions[
        "Predicted_Price"
    ],

    "RNN_Predicted": rnn_predictions[
        "Predicted_Price"
    ]

})

comparison_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_model_predictions.csv"
    ),
    index=False
)

print("\nPrediction comparison saved successfully!")


# ==================================================
# STEP 12: Generate Final Model Ranking
# ==================================================

ranking_df = metrics_df.copy()

# Lower error is better

ranking_df["MAE_Rank"] = ranking_df[
    "mae"
].rank(method="min", ascending=True)

ranking_df["RMSE_Rank"] = ranking_df[
    "rmse"
].rank(method="min", ascending=True)

# Higher R2 is better

ranking_df["R2_Rank"] = ranking_df[
    "r2_score"
].rank(method="min", ascending=False)

# Overall rank based on three metrics

ranking_df["Overall_Rank"] = ranking_df[
    ["MAE_Rank", "RMSE_Rank", "R2_Rank"]
].mean(axis=1).rank(method="min")

ranking_df = ranking_df.sort_values(
    "Overall_Rank"
)

ranking_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_ranking.csv"
    ),
    index=False
)

print("\nFINAL MODEL RANKING")
print("=" * 60)

print(
    ranking_df[
        ["model", "mae", "rmse", "r2_score", "Overall_Rank"]
    ].to_string(index=False)
)

print(
    "\nBest overall model:",
    ranking_df.iloc[0]["model"]
)


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n" + "=" * 60)

print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("\nGenerated Files:")

print("1. comparison/model_comparison_metrics.csv")

print("2. comparison/model_metrics_comparison.png")

print("3. comparison/all_model_predictions.csv")

print("4. comparison/all_model_predictions.png")

print("5. comparison/model_ranking.csv")
