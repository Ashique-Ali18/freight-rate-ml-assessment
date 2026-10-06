import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "data/train-test.csv"
DECEMBER_INPUT_PATH = "data/december-chart-inputs.csv"
OUTPUT_PATH = "outputs/december_predictions.csv"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("LOADING DATA")
print("=" * 90)

train = pd.read_csv(TRAIN_PATH)
december = pd.read_csv(DECEMBER_INPUT_PATH)

print(f"Training rows   : {len(train):,}")
print(f"December rows   : {len(december):,}")

print("\nDecember columns:")
print(december.columns.tolist())

# ============================================================
# RECOVER LOCATION COORDINATES FROM TRAINING DATA
# ============================================================

pickup_coords = (
    train.groupby("pickup")[["pickup_lat", "pickup_lon"]]
    .median()
)

delivery_coords = (
    train.groupby("delivery")[["delivery_lat", "delivery_lon"]]
    .median()
)

# Map pickup coordinates
december["pickup_lat"] = december["pickup"].map(
    pickup_coords["pickup_lat"]
)

december["pickup_lon"] = december["pickup"].map(
    pickup_coords["pickup_lon"]
)

# Map delivery coordinates
december["delivery_lat"] = december["delivery"].map(
    delivery_coords["delivery_lat"]
)

december["delivery_lon"] = december["delivery"].map(
    delivery_coords["delivery_lon"]
)

# Verify that all December locations were found
coordinate_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon"
]

if december[coordinate_columns].isna().any().any():
    print("\nERROR: Missing coordinates found:")
    print(
        december[
            december[coordinate_columns].isna().any(axis=1)
        ]
    )
    raise ValueError(
        "Could not recover coordinates for one or more December locations."
    )

print("\nLocation coordinates recovered successfully.")


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# FINAL MODEL
# ============================================================

model = GradientBoostingRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=4,
    min_samples_leaf=3,
    subsample=0.8,
    loss="huber",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# TRAIN ON ALL 48,000 LABELED ROWS
# ============================================================

print("\n" + "=" * 90)
print("TRAINING FINAL MODEL")
print("=" * 90)

pipeline.fit(
    train[FEATURES],
    train["posted_rate"]
)

print("Training complete.")


# ============================================================
# PREDICT DECEMBER
# ============================================================

print("\n" + "=" * 90)
print("GENERATING DECEMBER PREDICTIONS")
print("=" * 90)

predictions = pipeline.predict(
    december[FEATURES]
)

predictions = np.maximum(
    predictions,
    1.0
)


# ============================================================
# CREATE OUTPUT
# ============================================================

december_output = december.copy()

december_output["predicted_rate"] = predictions


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate"
]

assert all(
    col in december_output.columns
    for col in required_columns
)

december_output = december_output[
    required_columns
]


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

december_output.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("DECEMBER PREDICTIONS CREATED")
print("=" * 90)

print(f"Output: {OUTPUT_PATH}")
print(f"Rows  : {len(december_output)}")

print(
    f"\nPrediction min   : "
    f"${december_output['predicted_rate'].min():,.2f}"
)

print(
    f"Prediction median: "
    f"${december_output['predicted_rate'].median():,.2f}"
)

print(
    f"Prediction max   : "
    f"${december_output['predicted_rate'].max():,.2f}"
)

print("\nDecember predictions:")
print(
    december_output.to_string(
        index=False,
        formatters={
            "predicted_rate": "{:.2f}".format
        }
    )
)

print("\nDecember prediction file: PASSED")