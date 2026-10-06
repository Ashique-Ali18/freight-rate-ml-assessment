import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/train-test.csv")
df["date"] = pd.to_datetime(df["date"])

train = df[df["date"] < "2025-10-01"].copy()

valid = df[
    (df["date"] >= "2025-10-01") &
    (df["date"] < "2025-11-01")
].copy()


# ============================================================
# FEATURES
# ============================================================

numeric_features = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]

all_features = (
    numeric_features +
    categorical_features
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),
            numeric_features
        ),
        (
            "cat",
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
            categorical_features
        )
    ]
)


X_train = train[all_features]
y_train = train["posted_rate"]

X_valid = valid[all_features]
y_valid = valid["posted_rate"]


# ============================================================
# LOSS FUNCTIONS
# ============================================================

losses = [
    "huber",
    "squared_error",
    "absolute_error",
]


results = []


# ============================================================
# TRAIN
# ============================================================

for loss in losses:

    print("\n" + "=" * 90)
    print(f"LOSS: {loss}")
    print("=" * 90)

    model = GradientBoostingRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=3,
        subsample=0.8,
        loss=loss,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_valid
    )

    predictions = np.maximum(
        predictions,
        1
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            predictions
        )
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    results.append({
        "loss": loss,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
)

print("\n" + "=" * 90)
print("LOSS COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)