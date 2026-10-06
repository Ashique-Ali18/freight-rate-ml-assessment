import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/train-test.csv")
df["date"] = pd.to_datetime(df["date"])

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

all_features = numeric_features + categorical_features


# ============================================================
# MODEL
# ============================================================

def make_pipeline():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
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

    model = GradientBoostingRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=3,
        subsample=0.8,
        loss="huber",
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# ============================================================
# ROLLING TIME SPLITS
# ============================================================

splits = [
    ("2025-01-01", "2025-07-01", "2025-07-01", "2025-08-01"),
    ("2025-01-01", "2025-08-01", "2025-08-01", "2025-09-01"),
    ("2025-01-01", "2025-09-01", "2025-09-01", "2025-10-01"),
    ("2025-01-01", "2025-10-01", "2025-10-01", "2025-11-01"),
]


results = []


# ============================================================
# RUN VALIDATION
# ============================================================

for i, (
    train_start,
    train_end,
    valid_start,
    valid_end
) in enumerate(splits, start=1):

    print("\n" + "=" * 90)
    print(f"ROLLING SPLIT {i}")
    print(
        f"Train: {train_start} -> {train_end}"
    )
    print(
        f"Valid: {valid_start} -> {valid_end}"
    )
    print("=" * 90)

    train = df[
        (df["date"] >= train_start)
        &
        (df["date"] < train_end)
    ].copy()

    valid = df[
        (df["date"] >= valid_start)
        &
        (df["date"] < valid_end)
    ].copy()

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(valid):,}"
    )

    pipeline = make_pipeline()

    pipeline.fit(
        train[all_features],
        train["posted_rate"]
    )

    predictions = pipeline.predict(
        valid[all_features]
    )

    predictions = np.maximum(
        predictions,
        1
    )

    mae = mean_absolute_error(
        valid["posted_rate"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            valid["posted_rate"],
            predictions
        )
    )

    r2 = r2_score(
        valid["posted_rate"],
        predictions
    )

    results.append({
        "validation_period": f"{valid_start} -> {valid_end}",
        "train_rows": len(train),
        "valid_rows": len(valid),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\nMAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("ROLLING VALIDATION SUMMARY")
print("=" * 100)

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

print("\nAverage MAE:")
print(
    f"${results_df['MAE'].mean():,.2f}"
)

print("\nAverage RMSE:")
print(
    f"${results_df['RMSE'].mean():,.2f}"
)

print("\nAverage R²:")
print(
    f"{results_df['R2'].mean():.4f}"
)