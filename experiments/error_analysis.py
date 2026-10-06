import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/train-test.csv",
    parse_dates=["date"]
)

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

df["log_distance"] = np.log1p(df["distance"])

train = df[df["date"] < "2025-10-01"].copy()
valid = df[df["date"] >= "2025-10-01"].copy()

numeric_features = [
    "distance",
    "log_distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

categorical_features = [
    "equipment"
]

features = numeric_features + categorical_features


# --------------------------------------------------
# MODEL
# --------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_features
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            random_state=42
        )
    )
])


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.fit(
    train[features],
    train["posted_rate"]
)

pred = model.predict(valid[features])

pred = np.maximum(pred, 1.0)


# --------------------------------------------------
# ERROR DATAFRAME
# --------------------------------------------------

results = valid[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "posted_rate",
    ]
].copy()

results["predicted_rate"] = pred

results["error"] = (
    results["predicted_rate"]
    - results["posted_rate"]
)

results["absolute_error"] = (
    results["error"].abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    / results["posted_rate"]
    * 100
)


# --------------------------------------------------
# WORST ERRORS
# --------------------------------------------------

print("=" * 80)
print("WORST 20 ABSOLUTE ERRORS")
print("=" * 80)

print(
    results
    .sort_values("absolute_error", ascending=False)
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# ERROR SUMMARY
# --------------------------------------------------

print("\n" + "=" * 80)
print("ERROR QUANTILES")
print("=" * 80)

print(
    results["absolute_error"]
    .quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 1.00]
    )
    .round(2)
)


# --------------------------------------------------
# LARGE ERROR COUNTS
# --------------------------------------------------

print("\n" + "=" * 80)
print("LARGE ERROR COUNTS")
print("=" * 80)

for threshold in [100, 250, 500, 1000, 2000, 5000]:

    count = (
        results["absolute_error"] > threshold
    ).sum()

    print(
        f"Errors > ${threshold:,}: {count:,} "
        f"({count / len(results) * 100:.2f}%)"
    )


# --------------------------------------------------
# ERROR BY DISTANCE
# --------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY DISTANCE")
print("=" * 80)

results["distance_bucket"] = pd.cut(
    results["distance"],
    bins=[0, 500, 1000, 1500, 2000, 2500, 4000]
)

print(
    results
    .groupby("distance_bucket", observed=True)
    ["absolute_error"]
    .agg(["count", "mean", "median"])
    .round(2)
)


# --------------------------------------------------
# ERROR BY EQUIPMENT
# --------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY EQUIPMENT")
print("=" * 80)

print(
    results
    .groupby("equipment")
    ["absolute_error"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)


# --------------------------------------------------
# ERROR BY ACTUAL RATE
# --------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY ACTUAL POSTED RATE")
print("=" * 80)

results["rate_bucket"] = pd.cut(
    results["posted_rate"],
    bins=[0, 1000, 2000, 3000, 5000, 10000, 30000]
)

print(
    results
    .groupby("rate_bucket", observed=True)
    ["absolute_error"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)