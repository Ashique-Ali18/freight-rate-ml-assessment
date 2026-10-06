import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv(
    "data/train-test.csv",
    parse_dates=["date"]
)

# --------------------------------------------------
# DATE FEATURES
# --------------------------------------------------

df["days_since_start"] = (
    df["date"] - pd.Timestamp("2025-01-01")
).dt.days

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["log_distance"] = np.log1p(df["distance"])


# --------------------------------------------------
# TIME SPLIT
# --------------------------------------------------

train = df[df["date"] < "2025-10-01"].copy()
valid = df[df["date"] >= "2025-10-01"].copy()


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

BASE = [
    "distance",
    "log_distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

SIGNALS = [
    "market_index",
    "quote_signal",
]


VARIANTS = {
    "base + signals + dow":
        (BASE + SIGNALS + ["day_of_week"], False, False),

    "base + signals (no date)":
        (BASE + SIGNALS, False, False),

    "base + dow (no signals)":
        (BASE + ["day_of_week"], False, False),

    "base only":
        (BASE, False, False),

    "base + signals + all date":
        (
            BASE
            + SIGNALS
            + ["day_of_week", "month", "days_since_start"],
            False,
            False,
        ),

    "base + signals + dow, log target":
        (BASE + SIGNALS + ["day_of_week"], True, False),

    "base + signals + dow, clean negative weight":
        (BASE + SIGNALS + ["day_of_week"], False, True),

    "base + signals + dow, clean negative weight, log target":
        (BASE + SIGNALS + ["day_of_week"], True, True),
}


def run(num_cols, log_target, clean_negative_weight):

    train_local = train.copy()
    valid_local = valid.copy()

    if clean_negative_weight:

        train_local.loc[
            train_local["weight"] < 0,
            "weight"
        ] = np.nan

        valid_local.loc[
            valid_local["weight"] < 0,
            "weight"
        ] = np.nan

    preprocessor = ColumnTransformer([
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            ["equipment"]
        ),
    ])

    model = Pipeline([
        ("pre", preprocessor),
        (
            "gbm",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.06,
                random_state=42,
            ),
        ),
    ])

    if log_target:
        y = np.log1p(train_local["posted_rate"])
    else:
        y = train_local["posted_rate"]

    model.fit(
        train_local[num_cols + ["equipment"]],
        y
    )

    pred = model.predict(
        valid_local[num_cols + ["equipment"]]
    )

    if log_target:
        pred = np.expm1(pred)

    pred = np.maximum(pred, 1.0)

    y_true = valid_local["posted_rate"]

    mae = mean_absolute_error(y_true, pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, pred)
    )

    r2 = r2_score(y_true, pred)

    return mae, rmse, r2


print("=" * 80)
print("FEATURE / TARGET ABLATION")
print("=" * 80)

print(
    f"Train rows={len(train):,} | "
    f"Validation rows={len(valid):,}"
)

print()

results = []

for name, (cols, log_target, clean_negative) in VARIANTS.items():

    mae, rmse, r2 = run(
        cols,
        log_target,
        clean_negative
    )

    results.append(
        {
            "variant": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(
        f"{name:52s} "
        f"MAE=${mae:8.2f} "
        f"RMSE=${rmse:8.2f} "
        f"R2={r2:.4f}"
    )


print("\n" + "=" * 80)

results_df = pd.DataFrame(results)

print("\nSORTED BY MAE")
print(
    results_df
    .sort_values("MAE")
    .to_string(index=False)
)

print("\n" + "=" * 80)