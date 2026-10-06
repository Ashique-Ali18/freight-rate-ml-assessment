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
# FEATURE ENGINEERING
# ============================================================

def add_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Distance transformations
    # --------------------------------------------------------

    df["log_distance"] = np.log1p(
        np.maximum(df["distance"], 0)
    )

    df["distance_squared"] = (
        df["distance"] ** 2
    )

    # --------------------------------------------------------
    # Geographic differences
    # --------------------------------------------------------

    df["lat_diff"] = (
        df["delivery_lat"] -
        df["pickup_lat"]
    )

    df["lon_diff"] = (
        df["delivery_lon"] -
        df["pickup_lon"]
    )

    df["abs_lat_diff"] = np.abs(
        df["lat_diff"]
    )

    df["abs_lon_diff"] = np.abs(
        df["lon_diff"]
    )

    # --------------------------------------------------------
    # Approximate geographic distance
    # Haversine formula
    # --------------------------------------------------------

    lat1 = np.radians(df["pickup_lat"])
    lon1 = np.radians(df["pickup_lon"])

    lat2 = np.radians(df["delivery_lat"])
    lon2 = np.radians(df["delivery_lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    df["geo_distance"] = (
        2
        * 6371
        * np.arcsin(np.sqrt(a))
    )

    # --------------------------------------------------------
    # Route identifier
    # --------------------------------------------------------

    df["route"] = (
        df["pickup"].astype(str)
        + " -> "
        + df["delivery"].astype(str)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/train-test.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

train = df[
    df["date"] < "2025-10-01"
].copy()

valid = df[
    (df["date"] >= "2025-10-01")
    &
    (df["date"] < "2025-11-01")
].copy()


# ============================================================
# ADD FEATURES
# ============================================================

train = add_features(train)
valid = add_features(valid)


# ============================================================
# FEATURE SET
# ============================================================

numeric_features = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",

    # New features
    "log_distance",
    "distance_squared",
    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",
    "geo_distance",
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]


all_features = (
    numeric_features
    + categorical_features
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


# ============================================================
# MODEL
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
# TRAIN
# ============================================================

X_train = train[all_features]
y_train = train["posted_rate"]

X_valid = valid[all_features]
y_valid = valid["posted_rate"]

print("=" * 90)
print("TRAINING FEATURE-ENGINEERED MODEL")
print("=" * 90)

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

predictions = pipeline.predict(
    X_valid
)

predictions = np.maximum(
    predictions,
    1
)


# ============================================================
# METRICS
# ============================================================

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


print("\n" + "=" * 90)
print("FEATURE-ENGINEERED RESULTS")
print("=" * 90)

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# BASELINE COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("COMPARISON")
print("=" * 90)

print("Previous best:")
print("MAE  : $103.75")
print("RMSE : $648.45")
print("R²   : 0.8200")

print("\nCurrent:")
print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print("\nMAE improvement:")
print(
    f"${103.75 - mae:,.2f}"
)