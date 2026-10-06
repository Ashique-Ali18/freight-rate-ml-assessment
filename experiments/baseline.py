import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


print("=" * 70)
print("FREIGHT RATE - BASELINE MODEL")
print("=" * 70)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/train-test.csv",
    parse_dates=["date"]
)

print(f"\nTotal rows: {len(df):,}")


# --------------------------------------------------
# 2. BASIC FEATURE ENGINEERING
# --------------------------------------------------

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["days_since_start"] = (
    df["date"] - pd.Timestamp("2025-01-01")
).dt.days

df["log_distance"] = np.log1p(df["distance"])


# --------------------------------------------------
# 3. TIME-BASED SPLIT
# --------------------------------------------------

train = df[df["date"] < "2025-10-01"].copy()
valid = df[df["date"] >= "2025-10-01"].copy()

print(f"Training rows : {len(train):,}")
print(f"Validation rows: {len(valid):,}")

print(
    f"Training period : "
    f"{train['date'].min().date()} → {train['date'].max().date()}"
)

print(
    f"Validation period: "
    f"{valid['date'].min().date()} → {valid['date'].max().date()}"
)


# --------------------------------------------------
# 4. FEATURES
# --------------------------------------------------

numeric_features = [
    "distance",
    "log_distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
    "day_of_week",
    "month",
    "days_since_start",
]

categorical_features = [
    "equipment"
]

features = numeric_features + categorical_features

target = "posted_rate"


X_train = train[features]
y_train = train[target]

X_valid = valid[features]
y_valid = valid[target]


# --------------------------------------------------
# 5. PREPROCESSING
# --------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# --------------------------------------------------
# 6. MODEL
# --------------------------------------------------

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
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
# 7. TRAIN
# --------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. PREDICT
# --------------------------------------------------

predictions = model.predict(X_valid)

predictions = np.maximum(
    predictions,
    1.0
)


# --------------------------------------------------
# 9. METRICS
# --------------------------------------------------

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


# --------------------------------------------------
# 10. RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"MAE :  ${mae:,.2f}")
print(f"RMSE:  ${rmse:,.2f}")
print(f"R²  :  {r2:.4f}")

print("=" * 70)