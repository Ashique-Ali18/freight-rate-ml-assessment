import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

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


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
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


X_train = train[
    numeric_features + categorical_features
]

y_train = train["posted_rate"]

X_valid = valid[
    numeric_features + categorical_features
]

y_valid = valid["posted_rate"]


# ============================================================
# MODELS
# ============================================================

models = {

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        random_state=42
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=42
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=-1,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        loss="huber",
        random_state=42
    ),
}


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 90)
    print(f"TRAINING: {name}")
    print("=" * 90)

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
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "MAE"
)

print("\n" + "=" * 90)
print("MODEL COMPARISON")
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