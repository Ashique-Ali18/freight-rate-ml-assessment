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
VALIDATION_PATH = "data/validation.csv"
TEMPLATE_PATH = "data/validation-predictions-template.csv"

OUTPUT_PATH = "outputs/validation_predictions.csv"


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
validation = pd.read_csv(VALIDATION_PATH)
template = pd.read_csv(TEMPLATE_PATH)

print(f"Training rows   : {len(train):,}")
print(f"Validation rows : {len(validation):,}")
print(f"Template rows   : {len(template):,}")


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
# TRAIN ON ALL LABELED DATA
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
# PREDICT VALIDATION DATA
# ============================================================

print("\n" + "=" * 90)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 90)

predictions = pipeline.predict(
    validation[FEATURES]
)

# Rates must be positive
predictions = np.maximum(
    predictions,
    1.0
)


# ============================================================
# BUILD SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": predictions
})


# ============================================================
# ALIGN WITH TEMPLATE
# ============================================================

# Use the template's ID order.
submission = template[["load_id"]].merge(
    submission,
    on="load_id",
    how="left"
)


# ============================================================
# VALIDATION CHECKS
# ============================================================

assert len(submission) == 12000

assert list(submission.columns) == [
    "load_id",
    "predicted_rate"
]

assert submission["load_id"].is_unique

assert submission["predicted_rate"].notna().all()

assert np.isfinite(
    submission["predicted_rate"]
).all()

assert (
    submission["predicted_rate"] > 0
).all()


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

submission.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("SUBMISSION CREATED")
print("=" * 90)

print(f"Output: {OUTPUT_PATH}")

print(
    f"Rows: {len(submission):,}"
)

print(
    f"Prediction min: "
    f"${submission['predicted_rate'].min():,.2f}"
)

print(
    f"Prediction median: "
    f"${submission['predicted_rate'].median():,.2f}"
)

print(
    f"Prediction mean: "
    f"${submission['predicted_rate'].mean():,.2f}"
)

print(
    f"Prediction max: "
    f"${submission['predicted_rate'].max():,.2f}"
)

print("\nFirst 10 predictions:")
print(
    submission.head(10).to_string(
        index=False
    )
)

print("\nFinal validation checks: PASSED")