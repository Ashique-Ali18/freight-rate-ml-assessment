import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/train-test.csv")
df["date"] = pd.to_datetime(df["date"])

# October validation
train = df[df["date"] < "2025-10-01"].copy()
valid = df[
    (df["date"] >= "2025-10-01") &
    (df["date"] < "2025-11-01")
].copy()

# ============================================================
# TARGET: EXTREME RATE REGIME
# ============================================================

threshold = 7500

train["high_rate"] = (train["posted_rate"] >= threshold).astype(int)
valid["high_rate"] = (valid["posted_rate"] >= threshold).astype(int)

print("=" * 90)
print("HIGH-RATE REGIME CLASSIFICATION")
print("=" * 90)

print("\nTraining:")
print(train["high_rate"].value_counts())
print(train["high_rate"].value_counts(normalize=True))

print("\nOctober validation:")
print(valid["high_rate"].value_counts())
print(valid["high_rate"].value_counts(normalize=True))

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

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.06,
    max_leaf_nodes=31,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ============================================================
# TRAIN
# ============================================================

X_train = train[numeric_features + categorical_features]
y_train = train["high_rate"]

X_valid = valid[numeric_features + categorical_features]
y_valid = valid["high_rate"]

print("\nTraining classifier...")

pipeline.fit(X_train, y_train)

# ============================================================
# PREDICT
# ============================================================

pred_prob = pipeline.predict_proba(X_valid)[:, 1]
pred = (pred_prob >= 0.5).astype(int)

# ============================================================
# METRICS
# ============================================================

print("\n" + "=" * 90)
print("CLASSIFICATION RESULTS")
print("=" * 90)

print("\nROC-AUC:")
print(round(roc_auc_score(y_valid, pred_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_valid, pred))

print("\nClassification Report:")
print(classification_report(
    y_valid,
    pred,
    digits=4,
    zero_division=0
))

# ============================================================
# INSPECT ACTUAL HIGH-RATE LOADS
# ============================================================

result = valid[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "weight",
        "equipment",
        "posted_rate"
    ]
].copy()

result["high_prob"] = pred_prob
result["predicted_high"] = pred

actual_high = result[result["posted_rate"] >= threshold].sort_values(
    "posted_rate",
    ascending=False
)

print("\n" + "=" * 90)
print("ACTUAL OCTOBER HIGH-RATE LOADS")
print("=" * 90)

print(
    actual_high.to_string(
        index=False,
        formatters={
            "posted_rate": "{:.2f}".format,
            "high_prob": "{:.4f}".format
        }
    )
)

# ============================================================
# HOW MANY EXTREMES DID WE CATCH?
# ============================================================

print("\n" + "=" * 90)
print("HIGH-RATE DETECTION")
print("=" * 90)

print(
    "Actual high-rate loads:",
    len(actual_high)
)

print(
    "Detected at probability >= 0.5:",
    int(pred.sum())
)

print(
    "Actual high-rate detected:",
    int(
        ((pred == 1) & (y_valid == 1)).sum()
    )
)

print(
    "Recall:",
    round(
        ((pred == 1) & (y_valid == 1)).sum()
        / max((y_valid == 1).sum(), 1),
        4
    )
)

# ============================================================
# TOP PROBABILITIES
# ============================================================

print("\n" + "=" * 90)
print("TOP 30 HIGH-RATE PROBABILITIES")
print("=" * 90)

print(
    result.sort_values(
        "high_prob",
        ascending=False
    ).head(30).to_string(
        index=False,
        formatters={
            "posted_rate": "{:.2f}".format,
            "high_prob": "{:.4f}".format
        }
    )
)