import pandas as pd
import numpy as np

df = pd.read_csv("data/train-test.csv")

df["date"] = pd.to_datetime(df["date"])

print("=" * 60)
print("QUICK DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# NEGATIVE / ZERO WEIGHTS
# --------------------------------------------------

print("\nWEIGHT QUALITY")
print("-" * 60)

print("Missing weights :", df["weight"].isna().sum())
print("Zero weights    :", (df["weight"] == 0).sum())
print("Negative weights:", (df["weight"] < 0).sum())

negative = df[df["weight"] < 0]

if len(negative) > 0:
    print("\nNegative weight statistics:")
    print(negative["weight"].describe())

    print("\nNegative-weight posted_rate:")
    print(
        negative["posted_rate"]
        .describe()
        .round(2)
    )

# --------------------------------------------------
# CORRELATIONS
# --------------------------------------------------

print("\nNUMERIC CORRELATIONS WITH POSTED RATE")
print("-" * 60)

numeric_cols = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
]

print(
    df[numeric_cols]
    .corr(numeric_only=True)["posted_rate"]
    .sort_values(ascending=False)
    .round(4)
)

# --------------------------------------------------
# EQUIPMENT
# --------------------------------------------------

print("\nPOSTED RATE BY EQUIPMENT")
print("-" * 60)

print(
    df.groupby("equipment")["posted_rate"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

# --------------------------------------------------
# MONTHLY TARGET
# --------------------------------------------------

print("\nMONTHLY POSTED RATE")
print("-" * 60)

df["month"] = df["date"].dt.to_period("M")

print(
    df.groupby("month")["posted_rate"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

print("\nROUTE INFORMATION")
print("-" * 60)

print("Unique routes:", df.groupby(["pickup", "delivery"]).ngroups)

route_counts = (
    df.groupby(["pickup", "delivery"])
    .size()
    .sort_values(ascending=False)
)

print("\nTop 10 routes:")
print(route_counts.head(10))

print("\n" + "=" * 60)
print("QUICK ANALYSIS COMPLETE")
print("=" * 60)