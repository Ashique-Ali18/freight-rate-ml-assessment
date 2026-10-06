import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/train-test.csv",
    parse_dates=["date"]
)

print("=" * 80)
print("HIGH-RATE LOAD ANALYSIS")
print("=" * 80)

# --------------------------------------------------
# HIGH RATE THRESHOLDS
# --------------------------------------------------

for threshold in [5000, 7500, 10000]:

    subset = df[df["posted_rate"] >= threshold]

    print("\n" + "-" * 80)
    print(f"POSTED RATE >= ${threshold:,}")
    print("-" * 80)

    print(f"Rows: {len(subset):,}")
    print(f"Percentage: {len(subset) / len(df) * 100:.3f}%")

    if len(subset) > 0:

        print("\nAverage features:")

        print(
            subset[
                [
                    "distance",
                    "weight",
                    "market_index",
                    "quote_signal",
                    "posted_rate",
                ]
            ]
            .mean()
            .round(2)
        )

        print("\nEquipment:")
        print(subset["equipment"].value_counts())

        print("\nMonths:")
        print(subset["date"].dt.month.value_counts().sort_index())


# --------------------------------------------------
# TOP ROUTES BY HIGH-RATE LOADS
# --------------------------------------------------

print("\n" + "=" * 80)
print("ROUTES WITH HIGH-RATE LOADS")
print("=" * 80)

high = df[df["posted_rate"] >= 5000].copy()

route_stats = (
    high
    .groupby(["pickup", "delivery"])
    .agg(
        high_rate_count=("posted_rate", "count"),
        avg_rate=("posted_rate", "mean"),
        max_rate=("posted_rate", "max"),
    )
    .sort_values("high_rate_count", ascending=False)
)

print(route_stats.head(30).round(2).to_string())


# --------------------------------------------------
# TOP LOCATIONS
# --------------------------------------------------

print("\n" + "=" * 80)
print("PICKUP LOCATIONS WITH HIGH-RATE LOADS")
print("=" * 80)

print(
    high["pickup"]
    .value_counts()
    .head(30)
)


print("\n" + "=" * 80)
print("DELIVERY LOCATIONS WITH HIGH-RATE LOADS")
print("=" * 80)

print(
    high["delivery"]
    .value_counts()
    .head(30)
)


# --------------------------------------------------
# DISTANCE
# --------------------------------------------------

print("\n" + "=" * 80)
print("HIGH-RATE DISTANCE")
print("=" * 80)

print(
    high["distance"]
    .describe()
    .round(2)
)


# --------------------------------------------------
# WEIGHT
# --------------------------------------------------

print("\n" + "=" * 80)
print("HIGH-RATE WEIGHT")
print("=" * 80)

print(
    high["weight"]
    .describe()
    .round(2)
)


# --------------------------------------------------
# SIGNALS
# --------------------------------------------------

print("\n" + "=" * 80)
print("HIGH-RATE SIGNALS")
print("=" * 80)

print(
    high[
        [
            "market_index",
            "quote_signal"
        ]
    ]
    .describe()
    .round(4)
)


# --------------------------------------------------
# HIGH RATE BY EQUIPMENT
# --------------------------------------------------

print("\n" + "=" * 80)
print("HIGH-RATE LOADS BY EQUIPMENT")
print("=" * 80)

print(
    high
    .groupby("equipment")["posted_rate"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max"
    )
    .round(2)
)


# --------------------------------------------------
# RATE PER MILE
# --------------------------------------------------

df["rate_per_mile"] = (
    df["posted_rate"] / df["distance"]
)

print("\n" + "=" * 80)
print("RATE PER MILE")
print("=" * 80)

print(
    df["rate_per_mile"]
    .describe()
    .round(3)
)

print("\nHigh-rate loads rate/mile:")
print(
    high.assign(
        rate_per_mile=high["posted_rate"] / high["distance"]
    )[
        [
            "pickup",
            "delivery",
            "distance",
            "posted_rate",
            "rate_per_mile"
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# TARGET / DISTANCE RELATIONSHIP
# --------------------------------------------------

print("\n" + "=" * 80)
print("RATE PER MILE BY TARGET RANGE")
print("=" * 80)

df["rate_bucket"] = pd.cut(
    df["posted_rate"],
    bins=[
        0,
        1000,
        2000,
        3000,
        5000,
        7500,
        10000,
        15000,
        30000,
    ]
)

print(
    df
    .groupby("rate_bucket", observed=True)
    ["rate_per_mile"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max"
    )
    .round(3)
)