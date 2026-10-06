import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/train-test.csv",
    parse_dates=["date"]
)

# Time split
train = df[df["date"] < "2025-10-01"].copy()
valid = df[df["date"] >= "2025-10-01"].copy()

# --------------------------------------------------
# TRAIN ROUTE STATISTICS
# --------------------------------------------------

route_stats = (
    train
    .groupby(["pickup", "delivery"])
    .agg(
        route_count=("posted_rate", "size"),
        route_mean_rate=("posted_rate", "mean"),
        route_median_rate=("posted_rate", "median"),
        route_max_rate=("posted_rate", "max"),
        route_high_rate_count=(
            "posted_rate",
            lambda x: (x >= 7500).sum()
        ),
    )
    .reset_index()
)

# --------------------------------------------------
# HIGH-RATE ROUTES IN TRAINING
# --------------------------------------------------

print("=" * 90)
print("TRAINING ROUTES WITH HIGH-RATE LOADS")
print("=" * 90)

high_routes = (
    route_stats[
        route_stats["route_high_rate_count"] > 0
    ]
    .sort_values(
        ["route_high_rate_count", "route_max_rate"],
        ascending=False
    )
)

print(
    high_routes
    .head(30)
    .round(2)
    .to_string(index=False)
)

# --------------------------------------------------
# HOW MANY ROUTES HAVE HIGH-RATE LOADS?
# --------------------------------------------------

print("\n" + "=" * 90)
print("ROUTE COVERAGE")
print("=" * 90)

print(
    f"Total training routes: "
    f"{len(route_stats):,}"
)

print(
    f"Routes with >= $7,500 load: "
    f"{(route_stats['route_high_rate_count'] > 0).sum():,}"
)

print(
    f"Routes with >= $10,000 load: "
    f"{(
        train
        .groupby(['pickup', 'delivery'])['posted_rate']
        .max()
        >= 10000
    ).sum():,}"
)

# --------------------------------------------------
# VALIDATION ROUTE MATCHING
# --------------------------------------------------

valid_routes = valid[
    ["pickup", "delivery", "posted_rate"]
].merge(
    route_stats,
    on=["pickup", "delivery"],
    how="left"
)

print("\n" + "=" * 90)
print("OCTOBER VALIDATION ROUTE COVERAGE")
print("=" * 90)

print(
    f"Validation rows: {len(valid_routes):,}"
)

print(
    f"Rows with known training route: "
    f"{valid_routes['route_count'].notna().sum():,}"
)

print(
    f"Rows with unseen route: "
    f"{valid_routes['route_count'].isna().sum():,}"
)

# --------------------------------------------------
# HIGH-RATE OCTOBER LOADS
# --------------------------------------------------

oct_high = valid_routes[
    valid_routes["posted_rate"] >= 7500
].copy()

print("\n" + "=" * 90)
print("OCTOBER HIGH-RATE LOADS")
print("=" * 90)

print(
    f"October >= $7,500 rows: "
    f"{len(oct_high):,}"
)

print(
    f"Known training route among them: "
    f"{oct_high['route_count'].notna().sum():,}"
)

print(
    f"Unseen route among them: "
    f"{oct_high['route_count'].isna().sum():,}"
)

print("\nOctober high-rate routes:")

print(
    oct_high[
        [
            "pickup",
            "delivery",
            "posted_rate",
            "route_count",
            "route_mean_rate",
            "route_max_rate",
            "route_high_rate_count",
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(30)
    .round(2)
    .to_string(index=False)
)

# --------------------------------------------------
# ROUTE RATE DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 90)
print("TRAINING ROUTE STATISTICS")
print("=" * 90)

print(
    route_stats[
        [
            "route_count",
            "route_mean_rate",
            "route_median_rate",
            "route_max_rate",
            "route_high_rate_count",
        ]
    ]
    .describe()
    .round(2)
)