import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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

numeric_cols = [
    "distance",
    "weight",
]

# Median-fill numeric fields
train_numeric = train[numeric_cols].copy()
valid_numeric = valid[numeric_cols].copy()

for col in numeric_cols:
    median = train_numeric[col].median()

    train_numeric[col] = train_numeric[col].fillna(median)
    valid_numeric[col] = valid_numeric[col].fillna(median)


# Standardize
scaler = StandardScaler()

X_train_num = scaler.fit_transform(train_numeric)
X_valid_num = scaler.transform(valid_numeric)


# ============================================================
# NEAREST NEIGHBORS
# ============================================================

nn = NearestNeighbors(
    n_neighbors=20,
    metric="euclidean"
)

nn.fit(X_train_num)


distances, indices = nn.kneighbors(X_valid_num)


# ============================================================
# ANALYZE OCTOBER HIGH-RATE LOADS
# ============================================================

threshold = 7500

high = valid[valid["posted_rate"] >= threshold].copy()

print("=" * 100)
print("NEAREST-NEIGHBOR ANALYSIS OF OCTOBER HIGH-RATE LOADS")
print("=" * 100)

for idx in high.index:

    row_position = valid.index.get_loc(idx)

    load = valid.loc[idx]

    neighbor_positions = indices[row_position]

    neighbors = train.iloc[neighbor_positions].copy()

    # Same route
    same_route = neighbors[
        (neighbors["pickup"] == load["pickup"]) &
        (neighbors["delivery"] == load["delivery"])
    ]

    # Same equipment
    same_equipment = neighbors[
        neighbors["equipment"] == load["equipment"]
    ]

    print("\n" + "-" * 100)

    print(
        f"LOAD: {load['load_id']} | "
        f"{load['pickup']} -> {load['delivery']} | "
        f"{load['equipment']} | "
        f"distance={load['distance']:.1f} | "
        f"weight={load['weight']:.1f}"
    )

    print(
        f"ACTUAL RATE: ${load['posted_rate']:,.2f}"
    )

    print(
        f"\nSame-route neighbors: {len(same_route)}"
    )

    if len(same_route) > 0:

        print(
            same_route[
                [
                    "pickup",
                    "delivery",
                    "distance",
                    "weight",
                    "equipment",
                    "posted_rate",
                    "date"
                ]
            ]
            .sort_values("posted_rate", ascending=False)
            .head(10)
            .to_string(index=False)
        )

        print(
            "\nSame-route statistics:"
        )

        print(
            "mean:",
            round(same_route["posted_rate"].mean(), 2)
        )

        print(
            "median:",
            round(same_route["posted_rate"].median(), 2)
        )

        print(
            "max:",
            round(same_route["posted_rate"].max(), 2)
        )

    print(
        f"\nSame-equipment among nearest neighbors: "
        f"{len(same_equipment)}"
    )

    if len(same_equipment) > 0:

        print(
            "nearest-equipment mean:",
            round(
                same_equipment["posted_rate"].mean(),
                2
            )
        )

        print(
            "nearest-equipment max:",
            round(
                same_equipment["posted_rate"].max(),
                2
            )
        )


# ============================================================
# SUMMARY BY RATE REGIME
# ============================================================

print("\n" + "=" * 100)
print("RATE REGIME SUMMARY")
print("=" * 100)

for threshold in [5000, 7500, 10000]:

    subset = valid[
        valid["posted_rate"] >= threshold
    ]

    print(
        f"\nOctober loads >= ${threshold:,}: "
        f"{len(subset)}"
    )

    if len(subset) == 0:
        continue

    same_route_maxes = []

    for idx in subset.index:

        row_position = valid.index.get_loc(idx)

        load = valid.loc[idx]

        route_rows = train[
            (train["pickup"] == load["pickup"]) &
            (train["delivery"] == load["delivery"])
        ]

        if len(route_rows) > 0:
            same_route_maxes.append(
                route_rows["posted_rate"].max()
            )

    if same_route_maxes:

        same_route_maxes = np.array(
            same_route_maxes
        )

        print(
            "Historical same-route max:",
            "mean =",
            round(same_route_maxes.mean(), 2),
            "| median =",
            round(np.median(same_route_maxes), 2),
            "| max =",
            round(same_route_maxes.max(), 2)
        )

        print(
            "Routes with historical max >= threshold:",
            int(
                (same_route_maxes >= threshold).sum()
            ),
            "/",
            len(same_route_maxes)
        )