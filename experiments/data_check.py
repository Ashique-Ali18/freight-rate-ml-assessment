import pandas as pd

print("=" * 60)
print("FREIGHT RATE ML ASSESSMENT - DATA CHECK")
print("=" * 60)

# Load datasets
train = pd.read_csv("data/train-test.csv")
validation = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation-predictions-template.csv")
december = pd.read_csv("data/december-chart-inputs.csv")

# --------------------------------------------------
# BASIC SHAPES
# --------------------------------------------------

print("\nDATASET SHAPES")
print("-" * 60)
print(f"Training data       : {train.shape}")
print(f"Validation data     : {validation.shape}")
print(f"Prediction template : {template.shape}")
print(f"December inputs     : {december.shape}")

# --------------------------------------------------
# TRAINING COLUMNS
# --------------------------------------------------

print("\nTRAINING COLUMNS")
print("-" * 60)

for column in train.columns:
    print(f"{column:20s} {train[column].dtype}")

# --------------------------------------------------
# TARGET
# --------------------------------------------------

print("\nTARGET INFORMATION")
print("-" * 60)

print("Target: posted_rate")
print(f"Minimum : {train['posted_rate'].min():,.2f}")
print(f"Maximum : {train['posted_rate'].max():,.2f}")
print(f"Mean    : {train['posted_rate'].mean():,.2f}")
print(f"Median  : {train['posted_rate'].median():,.2f}")

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\nMISSING VALUES - TRAINING")
print("-" * 60)

print(train.isna().sum())

print("\nMISSING VALUES - VALIDATION")
print("-" * 60)

print(validation.isna().sum())

# --------------------------------------------------
# UNIQUE VALUES
# --------------------------------------------------

print("\nUNIQUE VALUES")
print("-" * 60)

for column in train.columns:
    print(f"{column:20s} {train[column].nunique()}")

# --------------------------------------------------
# CATEGORICAL DISTRIBUTIONS
# --------------------------------------------------

print("\nEQUIPMENT DISTRIBUTION")
print("-" * 60)

print(train["equipment"].value_counts())

print("\nPICKUP LOCATIONS")
print("-" * 60)

print(f"Training pickup locations : {train['pickup'].nunique()}")
print(f"Validation pickup locations: {validation['pickup'].nunique()}")

print("\nDELIVERY LOCATIONS")
print("-" * 60)

print(f"Training delivery locations : {train['delivery'].nunique()}")
print(f"Validation delivery locations: {validation['delivery'].nunique()}")

# --------------------------------------------------
# DATE RANGE
# --------------------------------------------------

train["date"] = pd.to_datetime(train["date"])
validation["date"] = pd.to_datetime(validation["date"])

print("\nDATE RANGE")
print("-" * 60)

print(f"Training   : {train['date'].min().date()} → {train['date'].max().date()}")
print(f"Validation : {validation['date'].min().date()} → {validation['date'].max().date()}")

# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\nDUPLICATES")
print("-" * 60)

print(f"Duplicate training rows : {train.duplicated().sum()}")
print(f"Duplicate validation rows: {validation.duplicated().sum()}")
print(f"Duplicate training IDs  : {train['load_id'].duplicated().sum()}")
print(f"Duplicate validation IDs: {validation['load_id'].duplicated().sum()}")

# --------------------------------------------------
# NUMERIC SUMMARY
# --------------------------------------------------

print("\nNUMERIC SUMMARY")
print("-" * 60)

print(
    train[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ].describe().round(2)
)

# --------------------------------------------------
# SAMPLE
# --------------------------------------------------

print("\nFIRST 5 TRAINING ROWS")
print("-" * 60)

print(train.head().to_string(index=False))

print("\nFIRST 5 VALIDATION ROWS")
print("-" * 60)

print(validation.head().to_string(index=False))

print("\n" + "=" * 60)
print("DATA CHECK COMPLETE")
print("=" * 60)