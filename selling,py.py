import pandas as pd
from pathlib import Path

# Folder locations
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data_raw"
OUTPUT = BASE_DIR / "data_cleaned"

OUTPUT.mkdir(exist_ok=True)

# Find all Sold CSV files
sold_files = sorted(RAW_DATA.glob("CRMLSSold*.csv"))

print(f"Found {len(sold_files)} Sold files.")

sold_dfs = []

for file in sold_files:
    df = pd.read_csv(file, low_memory=False)
    print(f"Loaded {file.name}: {len(df)} rows")
    sold_dfs.append(df)

# Combine all months
sold = pd.concat(sold_dfs, ignore_index=True)

print(f"\nRows after combining: {len(sold)}")

# Residential filter
before = len(sold)

sold = sold[sold["PropertyType"] == "Residential"]

print(f"Rows before Residential filter: {before}")
print(f"Rows after Residential filter: {len(sold)}")

# Save file
sold.to_csv(OUTPUT / "sold.csv", index=False)

print("\nFinished!")
print("Saved: data_cleaned/sold.csv")