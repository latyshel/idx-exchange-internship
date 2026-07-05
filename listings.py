import pandas as pd
from pathlib import Path

# Folder locations
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data_raw"
OUTPUT = BASE_DIR / "data_cleaned"

OUTPUT.mkdir(exist_ok=True)

# Find all Listing CSV files
listing_files = sorted(RAW_DATA.glob("CRMLSListing*.csv"))

print(f"Found {len(listing_files)} Listing files.")

listing_dfs = []

for file in listing_files:
    df = pd.read_csv(file, low_memory=False)
    print(f"Loaded {file.name}: {len(df)} rows")
    listing_dfs.append(df)

# Combine all months
listings = pd.concat(listing_dfs, ignore_index=True)

print(f"\nRows after combining: {len(listings)}")

# Residential filter
before = len(listings)

listings = listings[listings["PropertyType"] == "Residential"]

print(f"Rows before Residential filter: {before}")
print(f"Rows after Residential filter: {len(listings)}")

# Save file
listings.to_csv(OUTPUT / "listings.csv", index=False)

print("\nFinished!")
print("Saved: data_cleaned/listings.csv")