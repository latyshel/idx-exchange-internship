"""
Week 1 - Monthly Dataset Aggregation
IDX Exchange Internship

Objective:
- Load all monthly Sold and Listing CSV files
- Combine them into two datasets
- Filter to Residential properties
- Save combined CSVs
"""

import pandas as pd
from pathlib import Path

# -----------------------------
# Folder setup
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data_raw"
OUTPUT = BASE_DIR / "data_cleaned"

OUTPUT.mkdir(exist_ok=True)

# -----------------------------
# Find all CSV files
# -----------------------------
sold_files = sorted(RAW_DATA.glob("CRMLSSold*.csv"))
listing_files = sorted(RAW_DATA.glob("CRMLSListing*.csv"))

print(f"Found {len(sold_files)} sold files.")
print(f"Found {len(listing_files)} listing files.")

# -----------------------------
# Combine Sold files
# -----------------------------
sold_dataframes = []

for file in sold_files:
    df = pd.read_csv(file, low_memory=False)
    print(f"{file.name}: {len(df)} rows")
    sold_dataframes.append(df)

sold = pd.concat(sold_dataframes, ignore_index=True)

print(f"\nSold rows after concatenation: {len(sold)}")

# -----------------------------
# Combine Listing files
# -----------------------------
listing_dataframes = []

for file in listing_files:
    df = pd.read_csv(file, low_memory=False)
    print(f"{file.name}: {len(df)} rows")
    listing_dataframes.append(df)

listings = pd.concat(listing_dataframes, ignore_index=True)

print(f"Listing rows after concatenation: {len(listings)}")

# -----------------------------
# Residential filter
# -----------------------------
sold_before = len(sold)
listing_before = len(listings)

sold = sold[sold["PropertyType"] == "Residential"]
listings = listings[listings["PropertyType"] == "Residential"]

print("\nResidential Filter")
print("-------------------")
print(f"Sold before filter: {sold_before}")
print(f"Sold after filter : {len(sold)}")

print(f"Listings before filter: {listing_before}")
print(f"Listings after filter : {len(listings)}")

# -----------------------------
# Save combined datasets
# -----------------------------
sold.to_csv(OUTPUT / "sold.csv", index=False)
listings.to_csv(OUTPUT / "listings.csv", index=False)

print("\nFinished.")
print("Created:")
print(" - sold.csv")
print(" - listings.csv")