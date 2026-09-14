import pandas as pd
from datetime import date

# ---------------------------------------------------------------
# 1. Build the list of year-month strings to load (YYYYMM format)
#    Range: January 2024 -> most recently completed calendar month
# ---------------------------------------------------------------

last_completed_year = 2026
last_completed_month = 5
#today = date.today()

# The "most recently completed" month is the month before the current one.
# (If today is in month M, month M is still in progress, so the last
# completed month is M-1.)
#if today.month == 1:
# last_completed_year = today.year
#   last_completed_month = today.month - 1

year_months = []
year, month = 2026, 1
while (year, month) <= (last_completed_year, last_completed_month):
    year_months.append(f"{year}{month:02d}")
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1

print(f"Months to include: {year_months}")

# ---------------------------------------------------------------
# 2. Update these filename patterns if your files are named differently
# ---------------------------------------------------------------
DATA_DIR = r"C:\Users\ellen\OneDrive\Desktop\IDX Internship 2026\data raw"

SOLD_FILE_PATTERN = DATA_DIR + r"\CRMLSSold{ym}.csv"
LISTING_FILE_PATTERN = DATA_DIR + r"\CRMLSListing{ym}.csv"


def load_and_concat(file_pattern, year_months, label):
    """Reads each monthly file and concatenates them into one DataFrame."""
    monthly_frames = []
    row_counts_before = {}

    for ym in year_months:
        filename = file_pattern.format(ym=ym)
        df = pd.read_csv(filename)
        row_counts_before[filename] = len(df)
        monthly_frames.append(df)

    combined = pd.concat(monthly_frames, ignore_index=True)

    # --- Row count comments (before concatenation) ---
    print(f"\n--- {label}: row counts per monthly file (before concatenation) ---")
    for filename, count in row_counts_before.items():
        print(f"{filename}: {count} rows")

    total_before = sum(row_counts_before.values())
    print(f"Sum of individual monthly files: {total_before} rows")

    # --- Row count comment (after concatenation) ---
    print(f"{label}: combined dataset after concatenation = {len(combined)} rows")
    # These two totals should match; if not, check for duplicate index issues.

    return combined


# ---------------------------------------------------------------
# 3. Load and concatenate SOLD data
# ---------------------------------------------------------------
sold = load_and_concat(SOLD_FILE_PATTERN, year_months, "SOLD")

# Row count before Residential filter
print(f"\nSOLD: row count before Residential filter = {len(sold)}")

sold_residential = sold[sold["PropertyType"] == "Residential"].copy()

# Row count after Residential filter
print(f"SOLD: row count after Residential filter = {len(sold_residential)}")

sold_residential.to_csv("combined_sold_residential.csv", index=False)
print("Saved: combined_sold_residential.csv")

# ---------------------------------------------------------------
# 4. Load and concatenate LISTING data
# ---------------------------------------------------------------
listings = load_and_concat(LISTING_FILE_PATTERN, year_months, "LISTINGS")

# Row count before Residential filter
print(f"\nLISTINGS: row count before Residential filter = {len(listings)}")

listings_residential = listings[listings["PropertyType"] == "Residential"].copy()

# Row count after Residential filter
print(f"LISTINGS: row count after Residential filter = {len(listings_residential)}")

listings_residential.to_csv("combined_listings_residential.csv", index=False)
print("Saved: combined_listings_residential.csv")

print("\nDone. Combined, Residential-filtered CSVs have been created.")
