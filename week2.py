import pandas as pd
from datetime import date

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---------------------------------------------------------------
# 1. Path + file pattern setup (same as Week 1 - update if needed)
# ---------------------------------------------------------------
DATA_DIR = r"C:\Users\ellen\OneDrive\Desktop\IDX Internship 2026\data raw"

SOLD_FILE_PATTERN = DATA_DIR + r"\CRMLSSold{ym}.csv"
LISTING_FILE_PATTERN = DATA_DIR + r"\CRMLSListing{ym}.csv"

# Hardcode the range to match what's actually in the data raw folder.
# Switch back to auto-calculated "today - 1 month" once new files arrive.
last_completed_year = 2026
last_completed_month = 5

year_months = []
year, month = 2026, 1
while (year, month) <= (last_completed_year, last_completed_month):
    year_months.append(f"{year}{month:02d}")
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1


def load_and_concat(file_pattern, year_months):
    """Reads each monthly file and concatenates them into one DataFrame."""
    frames = [pd.read_csv(file_pattern.format(ym=ym), low_memory=False) for ym in year_months]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------
# 2. Reusable EDA functions
# ---------------------------------------------------------------

def inspect_structure(df, label):
    """Prints shape, dtypes, and unique property types for a dataset."""
    print(f"\n{'=' * 60}\n{label}: DATASET STRUCTURE\n{'=' * 60}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\nColumn data types:\n{df.dtypes}")
    print(f"\nUnique PropertyType values: {sorted(df['PropertyType'].dropna().unique())}")
    print(f"\nPropertyType value counts:\n{df['PropertyType'].value_counts()}")


def filter_residential(df, label):
    """Filters to PropertyType == 'Residential' and documents the logic."""
    before = len(df)
    filtered = df[df["PropertyType"] == "Residential"].copy()
    after = len(filtered)
    print(f"\n{label}: filtering applied -> df[df['PropertyType'] == 'Residential']")
    print(f"{label}: row count before filter = {before}")
    print(f"{label}: row count after filter  = {after}")
    print(f"{label}: Residential share = {after / before:.1%}")
    return filtered


def null_count_summary(df, label):
    """Returns a null-count / null-percentage table sorted worst-first."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"null_count": null_counts, "null_percent": null_pct})
    summary = summary.sort_values("null_percent", ascending=False)

    print(f"\n{label}: NULL COUNT SUMMARY (top 20 by % missing)")
    print(summary.head(20))
    return summary


def missing_value_report(summary, label, threshold=90.0):
    """Flags columns above the missing-value threshold (default 90%)."""
    flagged = summary[summary["null_percent"] > threshold]
    print(f"\n{label}: MISSING VALUE REPORT (>{threshold}% null)")
    if flagged.empty:
        print("No columns exceed the 90% missing threshold.")
    else:
        print(flagged)
        print(
            f"\nRecommendation: consider dropping these {len(flagged)} column(s) "
            "unless they are core analytical fields, in which case retain them "
            "as partially-missing rather than dropping outright."
        )
    return flagged


def numeric_distribution_summary(df, columns, label):
    """Prints min/max/mean/median/percentile summary for given numeric columns."""
    print(f"\n{label}: NUMERIC DISTRIBUTION SUMMARY")
    existing_cols = [c for c in columns if c in df.columns]
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        print(f"Note: these columns were not found in the dataset and were skipped: {missing_cols}")

    stats = df[existing_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats = stats.rename(columns={"50%": "median"})
    print(stats)
    return stats


# ---------------------------------------------------------------
# 3. Run the pipeline for SOLD
# ---------------------------------------------------------------
sold_raw = load_and_concat(SOLD_FILE_PATTERN, year_months)
inspect_structure(sold_raw, "SOLD")

sold = filter_residential(sold_raw, "SOLD")

sold_null_summary = null_count_summary(sold, "SOLD")
sold_missing_flagged = missing_value_report(sold_null_summary, "SOLD")

sold_numeric_summary = numeric_distribution_summary(
    sold, ["ClosePrice", "LivingArea", "DaysOnMarket"], "SOLD"
)

sold.to_csv(DATA_DIR + r"\sold_residential_filtered.csv", index=False)
print("\nSaved: sold_residential_filtered.csv")

# ---------------------------------------------------------------
# 4. Run the pipeline for LISTINGS
# ---------------------------------------------------------------
listings_raw = load_and_concat(LISTING_FILE_PATTERN, year_months)
inspect_structure(listings_raw, "LISTINGS")

listings = filter_residential(listings_raw, "LISTINGS")

listings_null_summary = null_count_summary(listings, "LISTINGS")
listings_missing_flagged = missing_value_report(listings_null_summary, "LISTINGS")

listings_numeric_summary = numeric_distribution_summary(
    listings, ["ListPrice", "OriginalListPrice", "LivingArea", "DaysOnMarket"], "LISTINGS"
)

listings.to_csv(DATA_DIR + r"\listings_residential_filtered.csv", index=False)
print("\nSaved: listings_residential_filtered.csv")

# ---------------------------------------------------------------
# 5. Suggested intern questions (quick answers to guide the write-up)
# ---------------------------------------------------------------
print(f"\n{'=' * 60}\nSUGGESTED INTERN QUESTIONS\n{'=' * 60}")

# Residential vs other property type share (from raw, pre-filter data)
res_share = sold_raw["PropertyType"].value_counts(normalize=True) * 100
print(f"\nProperty type share (SOLD, % of all types):\n{res_share.round(2)}")

# Median and average close price
print(f"\nSOLD ClosePrice - mean: {sold['ClosePrice'].mean():.2f}, median: {sold['ClosePrice'].median():.2f}")

# Days on Market distribution
print(f"\nSOLD DaysOnMarket describe():\n{sold['DaysOnMarket'].describe()}")

# % sold above vs below list price (requires both ClosePrice and ListPrice)
if "ListPrice" in sold.columns:
    above = (sold["ClosePrice"] > sold["ListPrice"]).mean() * 100
    below = (sold["ClosePrice"] < sold["ListPrice"]).mean() * 100
    at_list = (sold["ClosePrice"] == sold["ListPrice"]).mean() * 100
    print(f"\nSold above list price: {above:.1f}%  |  below list price: {below:.1f}%  |  at list price: {at_list:.1f}%")

# Date consistency: close date before listing date
if "CloseDate" in sold.columns and "ListingContractDate" in sold.columns:
    close_dt = pd.to_datetime(sold["CloseDate"], errors="coerce")
    list_dt = pd.to_datetime(sold["ListingContractDate"], errors="coerce")
    bad_dates = (close_dt < list_dt).sum()
    print(f"\nRecords where CloseDate is before ListingContractDate: {bad_dates}")

# Counties with highest median price
if "CountyOrParish" in sold.columns:
    county_median = sold.groupby("CountyOrParish")["ClosePrice"].median().sort_values(ascending=False)
    print(f"\nTop 10 counties by median ClosePrice:\n{county_median.head(10)}")

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---------------------------------------------------------------
# 0. Where to find the filtered datasets from the previous step
# ---------------------------------------------------------------
INPUT_DIR = r"C:\Users\ellen\OneDrive\Desktop\IDX Internship 2026\data raw"

sold = pd.read_csv(INPUT_DIR + r"\sold_residential_filtered.csv", low_memory=False)
listings = pd.read_csv(INPUT_DIR + r"\listings_residential_filtered.csv", low_memory=False)

print(f"Loaded SOLD: {len(sold)} rows")
print(f"Loaded LISTINGS: {len(listings)} rows")

# ---------------------------------------------------------------
# Step 1 - Fetch the mortgage rate data from FRED (no API key needed)
# ---------------------------------------------------------------
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=["observation_date"])
mortgage.columns = ["date", "rate_30yr_fixed"]

print(f"\nFetched {len(mortgage)} weekly mortgage rate observations from FRED")
print(mortgage.tail())

# ---------------------------------------------------------------
# Step 2 - Resample weekly rates to monthly averages
# ---------------------------------------------------------------
mortgage["year_month"] = mortgage["date"].dt.to_period("M")
mortgage_monthly = (
    mortgage.groupby("year_month")["rate_30yr_fixed"]
    .mean()
    .reset_index()
)

print(f"\nResampled to {len(mortgage_monthly)} monthly averages")
print(mortgage_monthly.tail())

# ---------------------------------------------------------------
# Step 3 - Create a matching year_month key on the MLS datasets
# ---------------------------------------------------------------
# Sold dataset - key off CloseDate
sold["year_month"] = pd.to_datetime(sold["CloseDate"], errors="coerce").dt.to_period("M")

# Listings dataset - key off ListingContractDate
listings["year_month"] = pd.to_datetime(
    listings["ListingContractDate"], errors="coerce"
).dt.to_period("M")

# ---------------------------------------------------------------
# Step 4 - Merge (left join keeps every MLS row)
# ---------------------------------------------------------------
sold_with_rates = sold.merge(mortgage_monthly, on="year_month", how="left")
listings_with_rates = listings.merge(mortgage_monthly, on="year_month", how="left")

# ---------------------------------------------------------------
# Step 5 - Validate the merge (rate should not be null)
# ---------------------------------------------------------------
sold_null_rates = sold_with_rates["rate_30yr_fixed"].isnull().sum()
listings_null_rates = listings_with_rates["rate_30yr_fixed"].isnull().sum()

print(f"\nSOLD: rows with missing rate_30yr_fixed after merge = {sold_null_rates}")
print(f"LISTINGS: rows with missing rate_30yr_fixed after merge = {listings_null_rates}")

if sold_null_rates > 0:
    # Rows with a missing/unparseable CloseDate, or a year_month outside the
    # range FRED returned, will not find a match - inspect these before
    # moving on.
    print("\nSOLD rows with null rate (check CloseDate / year_month):")
    print(sold_with_rates.loc[sold_with_rates["rate_30yr_fixed"].isnull(), ["CloseDate", "year_month"]].head(10))

if listings_null_rates > 0:
    print("\nLISTINGS rows with null rate (check ListingContractDate / year_month):")
    print(listings_with_rates.loc[listings_with_rates["rate_30yr_fixed"].isnull(), ["ListingContractDate", "year_month"]].head(10))

# ---------------------------------------------------------------
# Preview
# ---------------------------------------------------------------
print("\nSOLD preview:")
print(sold_with_rates[["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]].head())

print("\nLISTINGS preview:")
print(listings_with_rates[["ListingContractDate", "year_month", "ListPrice", "rate_30yr_fixed"]].head())

# ---------------------------------------------------------------
# Save enriched datasets
# ---------------------------------------------------------------
sold_with_rates.to_csv(INPUT_DIR + r"\sold_with_mortgage_rates.csv", index=False)
listings_with_rates.to_csv(INPUT_DIR + r"\listings_with_mortgage_rates.csv", index=False)

print("\nSaved: sold_with_mortgage_rates.csv")
print("Saved: listings_with_mortgage_rates.csv")
print("\nDone. Mortgage rate enrichment complete.")

print("\nDone. Structure, missing-value, and numeric distribution analysis complete.")
