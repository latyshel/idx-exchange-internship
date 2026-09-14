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

print("\nDone. Structure, missing-value, and numeric distribution analysis complete.")
