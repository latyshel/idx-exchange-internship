# WEEK 6 - FEATURE ENGINEERING

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---------------------------------------------------------------
# 0. Load the cleaned datasets from Weeks 4-5
# ---------------------------------------------------------------
INPUT_DIR = r"C:\Users\ellen\OneDrive\Desktop\IDX Internship 2026\data raw"

sold = pd.read_csv(INPUT_DIR + r"\sold_cleaned.csv", low_memory=False)
listings = pd.read_csv(INPUT_DIR + r"\listings_cleaned.csv", low_memory=False)

# Re-parse date columns - CSV round-trips lose datetime dtype, so these
# need to be converted again after loading from disk.
DATE_COLUMNS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
for col in DATE_COLUMNS:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")
    if col in listings.columns:
        listings[col] = pd.to_datetime(listings[col], errors="coerce")

print(f"Loaded SOLD: {len(sold)} rows")
print(f"Loaded LISTINGS: {len(listings)} rows")


# ---------------------------------------------------------------
# 1. Feature engineering function - applied to SOLD
#    (Note: most of these metrics require ClosePrice, which is ~92%
#    missing in LISTINGS since most listings haven't closed yet. So the
#    price-based metrics are computed on SOLD; LISTINGS gets the
#    time-based fields that don't depend on ClosePrice.)
# ---------------------------------------------------------------

def engineer_price_metrics(df, label):
    """Price Ratio, Close-to-Original-List Ratio, Price Per Sq Ft."""
    # Price Ratio == Close-to-Original-List Ratio per the handbook's own
    # formula table (both defined as ClosePrice / OriginalListPrice).
    # Guard against divide-by-zero with OriginalListPrice <= 0.
    safe_original_list = df["OriginalListPrice"].where(df["OriginalListPrice"] > 0)
    df["price_ratio"] = df["ClosePrice"] / safe_original_list
    df["close_to_original_list_ratio"] = df["price_ratio"]  # identical formula, kept as its own named column per spec

    safe_living_area = df["LivingArea"].where(df["LivingArea"] > 0)
    df["price_per_sqft"] = df["ClosePrice"] / safe_living_area

    print(f"\n{label}: price metrics engineered")
    print(f"  price_ratio - non-null: {df['price_ratio'].notnull().sum()}, mean: {df['price_ratio'].mean():.3f}")
    print(f"  price_per_sqft - non-null: {df['price_per_sqft'].notnull().sum()}, mean: {df['price_per_sqft'].mean():.2f}")
    return df


def engineer_time_dimensions(df, label, date_col="CloseDate"):
    """Year / Month / YrMo derived from a date column."""
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["yr_mo"] = df[date_col].dt.to_period("M").astype(str)

    print(f"\n{label}: time dimensions engineered (from {date_col})")
    print(f"  year range: {df['year'].min()} - {df['year'].max()}")
    print(f"  yr_mo unique values: {df['yr_mo'].nunique()}")
    return df


def engineer_days_on_market(df, label):
    """Days on Market - carried through from the raw cleaned field."""
    df["days_on_market"] = df["DaysOnMarket"]
    print(f"\n{label}: days_on_market carried through - mean: {df['days_on_market'].mean():.1f}, median: {df['days_on_market'].median():.1f}")
    return df


def engineer_timeline_durations(df, label):
    """Listing to Contract Days, Contract to Close Days."""
    df["listing_to_contract_days"] = (
        df["PurchaseContractDate"] - df["ListingContractDate"]
    ).dt.days
    df["contract_to_close_days"] = (
        df["CloseDate"] - df["PurchaseContractDate"]
    ).dt.days

    print(f"\n{label}: timeline duration metrics engineered")
    print(
        f"  listing_to_contract_days - non-null: {df['listing_to_contract_days'].notnull().sum()}, "
        f"mean: {df['listing_to_contract_days'].mean():.1f}, median: {df['listing_to_contract_days'].median():.1f}"
    )
    print(
        f"  contract_to_close_days - non-null: {df['contract_to_close_days'].notnull().sum()}, "
        f"mean: {df['contract_to_close_days'].mean():.1f}, median: {df['contract_to_close_days'].median():.1f}"
    )
    return df


# ---------------------------------------------------------------
# 2. Apply feature engineering to SOLD (full metric set - has ClosePrice)
# ---------------------------------------------------------------
sold = engineer_price_metrics(sold, "SOLD")
sold = engineer_time_dimensions(sold, "SOLD", date_col="CloseDate")
sold = engineer_days_on_market(sold, "SOLD")
sold = engineer_timeline_durations(sold, "SOLD")

# ---------------------------------------------------------------
# 3. Apply feature engineering to LISTINGS
#    Time dimensions keyed off ListingContractDate since most listings
#    lack a CloseDate. Price metrics computed only where ClosePrice
#    exists (i.e. listings that have since sold).
# ---------------------------------------------------------------
listings = engineer_time_dimensions(listings, "LISTINGS", date_col="ListingContractDate")
listings = engineer_days_on_market(listings, "LISTINGS")

# Price metrics only meaningful for the ~8% of listings that have closed
listings_with_close = listings["ClosePrice"].notnull().sum()
print(f"\nLISTINGS: {listings_with_close} of {len(listings)} rows have a ClosePrice - price metrics computed for these only")
listings = engineer_price_metrics(listings, "LISTINGS")
listings = engineer_timeline_durations(listings, "LISTINGS")

# ---------------------------------------------------------------
# 4. Sample output table - confirms new columns populated correctly
# ---------------------------------------------------------------
SAMPLE_COLUMNS = [
    "CloseDate",
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "days_on_market",
    "year",
    "month",
    "yr_mo",
    "listing_to_contract_days",
    "contract_to_close_days",
]
print("\n" + "=" * 60)
print("SOLD: SAMPLE OUTPUT TABLE (engineered metrics)")
print("=" * 60)
print(sold[[c for c in SAMPLE_COLUMNS if c in sold.columns]].head(10))

# ---------------------------------------------------------------
# 5. Segment analysis - summary statistics by key dimensions
# ---------------------------------------------------------------

def segment_summary(df, group_col, label):
    """Summary stats (count, mean/median price, price ratio, PPSF, DOM) by segment."""
    summary = (
        df.groupby(group_col)
        .agg(
            record_count=("ClosePrice", "count"),
            median_close_price=("ClosePrice", "median"),
            mean_close_price=("ClosePrice", "mean"),
            median_price_ratio=("price_ratio", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_days_on_market=("days_on_market", "median"),
        )
        .sort_values("record_count", ascending=False)
    )
    print(f"\n{label}: segment summary by {group_col} (top 15 by record count)")
    print(summary.head(15))
    return summary


print("\n" + "=" * 60)
print("SEGMENT ANALYSIS")
print("=" * 60)

# Required: at least one segmented summary by PropertyType or CountyOrParish
county_summary = segment_summary(sold, "CountyOrParish", "SOLD")
property_subtype_summary = segment_summary(sold, "PropertySubType", "SOLD")

# Bonus segments mentioned in the handbook (only run if columns exist)
if "MLSAreaMajor" in sold.columns:
    mls_area_summary = segment_summary(sold, "MLSAreaMajor", "SOLD")

if "ListOfficeName" in sold.columns:
    list_office_summary = segment_summary(sold, "ListOfficeName", "SOLD")

if "BuyerOfficeName" in sold.columns:
    buyer_office_summary = segment_summary(sold, "BuyerOfficeName", "SOLD")

# ---------------------------------------------------------------
# 6. Save engineered datasets and segment summary tables
# ---------------------------------------------------------------
sold.to_csv(INPUT_DIR + r"\sold_with_metrics.csv", index=False)
listings.to_csv(INPUT_DIR + r"\listings_with_metrics.csv", index=False)
county_summary.to_csv(INPUT_DIR + r"\segment_summary_by_county.csv")
property_subtype_summary.to_csv(INPUT_DIR + r"\segment_summary_by_property_subtype.csv")

print("\nSaved: sold_with_metrics.csv")
print("Saved: listings_with_metrics.csv")
print("Saved: segment_summary_by_county.csv")
print("Saved: segment_summary_by_property_subtype.csv")
print("\nDone. Week 6 feature engineering and market metrics complete.")
