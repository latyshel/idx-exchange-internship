
import matplotlib.pyplot as plt
#step 1
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

Reads:
    sold_with_mortgage_rates.csv
    listings_with_mortgage_rates.csv
Saves:
    sold_cleaned.csv
    listings_cleaned.csv
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---------------------------------------------------------------
# 0. Load the enriched datasets from Weeks 2-3
# ---------------------------------------------------------------
INPUT_DIR = r"C:\Users\ellen\OneDrive\Desktop\IDX Internship 2026\data raw"

sold = pd.read_csv(INPUT_DIR + r"\sold_with_mortgage_rates.csv", low_memory=False)
listings = pd.read_csv(INPUT_DIR + r"\listings_with_mortgage_rates.csv", low_memory=False)

print(f"Loaded SOLD: {len(sold)} rows, {sold.shape[1]} columns")
print(f"Loaded LISTINGS: {len(listings)} rows, {listings.shape[1]} columns")

# ---------------------------------------------------------------
# 1. Convert date fields to datetime
#    Reason: dates were read in as plain strings/objects from CSV;
#    downstream date-math (consistency checks, joins, plotting) requires
#    real datetime dtype.
# ---------------------------------------------------------------
DATE_COLUMNS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

for col in DATE_COLUMNS:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")
    if col in listings.columns:
        listings[col] = pd.to_datetime(listings[col], errors="coerce")

print("\nDate columns converted to datetime:")
for col in DATE_COLUMNS:
    if col in sold.columns:
        print(f"  SOLD.{col}: dtype = {sold[col].dtype}")
    if col in listings.columns:
        print(f"  LISTINGS.{col}: dtype = {listings[col].dtype}")

# ---------------------------------------------------------------
# 2. Remove unnecessary / redundant columns
#    Reason: these were identified in Weeks 2-3 as 100% empty (no data
#    at all) or as duplicate ".1"-suffixed columns produced by the
#    original MLS export/join, which just mirror another column.
# ---------------------------------------------------------------
COLUMNS_TO_DROP = [
    # 100% empty in both datasets (confirmed in Weeks 2-3 missing value report)
    "TaxYear",
    "ElementarySchoolDistrict",
    "BusinessType",
    "CoveredSpaces",
    "MiddleOrJuniorSchoolDistrict",
    "TaxAnnualAmount",
    "AboveGradeFinishedArea",
    "FireplacesTotal",
]

# Duplicate ".1" columns only exist in the Listings export
LISTINGS_DUPLICATE_COLUMNS = [c for c in listings.columns if c.endswith(".1")]

sold_cols_before = sold.shape[1]
listings_cols_before = listings.shape[1]

sold = sold.drop(columns=[c for c in COLUMNS_TO_DROP if c in sold.columns])
listings = listings.drop(
    columns=[c for c in COLUMNS_TO_DROP if c in listings.columns] + LISTINGS_DUPLICATE_COLUMNS
)

print(f"\nSOLD: columns before drop = {sold_cols_before}, after drop = {sold.shape[1]}")
print(f"LISTINGS: columns before drop = {listings_cols_before}, after drop = {listings.shape[1]}")
print(f"LISTINGS: duplicate '.1' columns removed: {LISTINGS_DUPLICATE_COLUMNS}")

# ---------------------------------------------------------------
# 3. Ensure numeric fields are properly typed
#    Reason: some numeric-looking fields can be read as object/str if
#    the raw CSV had stray text or inconsistent formatting in any row.
#    Coercing forces true numeric dtype; anything unparseable becomes NaN
#    (captured, not silently dropped).
# ---------------------------------------------------------------
NUMERIC_COLUMNS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
]


def coerce_numeric(df, columns, label):
    for col in columns:
        if col in df.columns:
            before_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"  {label}.{col}: {before_dtype} -> {df[col].dtype}")
    return df


print("\nNumeric type enforcement:")
sold = coerce_numeric(sold, NUMERIC_COLUMNS, "SOLD")
listings = coerce_numeric(listings, NUMERIC_COLUMNS, "LISTINGS")

# ---------------------------------------------------------------
# 4. Flag invalid numeric values (do not silently drop - flag for
#    transparency, so analysts downstream can decide whether to
#    exclude flagged rows from specific analyses)
#    Reason: ClosePrice/LivingArea <= 0 and negative DaysOnMarket or
#    Bedrooms/Bathrooms are physically impossible and indicate data
#    entry errors, not real market values.
# ---------------------------------------------------------------

def flag_invalid_numeric(df, label):
    df["invalid_close_price_flag"] = df["ClosePrice"] <= 0 if "ClosePrice" in df.columns else False
    df["invalid_living_area_flag"] = df["LivingArea"] <= 0 if "LivingArea" in df.columns else False
    df["invalid_days_on_market_flag"] = df["DaysOnMarket"] < 0 if "DaysOnMarket" in df.columns else False
    df["invalid_bedrooms_flag"] = df["BedroomsTotal"] < 0 if "BedroomsTotal" in df.columns else False
    df["invalid_bathrooms_flag"] = (
        df["BathroomsTotalInteger"] < 0 if "BathroomsTotalInteger" in df.columns else False
    )

    print(f"\n{label}: invalid numeric value flag counts")
    for flag in [
        "invalid_close_price_flag",
        "invalid_living_area_flag",
        "invalid_days_on_market_flag",
        "invalid_bedrooms_flag",
        "invalid_bathrooms_flag",
    ]:
        print(f"  {flag}: {df[flag].sum()}")
    return df


sold = flag_invalid_numeric(sold, "SOLD")
listings = flag_invalid_numeric(listings, "LISTINGS")

# ---------------------------------------------------------------
# 5. Handle missing values
#    Reason: core analytical fields (price, area, bed/bath, dates) are
#    retained even if partially missing, per the handbook's guidance -
#    dropping rows would bias the dataset. Instead we flag missingness
#    on the fields that matter most for analysis, and leave the actual
#    NaNs in place so downstream aggregations (mean/median) naturally
#    exclude them via pandas' default NaN handling.
# ---------------------------------------------------------------
CORE_FIELDS_FOR_MISSINGNESS_FLAG = [
    "ClosePrice",
    "ListPrice",
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
]


def flag_missing_core_fields(df, label):
    for col in CORE_FIELDS_FOR_MISSINGNESS_FLAG:
        if col in df.columns:
            flag_col = f"{col}_missing_flag"
            df[flag_col] = df[col].isnull()
    missing_summary = {
        col: df[col].isnull().sum()
        for col in CORE_FIELDS_FOR_MISSINGNESS_FLAG
        if col in df.columns
    }
    print(f"\n{label}: missing value counts on core fields (retained, not dropped)")
    for col, count in missing_summary.items():
        print(f"  {col}: {count} missing ({count / len(df):.1%})")
    return df


sold = flag_missing_core_fields(sold, "SOLD")
listings = flag_missing_core_fields(listings, "LISTINGS")

# ---------------------------------------------------------------
# 6. Date consistency checks
#    Reason: logically, a listing must go live before a buyer's offer
#    is accepted, and the offer must be accepted before the sale
#    closes. Violations indicate data entry errors in the MLS source.
# ---------------------------------------------------------------

def add_date_consistency_flags(df, label):
    has_listing = "ListingContractDate" in df.columns
    has_purchase = "PurchaseContractDate" in df.columns
    has_close = "CloseDate" in df.columns

    if has_listing and has_close:
        df["listing_after_close_flag"] = df["ListingContractDate"] > df["CloseDate"]
    else:
        df["listing_after_close_flag"] = False

    if has_purchase and has_close:
        df["purchase_after_close_flag"] = df["PurchaseContractDate"] > df["CloseDate"]
    else:
        df["purchase_after_close_flag"] = False

    # negative_timeline_flag: any violation of the expected order
    # ListingContractDate -> PurchaseContractDate -> CloseDate
    negative_timeline = pd.Series(False, index=df.index)
    if has_listing and has_purchase:
        negative_timeline |= df["ListingContractDate"] > df["PurchaseContractDate"]
    negative_timeline |= df["listing_after_close_flag"]
    negative_timeline |= df["purchase_after_close_flag"]
    df["negative_timeline_flag"] = negative_timeline

    print(f"\n{label}: date consistency flag counts")
    print(f"  listing_after_close_flag: {df['listing_after_close_flag'].sum()}")
    print(f"  purchase_after_close_flag: {df['purchase_after_close_flag'].sum()}")
    print(f"  negative_timeline_flag: {df['negative_timeline_flag'].sum()}")
    return df


sold = add_date_consistency_flags(sold, "SOLD")
listings = add_date_consistency_flags(listings, "LISTINGS")

# ---------------------------------------------------------------
# 7. Geographic data checks
#    Reason: missing, zero, or wrong-sign coordinates make a record
#    unusable for mapping/geospatial analysis. California longitudes
#    are always negative (west of the prime meridian); a positive
#    longitude signals a sign error or bad geocode. Rough CA bounding
#    box catches other implausible coordinates (e.g. from a different
#    state or a geocoding failure).
# ---------------------------------------------------------------
# Approximate California bounding box
CA_LAT_MIN, CA_LAT_MAX = 32.5, 42.0
CA_LON_MIN, CA_LON_MAX = -124.5, -114.0


def add_geo_flags(df, label):
    has_lat = "Latitude" in df.columns
    has_lon = "Longitude" in df.columns

    if not (has_lat and has_lon):
        print(f"\n{label}: Latitude/Longitude columns not found - skipping geo checks")
        return df

    df["missing_coordinates_flag"] = df["Latitude"].isnull() | df["Longitude"].isnull()
    df["zero_coordinates_flag"] = (df["Latitude"] == 0) | (df["Longitude"] == 0)
    df["positive_longitude_flag"] = df["Longitude"] > 0

    within_ca_box = (
        df["Latitude"].between(CA_LAT_MIN, CA_LAT_MAX)
        & df["Longitude"].between(CA_LON_MIN, CA_LON_MAX)
    )
    # Only flag as implausible if coordinates exist and aren't already
    # caught by the missing/zero flags above
    df["implausible_coordinates_flag"] = (
        ~within_ca_box & ~df["missing_coordinates_flag"] & ~df["zero_coordinates_flag"]
    )

    print(f"\n{label}: geographic data quality summary")
    print(f"  missing_coordinates_flag: {df['missing_coordinates_flag'].sum()}")
    print(f"  zero_coordinates_flag: {df['zero_coordinates_flag'].sum()}")
    print(f"  positive_longitude_flag: {df['positive_longitude_flag'].sum()}")
    print(f"  implausible_coordinates_flag (outside CA bounding box): {df['implausible_coordinates_flag'].sum()}")

    total_flagged = (
        df["missing_coordinates_flag"]
        | df["zero_coordinates_flag"]
        | df["positive_longitude_flag"]
        | df["implausible_coordinates_flag"]
    ).sum()
    print(f"  TOTAL records with at least one geo issue: {total_flagged} ({total_flagged / len(df):.1%})")
    return df


sold = add_geo_flags(sold, "SOLD")
listings = add_geo_flags(listings, "LISTINGS")

# ---------------------------------------------------------------
# 8. Final row count confirmation
#    Reason: this cleaning pass flags issues rather than deleting rows,
#    so row counts should be unchanged from the enriched input. This
#    confirms no accidental row loss occurred during cleaning.
# ---------------------------------------------------------------
print(f"\nSOLD: row count unchanged through cleaning = {len(sold)} (matches enriched input)")
print(f"LISTINGS: row count unchanged through cleaning = {len(listings)} (matches enriched input)")

print("\nFinal dtype confirmation - SOLD date columns:")
for col in DATE_COLUMNS:
    if col in sold.columns:
        print(f"  {col}: {sold[col].dtype}")

print("\nFinal dtype confirmation - SOLD numeric columns:")
for col in NUMERIC_COLUMNS:
    if col in sold.columns:
        print(f"  {col}: {sold[col].dtype}")

# ---------------------------------------------------------------
# 9. Save cleaned, analysis-ready datasets
# ---------------------------------------------------------------
sold.to_csv(INPUT_DIR + r"\sold_cleaned.csv", index=False)
listings.to_csv(INPUT_DIR + r"\listings_cleaned.csv", index=False)

print("\nSaved: sold_cleaned.csv")
print("Saved: listings_cleaned.csv")
print("\nDone. Weeks 4-5 cleaning and preparation complete.")
