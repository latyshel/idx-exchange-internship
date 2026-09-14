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
