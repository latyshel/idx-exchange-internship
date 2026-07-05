import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA = BASE_DIR / "data_cleaned"
OUTPUT = BASE_DIR / "week2_outputs"

OUTPUT.mkdir(exist_ok=True)


sold = pd.read_csv(CLEANED_DATA / "sold.csv", low_memory=False)
listings = pd.read_csv(CLEANED_DATA / "listings.csv", low_memory=False)

print("Sold dataset shape:", sold.shape)
print("Listings dataset shape:", listings.shape)


print("\nSold Property Types:")
print(sold["PropertyType"].value_counts(dropna=False))

print("\nListings Property Types:")
print(listings["PropertyType"].value_counts(dropna=False))


sold_before = len(sold)
listings_before = len(listings)

sold_residential = sold[sold["PropertyType"] == "Residential"].copy()
listings_residential = listings[listings["PropertyType"] == "Residential"].copy()

print("\nResidential Filter")
print("------------------")
print(f"Sold before filter: {sold_before}")
print(f"Sold after filter: {len(sold_residential)}")

print(f"Listings before filter: {listings_before}")
print(f"Listings after filter: {len(listings_residential)}")

def missing_value_report(df, dataset_name):
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_percent": (df.isnull().sum().values / len(df)) * 100
    })

    missing["over_90_percent_missing"] = missing["missing_percent"] > 90

    missing = missing.sort_values(by="missing_percent", ascending=False)

    missing.to_csv(OUTPUT / f"{dataset_name}_missing_value_report.csv", index=False)

    print(f"\n{dataset_name} columns over 90% missing:")
    print(missing[missing["over_90_percent_missing"] == True])

    return missing


sold_missing = missing_value_report(sold_residential, "sold")
listings_missing = missing_value_report(listings_residential, "listings")


numeric_fields = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]

available_numeric_fields = [
    col for col in numeric_fields if col in sold_residential.columns
]

numeric_summary = sold_residential[available_numeric_fields].describe(
    percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
)

numeric_summary.to_csv(OUTPUT / "sold_numeric_distribution_summary.csv")

print("\nNumeric Distribution Summary:")
print(numeric_summary)



if "ClosePrice" in sold_residential.columns:
    print("\nClose Price Summary")
    print("-------------------")
    print("Average Close Price:", sold_residential["ClosePrice"].mean())
    print("Median Close Price:", sold_residential["ClosePrice"].median())


if "DaysOnMarket" in sold_residential.columns:
    print("\nDays on Market Summary")
    print("----------------------")
    print(sold_residential["DaysOnMarket"].describe())


if "ClosePrice" in sold_residential.columns and "ListPrice" in sold_residential.columns:
    sold_residential["sold_above_list"] = sold_residential["ClosePrice"] > sold_residential["ListPrice"]
    sold_residential["sold_below_list"] = sold_residential["ClosePrice"] < sold_residential["ListPrice"]

    percent_above = sold_residential["sold_above_list"].mean() * 100
    percent_below = sold_residential["sold_below_list"].mean() * 100

    print("\nSale Price vs List Price")
    print("------------------------")
    print(f"Percent sold above list price: {percent_above:.2f}%")
    print(f"Percent sold below list price: {percent_below:.2f}%")

if "CloseDate" in sold_residential.columns and "ListingContractDate" in sold_residential.columns:
    sold_residential["CloseDate"] = pd.to_datetime(sold_residential["CloseDate"], errors="coerce")
    sold_residential["ListingContractDate"] = pd.to_datetime(
        sold_residential["ListingContractDate"], errors="coerce"
    )

    sold_residential["close_before_listing_flag"] = (
        sold_residential["CloseDate"] < sold_residential["ListingContractDate"]
    )

    print("\nDate Consistency Check")
    print("----------------------")
    print("Records where CloseDate is before ListingContractDate:")
    print(sold_residential["close_before_listing_flag"].sum())


if "CountyOrParish" in sold_residential.columns and "ClosePrice" in sold_residential.columns:
    county_prices = (
        sold_residential
        .groupby("CountyOrParish")["ClosePrice"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    county_prices.to_csv(OUTPUT / "county_median_close_prices.csv", index=False)

    print("\nTop Counties by Median Close Price:")
    print(county_prices.head(10))


sold_residential.to_csv(OUTPUT / "sold_residential_week2.csv", index=False)
listings_residential.to_csv(OUTPUT / "listings_residential_week2.csv", index=False)

print("\nWeek 2 validation complete.")
print("Files saved in week2_outputs folder.")