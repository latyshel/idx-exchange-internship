import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
#step 1
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

sold_file = INPUT_DIR / "sold.csv"
listings_file = INPUT_DIR / "listings.csv"

sold = pd.read_csv(sold_file, low_memory=False)
listings = pd.read_csv(listings_file, low_memory=False)

print("\nDATASET SHAPES")
print("--------------------------------------------------")
print(f"Sold dataset: {sold.shape[0]} rows and {sold.shape[1]} columns")
print(
    f"Listings dataset: {listings.shape[0]} rows "
    f"and {listings.shape[1]} columns"
)


print("\nUNIQUE SOLD PROPERTY TYPES")
print("--------------------------------------------------")
print(sold["PropertyType"].value_counts(dropna=False))

print("\nUNIQUE LISTING PROPERTY TYPES")
print("--------------------------------------------------")
print(listings["PropertyType"].value_counts(dropna=False))


sold_before_filter = len(sold)
listings_before_filter = len(listings)

sold = sold[sold["PropertyType"] == "Residential"].copy()
listings = listings[
    listings["PropertyType"] == "Residential"
].copy()

print("\nRESIDENTIAL FILTER")
print("--------------------------------------------------")
print(f"Sold rows before filter: {sold_before_filter}")
print(f"Sold rows after filter: {len(sold)}")

print(f"Listings rows before filter: {listings_before_filter}")
print(f"Listings rows after filter: {len(listings)}")

def create_missing_report(dataframe, dataset_name):
    """
    Create a table showing the number and percentage of missing
    values in every column.

    Columns with more than 90% missing values are flagged.
    """

    missing_count = dataframe.isnull().sum()
    missing_percent = (
        missing_count / len(dataframe) * 100
    )

    report = pd.DataFrame({
        "column": dataframe.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    })

    report["over_90_percent_missing"] = (
        report["missing_percent"] > 90
    )

    report = report.sort_values(
        by="missing_percent",
        ascending=False
    )

    output_file = (
        OUTPUT_DIR / f"{dataset_name}_missing_value_report.csv"
    )

    report.to_csv(output_file, index=False)

    print(f"\n{dataset_name.upper()} COLUMNS OVER 90% MISSING")
    print("--------------------------------------------------")

    high_missing = report[
        report["over_90_percent_missing"]
    ]

    if high_missing.empty:
        print("No columns are more than 90% missing.")
    else:
        print(
            high_missing[
                [
                    "column",
                    "missing_count",
                    "missing_percent"
                ]
            ]
        )

    return report


sold_missing_report = create_missing_report(sold, "sold")
listings_missing_report = create_missing_report(
    listings,
    "listings"
)

numeric_columns = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]

available_numeric_columns = [
    column
    for column in numeric_columns
    if column in sold.columns
]
for column in available_numeric_columns:
    sold[column] = pd.to_numeric(
        sold[column],
        errors="coerce"
    )

numeric_summary = sold[
    available_numeric_columns
].describe(
    percentiles=[
        0.01,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]
)

numeric_summary.loc["median"] = sold[
    available_numeric_columns
].median()

numeric_summary.to_csv(
    OUTPUT_DIR / "sold_numeric_distribution_summary.csv"
)

print("\nNUMERIC DISTRIBUTION SUMMARY")
print("--------------------------------------------------")
print(numeric_summary)


numeric_fields = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

GRAPH_DIR = OUTPUT_DIR / "distribution_graphs"
GRAPH_DIR.mkdir(exist_ok=True)


available_numeric_fields = [
    field for field in numeric_fields
    if field in sold.columns
]

for field in available_numeric_fields:
    sold[field] = pd.to_numeric(
        sold[field],
        errors="coerce"
    )

for field in available_numeric_fields:

   
    field_data = sold[field].dropna()

    if field_data.empty:
        print(f"No valid data available for {field}.")
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(
        field_data,
        bins=30,
        edgecolor="black"
    )

    plt.title(f"Distribution of {field}")
    plt.xlabel(field)
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{field}_histogram.png"
    )

    plt.close()


    plt.figure(figsize=(8, 3))

    plt.boxplot(
        field_data,
        vert=False
    )

    plt.title(f"Boxplot of {field}")
    plt.xlabel(field)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{field}_boxplot.png"
    )

    plt.close()

    percentile_summary = field_data.describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )

    percentile_summary.to_csv(
        OUTPUT_DIR / f"{field}_percentile_summary.csv"
    )


    q1 = field_data.quantile(0.25)
    q3 = field_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_count = (
        (field_data < lower_bound)
        | (field_data > upper_bound)
    ).sum()

    print(f"\n{field}")
    print("-" * 40)
    print(f"Minimum: {field_data.min()}")
    print(f"Maximum: {field_data.max()}")
    print(f"Mean: {field_data.mean()}")
    print(f"Median: {field_data.median()}")
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"IQR: {iqr}")
    print(f"Lower outlier boundary: {lower_bound}")
    print(f"Upper outlier boundary: {upper_bound}")
    print(f"Number of possible outliers: {outlier_count}")

print("\nDistribution graphs have been saved.")

fred_url = (
    "https://fred.stlouisfed.org/graph/"
    "fredgraph.csv?id=MORTGAGE30US"
)

print("\nDownloading mortgage-rate data from FRED...")

mortgage = pd.read_csv(
    fred_url,
    parse_dates=["observation_date"]
)

mortgage.columns = [
    "date",
    "rate_30yr_fixed"
]

mortgage["rate_30yr_fixed"] = pd.to_numeric(
    mortgage["rate_30yr_fixed"],
    errors="coerce"
)

mortgage = mortgage.dropna(
    subset=["rate_30yr_fixed"]
)

print(f"Downloaded {len(mortgage)} weekly rate observations.")


mortgage["year_month"] = (
    mortgage["date"].dt.to_period("M")
)

mortgage_monthly = (
    mortgage
    .groupby("year_month", as_index=False)[
        "rate_30yr_fixed"
    ]
    .mean()
)

mortgage_monthly.to_csv(
    OUTPUT_DIR / "mortgage_rates_monthly.csv",
    index=False
)

print("\nMONTHLY MORTGAGE-RATE PREVIEW")
print("--------------------------------------------------")
print(mortgage_monthly.tail())


sold["CloseDate"] = pd.to_datetime(
    sold["CloseDate"],
    errors="coerce"
)

sold["year_month"] = (
    sold["CloseDate"].dt.to_period("M")
)

listings["ListingContractDate"] = pd.to_datetime(
    listings["ListingContractDate"],
    errors="coerce"
)

listings["year_month"] = (
    listings["ListingContractDate"].dt.to_period("M")
)

sold_with_rates = sold.merge(
    mortgage_monthly,
    on="year_month",
    how="left"
)

listings_with_rates = listings.merge(
    mortgage_monthly,
    on="year_month",
    how="left"
)

sold_null_rates = (
    sold_with_rates["rate_30yr_fixed"]
    .isnull()
    .sum()
)

listings_null_rates = (
    listings_with_rates["rate_30yr_fixed"]
    .isnull()
    .sum()
)

print("\nMORTGAGE-RATE MERGE VALIDATION")
print("--------------------------------------------------")
print(
    f"Sold records with missing mortgage rates: "
    f"{sold_null_rates}"
)
print(
    f"Listing records with missing mortgage rates: "
    f"{listings_null_rates}"
)

if sold_null_rates == 0:
    print("Sold merge validation passed.")
else:
    print(
        "Warning: Some sold records did not match "
        "a mortgage-rate month."
    )

if listings_null_rates == 0:
    print("Listings merge validation passed.")
else:
    print(
        "Warning: Some listing records did not match "
        "a mortgage-rate month."
    )

sold_preview_columns = [
    column
    for column in [
        "CloseDate",
        "year_month",
        "ClosePrice",
        "rate_30yr_fixed"
    ]
    if column in sold_with_rates.columns
]

listings_preview_columns = [
    column
    for column in [
        "ListingContractDate",
        "year_month",
        "ListPrice",
        "rate_30yr_fixed"
    ]
    if column in listings_with_rates.columns
]

print("\nSOLD DATA PREVIEW")
print("--------------------------------------------------")
print(
    sold_with_rates[
        sold_preview_columns
    ].head()
)

print("\nLISTINGS DATA PREVIEW")
print("--------------------------------------------------")
print(
    listings_with_rates[
        listings_preview_columns
    ].head()
)


sold_with_rates.to_csv(
    OUTPUT_DIR / "sold_residential_with_rates.csv",
    index=False
)

listings_with_rates.to_csv(
    OUTPUT_DIR / "listings_residential_with_rates.csv",
    index=False
)

print("\nWEEKS 2–3 WORK COMPLETE")
print("--------------------------------------------------")
print("Saved locally:")
print("1. sold_residential_with_rates.csv")
print("2. listings_residential_with_rates.csv")
print("3. sold_missing_value_report.csv")
print("4. listings_missing_value_report.csv")
print("5. sold_numeric_distribution_summary.csv")
print("6. mortgage_rates_monthly.csv")
