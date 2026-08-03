# WEEK 6 - FEATURE ENGINEERING

sold["PriceRatio"] = (
    sold["ClosePrice"] /
    sold["OriginalListPrice"]
)

sold["CloseToOriginalListRatio"] = (
    sold["ClosePrice"] /
    sold["OriginalListPrice"]
)

sold["PricePerSqFt"] = (
    sold["ClosePrice"] /
    sold["LivingArea"]
)

sold["DaysOnMarketMetric"] = sold["DaysOnMarket"]

sold["Year"] = sold["CloseDate"].dt.year

sold["Month"] = sold["CloseDate"].dt.month

sold["YrMo"] = (
    sold["CloseDate"]
    .dt.to_period("M")
    .astype(str)
)

sold["ListingToContractDays"] = (
    sold["PurchaseContractDate"]
    - sold["ListingContractDate"]
).dt.days

sold["ContractToCloseDays"] = (
    sold["CloseDate"]
    - sold["PurchaseContractDate"]
).dt.days

# Property type summary

property_summary = (
    sold.groupby("PropertyType")
    .agg({
        "ClosePrice": ["mean", "median", "count"],
        "PricePerSqFt": "mean",
        "DaysOnMarketMetric": "mean",
        "PriceRatio": "mean"
    })
)

property_summary.to_csv(
    OUTPUT_DIR / "property_type_summary.csv"
)

# County summary

county_summary = (
    sold.groupby("CountyOrParish")
    .agg({
        "ClosePrice": ["mean", "median", "count"],
        "PricePerSqFt": "mean",
        "DaysOnMarketMetric": "mean",
        "PriceRatio": "mean"
    })
)

county_summary.to_csv(
    OUTPUT_DIR / "county_summary.csv"
)

print("\nCOUNTY SUMMARY")
print(county_summary)

# Listing office summary

listing_office_summary = (
    sold.groupby("ListOfficeName")
    .agg({
        "ClosePrice": ["mean", "count"],
        "PriceRatio": "mean",
        "DaysOnMarketMetric": "mean"
    })
)

listing_office_summary.to_csv(
    OUTPUT_DIR / "listing_office_summary.csv"
)

# Buyer office summary

buyer_office_summary = (
    sold.groupby("BuyerOfficeName")
    .agg({
        "ClosePrice": ["mean", "count"],
        "PriceRatio": "mean",
        "DaysOnMarketMetric": "mean"
    })
)

buyer_office_summary.to_csv(
    OUTPUT_DIR / "buyer_office_summary.csv"
)

sold.to_csv(
    OUTPUT_DIR / "sold_feature_engineered.csv",
    index=False
)

print("\nPROPERTY TYPE SUMMARY")
print(property_summary)
print("\nFEATURE ENGINEERING COMPLETE")
print("--------------------------------------------")
print(
    sold[
        [
            "ClosePrice",
            "OriginalListPrice",
            "PriceRatio",
            "CloseToOriginalListRatio",
            "LivingArea",
            "PricePerSqFt",
            "DaysOnMarketMetric",
            "Year",
            "Month",
            "YrMo",
            "ListingToContractDays",
            "ContractToCloseDays"
        ]
    ].head()
)
