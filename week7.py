
outlier_fields = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]

# step 1

for field in outlier_fields:

    Q1 = sold[field].quantile(0.25)
    Q3 = sold[field].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    sold[f"{field}_outlier_flag"] = (
        (sold[field] < lower)
        | (sold[field] > upper)
    )

    print("\n" + "=" * 50)
    print(f"IQR OUTLIER ANALYSIS: {field}")
    print("=" * 50)
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower}")
    print(f"Upper Bound: {upper}")
    print(
        f"Number of outliers: "
        f"{sold[f'{field}_outlier_flag'].sum()}"
    )


# step 2

sold["invalid_close_price_flag"] = (
    sold["ClosePrice"] <= 0
)

sold["invalid_living_area_flag"] = (
    sold["LivingArea"] <= 0
)

sold["invalid_days_on_market_flag"] = (
    sold["DaysOnMarket"] < 0
)


# step 3

sold["any_outlier_flag"] = (
    sold["ClosePrice_outlier_flag"]
    | sold["LivingArea_outlier_flag"]
    | sold["DaysOnMarket_outlier_flag"]
    | sold["invalid_close_price_flag"]
    | sold["invalid_living_area_flag"]
    | sold["invalid_days_on_market_flag"]
)

print("\nOVERALL OUTLIER SUMMARY")
print("=" * 50)

print(
    "ClosePrice outliers:",
    sold["ClosePrice_outlier_flag"].sum()
)

print(
    "LivingArea outliers:",
    sold["LivingArea_outlier_flag"].sum()
)

print(
    "DaysOnMarket outliers:",
    sold["DaysOnMarket_outlier_flag"].sum()
)

print(
    "Invalid ClosePrice records:",
    sold["invalid_close_price_flag"].sum()
)

print(
    "Invalid LivingArea records:",
    sold["invalid_living_area_flag"].sum()
)

print(
    "Invalid DaysOnMarket records:",
    sold["invalid_days_on_market_flag"].sum()
)

print(
    "Total records flagged for filtering:",
    sold["any_outlier_flag"].sum()
)


# step 4

full_flagged = sold.copy()

full_flagged.to_csv(
    OUTPUT_DIR / "sold_full_flagged.csv",
    index=False
)

print("\nFull flagged dataset saved.")

# step 5

sold_clean = sold[
    ~sold["any_outlier_flag"]
].copy()

print("\nDATASET SIZE COMPARISON")
print("=" * 50)

print("Rows before filtering:", len(sold))
print("Rows after filtering:", len(sold_clean))
print("Rows removed:", len(sold) - len(sold_clean))


# step 6

print("\nMEDIAN COMPARISON")
print("=" * 50)

for field in outlier_fields:

    before_median = sold[field].median()
    after_median = sold_clean[field].median()

    print(f"\n{field}")
    print(f"Median before filtering: {before_median}")
    print(f"Median after filtering:  {after_median}")


sold_clean.to_csv(
    OUTPUT_DIR / "sold_clean_filtered.csv",
    index=False
)

print("\nClean filtered dataset saved.")
