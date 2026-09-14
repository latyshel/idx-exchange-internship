import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# ---------------------------------------------------------------
# 0. Load the metrics-enriched Sold dataset from Week 6
# ---------------------------------------------------------------
INPUT_DIR = r"C:\Users\ellen\OneDrive\Desktop\IDX Internship 2026\data raw"

sold = pd.read_csv(INPUT_DIR + r"\sold_with_metrics.csv", low_memory=False)
print(f"Loaded SOLD: {len(sold)} rows")

FIELDS_FOR_OUTLIER_CHECK = ["ClosePrice", "LivingArea", "DaysOnMarket"]

# ---------------------------------------------------------------
# 1. Tier 1 - Business rule flags (always invalid, independent of IQR)
#    Reason: a $0 or negative ClosePrice, non-positive LivingArea, or
#    negative DaysOnMarket is not a statistical outlier - it's an
#    impossible value. These were already flagged in Weeks 4-5
#    (invalid_close_price_flag, invalid_living_area_flag,
#    invalid_days_on_market_flag); re-confirmed here as Tier 1 for
#    completeness in case this script is run standalone.
# ---------------------------------------------------------------
sold["invalid_close_price_flag"] = sold["ClosePrice"] <= 0
sold["invalid_living_area_flag"] = sold["LivingArea"] <= 0
sold["invalid_days_on_market_flag"] = sold["DaysOnMarket"] < 0

print("\nTier 1 - business rule flags (always invalid):")
print(f"  invalid_close_price_flag: {sold['invalid_close_price_flag'].sum()}")
print(f"  invalid_living_area_flag: {sold['invalid_living_area_flag'].sum()}")
print(f"  invalid_days_on_market_flag: {sold['invalid_days_on_market_flag'].sum()}")

# ---------------------------------------------------------------
# 2. Tier 2 - IQR-based statistical outlier flags
#    Reason: values that are technically valid (positive price, positive
#    area) can still be so extreme relative to the rest of the market
#    that they distort averages - e.g. a $50M estate or a $10K
#    distressed sale. IQR is computed on business-rule-valid rows only,
#    so a handful of $0 records don't distort the quartiles themselves.
# ---------------------------------------------------------------

def compute_iqr_bounds(series):
    """Returns (lower_bound, upper_bound, Q1, Q3, IQR) for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper, q1, q3, iqr


print("\nTier 2 - IQR outlier detection:")

iqr_bounds = {}
for field in FIELDS_FOR_OUTLIER_CHECK:
    business_rule_flag_col = {
        "ClosePrice": "invalid_close_price_flag",
        "LivingArea": "invalid_living_area_flag",
        "DaysOnMarket": "invalid_days_on_market_flag",
    }[field]

    # Compute IQR only on rows that pass the Tier 1 business rule check
    valid_values = sold.loc[~sold[business_rule_flag_col], field].dropna()
    lower, upper, q1, q3, iqr = compute_iqr_bounds(valid_values)
    iqr_bounds[field] = (lower, upper)

    flag_col = f"{field.lower()}_iqr_outlier_flag" if field != "ClosePrice" else "closeprice_iqr_outlier_flag"
    flag_col = f"{field}_iqr_outlier_flag"
    sold[flag_col] = (sold[field] < lower) | (sold[field] > upper)

    print(f"\n  {field}:")
    print(f"    Q1 = {q1:,.2f}, Q3 = {q3:,.2f}, IQR = {iqr:,.2f}")
    print(f"    lower bound = {lower:,.2f}, upper bound = {upper:,.2f}")
    print(f"    {flag_col}: {sold[flag_col].sum()} records flagged")

# ---------------------------------------------------------------
# 3. Combined outlier flag - any business rule OR IQR violation
# ---------------------------------------------------------------
sold["any_outlier_flag"] = (
    sold["invalid_close_price_flag"]
    | sold["invalid_living_area_flag"]
    | sold["invalid_days_on_market_flag"]
    | sold["ClosePrice_iqr_outlier_flag"]
    | sold["LivingArea_iqr_outlier_flag"]
    | sold["DaysOnMarket_iqr_outlier_flag"]
)

print(f"\nany_outlier_flag: {sold['any_outlier_flag'].sum()} of {len(sold)} records flagged on at least one criterion")
print(f"  ({sold['any_outlier_flag'].sum() / len(sold):.1%} of dataset)")

# ---------------------------------------------------------------
# 4. Before / after comparison - size and median values
# ---------------------------------------------------------------
sold_filtered = sold[~sold["any_outlier_flag"]].copy()

print("\n" + "=" * 60)
print("BEFORE vs AFTER FILTERING COMPARISON")
print("=" * 60)

print(f"\nRow count:")
print(f"  Full (flagged) dataset:     {len(sold)}")
print(f"  Filtered (clean) dataset:   {len(sold_filtered)}")
print(f"  Rows removed:               {len(sold) - len(sold_filtered)} ({(len(sold) - len(sold_filtered)) / len(sold):.1%})")

print(f"\nMedian values - before vs after:")
for field in FIELDS_FOR_OUTLIER_CHECK:
    median_before = sold[field].median()
    median_after = sold_filtered[field].median()
    pct_change = (median_after - median_before) / median_before * 100 if median_before else float("nan")
    print(f"  {field}: before = {median_before:,.2f}, after = {median_after:,.2f} ({pct_change:+.2f}% change)")

print(f"\nMean values - before vs after (means are more sensitive to outliers than medians):")
for field in FIELDS_FOR_OUTLIER_CHECK:
    mean_before = sold[field].mean()
    mean_after = sold_filtered[field].mean()
    pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before else float("nan")
    print(f"  {field}: before = {mean_before:,.2f}, after = {mean_after:,.2f} ({pct_change:+.2f}% change)")

# ---------------------------------------------------------------
# 5. Save both the full flagged dataset and the filtered clean dataset
# ---------------------------------------------------------------
sold.to_csv(INPUT_DIR + r"\sold_flagged_full.csv", index=False)
sold_filtered.to_csv(INPUT_DIR + r"\sold_filtered_clean.csv", index=False)

print("\nSaved: sold_flagged_full.csv (all records preserved, flags added)")
print("Saved: sold_filtered_clean.csv (outlier records excluded)")
print("\nDone. Week 7 outlier detection and data quality complete.")
