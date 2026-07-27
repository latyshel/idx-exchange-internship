# week 5

sold["missing_coordinates_flag"] = (
    sold["Latitude"].isnull()
    | sold["Longitude"].isnull()
)

sold["zero_coordinates_flag"] = (
    (sold["Latitude"] == 0)
    | (sold["Longitude"] == 0)
)

sold["positive_longitude_flag"] = (
    sold["Longitude"] > 0
)

# Approximate California boundaries:
# Latitude: 32° to 42°
# Longitude: -125° to -114°

sold["invalid_coordinate_range_flag"] = (
    (sold["Latitude"] < 32)
    | (sold["Latitude"] > 42)
    | (sold["Longitude"] < -125)
    | (sold["Longitude"] > -114)
)

missing_coordinates_count = sold[
    "missing_coordinates_flag"
].sum()

zero_coordinates_count = sold[
    "zero_coordinates_flag"
].sum()

positive_longitude_count = sold[
    "positive_longitude_flag"
].sum()

invalid_coordinate_range_count = sold[
    "invalid_coordinate_range_flag"
].sum()

print("\nGEOGRAPHIC DATA QUALITY SUMMARY")
print("--------------------------------------------------")
print(
    "Missing coordinate records:",
    missing_coordinates_count
)
print(
    "Zero coordinate records:",
    zero_coordinates_count
)
print(
    "Positive longitude records:",
    positive_longitude_count
)
print(
    "Out-of-state/invalid coordinate records:",
    invalid_coordinate_range_count
)
