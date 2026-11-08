from src.data_loader import load_data
from src.feature_engineering import prepare_feature

df = load_data("Cars.csv")
df = prepare_feature(df)


print(
    df[
        [
            "engine",
            "horsepower",
            "engine_liters",
            "cylinders",
            "price"
        ]
    ].head(20)
)

print("\nMissing values in engineered features: ")

print(
    df[
        [
            "horsepower",
            "engine_liters",
            "cylinders"
        ]
    ].isna().sum()
)

print("\nMissing values:")
print(
    df[
        [
            "horsepower",
            "engine_liters",
            "cylinders"
        ]
    ].isna().sum()
)

print("\nMissing percentage:")
print(
    (
        df[
            [
                "horsepower",
                "engine_liters",
                "cylinders"
            ]
        ].isna().mean() * 100
    ).round(2)
)

