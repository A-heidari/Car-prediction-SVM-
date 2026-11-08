def prepare_feature(df):

    # ----------------------------------
    # Convert price to numerical
    # ----------------------------------

    df["price"] = (
        df["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # ----------------------------------
    # Convert milage to numerical
    # ----------------------------------

    df["milage"] = (
        df["milage"]
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

    # ----------------------------------
    # Car age
    # ----------------------------------

    current_year = 2026

    df["car_age"] = current_year - df["model_year"]

    # ----------------------------------
    # Mileage per year
    # ----------------------------------

    df["milage_per_year"] = (
        df["milage"] / df["car_age"].replace(0, 1)
    )

    # ----------------------------------
    # Engine features
    # ----------------------------------

    # Horsepower
    df["horsepower"] = (
        df["engine"]
        .str.extract(r"(\d+(?:\.\d+)?)HP")[0]
        .astype(float)
    )

    # Engine liters
    df["engine_liters"] = (
        df["engine"]
        .str.extract(r"(\d+(?:\.\d+)?)L")[0]
        .astype(float)
    )

    # Cylinders
    cylinders_v = (
        df["engine"]
        .str.extract(r"[VI](\d+)")[0]
    )

    cylinders_text = (
        df["engine"]
        .str.extract(r"(\d+)\s*Cylinder")[0]
    )

    df["cylinders"] = (
        cylinders_v
        .fillna(cylinders_text)
        .astype(float)
    )

    # ----------------------------------
    # Delete unnecessary features
    # ----------------------------------

    df = df.drop(
        columns=[
            "model",
            "engine",
            "ext_col",
            "int_col",
            "clean_title",
        ]
    )

    return df
