import numpy as np


def augment_by_lot_frontage_missing(df):
    df["lot_frontage_missing"] = df["LotFrontage"].isna()
    return df


def augment_by_has_garage(df):
    df["has_garage"] = np.where(
        (df["GarageType"].astype(str) == "NA") |
        (df["GarageCars"] == 0) |
        (df["GarageArea"] == 0), 0, 1)
    return df


def augment_by_has_basement(df):
    df["has_basement"] = np.where(
        df["BsmtFinType1"].astype(str) == "NA", 0, 1
    )
    return df


def engineer_features(df):
    df = augment_by_lot_frontage_missing(df)
    df = augment_by_has_garage(df)
    df = augment_by_has_basement(df)

    return df
