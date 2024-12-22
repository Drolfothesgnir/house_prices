import numpy as np


def augment_by_lot_frontage_missing(df):
    df["lot_frontage_missing"] = df["LotFrontage"].isna()
    return df


def augment_by_has_garage(df):
    df["has_garage"] = np.where(
        (df["GarageType"] == "NA") |
        (df["GarageCars"] == 0) |
        (df["GarageArea"] == 0), 0, 1)
    return df


def engineer_features(df):
    df = augment_by_lot_frontage_missing(df)
    df = augment_by_has_garage(df)
    return df
