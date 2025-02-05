import numpy as np
import pandas as pd


def augment_by_lot_frontage_missing(df):
    df["lot_frontage_missing"] = df["LotFrontage"].isna()
    return df


def augment_by_has_garage(df):
    df["has_garage"] = np.where(
        (df["GarageType"].astype(str) == "NA")
        | (df["GarageCars"] == 0)
        | (df["GarageArea"] == 0),
        0,
        1,
    )
    return df


def augment_by_has_basement(df):
    df["has_basement"] = np.where(df["BsmtFinType1"].astype(str) == "NA", 0, 1)
    return df


def augment_by_has_2nd_floor(df):
    df["has_2nd_floor"] = np.where(df["2ndFlrSF"] < 0.1, 0, 1)
    return df


def augment_by_has_wood_deck(df):
    df["has_wood_deck"] = np.where(df["WoodDeckSF"] < 0.1, 0, 1)
    return df


def augment_by_has_open_porch(df):
    df["has_open_porch"] = np.where(df["OpenPorchSF"] < 0.1, 0, 1)
    return df


def augment_by_has_enclosed_porch(df):
    df["has_enclosed_porch"] = np.where(df["EnclosedPorch"] < 0.1, 0, 1)
    return df


def augment_by_has_3sn_porch(df):
    df["has_3sn_porch"] = np.where(df["3SsnPorch"] < 0.1, 0, 1)
    return df


def augment_by_has_screen_porch(df):
    df["has_screen_porch"] = np.where(df["ScreenPorch"] < 0.1, 0, 1)
    return df


def augment_by_has_pool(df):
    df["has_pool"] = np.where(
        (df["PoolArea"] < 0.1) | (df["PoolQC"].astype(str) == "NA"), 0, 1
    )
    return df


def augment_by_has_fence(df):
    df["has_fence"] = np.where(df["Fence"].astype(str) == "NA", 0, 1)
    return df


def augment_by_has_misc_feature(df):
    df["has_misc_feature"] = np.where(df["MiscFeature"].astype(str) == "NA", 0, 1)
    return df


def augment_by_grouped_overall_qual_cond(df):
    bins = [0, 4, 7, 10]
    labels = ["Low", "Average", "High"]
    df["grouped_qual"] = pd.cut(
        df["OverallQual"], bins=bins, labels=labels, include_lowest=True
    )

    df["grouped_cond"] = pd.cut(
        df["OverallCond"], bins=bins, labels=labels, include_lowest=True
    )
    return df


def augment_by_log_sale_price(df):
    df["log_sale_price"] = np.log(df["SalePrice"])
    return df


def create_groups_from_series(series):
    def f(cond):
        if cond in ["NA", "Po", "Fa"]:
            return "Low"
        if cond == "TA":
            return "Average"
        if cond in ["Gd", "Ex"]:
            return "High"
        return cond

    result = series.map(f)
    return result


def augment_by_grouped_cond(df):
    cond_vars = [
        "ExterCond",
        "BsmtCond",
        "GarageCond",
    ]

    cat_dtype = pd.CategoricalDtype(categories=["Low", "Average", "High"], ordered=True)

    for var in cond_vars:

        df[f"{var}_grouped"] = create_groups_from_series(df[var]).astype(cat_dtype)

    return df


def augment_by_grouped_qual(df):
    qual_vars = [
        "ExterQual",
        "BsmtQual",
        "KitchenQual",
        "GarageQual",
        "HeatingQC",
        "FireplaceQu",
        "PoolQC",
    ]
    cat_dtype = pd.CategoricalDtype(categories=["Low", "Average", "High"], ordered=True)
    for var in qual_vars:

        df[f"{var}_grouped"] = create_groups_from_series(df[var]).astype(cat_dtype)

    return df


def engineer_features(df):
    df = augment_by_lot_frontage_missing(df)
    df = augment_by_has_garage(df)
    df = augment_by_has_basement(df)
    df = augment_by_has_2nd_floor(df)
    df = augment_by_has_wood_deck(df)
    df = augment_by_has_open_porch(df)
    df = augment_by_has_enclosed_porch(df)
    df = augment_by_has_3sn_porch(df)
    df = augment_by_has_screen_porch(df)
    df = augment_by_has_pool(df)
    df = augment_by_has_misc_feature(df)
    df = augment_by_log_sale_price(df)
    df = augment_by_grouped_overall_qual_cond(df)
    df = augment_by_grouped_cond(df)
    df = augment_by_grouped_qual(df)

    return df
