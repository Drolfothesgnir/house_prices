import os
import sys
from pathlib import Path

# Get the notebook's directory
notebook_dir = os.getcwd()

# Add the notebook directory and its parent to Python path
sys.path.append(notebook_dir)
sys.path.append(str(Path(notebook_dir).parent))


# Function to add all subdirectories to Python path
def add_subdirs_to_path():
    for root, dirs, files in os.walk(notebook_dir):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if full_path not in sys.path:
                sys.path.append(full_path)


# Add all subdirectories
add_subdirs_to_path()


# Function to get absolute path for any relative path
def get_abs_path(relative_path):
    return str(Path(relative_path).absolute())


from .data_consistency_check import check_data_consistency
from .LotFrontage import impute_lot_frontage


def fix_electrical_1380(df):
    # imputation of single Electrical value
    mask = (df["Id"] == 1380) & (df["Electrical"].isna())
    if mask.any():
        df.loc[mask, "Electrical"] = "SBrkr"

    return df


def fix_garage_type(df):
    # there is one suspicious garage in the test set, which has type detached and no other info
    # I'll convert it into an NA garage type
    qry = (df["has_garage"] == 0) & (df["GarageType"].astype(str) != "NA")
    df.loc[qry, "GarageType"] = "NA"
    return df


def fix_masonry_veneer(df):
    """
    Fixes masonry veneer inconsistencies:
    1. For MasVnrType='None' & area < 10: sets area to 0
    2. For MasVnrType='None' & area >= 10: changes type to 'BrkFace'
    3. For non-None MasVnrType & area = 0: changes type to 'None'
    """
    # Find houses with None type but non-zero area
    none_type_mask = (df["MasVnrType"].astype(str) == "None") & (df["MasVnrArea"] != 0)

    # For small areas (< 10), set area to 0
    small_area_mask = none_type_mask & (df["MasVnrArea"] < 10)
    df.loc[small_area_mask, "MasVnrArea"] = 0

    # For larger areas (>= 10), change type to BrkFace
    large_area_mask = none_type_mask & (df["MasVnrArea"] >= 10)
    df.loc[large_area_mask, "MasVnrType"] = "BrkFace"

    # For zero area, set type to None
    zero_area_mask = (df["MasVnrType"].astype(str) != "None") & (df["MasVnrArea"] == 0)
    df.loc[zero_area_mask, "MasVnrType"] = "None"

    return df


def impute_detchd_garage_2127(df):
    """
    Impute missing garage features for house ID 2127 which has missing:
    - GarageFinish
    - GarageQual
    - GarageCond

    Based on analysis of training data:
    - GarageFinish = 'Unf' (91.5% of detached garages are unfinished)
    - GarageQual = 'TA' (87.9% of detached garages have typical quality)
    - GarageCond = 'TA' (89.1% of detached garages have typical condition)

    Args:
        df: DataFrame with house data
    Returns:
        DataFrame with imputed values for house ID 2127
    """
    mask = (
        (df["Id"] == 2127)
        & (df["GarageFinish"].astype(str) == "NA")
        & (df["GarageQual"].astype(str) == "NA")
        & (df["GarageCond"].astype(str) == "NA")
    )

    if mask.any():
        df.loc[mask, "GarageFinish"] = "Unf"
        df.loc[mask, "GarageQual"] = "TA"
        df.loc[mask, "GarageCond"] = "TA"

    return df


def impute_basement_features(df):
    basement_features_consistent = check_data_consistency(df)[
        "basement_features_consistent"
    ]
    df.loc[
        ~basement_features_consistent & (df["BsmtExposure"].astype(str) == "NA"),
        "BsmtExposure",
    ] = "No"
    df.loc[
        ~basement_features_consistent & (df["BsmtCond"].astype(str) == "NA"), "BsmtCond"
    ] = "TA"

    qual_mask = ~basement_features_consistent & (df["BsmtQual"].astype(str) == "NA")
    df.loc[qual_mask & (df["BsmtFinType1"].astype(str) == "GLQ"), "BsmtQual"] = "Gd"
    df.loc[qual_mask & (df["BsmtFinType1"].astype(str) != "GLQ"), "BsmtQual"] = "TA"

    return df


def check_garage_consistency(df):
    """
    Check consistency of garage-related features.

    Args:
        df: DataFrame with garage features
    """
    # Basic feature consistency check
    garage_features_consistent = (df["has_garage"] == 0) | (
        (df["GarageType"].astype(str) != "NA")
        & (df["GarageFinish"].astype(str) != "NA")
        & (df["GarageQual"].astype(str) != "NA")
        & (df["GarageCond"].astype(str) != "NA")
        & (df["GarageYrBlt"].notna())
        & (df["GarageCars"] > 0)
        & (df["GarageArea"] > 0)
    )

    # Check if garage area is reasonable given car capacity
    # A typical 1-car garage is ~200-280 sq ft, 2-car is ~400-560 sq ft
    garage_area_reasonable = (df["has_garage"] == 0) | (
        (df["GarageArea"] >= df["GarageCars"] * 180)  # min area per car
        & (df["GarageArea"] <= df["GarageCars"] * 600)
    )  # max area per car

    df["garage_features_consistent"] = garage_features_consistent
    df["garage_area_reasonable"] = garage_area_reasonable

    return df


def impute_garage_2127(df):
    """
    Impute missing garage features for house ID 2127.
    Based on cross-tabulation analysis:
    - GarageFinish = 'Unf' (most common for Detchd, 354/387 cases)
    - GarageQual = 'TA' (most common for Detchd, 340/387 cases)
    - GarageCond = 'TA' (most common for Detchd, 345/387 cases)

    Args:
        df: DataFrame with house data
    Returns:
        DataFrame with imputed values for house ID 2127
    """
    mask = (
        (df["Id"] == 2127)
        & (df["GarageFinish"].astype(str) == "NA")
        & (df["GarageQual"].astype(str) == "NA")
        & (df["GarageCond"].astype(str) == "NA")
    )

    if mask.any():
        df.loc[mask, "GarageFinish"] = "Unf"
        df.loc[mask, "GarageQual"] = "TA"
        df.loc[mask, "GarageCond"] = "TA"

    return df


def impute_mszoning(df):
    imputation_data = {1916: "RM", 2217: "RL", 2251: "RM", 2905: "RL"}

    for id, msz in imputation_data.items():
        mask = (df["Id"] == id) & (df["MSZoning"].isna())
        if mask.any():
            df.loc[mask, "MSZoning"] = msz

    return df


def impute_basement_area_333(df):
    mask = (df["Id"] == 333) & (df["BsmtFinType2"].astype(str) == "NA")

    if mask.any():
        df.loc[mask, "BsmtFinType2"] = "Unf"

    return df


def impute_utilities(df):
    imputation_data = {1916: "NoSeWa", 1946: "NoSeWa"}

    for id, ut in imputation_data.items():
        mask = (df["Id"] == id) & (df["Utilities"].isna())
        if mask.any():
            df.loc[mask, "Utilities"] = ut

    return df


def impute_functional(df):
    imputation_data = {2217: "Typ", 2474: "Maj1"}

    for id, func in imputation_data.items():
        mask = (df["Id"] == id) & (df["Functional"].isna())
        if mask.any():
            df.loc[mask, "Functional"] = func

    return df


def impute_exterior_2152(df):
    mask = (df["Id"] == 2152) & (df["Exterior1st"].isna()) & (df["Exterior2nd"].isna())

    if mask.any():
        df.loc[mask, "Exterior1st"] = "VinylSd"
        df.loc[mask, "Exterior2nd"] = "VinylSd"

    return df


def impute_kitchen_qual_1556(df):
    mask = (df["Id"] == 1556) & (df["KitchenQual"].isna())

    if mask.any():
        df.loc[mask, "KitchenQual"] = "TA"

    return df


def impute_sale_type_2490(df):
    mask = (df["Id"] == 2490) & (df["SaleType"].isna())

    if mask.any():
        df.loc[mask, "SaleType"] = "WD"

    return df


def clean_data(df):
    df = fix_electrical_1380(df)
    df = fix_garage_type(df)
    df = fix_masonry_veneer(df)
    df = impute_detchd_garage_2127(df)
    df = impute_basement_features(df)
    df = impute_garage_2127(df)
    df = impute_basement_area_333(df)
    df = impute_mszoning(df)
    df = impute_utilities(df)
    df = impute_functional(df)
    df = impute_exterior_2152(df)
    df = impute_kitchen_qual_1556(df)
    df = impute_sale_type_2490(df)
    df = impute_lot_frontage(df)
    return df
