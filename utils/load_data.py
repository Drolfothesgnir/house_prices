# import os
# import sys
# from pathlib import Path

# # Get the notebook's directory
# notebook_dir = os.getcwd()

# # Add the notebook directory and its parent to Python path
# sys.path.append(notebook_dir)
# sys.path.append(str(Path(notebook_dir).parent))

# # Function to add all subdirectories to Python path
# def add_subdirs_to_path():
#     for root, dirs, files in os.walk(notebook_dir):
#         for dir_name in dirs:
#             full_path = os.path.join(root, dir_name)
#             if full_path not in sys.path:
#                 sys.path.append(full_path)

# # Add all subdirectories
# add_subdirs_to_path()

# from .path_utils import resolve_data_file

# Function to get absolute path for any relative path
# def get_abs_path(relative_path):
#     return str(Path(relative_path).absolute())
import pandas as pd
from utils.path_utils import get_data_path

def load_data(pathname):
    # pathname = resolve_data_file(filename)
    df = pd.read_csv(get_data_path(pathname))
    # converting int64 columns to int32
    int_vars = df.select_dtypes(include=["int64"]).columns

    for var in int_vars:
        df[var] = df[var].astype("int32")

    # converting float64 columns to float32
    float_vars = df.select_dtypes(include=["float64"]).columns

    for var in float_vars:
        df[var] = df[var].astype("float32")

    na_values = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        # "MasVnrType"
    ]

    for var in na_values:
        df[var] = df[var].fillna("NA")

    # MasVnrType has special NA value - None
    df["MasVnrType"] = df["MasVnrType"].fillna("None")

    # filling numerical NA values
    num_vars = [
        "MasVnrArea",
        "GarageCars",
        "GarageArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "TotalBsmtSF"
    ]
    for var in num_vars:
        df[var] = df[var].fillna(0)

    # filling missing garage year build values with house date built
    df['GarageYrBlt'] = df["GarageYrBlt"].fillna(df["YearBuilt"]).astype('int32')

    # converting ordinal categories
    ordered_cats = {
        "LotShape": pd.CategoricalDtype(
            categories=['Reg', 'IR1', 'IR2', 'IR3'],
            ordered=True

        ),
        "Utilities": pd.CategoricalDtype(
            categories=['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
            ordered=True
        ),
        "LandSlope": pd.CategoricalDtype(
            categories=["Gtl", "Mod", "Sev"],
            ordered=True
        ),
        "OverallQual": pd.CategoricalDtype(
            categories=list(range(1, 11)),
            ordered=True
        ),
        "OverallCond": pd.CategoricalDtype(
            categories=list(range(1, 11)),
            ordered=True
        ),
        "ExterQual": pd.CategoricalDtype(
            categories=["Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "ExterCond": pd.CategoricalDtype(
            categories=["Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "BsmtQual": pd.CategoricalDtype(
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "BsmtCond": pd.CategoricalDtype(
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "BsmtExposure": pd.CategoricalDtype(
            categories=["NA", "No", "Mn", "Av", "Gd"],
            ordered=True
        ),
        "BsmtFinType1": pd.CategoricalDtype(
            categories=["GLQ",
                        "ALQ",
                        "BLQ",
                        "Rec",
                        "LwQ",
                        "Unf",
                        "NA"],
            ordered=True
        ),
        "BsmtFinType2": pd.CategoricalDtype(
            categories=["GLQ",
                        "ALQ",
                        "BLQ",
                        "Rec",
                        "LwQ",
                        "Unf",
                        "NA"],
            ordered=True
        ),
        "HeatingQC": pd.CategoricalDtype(
            categories=["Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "KitchenQual": pd.CategoricalDtype(
            categories=["Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "FireplaceQu": pd.CategoricalDtype(
            categories=["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "Functional": pd.CategoricalDtype(
            categories=[
                "Typ",
                "Min1",
                "Min2",
                "Mod",
                "Maj1",
                "Maj2",
                "Sev",
                "Sal"
            ],
            ordered=True
        ),
        "GarageFinish": pd.CategoricalDtype(
            categories=["NA", "Unf", "RFn", "Fin"],
            ordered=True
        ),
        "GarageQual": pd.CategoricalDtype(
            categories=[
                "Ex",
                "Gd",
                "TA",
                "Fa",
                "Po",
                "NA"
            ],
            ordered=True
        ),
        "GarageCond": pd.CategoricalDtype(
            categories=[
                "Ex",
                "Gd",
                "TA",
                "Fa",
                "Po",
                "NA"
            ],
            ordered=True
        ),
        "PoolQC": pd.CategoricalDtype(
            categories=["NA", "Fa", "TA", "Gd", "Ex"],
            ordered=True
        ),
        "Fence": pd.CategoricalDtype(
            categories=["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
            ordered=True
        )
    }

    for (var, dtype) in ordered_cats.items():
        df[var] = df[var].astype(dtype)

    # converting nominal categories
    cat_vars = df.select_dtypes(include=["object"]).columns
    for var in cat_vars:
        df[var] = df[var].astype("category")

    # converting MSSubClass into categorical type
    df["MSSubClass"] = df["MSSubClass"].astype("category")

    return df
