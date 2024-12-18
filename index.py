import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data import load_data
from feature_engineering import augment_by_lot_frontage_missing

train_df = load_data("data/train.csv")
test_df = load_data("data/test.csv")
print(train_df.shape)
print(test_df.shape)
# (1460, 81)

# print(train_df.columns)

# print(train_df.describe())
#
print(train_df.info())
#
na_info = train_df.isna().sum()[train_df.isna().sum() > 0]
print("train df NA counts")
print(na_info)
# for col in na_info.index:
#     print(col)
#     print(train_df[col].unique())
na_info = test_df.isna().sum()[test_df.isna().sum() > 0]
print("test df NA counts")
print(na_info)


# Exploration of single missing Electrical values
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

print(train_df[train_df["Electrical"].isna()].transpose())

# Exploration of LotFrontage missing values

# train_df[train_df["LotFrontage"].isna()]["LotArea"].hist()
# # plt.show()
#
#
# train_df = augment_by_lot_frontage_missing(train_df)
#
# related_vars = ["Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood"]
#
# for var in related_vars:
#     print(f"\n\n\nDistribution of {var} for missing lot frontage")
#     print(train_df[train_df["lot_frontage_missing"]][var].value_counts(normalize=True))
#     print(f"\n\nDistribution of {var} for available lot frontage")
#     print(train_df[~train_df["lot_frontage_missing"]][var].value_counts(normalize=True))

