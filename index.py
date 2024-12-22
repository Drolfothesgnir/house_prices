import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.load_data import load_data
from feature_engineering import engineer_features

train_df = load_data("data/train.csv")
train_df = engineer_features(train_df)
test_df = load_data("data/test.csv")
test_df = engineer_features(test_df)

print(train_df.shape)
print(test_df.shape)
# (1460, 81)

print(train_df.info())

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

print(train_df["Electrical"].value_counts())

# Given other features are normal, particularly Utilities is AllPub, and construction year is 2006
# I think Electrical value is missing completely at random. I will impute it with SBrkr
train_df["Electrical"] = train_df["Electrical"].fillna("SBrkr")

na_info = train_df.isna().sum()[train_df.isna().sum() > 0]
print("train df NA counts")
print(na_info)

na_info = test_df.isna().sum()[test_df.isna().sum() > 0]
print("test df NA counts")
print(na_info)
# Exploration of missing GarageYrBlt

# there is one suspicious garage in the test set, which has type detached and no other info
# I'll convert it into an NA garage type
qry = (test_df["has_garage"] == 0) & (test_df["GarageType"].astype(str) != "NA")
test_df.loc[qry, "GarageType"] = "NA"


