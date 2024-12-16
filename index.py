import pandas as pd
from utils.load_data import load_train_data

train_df = load_train_data()
# print(train_df.shape)
# (1460, 81)

# print(train_df.columns)
# Id -> int
# MSSubClass -> fct
# MSZoning -> fct
# LotFrontage -> float
# LotArea -> float
# Street -> fct
# Alley -> fct
# LotShape -> fct ordered
# LandContour -> fct
# Utilities -> fct ordered
# LotConfig -> fct
# LandSlope -> fct ordered
# Neighborhood -> fct
# Condition1 -> fct
# Condition2 -> fct
# BldgType -> fct
# HouseStyle -> fct
# OverallQual -> fct ordered
# OverallCond -> fct ordered
# YearBuilt -> int
# YearRemodAdd -> int
# RoofStyle -> fct
# RoofMatl -> fct
# Exterior1st -> fct
# Exterior2nd -> fct
# MasVnrType -> fct
# MasVnrArea -> float
# ExterQual -> fct ordered
# ExterCond -> fct ordered
# Foundation -> fct
# BsmtQual -> fct ordered
# BsmtCond -> fct ordered
# BsmtExposure -> fct ordered
# BsmtFinType1 -> fct ordered
# BsmtFinSF1 -> float
# BsmtFinType2 -> fct ordered
# BsmtFinSF2 -> float
# BsmtUnfSF -> float
# TotalBsmtSF -> float
# Heating -> fct
# HeatingQC -> fct ordered
# CentralAir -> fct
# Electrical -> fct
# 1stFlrSF -> float
# 2ndFlrSF -> float
# LowQualFinSF -> float
# GrLivArea -> float
# BsmtFullBath -> int
# BsmtHalfBath -> int
# FullBath -> int
# HalfBath -> int
# BedroomAbvGr -> int
# KitchenAbvGr -> int
# KitchenQual -> fct ordered
# TotRmsAbvGrd -> int
# Functional -> fct ordered
# Fireplaces -> int
# FireplaceQu -> fct ordered
# GarageType -> fct
# GarageYrBlt -> int
# GarageFinish -> fct ordered
# GarageCars -> int
# GarageArea -> float
# GarageQual -> fct ordered
# GarageCond -> fct ordered
# PavedDrive -> fct
# WoodDeckSF -> float
# OpenPorchSF -> float
# EnclosedPorch -> float
# 3SsnPorch -> float
# ScreenPorch -> float
# PoolArea -> float
# PoolQC -> fct ordered
# Fence -> fct ordered
# MiscFeature -> fct
# MiscVal -> float
# MoSold -> fct
# YrSold -> fct
# SaleType -> fct
# SaleCondition -> fct
# SalePrice -> float

# print(train_df.describe())
#
# print(train_df.info())
#
# print(train_df.isna().sum()[train_df.isna().sum() > 0])
# LotFrontage      259  // need to investigate
# Alley           1369
# MasVnrType       872
# MasVnrArea         8
# BsmtQual          37
# BsmtCond          37
# BsmtExposure      38
# BsmtFinType1      37
# BsmtFinType2      38
# Electrical         1 // need to investigate
# FireplaceQu      690
# GarageType        81
# GarageYrBlt       81
# GarageFinish      81
# GarageQual        81
# GarageCond        81
# PoolQC          1453
# Fence           1179
# MiscFeature     1406

# print(train_df["OverallQual"].unique())