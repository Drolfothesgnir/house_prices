import pandas as pd

train_df = pd.read_csv("data/train.csv")
print(train_df.shape)
# (1460, 81)

print(train_df.columns)
print(train_df.describe())

print(train_df.isna().sum()[train_df.isna().sum() > 0])
# LotFrontage      259
# Alley           1369
# MasVnrType       872
# MasVnrArea         8
# BsmtQual          37
# BsmtCond          37
# BsmtExposure      38
# BsmtFinType1      37
# BsmtFinType2      38
# Electrical         1
# FireplaceQu      690
# GarageType        81
# GarageYrBlt       81
# GarageFinish      81
# GarageQual        81
# GarageCond        81
# PoolQC          1453
# Fence           1179
# MiscFeature     1406