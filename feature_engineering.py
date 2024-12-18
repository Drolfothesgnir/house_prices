def augment_by_lot_frontage_missing(df):
    df["lot_frontage_missing"] = df["LotFrontage"].isna()
    return df