def clean_electrical(df):
    # imputation of single Electrical value 
    df["Electrical"] = train_df["Electrical"].fillna("SBrkr")
    return df

def clean_data(df):
    df = clean_electrical(df)
    return df