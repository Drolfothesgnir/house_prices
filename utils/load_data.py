import pandas as pd

def load_train_data():
    df = pd.read_csv("data/train.csv")
    # converting int64 columns to int32
    int_vars = df.select_dtypes(include=["int64"]).columns

    for var in int_vars:
        df[var] = df[var].astype("int32")


    # converting float64 columns to float32
    float_vars = df.select_dtypes(include=["float64"]).columns

    for var in float_vars:
        df[var] = df[var].astype("float32")

    print(df.info())
    return df