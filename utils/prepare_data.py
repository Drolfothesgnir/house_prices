import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_data(df):
    """
    Prepare data for Lasso: encode categoricals and scale features
    """
    # Create copy to avoid modifying original
    data = df.copy()

    # Get dummies for categorical columns
    categorical_columns = data.select_dtypes(include=["category"]).columns
    data = pd.get_dummies(data, columns=categorical_columns)

    # Separate features and target
    X = data.drop(["Id", "SalePrice"], axis=1)
    y = np.log(data["SalePrice"])  # log transform target

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler
