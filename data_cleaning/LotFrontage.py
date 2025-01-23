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

# # Function to get absolute path for any relative path
# def get_abs_path(relative_path):
#     return str(Path(relative_path).absolute())
    
# First, let's train the model once
from utils.load_data import load_data
from feature_engineering import engineer_features
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = load_data("../data/train.csv")
df = engineer_features(df)
df = df[~df["lot_frontage_missing"] == 1] 

def train_imputer(train_df):
    """Train the RF model for imputation."""
    features = [
        'LotArea', 'BldgType', 'Neighborhood',
        'LotShape', 'LotConfig', 'LandContour',
        'YearBuilt', 'OverallQual'
    ]
    
    # Create initial dataframe
    X = train_df[features].copy()
    
    # Transform LotArea BEFORE getting feature names
    X['log_LotArea'] = np.log(X['LotArea'])
    X = X.drop('LotArea', axis=1)
    
    # Get dummies after the transformation
    cat_features = ['BldgType', 'Neighborhood', 'LotShape', 'LotConfig', 'LandContour']
    X = pd.get_dummies(X, columns=cat_features, drop_first=False)
    
    # Now feature_names will include 'log_LotArea' instead of 'LotArea'
    feature_names = X.columns.tolist()
    
    y = np.log(train_df['LotFrontage'])
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        min_samples_split=8,
        random_state=42
    )
    rf.fit(X, y)
    
    return rf, feature_names

# Train and get the model (run once)
rf_model, features = train_imputer(df)

# In LotFrontage.py first let's modify the function:
def impute_lot_frontage(df_to_impute, rf_model=rf_model, feature_names=features):
    df = df_to_impute.copy()
    
    # Find rows with missing LotFrontage
    missing_mask = df['LotFrontage'].isna()
    if not missing_mask.any():
        return df
    
    features = [
        'LotArea', 'BldgType', 'Neighborhood',
        'LotShape', 'LotConfig', 'LandContour',
        'YearBuilt', 'OverallQual'
    ]
    cat_features = ['BldgType', 'Neighborhood', 'LotShape', 'LotConfig', 'LandContour']
    
    # Get features for missing values
    df_subset = df[missing_mask][features].copy()
    
    # Transform LotArea to log_LotArea BEFORE creating dummies
    df_subset['log_LotArea'] = np.log(df_subset['LotArea'])
    df_subset = df_subset.drop('LotArea', axis=1)
    
    # Now create dummies
    X_impute = pd.get_dummies(df_subset, columns=cat_features, drop_first=False)
    
    # Align columns
    X_impute = X_impute.reindex(columns=feature_names, fill_value=0)
    
    imputed_values = np.exp(rf_model.predict(X_impute))
    df.loc[missing_mask, 'LotFrontage'] = imputed_values.astype("float32")
    
    return df