import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add it to sys.path
sys.path.insert(0, parent_dir)
# First, let's train the model once
from utils.load_data import load_data
from feature_engineering import engineer_features
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = load_data("data/train.csv")
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