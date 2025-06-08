"""
Some utility functions for data processing and loading.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path, separator=";"):
    """Load data and perform basic cleaning"""
    df = pd.read_csv(file_path, sep=separator)
    
    # Drop duplicates
    df_silver = df.drop_duplicates()
    
    # Replace NaN in TARGET(y = sedimentation velocity) with mean
    df_silver = df_silver.copy()  # Make explicit copy
    df_silver['TARGET'] = df_silver['TARGET'].fillna(df_silver['TARGET'].mean())
    
    print(f"Raw data loaded shape: {df.shape}")
    print(f"Cleaned data shape (duplicates dropped/NaNs replaced with mean): {df_silver.shape}")
    df_silver.to_csv("/Users/litani/Documents/myCode/sedimentation-rate/data/interim/df_silver.csv", index=False)
    return df_silver

def split_features_target(df, target_col="TARGET", test_size=0.2, random_state=42):
    """Split data into features and target, then train/test"""
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test