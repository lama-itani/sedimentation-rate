"""
Preprocessing pipelines.
"""
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def transform_time_feature(data: pd.DataFrame):
    """Convert date columns ("DATE_DEBUT_DELTA" + "DATE_FIN_DELTA") to duration in days (DUREE_JOURS)"""
    X = data.copy()
    X["DATE_DEBUT_DELTA"] = pd.to_datetime(X["DATE_DEBUT_DELTA"], format="%d/%m/%Y %H:%M")
    X["DATE_FIN_DELTA"] = pd.to_datetime(X["DATE_FIN_DELTA"], format="%d/%m/%Y %H:%M")
    X["DUREE_JOURS"] = (X["DATE_FIN_DELTA"] - X["DATE_DEBUT_DELTA"]).dt.days
    return X[["DUREE_JOURS"]]


def create_preprocessing_pipeline():
    """Create the complete preprocessing pipeline"""
    
    # Duration pipeline
    duration_pipeline = make_pipeline(
        FunctionTransformer(transform_time_feature),
        PowerTransformer(method="yeo-johnson", standardize=True)
    )
    
    # Numeric pipeline
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler()
    )
    
    # Categorical pipeline
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    
    # Column selectors
    date_cols = ["DATE_DEBUT_DELTA", "DATE_FIN_DELTA"]
    num_cols = ["TENEUR", "indicateur_1", "VALEUR", "CHIMIE3"]
    cat_cols = ["TRIG"]
    
    # Complete preprocessor
    preproc = ColumnTransformer([
        ("duration_preprocessor", duration_pipeline, date_cols),
        ("numeric_preprocessor", num_pipeline, num_cols),
        ("categorical_preprocessor", cat_pipeline, cat_cols)
    ])
    
    print("Preprocessing pipeline created")
    return preproc