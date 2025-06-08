"""
Functions to create and train models.
"""
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


def create_models(preprocessor):
    """Create Lasso and ElasticNet pipelines"""
    
    # Define models
    model_lasso = LassoCV(cv=5, random_state=42)
    model_enc = ElasticNetCV(cv=5, random_state=42)
    
    # Create full pipelines
    pipeline_lasso = make_pipeline(preprocessor, model_lasso)
    pipeline_enc = make_pipeline(preprocessor, model_enc)
    
    print("Model pipelines created")
    return pipeline_lasso, pipeline_enc


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, model_name):
    """Train model and return results"""
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate R²
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} R² score: {r2:.4f}")
    return pipeline, y_pred, r2