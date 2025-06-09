"""
MLflow utilities aligned with MLOps best practices: 
data versioning, dataset and env tracking, 
automated best model selection, metrics (R2, RMSE) 
"""
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import r2_score
import platform
import pandas as pd
import sklearn
from pathlib import Path

def setup_mlflow_tracking():
    """Configure MLflow project tracking"""
    # Get project root
    project_root = Path(__file__).parent.parent
    mlflow_path = project_root / "mlruns"
    # Set tracking URI to project root
    mlflow.set_tracking_uri(f"file://{mlflow_path.absolute()}")
    print(f"🔷 MLflow tracking: {mlflow_path}")

# Call setup when module is imported
setup_mlflow_tracking()

def get_data_version(df):
    """Create data version hash for tracking"""
    data_string = pd.util.hash_pandas_object(df).sum()
    return str(data_string)[:10]


def get_environment_info():
    """Track environment for reproducibility"""
    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.system()
    }


def log_model(pipeline, model_name, metrics, X_test, data_version, experiment_name="ml-pred-sediment"):
    """Log model after traning w/ MLOps best practices"""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_name}_Model"):
        
        # 1. Log model parameters
        model_step = pipeline.named_steps[pipeline.steps[-1][0]]
        if hasattr(model_step, 'alpha_'):
            mlflow.log_param(f"{model_name.lower()}_alpha", model_step.alpha_)
        if hasattr(model_step, 'l1_ratio_'):
            mlflow.log_param(f"{model_name.lower()}_l1_ratio", model_step.l1_ratio_)
        
        # 2. Log all metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # 3. Log data versioning
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("data_shape", f"{X_test.shape[0] + len(X_test)*4}")  # Approximate total samples
        
        # 4. Log environment info
        env_info = get_environment_info()
        for key, value in env_info.items():
            mlflow.log_param(key, value)
        
        # 5. Prepare input example (fix schema issues)
        input_example = X_test.iloc[1:2].copy()
        integer_cols = input_example.select_dtypes(include=['int64']).columns
        input_example[integer_cols] = input_example[integer_cols].astype('float64')
        
        # 6. Create signature and log model
        signature = infer_signature(input_example, pipeline.predict(input_example))
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"{model_name.lower()}_pipeline",
            signature=signature,
            input_example=input_example,
            registered_model_name=f"sk-learn-{model_name}-model"
        )
        
        print(f"✅ {model_name} logged (R² = {metrics['test_r2']:.4f})")
        return mlflow.active_run().info.run_id


def compare_and_log_models(models_dict, data_version, experiment_name="ml-pred-sediment"):
    """Compare models and log the best one with comparison metrics"""
    
    # Find best model
    best_model_name = max(models_dict.keys(), key=lambda k: models_dict[k]['metrics']['test_r2'])
    best_r2 = models_dict[best_model_name]['metrics']['test_r2']
    
    # Log comparison run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="Model_Comparison"):
        
        # Log all model performances
        for name, model_info in models_dict.items():
            mlflow.log_metric(f"{name.lower()}_test_r2", model_info['metrics']['test_r2'])
            mlflow.log_metric(f"{name.lower()}_train_r2", model_info['metrics']['train_r2'])
        
        # Log best model info
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_test_r2", best_r2)
        mlflow.log_param("data_version", data_version)
        
        # Log environment
        env_info = get_environment_info()
        for key, value in env_info.items():
            mlflow.log_param(key, value)
        
        print(f"🔷 Best model: {best_model_name} (R² = {best_r2:.4f})")
        return best_model_name


def create_comprehensive_metrics(pipeline, X_train, y_train, y_test, y_test_pred):
    """Create comprehensive metrics from already-trained model"""
    
    # Get train predictions (model already trained)
    y_train_pred = pipeline.predict(X_train)
    
    # shensive metrics using existing predictions
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),  # Use existing predictions
        'overfit_score': r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred)
    }
    
    return metrics