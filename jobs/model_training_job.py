"""
This job is for model training.
It handles data loading, preprocessing, model training, and evaluation.
"""
import sys
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

class ModelTrainingJob:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_paths()
        
    def _setup_paths(self):
        """Add source path for imports"""
        src_path = self.config.get("src_path", "src")
        if src_path not in sys.path:
            sys.path.append(src_path)
    
    def load_and_preprocess_data(self) -> Tuple[Any, Any, Any, Any, str]:
        """Load and preprocess data for training"""
        try:
            from data_utils import load_and_clean_data, split_features_target
            from preprocessing import create_preprocessing_pipeline
            from mlflow_utils import get_data_version
            
            self.logger.info("🔷 Loading and processing data")
            
            data_path = self.config.get("data_path", "..data/raw/Data_set_cas_1.csv")
            df_silver = load_and_clean_data(data_path)
            
            X_train, X_test, y_train, y_test = split_features_target(df_silver)
            data_version = get_data_version(df_silver)
            
            self.logger.info(f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            
            return X_train, X_test, y_train, y_test, data_version
            
        except Exception as e:
            self.logger.error(f"🚨 Data loading failed: {str(e)}")
            raise
    
    def train_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train and evaluate all models"""
        try:
            from preprocessing import create_preprocessing_pipeline
            from models import create_models, train_and_evaluate
            from mlflow_utils import create_comprehensive_metrics
            
            self.logger.info("🔷 Creating preprocessing pipeline")
            preproc = create_preprocessing_pipeline()
            
            self.logger.info("🔷 Training models")
            pipeline_lasso, pipeline_enc = create_models(preproc)
            
            # Train Lasso
            lasso_model, lasso_pred, lasso_r2 = train_and_evaluate(
                pipeline_lasso, X_train, y_train, X_test, y_test, "Lasso"
            )
            
            # Train ElasticNet
            enc_model, enc_pred, enc_r2 = train_and_evaluate(
                pipeline_enc, X_train, y_train, X_test, y_test, "ElasticNet"
            )
            
            # Create comprehensive metrics
            lasso_metrics = create_comprehensive_metrics(
                lasso_model, X_train, y_train, y_test, lasso_pred
            )
            
            enc_metrics = create_comprehensive_metrics(
                enc_model, X_train, y_train, y_test, enc_pred
            )
            
            # Determine best model
            if lasso_r2 > enc_r2:
                best_model_name = "Lasso"
                best_model = lasso_model
                best_metrics = lasso_metrics
                best_score = lasso_r2
            else:
                best_model_name = "ElasticNet"
                best_model = enc_model
                best_metrics = enc_metrics
                best_score = enc_r2
            
            self.logger.info(f"🏆 Best model: {best_model_name} (R² = {best_score:.4f})")
            
            return {
                "models": {
                    "Lasso": {
                        "pipeline": lasso_model,
                        "metrics": lasso_metrics,
                        "score": lasso_r2
                    },
                    "ElasticNet": {
                        "pipeline": enc_model,
                        "metrics": enc_metrics,
                        "score": enc_r2
                    }
                },
                "best_model": {
                    "name": best_model_name,
                    "pipeline": best_model,
                    "metrics": best_metrics,
                    "score": best_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"🚨 Model training failed: {str(e)}")
            raise
    
    def run_training(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        self.logger.info("🔷 Starting model training job")
        
        try:
            # Load and preprocess data
            X_train, X_test, y_train, y_test, data_version = self.load_and_preprocess_data()
            
            # Train models
            training_results = self.train_models(X_train, X_test, y_train, y_test)
            
            # Add metadata
            training_results.update({
                "data_version": data_version,
                "test_data_shape": X_test.shape,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
            
            self.logger.info("✅ Model training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"🚨 Training job failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

if __name__ == "__main__":
    # Configuration
    config = {
        "src_path": "src",
        "data_path": "data/raw/Data_set_cas_1.csv"
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run training
    trainer = ModelTrainingJob(config)
    result = trainer.run_training()
    
    # Exit with appropriate code
    exit_code = 0 if result.get("status") == "success" else 1
    exit(exit_code)