"""
This is an automated training pipeline with scheduling at 2 AM every Monday.
It checks for new data, runs training jobs, logs models with MLflow,
and deploys the best model if it meets performance criterion (R² > 0.45).
"""
import os
import sys
import logging
from datetime import datetime

# Setup logging for job execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/training_job.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_data_freshness():
    """Check if new data is available for retraining"""
    # In real scenario, check data lake timestamps
    data_path = "data/raw/Data_set_cas_1.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"🚨 Data file not found: {data_path}")
        return False
        
    # Simple freshness check - in production, compare with last training date
    logger.info("✅ Data available for training")
    return True

def run_training_job():
    """Main training job function"""
    logger.info("🔷 Starting automated training job")
    
    try:
        # Step 1: Data validation
        if not check_data_freshness():
            logger.warning("‼️ No fresh data, skipping training")
            return {"status": "skipped", "reason": "no_fresh_data"}
        
        # Step 2: Import training modules
        sys.path.append("src")
        from data_utils import load_and_clean_data, split_features_target
        from preprocessing import create_preprocessing_pipeline
        from models import create_models, train_and_evaluate
        from mlflow_utils import get_data_version, create_comprehensive_metrics, log_model
        
        # Step 3: Execute training pipeline
        logger.info("🔷 Loading and processing data")
        df_silver = load_and_clean_data("data/raw/Data_set_cas_1.csv")
        X_train, X_test, y_train, y_test = split_features_target(df_silver)
        
        logger.info("🔷 Creating preprocessing pipeline")
        preproc = create_preprocessing_pipeline()
        
        logger.info("🔷 Training models")
        pipeline_lasso, pipeline_enc = create_models(preproc)
        
        lasso_model, lasso_pred, lasso_r2 = train_and_evaluate(
            pipeline_lasso, X_train, y_train, X_test, y_test, "Lasso"
        )
        
        enc_model, enc_pred, enc_r2 = train_and_evaluate(
            pipeline_enc, X_train, y_train, X_test, y_test, "ElasticNet"
        )
        
        # Step 4: Model selection and registration
        logger.info("🔷 Registering best model")
        data_version = get_data_version(df_silver)
        
        if lasso_r2 > enc_r2:
            best_model = lasso_model
            best_metrics = create_comprehensive_metrics(lasso_model, X_train, y_train, y_test, lasso_pred)
            model_name = "Lasso"
            best_score = lasso_r2
        else:
            best_model = enc_model
            best_metrics = create_comprehensive_metrics(enc_model, X_train, y_train, y_test, enc_pred)
            model_name = "ElasticNet"
            best_score = enc_r2
        
        # Step 5: Register in Cloudera Model Registry
        model_id = log_model(best_model, model_name, best_metrics, X_test, data_version)
        
        # Step 6: Auto-deploy if performance threshold met
        PERFORMANCE_THRESHOLD = 0.45  # Minimum R² for auto-deployment
        
        if best_score >= PERFORMANCE_THRESHOLD:
            logger.info(f"✅ Model performance ({best_score:.3f}) meets threshold. Auto-deploying...")
            # Note: In real Cloudera, use cml.models.deploy_model()
            logger.info("✅ Model deployed to production endpoint")
            deployment_status = "deployed"
        else:
            logger.warning(f"⚠️ Model performance ({best_score:.3f}) below threshold ({PERFORMANCE_THRESHOLD})")
            deployment_status = "held_for_review"
        
        logger.info(f"✅ Training job completed successfully")
        
        return {
            "status": "success",
            "best_model": model_name,
            "best_score": best_score,
            "data_version": data_version,
            "deployment_status": deployment_status,
            "model_id": model_id
        }
        
    except Exception as e:
        logger.error(f"🚨 Training job failed: {str(e)}")
        
        # Optional: Send alert notification
        # send_slack_alert(f"Training job failed: {str(e)}")
        
        return {"status": "failed", "error": str(e)}

def send_job_summary(results):
    """Send job completion summary"""
    logger.info("🔷 Sending job summary")
    
    if results["status"] == "success":
        message = f"""
        ✅ Training Job Completed
        Model: {results['best_model']}
        Performance: R² = {results['best_score']:.3f}
        Status: {results['deployment_status']}
        Data Version: {results['data_version']}
        """
    else:
        message = f"🚨 Training Job Failed: {results.get('error', 'Unknown error')}"
    
    logger.info(message)
    # In production: send to Slack/email/dashboard

if __name__ == "__main__":
    # This runs as a Cloudera AI Job
    job_results = run_training_job()
    send_job_summary(job_results)
    
    # Exit with appropriate code for job scheduler
    exit_code = 0 if job_results["status"] == "success" else 1
    sys.exit(exit_code)